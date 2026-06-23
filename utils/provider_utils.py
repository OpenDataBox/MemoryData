"""Shared provider utilities for multi-model LLM/embedding client creation.

Centralises provider normalisation, env-var resolution, LangChain client
factories, and the ApproximateTokenizer fallback used across method modules.
"""
import os
from pathlib import Path


class ApproximateTokenizer:
    """Character-level fallback tokenizer for non-GPT providers.

    Used when tiktoken cannot load an encoding (e.g. non-OpenAI model names).
    Token counts are approximate (1 char ≈ 1 token) but sufficient for
    context-length guards that only need rough estimates.
    """

    def encode(self, text, disallowed_special=()):
        return list(text or "")

    def decode(self, tokens):
        return "".join(tokens)


class HuggingFaceTokenizerAdapter:
    """Thin adapter that makes a Hugging Face tokenizer look like tiktoken."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text, disallowed_special=(), add_special_tokens=False):
        del disallowed_special
        return self._tokenizer.encode(text or "", add_special_tokens=add_special_tokens)

    def decode(self, tokens):
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


def load_local_hf_tokenizer(candidate):
    """Load a local Hugging Face tokenizer when a matching model path exists."""
    normalized_candidate = str(candidate or "").strip()
    if not normalized_candidate:
        return None

    resolved_candidates = []
    candidate_path = Path(normalized_candidate).expanduser()
    if candidate_path.exists():
        resolved_candidates.append(candidate_path)

    configured_models_root = os.getenv("LOCAL_MODELS_ROOT")
    candidate_roots = []
    if configured_models_root:
        candidate_roots.append(Path(configured_models_root).expanduser())

    repo_models_root = Path(__file__).resolve().parents[1] / "models"
    candidate_roots.append(repo_models_root)
    candidate_roots.append(Path("/data/models"))

    for candidate_root in candidate_roots:
        models_root_candidate = candidate_root / normalized_candidate
        if models_root_candidate.exists():
            resolved_candidates.append(models_root_candidate)

    tried_paths = set()
    for resolved_path in resolved_candidates:
        resolved_str = str(resolved_path)
        if resolved_str in tried_paths:
            continue
        tried_paths.add(resolved_str)
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                resolved_str,
                local_files_only=True,
                trust_remote_code=True,
            )
            return HuggingFaceTokenizerAdapter(tokenizer)
        except Exception:
            continue

    return None

_PROVIDER_ALIASES = {
    "azure": "azure_openai",
    "azure_openai": "azure_openai",
    "openai": "openai",
    "openai-compatible": "openai_compatible",
    "openai_compatible": "openai_compatible",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "google": "gemini",
    "google_genai": "gemini",
    "gemini": "gemini",
    "deepseek": "openai_compatible",
}


def normalize_provider(provider_name):
    """Normalize provider name aliases to a stable internal identifier."""
    normalized = (provider_name or "").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def resolve_env_value(explicit_env_name=None, fallback_env_names=None):
    """Resolve a config value from an explicit or fallback environment variable."""
    if explicit_env_name:
        return os.environ.get(explicit_env_name)
    for env_name in fallback_env_names or []:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def resolve_base_url(base_url=None, base_url_env=None):
    """Resolve an OpenAI-compatible base URL from config or environment."""
    if base_url:
        return base_url
    if base_url_env:
        return os.environ.get(base_url_env)
    return os.environ.get("OPENAI_BASE_URL")


def use_azure_openai(provider, azure_endpoint=None, base_url=None, base_url_env=None):
    """Return True when the resolved configuration should use Azure OpenAI."""
    return provider == "azure_openai" or (
        provider == "openai"
        and (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"))
        and not base_url
        and not base_url_env
    )


def create_chat_llm(model, temperature=0.7, max_tokens=None, provider="openai",
                    base_url=None, base_url_env=None, api_key_env=None,
                    azure_endpoint=None, azure_api_version=None):
    """Create a LangChain chat LLM for the given provider.

    Supports: openai, azure_openai, openai_compatible, gemini.
    max_tokens=None leaves the parameter unset (uses provider default).
    """
    provider = normalize_provider(provider)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {
            "model": model,
            "temperature": temperature,
            "google_api_key": resolve_env_value(api_key_env, ["GOOGLE_API_KEY"]),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**kwargs)

    if use_azure_openai(provider, azure_endpoint=azure_endpoint,
                        base_url=base_url, base_url_env=base_url_env):
        from langchain_openai import AzureChatOpenAI
        kwargs = {
            "azure_deployment": model,
            "api_version": azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
            "azure_endpoint": azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            "api_key": resolve_env_value(api_key_env, ["AZURE_OPENAI_API_KEY"]),
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return AzureChatOpenAI(**kwargs)

    resolved_base_url = resolve_base_url(base_url, base_url_env)
    api_key = resolve_env_value(api_key_env, ["OPENAI_API_KEY"])
    if provider == "openai_compatible" and not resolved_base_url:
        raise RuntimeError("OpenAI-compatible models require 'base_url' or 'base_url_env'.")

    from langchain_openai import ChatOpenAI
    kwargs = {"model": model, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def create_embedding_model(model=None, provider="openai", base_url=None, base_url_env=None,
                           api_key_env=None, azure_endpoint=None, azure_api_version=None,
                           azure_deployment=None):
    """Create a LangChain embedding model for the given provider.

    Supports: openai, azure_openai, openai_compatible.
    azure_deployment defaults to the model name when using Azure.
    """
    provider = normalize_provider(provider)
    embedding_model_name = model or "text-embedding-3-small"

    if use_azure_openai(provider, azure_endpoint=azure_endpoint,
                        base_url=base_url, base_url_env=base_url_env):
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            model=embedding_model_name,
            azure_deployment=azure_deployment or embedding_model_name,
            api_version=azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=resolve_env_value(api_key_env, ["AZURE_OPENAI_API_KEY"]),
        )

    resolved_base_url = resolve_base_url(base_url, base_url_env)
    api_key = resolve_env_value(api_key_env, ["OPENAI_API_KEY"])
    if provider == "openai_compatible" and not resolved_base_url:
        raise RuntimeError("OpenAI-compatible embedding models require 'base_url' or 'base_url_env'.")

    from langchain_openai import OpenAIEmbeddings
    kwargs = {"model": embedding_model_name}
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)
