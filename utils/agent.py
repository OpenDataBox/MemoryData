import os
import sys
import json
import shutil
import importlib
import importlib.machinery
import types
import torch
import tiktoken
import httpx
from openai import OpenAI
from benchmark.memoryagentbench.prompts.benchmark_templates import get_template
from benchmark.memoryagentbench.loader import format_chat
import re
import time
from utils.locomo_utils import (
    dedupe_preserve_order,
    parse_locomo_metadata,
    parse_locomo_source_ids,
    strip_locomo_metadata,
)
from utils.provider_utils import (
    normalize_provider as _normalize_provider_fn,
    ApproximateTokenizer,
    load_local_hf_tokenizer,
)
from utils.artifact_paths import (
    build_memorag_cache_dir,
    resolve_artifact_root,
    resolve_results_artifact_path,
)

from langchain_core.documents import Document
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig

for extra_path in [
    "./methods/letta/source",
    "./methods/mem0/source",
    "./methods/cognee/source",
    "./methods/memochat/source",
    "./methods/memoryos/source",
    "./methods/simplemem/source",
    "./methods/lightmem/source",
    "./methods/a_mem/source",
]:
    absolute_extra_path = os.path.abspath(extra_path)
    if absolute_extra_path not in sys.path:
        sys.path.insert(0, absolute_extra_path)


def _ensure_namespace_package(package_name, package_root):
    """Expose a local source tree as a package namespace without touching sys.path."""
    abs_root = os.path.abspath(package_root)
    package = sys.modules.get(package_name)

    if package is None:
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [abs_root]
        package.__file__ = os.path.join(abs_root, "__init__.py")
        spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        spec.submodule_search_locations = [abs_root]
        package.__spec__ = spec
        sys.modules[package_name] = package
        return package

    package_path = list(getattr(package, "__path__", []))
    if abs_root not in package_path:
        package_path.insert(0, abs_root)
        package.__path__ = package_path
        if getattr(package, "__spec__", None) is not None:
            package.__spec__.submodule_search_locations = package_path
    return package


def _import_from_vendor_namespace(module_name, package_name, package_root):
    """Import a vendored module from a local namespace package root."""
    _ensure_namespace_package(package_name, package_root)
    return importlib.import_module(module_name)


def _sanitize_storage_identifier(name, suffix=""):
    """Make dataset-derived identifiers safe for local storage backends."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    safe_name = re.sub(r"_+", "_", safe_name).strip("_.-")
    if not safe_name:
        safe_name = "benchmark"
    if suffix:
        safe_name = f"{safe_name}_{suffix}"
    return safe_name


def _sanitize_graph_group_id(name):
    """Make dataset-derived identifiers safe for graph backends like Graphiti/Neo4j."""
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    safe_name = re.sub(r"_+", "_", safe_name).strip("_-")
    return safe_name or "benchmark"


def _build_zep_local_namespace(context_id, sub_dataset, namespace_prefix=""):
    """Build a stable, Graphiti-safe namespace for zep_local state."""
    namespace = _sanitize_graph_group_id(f"context_{context_id}_{sub_dataset}")
    if namespace_prefix:
        namespace = _sanitize_graph_group_id(f"{namespace_prefix}_{namespace}")
    return namespace



class AgentWrapper:
    """
    A wrapper class for different types of memory agents including:
    - Long context agents (OpenAI-compatible, Claude, Gemini)
    - Letta agents
    - Mem0 agents
    - Cognee agents
    - RAG agents (various implementations)
    """

    def __init__(self, agent_config, dataset_config, load_agent_from):
        """
        Initialize the agent wrapper with specified configuration.

        Args:
            agent_config: Configuration dictionary for the agent
            dataset_config: Configuration dictionary for the dataset
            load_agent_from: Optional path to load existing agent state from
        """
        # Basic agent configuration
        self.agent_config = agent_config
        self.dataset_config = dataset_config
        self.agent_name = agent_config['agent_name']
        self.sub_dataset = dataset_config['sub_dataset']
        self.context_max_length = dataset_config['context_max_length']
        self.dataset = dataset_config['dataset']

        # Output and storage configuration
        self.output_dir = agent_config['output_dir']
        self.artifact_root = resolve_artifact_root(agent_config=agent_config)
        self.agent_save_to_folder = load_agent_from
        self.record_llm_io = bool(agent_config.get("record_llm_io", False))
        self._last_llm_trace = None
        self._current_query_id = None
        self._current_context_id = None
        self._current_eval_metadata = None
        self._retrieval_debug_cleanup_roots = set()
        self.backfill_longmemeval_recall_debug = bool(
            agent_config.get("_backfill_longmemeval_recall_debug", False)
        )

        self.memorag_cuda_visible_devices = agent_config.get("memorag_cuda_visible_devices")
        if self.memorag_cuda_visible_devices and "memo_rag" in agent_config["agent_name"].lower():
            if isinstance(self.memorag_cuda_visible_devices, (list, tuple)):
                visible_devices = ",".join(str(device) for device in self.memorag_cuda_visible_devices)
            else:
                visible_devices = str(self.memorag_cuda_visible_devices)
            # MemoRAG is launched as its own Python process in batch/smoke runs, so
            # setting CUDA_VISIBLE_DEVICES here is enough to constrain its model load.
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        # Context and token limits
        self.input_length_limit = (agent_config['input_length_limit'] -
                                 agent_config['buffer_length'] -
                                 dataset_config['generation_max_length'])

        # Model configuration
        self.model = agent_config['model']
        self.max_tokens = dataset_config['generation_max_length']
        self.temperature = agent_config.get('temperature', 0.0)
        self.model_context_window = agent_config.get('model_context_window')
        self.model_provider = self._normalize_provider(
            agent_config.get('provider') or self._infer_model_provider(self.model, agent_config)
        )
        self.api_key = agent_config.get('api_key')
        self.api_key_env = agent_config.get('api_key_env')
        self.base_url = agent_config.get('base_url')
        self.base_url_env = agent_config.get('base_url_env')
        self.azure_endpoint = agent_config.get('azure_endpoint')
        self.azure_api_version = agent_config.get('azure_api_version')
        self.tokenizer_model = agent_config.get('tokenizer_model')
        self.tokenizer_encoding = agent_config.get('tokenizer_encoding')
        self.embedding_model = agent_config.get('embedding_model')
        configured_embedding_provider = agent_config.get('embedding_provider')
        self.embedding_provider = (
            self._normalize_provider(configured_embedding_provider)
            if configured_embedding_provider
            else None
        )
        self.embedding_api_key = agent_config.get('embedding_api_key')
        self.embedding_api_key_env = agent_config.get('embedding_api_key_env')
        self.embedding_base_url = agent_config.get('embedding_base_url')
        self.embedding_base_url_env = agent_config.get('embedding_base_url_env')
        self.embedding_azure_endpoint = agent_config.get('embedding_azure_endpoint')
        self.embedding_azure_api_version = agent_config.get('embedding_azure_api_version')
        self.graph_rag_internal_max_tokens = agent_config.get('graph_rag_internal_max_tokens')

        self.tokenizer = self._initialize_tokenizer()

        # Initialize agent based on type
        self._initialize_agent_by_type(agent_config, dataset_config)

    def _initialize_agent_by_type(self, agent_config, dataset_config):
        """Initialize the specific agent type based on agent name."""

        if 'Long_context_agent' in self.agent_name:
            self._initialize_long_context_agent()
        elif self._is_agent_type("letta"):
            self._initialize_letta_agent(agent_config, dataset_config)
        elif self._is_agent_type("mem0"):
            self._initialize_mem0_agent(agent_config, dataset_config)
        elif self._is_agent_type("simplemem"):
            self._initialize_simplemem_agent(agent_config, dataset_config)
        elif self._is_agent_type("lightmem"):
            self._initialize_lightmem_agent(agent_config, dataset_config)
        elif self._is_agent_type("a_mem"):
            self._initialize_a_mem_agent(agent_config, dataset_config)
        elif self._is_agent_type("memtree"):
            self._initialize_memtree_agent(agent_config, dataset_config)
        elif self._is_agent_type("everos"):
            self._initialize_everos_agent(agent_config, dataset_config)
        elif self._is_agent_type("memochat"):
            self._initialize_memochat_agent(agent_config, dataset_config)
        elif self._is_agent_type("memoryos"):
            self._initialize_memoryos_agent(agent_config, dataset_config)
        elif self._is_agent_type("cognee"):
            self._initialize_cognee_agent(agent_config, dataset_config)
        elif self._is_agent_type("zep_local"):
            self._initialize_zep_local_agent(agent_config)
        elif self._is_agent_type("zep"):
            self._initialize_zep_agent(agent_config)
        elif self._is_agent_type("memagent"):
            self._initialize_memagent(agent_config)
        elif self._is_agent_type("MemOS"):
            self._initialize_memos_agent(agent_config, dataset_config)
        elif self._is_agent_type("rag"):
            self._initialize_rag_agent(agent_config, dataset_config)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _build_memorag_cache_dir(self, context_id):
        return build_memorag_cache_dir(
            self.agent_config,
            self.dataset_config,
            self.chunk_size,
            context_id,
            self.artifact_root,
        )

    def _prepare_memorag_context(self, context, token_counter=None):
        max_context_tokens = self.agent_config.get("memorag_max_context_tokens")
        reserve_tokens = int(self.agent_config.get("memorag_context_token_reserve", 0) or 0)
        truncation_strategy = self.agent_config.get("memorag_truncation_strategy", "head_tail")

        if token_counter is not None:
            context_tokens = token_counter.encode(context, add_special_tokens=False)
            decode_tokens = lambda tokens: token_counter.decode(tokens, skip_special_tokens=True)
        else:
            encoding_name = self.tokenizer_encoding or "cl100k_base"
            encoding = tiktoken.get_encoding(encoding_name)
            context_tokens = encoding.encode(context, disallowed_special=())
            decode_tokens = lambda tokens: encoding.decode(tokens)

        if not max_context_tokens or len(context_tokens) <= max_context_tokens:
            return context

        effective_max_tokens = max_context_tokens
        if reserve_tokens > 0 and max_context_tokens > reserve_tokens:
            effective_max_tokens = max_context_tokens - reserve_tokens

        if truncation_strategy == "head":
            kept_tokens = context_tokens[:effective_max_tokens]
        elif truncation_strategy == "tail":
            kept_tokens = context_tokens[-effective_max_tokens:]
        else:
            head_tokens = effective_max_tokens // 2
            tail_tokens = effective_max_tokens - head_tokens
            kept_tokens = context_tokens[:head_tokens] + context_tokens[-tail_tokens:]

        truncated_context = decode_tokens(kept_tokens)
        print(
            "MemoRAG context truncated "
            f"from {len(context_tokens)} to {len(kept_tokens)} tokens "
            f"using strategy={truncation_strategy} with reserve={reserve_tokens}."
        )
        return truncated_context

    def _is_agent_type(self, agent_type):
        """Check if the current agent is of a specific type."""
        return agent_type in self.agent_name

    def _normalize_provider(self, provider_name):
        """Normalize provider aliases to a stable internal name."""
        return _normalize_provider_fn(provider_name)

    def _infer_model_provider(self, model_name, agent_config):
        """Infer a provider from the model name while preserving old configs."""
        configured_base_url = agent_config.get("base_url") or agent_config.get("base_url_env")
        if configured_base_url:
            return "openai_compatible"

        normalized_model_name = (model_name or "").strip().lower()
        if "claude" in normalized_model_name:
            return "anthropic"
        if "gemini" in normalized_model_name:
            return "gemini"
        if normalized_model_name.startswith(("gpt", "o1", "o3", "o4")):
            return "openai"
        if any(
            provider_hint in normalized_model_name
            for provider_hint in ["qwen", "deepseek", "mistral", "moonshot", "glm", "llama", "openrouter"]
        ):
            return "openai_compatible"
        return "openai"

    def _normalize_cognee_openai_compatible_model(self, model_name):
        """Normalize OpenAI-compatible model names for Cognee's LiteLLM-backed OpenAI adapter."""
        normalized_model_name = str(model_name or "").strip()
        if not normalized_model_name:
            return normalized_model_name

        explicit_provider_prefixes = {
            "openai",
            "azure",
            "anthropic",
            "gemini",
            "mistral",
            "bedrock",
            "ollama",
            "openrouter",
            "hosted_vllm",
            "llama_cpp",
        }
        if "/" in normalized_model_name:
            provider_prefix = normalized_model_name.split("/", 1)[0].strip().lower()
            if provider_prefix in explicit_provider_prefixes:
                return normalized_model_name

        return f"openai/{normalized_model_name}"

    def _get_env_value(self, explicit_env_name=None, fallback_env_names=None):
        """Resolve a configuration value from one explicit or multiple fallback env vars."""
        if explicit_env_name:
            return os.environ.get(explicit_env_name)
        for env_name in fallback_env_names or []:
            value = os.environ.get(env_name)
            if value:
                return value
        return None

    def _resolve_api_key(self, explicit_key=None, explicit_env_name=None, fallback_env_names=None):
        """Resolve a secret from config first, then env vars."""
        if explicit_key:
            return explicit_key
        return self._get_env_value(explicit_env_name, fallback_env_names)

    def _resolve_llm_api_key(self, fallback_env_names=None):
        """Resolve the configured chat-model API key."""
        return self._resolve_api_key(
            explicit_key=self.api_key,
            explicit_env_name=self.api_key_env,
            fallback_env_names=fallback_env_names or ["OPENAI_API_KEY"],
        )

    def _resolve_embedding_api_key(self, fallback_env_names=None):
        """Resolve the configured embedding API key, falling back to the chat key."""
        return (
            self._resolve_api_key(
                explicit_key=self.embedding_api_key,
                explicit_env_name=self.embedding_api_key_env,
                fallback_env_names=fallback_env_names or ["OPENAI_API_KEY"],
            )
            or self._resolve_llm_api_key(fallback_env_names)
        )

    def _resolve_base_url(self):
        """Resolve an explicit or env-backed OpenAI-compatible base URL."""
        return self.base_url or self._get_env_value(self.base_url_env, ["OPENAI_BASE_URL"])

    def _ensure_explicit_embedding_config(self):
        """Require an explicit embedding_model for retrieval-style non-OpenAI chat backends.

        When the chat backend is openai_compatible (e.g. DeepSeek), the embedding pipeline
        is independent and must be declared explicitly so results are reproducible and the
        skip-on-existing-results logic does not silently reuse stale output.

        Only embedding_model is required:
        - Local / HuggingFace models (facebook/contriever, Qwen/*, nvidia/*) need no
          API endpoint — naming the model is sufficient.
        - OpenAI-API models (text-embedding-3-small, etc.) default to the standard
          OpenAI endpoint; add embedding_provider / embedding_base_url only when
          using a non-default endpoint.
        """
        if self.model_provider in {"openai", "azure_openai"}:
            return
        if self.embedding_model:
            return

        raise RuntimeError(
            "Retrieval-style methods using non-OpenAI chat backends require 'embedding_model' "
            "to be set explicitly in the agent config (e.g. embedding_model: text-embedding-3-small). "
            "Also set embedding_provider and embedding endpoint fields when not using the "
            "default OpenAI embedding API."
        )

    def _initialize_tokenizer(self):
        """Initialize a tokenizer with optional config overrides for non-GPT models."""
        last_exception = None

        local_tokenizer_candidates = []
        if self.tokenizer_model:
            local_tokenizer_candidates.append(self.tokenizer_model)
        if self.model_provider in {"openai_compatible", "anthropic", "gemini"}:
            local_tokenizer_candidates.append(self.model)

        seen_local_candidates = set()
        for tokenizer_candidate in local_tokenizer_candidates:
            normalized_candidate = str(tokenizer_candidate or "").strip()
            if not normalized_candidate or normalized_candidate in seen_local_candidates:
                continue
            seen_local_candidates.add(normalized_candidate)
            try:
                local_tokenizer = load_local_hf_tokenizer(normalized_candidate)
                if local_tokenizer is not None:
                    return local_tokenizer
            except Exception as exc:
                last_exception = exc

        if self.tokenizer_encoding:
            try:
                return tiktoken.get_encoding(self.tokenizer_encoding)
            except Exception as exc:
                last_exception = exc

        tokenizer_candidates = []
        if self.tokenizer_model:
            tokenizer_candidates.append(self.tokenizer_model)
        if self.model_provider in {"openai", "azure_openai"}:
            tokenizer_candidates.append(self.model)
        tokenizer_candidates.append("gpt-4o-mini")

        seen_candidates = set()
        for model_for_tokenizer in tokenizer_candidates:
            if not model_for_tokenizer or model_for_tokenizer in seen_candidates:
                continue
            seen_candidates.add(model_for_tokenizer)
            try:
                return tiktoken.encoding_for_model(model_for_tokenizer)
            except Exception as exc:
                last_exception = exc

        if self.model_provider in {"openai_compatible", "anthropic", "gemini"}:
            return ApproximateTokenizer()

        raise RuntimeError(
            f"Failed to initialize a tokenizer for model '{self.model}'. "
            "Set 'tokenizer_model' or 'tokenizer_encoding' in the agent config for non-GPT providers. "
            "tiktoken may also need to download encoding files on first use, so this can fail "
            "when network access is blocked. Pre-populate the tiktoken cache or enable network access."
        ) from last_exception

    def _create_oai_client(self):
        """Create an OpenAI-compatible client.

        Supports OpenAI, Azure OpenAI, and OpenAI-compatible providers.
        """
        legacy_azure_endpoint = self.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if self.model_provider == "azure_openai" or (
            self.model_provider == "openai" and legacy_azure_endpoint and not self.base_url and not self.base_url_env
        ):
            from openai import AzureOpenAI

            return AzureOpenAI(
                api_key=self._get_env_value(self.api_key_env, ["AZURE_OPENAI_API_KEY"]),
                api_version=self.azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=legacy_azure_endpoint,
            )

        base_url = self._resolve_base_url()
        api_key = self._resolve_llm_api_key(["OPENAI_API_KEY"])
        if self.model_provider == "openai_compatible" and not base_url:
            raise RuntimeError(
                "OpenAI-compatible models require 'base_url' or 'base_url_env' in the agent config."
            )
        if base_url:
            return OpenAI(
                base_url=base_url,
                api_key=api_key or "EMPTY",
                http_client=httpx.Client(trust_env=False),
            )
        return OpenAI(api_key=api_key)

    def _create_standard_response(self, output, input_tokens, output_tokens, memory_time, query_time):
        """Create standardized response dictionary."""
        response = {
            "output": output,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
        }
        if self.record_llm_io and self._last_llm_trace is not None:
            response["llm_trace"] = self._last_llm_trace
        return response

    def _supports_locomo_recall(self):
        """Return whether the current agent should persist LoCoMo source metadata."""
        return any([
            self._is_agent_type("cognee"),
            self._is_agent_type("mem0"),
            self._is_agent_type("memoryos"),
            self._is_agent_type("simplemem"),
            self._is_agent_type("lightmem"),
            self._is_agent_type("a_mem"),
            self._is_agent_type("memtree"),
            self._is_agent_type("everos"),
            self._is_agent_type("zep_local"),
            self._is_agent_type("zep"),
            self._is_agent_type("MemOS"),
            self._is_agent_type("rag"),
        ])

    def _normalize_message_payload(self, message, memorizing=False):
        """Convert structured dataset chunks into the text each agent should see."""
        if not isinstance(message, dict):
            return message

        plain_text = str(message.get("text", "") or "")
        storage_text = str(message.get("storage_text", "") or plain_text)

        if memorizing and self._supports_locomo_recall():
            return storage_text
        return plain_text

    def _serialize_trace_messages(self, messages):
        """Convert OpenAI-style messages into a JSON-safe trace format."""
        serialized_messages = []
        for message in messages or []:
            if not isinstance(message, dict):
                serialized_messages.append({"role": "unknown", "content": str(message)})
                continue

            content = message.get("content")
            if isinstance(content, list):
                normalized_parts = []
                for part in content:
                    if isinstance(part, dict):
                        normalized_parts.append({
                            "type": part.get("type"),
                            "text": part.get("text") or part.get("content"),
                        })
                    else:
                        normalized_parts.append(str(part))
                normalized_content = normalized_parts
            else:
                normalized_content = content

            serialized_messages.append(
                {
                    "role": message.get("role", "unknown"),
                    "content": normalized_content,
                }
            )
        return serialized_messages

    def _record_llm_trace(
        self,
        *,
        stage,
        messages,
        response_text,
        prompt_tokens=None,
        completion_tokens=None,
        extra=None,
    ):
        """Persist the latest query-time LLM prompt/response for debugging."""
        if not self.record_llm_io:
            return

        trace = {
            "stage": stage,
            "query_id": self._current_query_id,
            "context_id": self._current_context_id,
            "model": self.model,
            "base_url": self._resolve_base_url(),
            "messages": self._serialize_trace_messages(messages),
            "response": response_text,
        }
        if prompt_tokens is not None:
            trace["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            trace["completion_tokens"] = completion_tokens
        if extra:
            trace.update(extra)
        self._last_llm_trace = trace

    def _encode_text(self, text):
        """Encode text with the current tokenizer, handling approximate tokenizers."""
        try:
            return self.tokenizer.encode(text, disallowed_special=())
        except TypeError:
            return self.tokenizer.encode(text)

    def _count_tokens(self, text):
        return len(self._encode_text(text))

    def _truncate_text_to_token_limit(self, text, max_tokens, truncation_strategy="head"):
        """Truncate plain text to a token ceiling using the configured tokenizer."""
        tokens = self._encode_text(text)
        if len(tokens) <= max_tokens:
            return text

        if truncation_strategy == "tail":
            truncated_tokens = tokens[-max_tokens:]
        elif truncation_strategy == "head_tail" and max_tokens > 1:
            head_tokens = max_tokens // 2
            tail_tokens = max_tokens - head_tokens
            truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
        else:
            truncated_tokens = tokens[:max_tokens]

        if hasattr(self.tokenizer, "decode"):
            truncated_text = self.tokenizer.decode(truncated_tokens)
        else:
            truncated_text = "".join(truncated_tokens)
        return truncated_text.strip()

    def _fit_text_to_prompt_token_limit(
        self,
        text,
        message,
        system_message="",
        reserved_tokens=1024,
        truncation_strategy="head",
    ):
        """Fit one long text block into the remaining prompt budget."""
        prompt_budget_limit = self.input_length_limit
        if self.model_context_window:
            prompt_budget_limit = min(prompt_budget_limit, int(self.model_context_window))

        available_budget = max(
            0,
            prompt_budget_limit
            - self._count_tokens(system_message or "")
            - self._count_tokens(message or "")
            - reserved_tokens,
        )
        if available_budget <= 0:
            return ""

        return self._truncate_text_to_token_limit(
            text or "",
            available_budget,
            truncation_strategy=truncation_strategy,
        )

    def _extract_chat_message_text(self, message):
        """Extract plain text from OpenAI-style chat messages."""
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_value = part.get("text") or part.get("content")
                else:
                    text_value = getattr(part, "text", None) or getattr(part, "content", None)
                if text_value:
                    text_parts.append(str(text_value).strip())
            if text_parts:
                return "\n".join(part for part in text_parts if part)

        reasoning_content = getattr(message, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return reasoning_content.strip()

        return ""

    def _should_disable_qwen3_thinking(self):
        """Return whether Qwen3 requests should disable thinking mode."""
        model_name = str(self.model or "").strip().lower()
        if "qwen3" not in model_name:
            return False
        return bool(self.agent_config.get("qwen3_disable_thinking", True))

    def _prepare_openai_request_kwargs(self, request_kwargs):
        """Inject request-time compatibility knobs for specific model families."""
        if not self._should_disable_qwen3_thinking():
            return request_kwargs

        extra_body = dict(request_kwargs.get("extra_body") or {})
        extra_body.setdefault("enable_thinking", False)
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        request_kwargs["extra_body"] = extra_body
        return request_kwargs

    def _request_openai_completion(self, messages):
        """Run a chat completion against the configured OpenAI-compatible client."""
        request_kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if "o4" not in self.model:
            request_kwargs["temperature"] = self.temperature
            request_kwargs["max_tokens"] = self.max_tokens
        request_kwargs = self._prepare_openai_request_kwargs(request_kwargs)

        response = self.client.chat.completions.create(**request_kwargs)
        if not getattr(response, "choices", None):
            return "", None, None

        message = response.choices[0].message
        usage = getattr(response, "usage", None)
        response_text = self._extract_chat_message_text(message)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        self._record_llm_trace(
            stage="openai_chat_completion",
            messages=messages,
            response_text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            extra={
                "temperature": request_kwargs.get("temperature"),
                "max_tokens": request_kwargs.get("max_tokens"),
            },
        )
        return (
            response_text,
            prompt_tokens,
            completion_tokens,
        )

    def _generate_answer_from_memories(self, question, memories_text, prompt_override=None):
        """Generate a benchmark answer from retrieved memories using the shared templates."""
        memory_answer_template = (
            prompt_override
            or self.agent_config.get("memory_answer_prompt")
            or get_template(self.sub_dataset, 'memory_answer', self.agent_name)
        )
        llm_messages = [
            {"role": "system", "content": get_template(self.sub_dataset, 'system', self.agent_name)},
            {
                "role": "user",
                "content": (
                    memory_answer_template.format(
                        memories=memories_text or "(No retrieved memories found.)",
                        question=question,
                    )
                    + "\n\nCurrent Time: "
                    + time.strftime("%Y-%m-%d %H:%M:%S")
                ),
            },
        ]
        try:
            response_text, prompt_tokens, completion_tokens = self._request_openai_completion(llm_messages)
        except Exception as exc:
            error_text = str(exc).lower()
            if "maximum context length" not in error_text:
                raise

            memory_token_count = max(self._count_tokens(memories_text or ""), 1)
            retry_budgets = []
            for ratio in (0.75, 0.5, 0.33, 0.25, 0.125):
                budget = max(256, int(memory_token_count * ratio))
                if budget < memory_token_count:
                    retry_budgets.append(budget)

            last_exc = exc
            for budget in retry_budgets:
                truncated_memories = self._truncate_text_to_token_limit(memories_text or "", budget)
                llm_messages[1]["content"] = (
                    memory_answer_template.format(
                        memories=truncated_memories or "(No retrieved memories found.)",
                        question=question,
                    )
                    + "\n\nCurrent Time: "
                    + time.strftime("%Y-%m-%d %H:%M:%S")
                )
                try:
                    response_text, prompt_tokens, completion_tokens = self._request_openai_completion(llm_messages)
                    break
                except Exception as retry_exc:
                    if "maximum context length" not in str(retry_exc).lower():
                        raise
                    last_exc = retry_exc
            else:
                raise last_exc
        if prompt_tokens is None:
            prompt_tokens = self._count_tokens((memories_text or "") + "\n" + question)
        if completion_tokens is None:
            completion_tokens = self._count_tokens(response_text)
        return response_text, prompt_tokens, completion_tokens

    def _fit_memories_for_answer(self, question, contexts, prompt_override=None, label_prefix="Memory"):
        """Trim retrieved memories so the downstream answer prompt stays within model limits."""
        if not contexts:
            return []

        memory_answer_template = (
            prompt_override
            or self.agent_config.get("memory_answer_prompt")
            or get_template(self.sub_dataset, 'memory_answer', self.agent_name)
        )
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        prompt_without_memories = (
            memory_answer_template.format(memories="", question=question)
            + "\n\nCurrent Time: "
            + time.strftime("%Y-%m-%d %H:%M:%S")
        )
        return self._fit_retrieved_contexts_to_token_limit(
            contexts=contexts,
            message=prompt_without_memories,
            tokenizer=self.tokenizer,
            label_prefix=label_prefix,
            system_message=system_message,
            reserved_tokens=max(self.max_tokens + 2048, 4096),
        )

    def _prepare_memory_chunk_for_storage(self, text):
        """Store raw benchmark content instead of the benchmark-side memorize wrapper."""
        if text is None:
            return ""
        normalized = strip_locomo_metadata(str(text or ""))
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").strip()
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized

    def _clean_retrieved_memory_text(self, text):
        """Remove benchmark wrappers and runtime metadata from retrieved memory snippets."""
        normalized = self._prepare_memory_chunk_for_storage(text)
        if not normalized:
            return ""

        skip_patterns = (
            re.compile(r"^Dialogue between User and Assistant\b.*$", re.IGNORECASE),
            re.compile(r"^<User>\s*The following context is\b.*$", re.IGNORECASE),
            re.compile(r"^<Assistant>\s*I have (?:learned|memorized|read)\b.*$", re.IGNORECASE),
            re.compile(
                r"^You are a helpful assistant that can read the context and memorize it for future retrieval\.?.*$",
                re.IGNORECASE,
            ),
            re.compile(r"^Search Archival Memory\b.*$", re.IGNORECASE),
            re.compile(r"^Use the provided mapping .*?$", re.IGNORECASE),
            re.compile(r"^Pretend you are a knowledge management system\..*$", re.IGNORECASE),
            re.compile(r"^Question:\s*Based on the provided Knowledge Pool,.*$", re.IGNORECASE),
            re.compile(r"^Retrieved memories:\s*$", re.IGNORECASE),
            re.compile(r"^label:\s*$", re.IGNORECASE),
            re.compile(r"^Answer:\s*$", re.IGNORECASE),
        )
        metadata_prefixes = (
            "[historical-memory",
            "[user-profile",
            "[assistant-knowledge",
            "[long-term-knowledge",
        )
        drop_prefixes = (
            "Time:",
            "Conversation chain overview:",
        )

        cleaned_lines = []
        for raw_line in normalized.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(metadata_prefixes):
                continue
            if line.startswith(drop_prefixes):
                continue
            if line.startswith("User:"):
                line = line[len("User:"):].strip()
            elif line.startswith("Assistant:"):
                line = line[len("Assistant:"):].strip()
            if not line:
                continue
            if any(pattern.match(line) for pattern in skip_patterns):
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned

    def _clean_retrieved_memory_contexts(self, contexts):
        """Apply retrieved-memory cleanup conservatively and deduplicate snippets."""
        if not contexts:
            return []

        cleaned_contexts = []
        seen = set()
        for context in contexts:
            cleaned = self._clean_retrieved_memory_text(strip_locomo_metadata(context))
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            cleaned_contexts.append(cleaned)
        return cleaned_contexts

    def _extract_locomo_source_id_groups_from_texts(self, texts):
        """Parse ranked LoCoMo retrieval units from raw retrieved text snippets."""
        groups = []
        for text in texts or []:
            metadata_entries = parse_locomo_metadata(text)
            if metadata_entries:
                for metadata in metadata_entries:
                    if metadata["source_ids"]:
                        groups.append(metadata["source_ids"])
                continue

            source_ids = parse_locomo_source_ids(text)
            if source_ids:
                groups.append(source_ids)
        return groups

    def _extract_locomo_source_id_groups_from_items(self, items, candidate_keys=None):
        """Parse LoCoMo source ids from a ranked list of dict/object retrieval items."""
        candidate_keys = candidate_keys or (
            "text", "memory", "content", "user_input", "agent_response", "page_content", "summary"
        )
        groups = []

        for item in items or []:
            text_candidates = []
            if isinstance(item, dict):
                for key in candidate_keys:
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        text_candidates.append(value)
            else:
                for key in candidate_keys:
                    value = getattr(item, key, None)
                    if isinstance(value, str) and value.strip():
                        text_candidates.append(value)

            source_ids = dedupe_preserve_order(
                source_id
                for text in text_candidates
                for source_id in parse_locomo_source_ids(text)
            )
            if source_ids:
                groups.append(source_ids)

        return groups

    def _normalize_mem0_add_results(self, payload):
        """Normalize mem0 add responses into a list of memory event dicts."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("results", "memory", "memories"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            if payload.get("id") is not None:
                return [payload]
        return []

    def _normalize_mem0_search_results(self, payload):
        """Normalize mem0 search responses into a ranked list of result dicts."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("results", "memory", "memories"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    def _remember_locomo_source_ids_for_text(self, source_map, text, source_ids):
        """Persist source ids for a normalized stored text snippet."""
        normalized_text = self._prepare_memory_chunk_for_storage(text)
        normalized_source_ids = dedupe_preserve_order(source_ids)
        if not normalized_text or not normalized_source_ids:
            return

        existing_source_ids = source_map.get(normalized_text, [])
        source_map[normalized_text] = dedupe_preserve_order(
            list(existing_source_ids) + normalized_source_ids
        )

    def _lookup_locomo_source_id_groups_by_texts(self, source_map, texts):
        """Resolve ranked source-id groups from normalized stored text snippets."""
        groups = []
        for text in texts or []:
            normalized_text = self._prepare_memory_chunk_for_storage(text)
            if not normalized_text:
                continue
            source_ids = source_map.get(normalized_text, [])
            if source_ids:
                groups.append(source_ids)
        return groups

    def _attach_locomo_recall_metadata(self, output, retrieved_source_id_groups):
        """Attach strict-recall metadata to query outputs when available."""
        if retrieved_source_id_groups is not None:
            output["retrieved_source_id_groups"] = retrieved_source_id_groups
        if getattr(self, "retrieve_num", None):
            output["requested_recall_k"] = self.retrieve_num
        return output

    def _require_locomo_provenance_sidecar(self, sidecar_path, agent_label):
        """Fail fast when a saved LoCoMo state predates provenance tracking."""
        if self.sub_dataset != "locomo_qa":
            return
        if os.path.exists(sidecar_path):
            return
        raise RuntimeError(
            f"{agent_label} saved state at {self.agent_save_to_folder} is missing "
            f"LoCoMo provenance metadata ({sidecar_path}). This cache predates the "
            "current recall tracking. Rebuild the agent state with --force before "
            "running LoCoMo recall evaluation."
        )

    def _get_retrieval_debug_root(self):
        """Return the common retrieval-debug directory for the current agent/dataset."""
        output_dir = str(self.agent_config.get("output_dir") or "").strip()
        output_label = os.path.basename(os.path.normpath(output_dir)) if output_dir else "unknown_output"
        run_scope = (
            f"in{self.dataset_config.get('context_max_length', 'unknown')}"
            f"_max_samples{self.dataset_config.get('max_test_samples', 'unknown')}"
        )
        return os.path.join(
            resolve_results_artifact_path(self.artifact_root, "outputs", "rag_retrieved"),
            output_label,
            self.agent_name,
            f"k_{self.retrieve_num}",
            self.sub_dataset,
            run_scope,
            f"chunksize_{self.chunk_size}",
        )

    def _build_retrieval_debug_record(self, payload, query_id, context_id, extra_fields=None):
        """Build a normalized retrieval-debug record with stable LoCoMo metadata."""
        record = dict(payload) if isinstance(payload, dict) else {"retrieval_context": payload}
        record.update(extra_fields or {})
        record["query_id"] = query_id
        record["context_id"] = context_id

        eval_metadata = self._current_eval_metadata or {}
        if eval_metadata:
            record["eval_metadata"] = eval_metadata
            for key in ("qa_pair_id", "question_id", "sample_id", "category"):
                value = eval_metadata.get(key)
                if value is not None:
                    record[key] = value

        record_key = (
            eval_metadata.get("qa_pair_id")
            or eval_metadata.get("question_id")
            or f"query_{query_id}_context_{context_id}"
        )
        record["record_key"] = str(record_key)
        return record

    def _save_retrieval_debug_payload(self, payload, query_id, context_id, extra_fields=None):
        """Persist retrieval-debug payloads, aggregating LoCoMo outputs by category."""
        if query_id is None or context_id is None:
            return

        retrieval_root = self._get_retrieval_debug_root()
        eval_metadata = self._current_eval_metadata or {}

        if eval_metadata.get("dataset") == "locomo_qa":
            category = str(eval_metadata.get("category") or "unknown")
            save_path = os.path.join(retrieval_root, "by_category", f"category_{category}.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            cleanup_root = os.path.abspath(retrieval_root)
            if cleanup_root not in self._retrieval_debug_cleanup_roots and os.path.isdir(retrieval_root):
                for filename in os.listdir(retrieval_root):
                    if filename.startswith("query_") and filename.endswith(".json"):
                        legacy_path = os.path.join(retrieval_root, filename)
                        if os.path.isfile(legacy_path):
                            os.remove(legacy_path)
                self._retrieval_debug_cleanup_roots.add(cleanup_root)

            record = self._build_retrieval_debug_record(payload, query_id, context_id, extra_fields=extra_fields)
            container = {
                "dataset": "locomo_qa",
                "category": category,
                "agent_name": self.agent_name,
                "sub_dataset": self.sub_dataset,
                "retrieve_num": self.retrieve_num,
                "chunk_size": self.chunk_size,
                "entries": [],
            }

            if os.path.exists(save_path):
                try:
                    with open(save_path, "r", encoding="utf-8") as file:
                        existing_data = json.load(file)
                    if isinstance(existing_data, dict):
                        container.update({k: v for k, v in existing_data.items() if k != "entries"})
                        if isinstance(existing_data.get("entries"), list):
                            container["entries"] = existing_data["entries"]
                    elif isinstance(existing_data, list):
                        container["entries"] = existing_data
                except (json.JSONDecodeError, OSError):
                    pass

            replaced = False
            for index, existing_record in enumerate(container["entries"]):
                if str(existing_record.get("record_key", "")) == record["record_key"]:
                    container["entries"][index] = record
                    replaced = True
                    break
            if not replaced:
                container["entries"].append(record)

            container["entries"].sort(
                key=lambda item: (
                    int(item.get("context_id", 0)),
                    int(item.get("query_id", 0)),
                    str(item.get("record_key", "")),
                )
            )

            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(container, file, ensure_ascii=False, indent=2)
            return

        save_path = os.path.join(retrieval_root, f"query_{query_id}_context_{context_id}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        record = self._build_retrieval_debug_record(payload, query_id, context_id, extra_fields=extra_fields)
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)

    def _should_backfill_longmemeval_recall_debug(self):
        """Return whether the current run should emit LongMemEval recall debug sidecars."""
        return (
            self.backfill_longmemeval_recall_debug
            and str(self.dataset or "").strip().lower() == "accurate_retrieval"
            and "longmemeval" in str(self.sub_dataset or "").strip().lower()
        )

    def _normalize_retrieval_debug_paragraphs(self, paragraphs):
        """Convert retrieval snippets into a stable list of non-empty text paragraphs."""
        normalized = []
        seen = set()
        for paragraph in self._flatten_text_items(paragraphs):
            text = str(paragraph or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    def _maybe_save_longmemeval_backfill_debug(
        self,
        *,
        query_id,
        context_id,
        retrieved_paragraphs,
        query_text,
        response_text=None,
        retrieved_source_id_groups=None,
        extra_fields=None,
    ):
        """Persist query-only LongMemEval recall debug files without touching the main results flow."""
        if not self._should_backfill_longmemeval_recall_debug():
            return

        payload = {
            "retrieved_context_paragraphs": self._normalize_retrieval_debug_paragraphs(
                retrieved_paragraphs
            ),
            "query": query_text,
            "backfill_longmemeval_recall_debug": True,
        }
        if response_text is not None:
            payload["response"] = response_text
        if retrieved_source_id_groups is not None:
            payload["retrieved_source_id_groups"] = retrieved_source_id_groups
        if extra_fields:
            payload.update(extra_fields)

        self._save_retrieval_debug_payload(payload, query_id, context_id)

    def _flatten_text_items(self, value):
        """Coerce nested search/retrieval results into a flat list of text snippets."""
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if hasattr(value, "model_dump"):
            return self._flatten_text_items(value.model_dump())
        if hasattr(value, "page_content"):
            return self._flatten_text_items(value.page_content)
        if isinstance(value, dict):
            results = []
            preferred_keys = [
                "search_result",
                "context_result",
                "text_result",
                "context",
                "memory",
                "knowledge",
                "content",
                "summary",
                "text",
            ]
            for key in preferred_keys:
                if key in value:
                    results.extend(self._flatten_text_items(value[key]))
            if results:
                return results
            serialized = json.dumps(value, ensure_ascii=False)
            return [serialized] if serialized else []
        if isinstance(value, (list, tuple, set)):
            flattened = []
            for item in value:
                flattened.extend(self._flatten_text_items(item))
            return flattened

        text = str(value).strip()
        return [text] if text else []

    def _infer_embedding_dimensions(self, embedding_model, configured_dimensions=None):
        """Infer common embedding dimensions when the config does not provide them."""
        if configured_dimensions is not None:
            return int(configured_dimensions)
        if not embedding_model:
            return None
        if "Qwen3-Embedding-4B" in embedding_model:
            return 2560
        if embedding_model == "text-embedding-3-large":
            return 3072
        if embedding_model in {"text-embedding-3-small", "text-embedding-ada-002"}:
            return 1536
        return None

    def _initialize_long_context_agent(self):
        """Initialize long context agent with appropriate client."""
        self.context = ''

        if self.model_provider in {"openai", "azure_openai", "openai_compatible"}:
            self.client = self._create_oai_client()
        elif self.model_provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=self._get_env_value(self.api_key_env, ['Anthropic_API_KEY']),
            )
        elif self.model_provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self._get_env_value(self.api_key_env, ['Google_API_KEY']))
        else:
            raise NotImplementedError(
                f"Provider '{self.model_provider}' is not supported for long context agent: {self.model}"
            )

    def _initialize_letta_agent(self, agent_config, dataset_config):
        """Initialize Letta agent with proper configuration."""
        if "api" not in agent_config['agent_name']:
            self.chunk_size = agent_config['agent_chunk_size']
            self.letta_mode = agent_config['letta_mode']
            chat_api_key = self._resolve_llm_api_key(["OPENAI_API_KEY"])
            embedding_api_key = self._resolve_embedding_api_key(["OPENAI_API_KEY"])
            self.letta_runtime_dir = os.path.join(self.agent_save_to_folder, "letta_runtime")
            self.letta_runtime_db_path = os.path.join(self.letta_runtime_dir, "sqlite.db")
            self.letta_query_baseline_dir = os.path.join(
                self.agent_save_to_folder,
                "letta_query_baseline",
            )
            os.makedirs(self.letta_runtime_dir, exist_ok=True)
            os.environ["LETTA_DIR"] = self.letta_runtime_dir
            os.environ["LETTA_LETTA_DIR"] = self.letta_runtime_dir
            os.environ.setdefault(
                "COMPOSIO_CACHE_DIR",
                os.path.join(self.letta_runtime_dir, "composio_cache"),
            )
            if agent_config.get("letta_retrieval_page_size"):
                os.environ["LETTA_RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE"] = str(
                    agent_config["letta_retrieval_page_size"]
                )
            if chat_api_key:
                os.environ["OPENAI_API_KEY"] = chat_api_key
            if embedding_api_key:
                # Letta's vendored runtime historically assumed one OpenAI-style key
                # for both chat and embeddings. Keep chat on OPENAI_API_KEY, and expose
                # a separate embedding key for local embedding endpoints when needed.
                os.environ["OPENAI_EMBEDDING_API_KEY"] = embedding_api_key
            base_url = self._resolve_base_url()
            if base_url:
                # Letta's vendored runtime still consults OPENAI_API_BASE in a few
                # internal provider paths, so keep it aligned with the configured
                # local endpoint.
                os.environ["OPENAI_API_BASE"] = base_url
            from letta import create_client, LLMConfig, EmbeddingConfig, BasicBlockMemory

            self.client = create_client()
            if base_url:
                # Custom endpoint: build LLMConfig explicitly.
                llm_config = LLMConfig(
                    model=agent_config['model'],
                    model_endpoint_type="openai",
                    model_endpoint=base_url,
                    context_window=agent_config.get('context_window', 128000),
                )
            else:
                # default_config only supports known GPT model names; raise a clear
                # error for unsupported models instead of letting Letta crash later.
                try:
                    llm_config = LLMConfig.default_config(agent_config['model'])
                except ValueError:
                    raise ValueError(
                        f"Letta default_config does not support model '{agent_config['model']}'. "
                        "Add 'base_url' (or 'base_url_env') to the agent config to use a custom endpoint."
                    )
            self.client.set_default_llm_config(llm_config)
            self.agent_start_time = time.time()

            # Configure embedding.
            # Resolve endpoint: prefer embedding_base_url from config, fall back to OpenAI default.
            embedding_model = agent_config['text_embedding']
            embedding_endpoint = (
                self.embedding_base_url
                or self._get_env_value(self.embedding_base_url_env)
                or "https://api.openai.com/v1"
            )
            # Known dimensions for common OpenAI embedding models.
            _embedding_dim_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            embedding_dim = (
                agent_config.get('embedding_dim')
                or _embedding_dim_map.get(embedding_model)
            )
            if embedding_dim:
                # Explicit construction: handles any OpenAI-compatible endpoint + model.
                if embedding_api_key and not chat_api_key:
                    os.environ["OPENAI_API_KEY"] = embedding_api_key
                self.client.set_default_embedding_config(EmbeddingConfig(
                    embedding_model=embedding_model,
                    embedding_endpoint_type="openai",
                    embedding_endpoint=embedding_endpoint,
                    embedding_dim=embedding_dim,
                    embedding_chunk_size=self.chunk_size * 2,
                ))
            else:
                # Fall back to Letta's default_config for non-OpenAI models (e.g. hugging-face).
                # Raises ValueError for unsupported models with a helpful message.
                try:
                    self.client.set_default_embedding_config(
                        EmbeddingConfig.default_config(embedding_model)
                    )
                except ValueError:
                    raise ValueError(
                        f"Letta EmbeddingConfig.default_config does not support model '{embedding_model}'. "
                        "Add 'embedding_dim' (and optionally 'embedding_base_url') to the agent config "
                        "to use a custom OpenAI-compatible embedding endpoint."
                    )

            # Load system prompt
            system_path = agent_config['system_path']
            with open(system_path, 'r') as f:
                self.system = f.read()

            # Load or create agent
            agent_id_path = os.path.join(self.agent_save_to_folder, "agent_id.txt")
            runtime_db_path = self.letta_runtime_db_path
            if os.path.exists(agent_id_path) and os.path.exists(runtime_db_path):
                self.load_agent()
            else:
                human_block = self.client.create_block(
                    label='human',
                    value='User is sharing the contents they are reading recently.',
                    limit=2000000
                )
                persona_block = self.client.create_block(
                    label='persona',
                    value='You are a helpful assistant that can help memorize details in the conversation.',
                    limit=2000000
                )
                memory = BasicBlockMemory(blocks=[human_block, persona_block])
                self.agent_state = self.client.create_agent(
                    name='mm_agent',
                    memory=memory,
                    system=self.system
                )
        ## use the letta api to create the agent
        else:
            from letta_client import Letta

            self.chunk_size = agent_config['agent_chunk_size']
            self.letta_mode = agent_config['letta_mode']
            self.agent_start_time = time.time()

            # base_url: if set, points to a self-hosted Letta server;
            # otherwise defaults to Letta cloud (https://api.letta.com).
            base_url = self._resolve_base_url()
            token = self._get_env_value(self.api_key_env, ["Letta_API_KEY"])
            client_kwargs = {"api_key": token}
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = Letta(**client_kwargs)

            # Self-hosted Letta uses bare model names; Letta cloud requires "openai/" prefix.
            if base_url:
                model_name = agent_config['model']
                embedding_name = agent_config['text_embedding']
            else:
                model_name = f"openai/{agent_config['model']}"
                embedding_name = f"openai/{agent_config['text_embedding']}"

            self.agent_state = self.client.agents.create(
                memory_blocks=[
                    {
                        "label": "human",
                        "limit": 2000000,
                        "value": "User is sharing the contents they are reading recently.",
                    },
                    {
                        "label": "persona",
                        "limit": 2000000,
                        "value": "You are a helpful assistant that can help memorize details in the conversation.",
                    },
                ],
                model=model_name,
                embedding=embedding_name,
            )

    def _snapshot_letta_query_baseline(self):
        """Persist a clean local Letta runtime baseline after memorization."""
        if not self._is_agent_type("letta") or "api" in self.agent_name:
            return

        runtime_db_path = getattr(self, "letta_runtime_db_path", None)
        baseline_dir = getattr(self, "letta_query_baseline_dir", None)
        if not runtime_db_path or not baseline_dir:
            return
        if not os.path.exists(runtime_db_path):
            print(f"Letta runtime database not found at {runtime_db_path}; skipping baseline snapshot.")
            return

        os.makedirs(baseline_dir, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            src = f"{runtime_db_path}{suffix}"
            dst = os.path.join(baseline_dir, f"sqlite.db{suffix}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
            elif os.path.exists(dst):
                os.remove(dst)

    def _restore_letta_query_baseline(self):
        """Restore the local Letta runtime baseline before each query."""
        if not self._is_agent_type("letta") or "api" in self.agent_name:
            return False

        runtime_db_path = getattr(self, "letta_runtime_db_path", None)
        baseline_dir = getattr(self, "letta_query_baseline_dir", None)
        baseline_db_path = os.path.join(baseline_dir, "sqlite.db") if baseline_dir else None
        if not runtime_db_path or not baseline_db_path or not os.path.exists(baseline_db_path):
            return False

        os.makedirs(self.letta_runtime_dir, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            dst = f"{runtime_db_path}{suffix}"
            src = os.path.join(baseline_dir, f"sqlite.db{suffix}")
            if os.path.exists(dst):
                os.remove(dst)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                os.chmod(dst, os.stat(dst).st_mode | 0o200)

        return True

    def _reset_letta_local_runtime_state(self):
        """Tear down local Letta runtime globals so sqlite can be safely restored."""
        if not self._is_agent_type("letta") or "api" in self.agent_name:
            return

        interface = getattr(getattr(self, "client", None), "interface", None)
        if interface is not None and hasattr(interface, "clear"):
            try:
                interface.clear()
            except Exception:
                pass

        try:
            from letta.server import db as letta_db

            if getattr(letta_db, "engine", None) is not None:
                try:
                    letta_db.engine.dispose()
                except Exception:
                    pass
            letta_db.engine = None
            letta_db.SessionLocal = None
            letta_db._engine_initialized = False
        except Exception:
            pass

    def _reload_letta_query_runtime(self):
        """Reload Letta state from the memorization-only baseline when available."""
        if not self._is_agent_type("letta"):
            return

        if "api" in self.agent_name:
            self.load_agent()
            return

        preserved_start_time = self.agent_start_time
        if not self._restore_letta_query_baseline():
            print(
                "\n\nLetta query baseline not found; falling back to shared runtime state. "
                "Rebuild this context from scratch to enable per-query isolation.\n\n"
            )
            self.load_agent()
            return

        self._reset_letta_local_runtime_state()
        self._initialize_letta_agent(self.agent_config, {"sub_dataset": self.sub_dataset})
        self.agent_start_time = preserved_start_time



    def _initialize_mem0_agent(self, agent_config, dataset_config):
        """Initialize Mem0 agent with retrieval configuration."""
        os.environ.setdefault("MEM0_DIR", os.path.join(self.agent_save_to_folder, "mem0_home"))
        from mem0.memory.main import Memory

        self.retrieve_num = agent_config['retrieve_num']
        self.mem0_add_infer = bool(agent_config.get("mem0_add_infer", True))
        self.chunk_size = agent_config.get('agent_chunk_size', dataset_config.get('chunk_size'))
        self.mem0_source_map_path = os.path.join(self.agent_save_to_folder, "mem0_source_map.json")
        self.mem0_source_map = {}
        self.context = ''
        self.client = self._create_oai_client()
        mem0_fact_prompt = (
            agent_config.get("mem0_fact_extraction_prompt")
            or get_template(self.sub_dataset, 'fact_extraction', self.agent_name)
        )
        mem0_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.model,
                    "temperature": self.temperature,
                },
            },
            "custom_fact_extraction_prompt": mem0_fact_prompt,
        }

        base_url = self._resolve_base_url()
        api_key = self._resolve_llm_api_key(["OPENAI_API_KEY"])
        if base_url:
            mem0_config["llm"]["config"]["openai_base_url"] = base_url
        if api_key:
            mem0_config["llm"]["config"]["api_key"] = api_key

        mem0_runtime_dir = os.path.join(self.agent_save_to_folder, "mem0_runtime")
        mem0_qdrant_path = os.path.join(mem0_runtime_dir, "qdrant")
        os.makedirs(mem0_runtime_dir, exist_ok=True)

        mem0_embedder_model = agent_config.get("mem0_embedder_model", "")
        embedding_dims = 2560 if "Qwen3-Embedding-4B" in mem0_embedder_model else 1536
        mem0_config["history_db_path"] = os.path.join(mem0_runtime_dir, "history.db")
        mem0_config["vector_store"] = {
            "provider": "qdrant",
            "config": {
                "path": mem0_qdrant_path,
                "collection_name": f"mem0_{self.sub_dataset}",
                "embedding_model_dims": embedding_dims,
                "on_disk": True,
            },
        }

        if agent_config.get("mem0_embedder_model"):
            embedding_base_url = (
                self.embedding_base_url
                or self._get_env_value(self.embedding_base_url_env)
                or base_url
            )
            embedder_model_name = agent_config["mem0_embedder_model"]
            if embedding_base_url and "/" in embedder_model_name:
                # Local OpenAI-compatible embedding services often expose the
                # deployed short name instead of the Hugging Face repo id.
                embedder_model_name = embedder_model_name.split("/")[-1]
            embedder_config = {
                "model": embedder_model_name,
            }
            embedding_api_key = self._resolve_embedding_api_key(["OPENAI_API_KEY"]) or api_key
            if embedding_base_url:
                embedder_config["openai_base_url"] = embedding_base_url
            if embedding_api_key:
                embedder_config["api_key"] = embedding_api_key
            mem0_config["embedder"] = {
                "provider": "openai",
                "config": embedder_config,
            }
        if agent_config.get("mem0_graph_store"):
            mem0_config["graph_store"] = agent_config["mem0_graph_store"]
        self.memory = Memory.from_config(mem0_config)
        if os.path.exists(self.mem0_source_map_path):
            with open(self.mem0_source_map_path, "r", encoding="utf-8") as f:
                self.mem0_source_map = json.load(f)
        self.agent_start_time = time.time()

    def _initialize_simplemem_agent(self, agent_config, dataset_config):
        """Initialize SimpleMem agent with a per-context LanceDB path."""
        from methods.simplemem.simplemem_adapter import SimpleMemAdapter

        self.retrieve_num = int(
            agent_config.get("retrieve_num", agent_config.get("simplemem_retrieve_num", 10))
        )
        self.chunk_size = agent_config.get('agent_chunk_size', dataset_config.get('chunk_size'))
        self.agent_start_time = time.time()
        self.client = self._create_oai_client()
        self.simplemem_marker_path = os.path.join(self.agent_save_to_folder, "simplemem_ready.txt")
        self.simplemem_source_map_path = os.path.join(self.agent_save_to_folder, "simplemem_source_map.json")
        self.simplemem_db_path = os.path.join(self.agent_save_to_folder, "lancedb_data")
        self.simplemem_table_name = _sanitize_storage_identifier(dataset_config['sub_dataset'], suffix="memory")
        api_key = (
            os.environ.get("SIMPLEMEM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "SimpleMem requires SIMPLEMEM_API_KEY or OPENAI_API_KEY to be set."
            )

        embedding_model = (
            self.embedding_model
            or agent_config.get('simplemem_embedding_model')
        )
        embedding_base_url = (
            self.embedding_base_url
            or self._get_env_value(self.embedding_base_url_env)
            or agent_config.get('simplemem_embedding_base_url')
            or os.environ.get("SIMPLEMEM_EMBEDDING_BASE_URL")
            or self._resolve_base_url()
        )
        embedding_api_key = (
            self._resolve_embedding_api_key(["SIMPLEMEM_EMBEDDING_API_KEY", "OPENAI_API_KEY"])
            or api_key
        )
        embedding_dimensions = self._infer_embedding_dimensions(
            embedding_model,
            agent_config.get("embedding_dim") or agent_config.get("simplemem_embedding_dimension"),
        )

        self.simplemem = SimpleMemAdapter(
            api_key=api_key,
            model=agent_config['model'],
            base_url=agent_config.get('simplemem_base_url') or os.environ.get("SIMPLEMEM_BASE_URL"),
            db_path=self.simplemem_db_path,
            table_name=self.simplemem_table_name,
            clear_db=not os.path.exists(self.simplemem_marker_path),
            enable_thinking=agent_config.get('simplemem_enable_thinking', False),
            use_streaming=agent_config.get('simplemem_use_streaming', False),
            enable_planning=agent_config.get('simplemem_enable_planning', True),
            enable_reflection=agent_config.get('simplemem_enable_reflection', True),
            max_reflection_rounds=agent_config.get('simplemem_max_reflection_rounds', 2),
            enable_parallel_processing=agent_config.get('simplemem_enable_parallel_processing', True),
            max_parallel_workers=agent_config.get('simplemem_max_parallel_workers', 4),
            enable_parallel_retrieval=agent_config.get('simplemem_enable_parallel_retrieval', True),
            max_retrieval_workers=agent_config.get('simplemem_max_retrieval_workers', 3),
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimensions,
            retrieve_limit=self.retrieve_num,
            semantic_top_k=agent_config.get('simplemem_semantic_top_k', 25),
            keyword_top_k=agent_config.get('simplemem_keyword_top_k', 5),
            structured_top_k=agent_config.get('simplemem_structured_top_k', 5),
            window_size=agent_config.get('simplemem_window_size', 40),
            overlap_size=agent_config.get('simplemem_overlap_size', 2),
        )
        if os.path.exists(self.simplemem_source_map_path):
            with open(self.simplemem_source_map_path, "r", encoding="utf-8") as f:
                self.simplemem.entry_source_map = json.load(f)

    def _initialize_lightmem_agent(self, agent_config, dataset_config):
        """Initialize LightMem with a per-context local persistent store."""
        from methods.lightmem.lightmem_adapter import LightMemAdapter

        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config.get('agent_chunk_size', dataset_config.get('chunk_size'))
        self.agent_start_time = time.time()
        self.lightmem_marker_path = os.path.join(self.agent_save_to_folder, "lightmem_ready.txt")
        self.lightmem_db_path = os.path.join(self.agent_save_to_folder, "lightmem_data")
        self.lightmem_collection_name = f"{dataset_config['sub_dataset']}_lightmem"

        api_key_env = agent_config.get("lightmem_api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"LightMem requires {api_key_env} or OPENAI_API_KEY to be set."
            )

        lightmem_model = os.environ.get("LIGHTMEM_MODEL", agent_config['model'])
        lightmem_base_url = (
            os.environ.get("LIGHTMEM_BASE_URL")
            or agent_config.get('lightmem_base_url')
            or os.environ.get("OPENAI_BASE_URL")
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=lightmem_base_url,
            http_client=httpx.Client(trust_env=False),
        )
        lightmem_embedding_model = os.environ.get(
            "LIGHTMEM_EMBEDDING_MODEL",
            agent_config.get('lightmem_embedding_model', 'text-embedding-3-small')
        )
        lightmem_embedding_dims = int(
            os.environ.get(
                "LIGHTMEM_EMBEDDING_DIMENSION",
                str(agent_config.get('lightmem_embedding_dims', 1536))
            )
        )

        self.lightmem = LightMemAdapter(
            api_key=api_key,
            model=lightmem_model,
            base_url=lightmem_base_url,
            embedding_base_url=(
                agent_config.get('lightmem_embedding_base_url')
                or self.embedding_base_url
                or self._get_env_value(self.embedding_base_url_env)
            ),
            db_path=self.lightmem_db_path,
            collection_name=self.lightmem_collection_name,
            embedding_model=lightmem_embedding_model,
            embedding_dims=lightmem_embedding_dims,
            retrieve_num=self.retrieve_num,
            ingest_mode=agent_config.get('lightmem_ingest_mode', 'direct'),
            memory_manager_backend=agent_config.get('lightmem_memory_manager_backend', 'openai'),
            embedding_backend=agent_config.get('lightmem_embedding_backend', 'openai'),
            qdrant_on_disk=agent_config.get('lightmem_qdrant_on_disk', True),
            messages_use=agent_config.get('lightmem_messages_use', 'user_only'),
            metadata_generate=agent_config.get('lightmem_metadata_generate', False),
            text_summary=agent_config.get('lightmem_text_summary', False),
            pre_compress=agent_config.get('lightmem_pre_compress', False),
            topic_segment=agent_config.get('lightmem_topic_segment', False),
            index_strategy=agent_config.get('lightmem_index_strategy', 'embedding'),
            retrieve_strategy=agent_config.get('lightmem_retrieve_strategy', 'embedding'),
            update_mode=agent_config.get('lightmem_update', 'offline'),
        )

    def _initialize_a_mem_agent(self, agent_config, dataset_config):
        """Initialize A-MEM with a per-context local state file."""
        from methods.a_mem.a_mem_adapter import AMemAdapter

        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.agent_start_time = time.time()
        self.a_mem_marker_path = os.path.join(self.agent_save_to_folder, "a_mem_ready.txt")
        self.a_mem_state_path = os.path.join(self.agent_save_to_folder, "a_mem_state.json")

        api_key_env = agent_config.get("a_mem_api_key_env", "MODELSCOPE_API_KEY")
        api_key = (
            os.environ.get(api_key_env)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                f"A-MEM requires {api_key_env} or OPENAI_API_KEY to be set."
            )

        embedding_api_key_env = agent_config.get("a_mem_embedding_api_key_env")
        embedding_api_key = (
            os.environ.get(embedding_api_key_env) if embedding_api_key_env else None
        ) or api_key

        embedding_base_url_env = agent_config.get("a_mem_embedding_base_url_env")
        embedding_base_url = (
            agent_config.get("a_mem_embedding_base_url")
            or (os.environ.get(embedding_base_url_env) if embedding_base_url_env else None)
            or agent_config.get('a_mem_base_url')
            or os.environ.get("OPENAI_BASE_URL")
        )
        llm_base_url_env = agent_config.get("a_mem_base_url_env")
        llm_base_url = (
            agent_config.get("a_mem_base_url")
            or (os.environ.get(llm_base_url_env) if llm_base_url_env else None)
            or os.environ.get("OPENAI_BASE_URL")
        )

        # Keep the wrapper's generic OpenAI-compatible client aligned with the
        # method-specific endpoint configuration. A-MEM currently answers via
        # its own adapter path, but the wrapper client may still be used by
        # shared helper flows and debug utilities.
        if not self.api_key_env:
            self.api_key_env = api_key_env
        if not self.base_url_env and llm_base_url_env:
            self.base_url_env = llm_base_url_env
        if not self.base_url:
            self.base_url = llm_base_url
        self.client = self._create_oai_client()

        self.a_mem = AMemAdapter(
            model=agent_config['model'],
            backend=agent_config.get('a_mem_backend', 'openai'),
            retrieve_k=self.retrieve_num,
            embedding_model=agent_config.get('a_mem_embedding_model', 'all-MiniLM-L6-v2'),
            embedding_provider=agent_config.get('a_mem_embedding_provider'),
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_base_url,
            api_key=api_key,
            api_base=agent_config.get('a_mem_base_url') or os.environ.get("OPENAI_BASE_URL"),
            sglang_host=agent_config.get('a_mem_sglang_host', 'http://localhost'),
            sglang_port=agent_config.get('a_mem_sglang_port', 30000),
            state_path=self.a_mem_state_path,
            retriever_type=agent_config.get('a_mem_retriever_type', 'dense'),
            hybrid_bm25_weight=agent_config.get('a_mem_hybrid_bm25_weight', 0.5),
        )

    def _initialize_memochat_agent(self, agent_config, dataset_config):
        """Initialize the upstream-style MemoChat adapter."""
        from methods.memochat.memochat_adapter import MemoChatAdapter

        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.context = ''
        self.client = self._create_oai_client()
        self.agent_start_time = time.time()
        self.memochat_marker_path = os.path.join(self.agent_save_to_folder, "memochat_ready.txt")
        self.memochat_state_path = os.path.join(self.agent_save_to_folder, "memochat_state.json")

        embedding_base_url = (
            self.embedding_base_url
            or self._get_env_value(self.embedding_base_url_env)
            or self._resolve_base_url()
        )
        embedding_api_key = self._resolve_embedding_api_key(["OPENAI_API_KEY"])
        embedding_dimensions = self._infer_embedding_dimensions(
            self.embedding_model,
            agent_config.get("embedding_dim"),
        )

        self.memochat = MemoChatAdapter(
            state_path=self.memochat_state_path,
            model=self.model,
            base_url=self._resolve_base_url(),
            api_key=self._resolve_llm_api_key(["OPENAI_API_KEY"]),
            embedding_model=self.embedding_model,
            embedding_provider=self.embedding_provider,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_dimensions=embedding_dimensions,
            retrieve_num=self.retrieve_num,
            summary_trigger_chunks=agent_config.get("memochat_summary_trigger_chunks", 6),
            keep_recent_chunks=agent_config.get("memochat_keep_recent_chunks", 2),
            max_topics_per_window=agent_config.get("memochat_max_topics_per_window", 3),
            summary_chars=agent_config.get("memochat_summary_chars", 320),
            keyword_limit=agent_config.get("memochat_keyword_limit", 16),
            topic_top_k=agent_config.get("memochat_topic_top_k", max(3, min(4, self.retrieve_num))),
            recent_top_k=agent_config.get("memochat_recent_top_k", 2),
            dialogs_per_topic=agent_config.get("memochat_dialogs_per_topic", 2),
            llm_max_tokens=agent_config.get("memochat_llm_max_tokens", 384),
            use_llm_topic_segmentation=agent_config.get("memochat_use_llm_topic_segmentation", True),
        )

    def _initialize_memoryos_agent(self, agent_config, dataset_config):
        """Initialize the upstream-style MemoryOS adapter."""
        from methods.memoryos.memoryos_adapter import MemoryOSAdapter

        self._ensure_explicit_embedding_config()
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.context = ''
        self.client = self._create_oai_client()
        self.agent_start_time = time.time()
        self.memoryos_marker_path = os.path.join(self.agent_save_to_folder, "memoryos_ready.txt")
        self.memoryos_state_path = os.path.join(self.agent_save_to_folder, "memoryos_state.json")
        self.memoryos_source_map_path = os.path.join(self.agent_save_to_folder, "memoryos_source_map.json")
        self.memoryos_source_map = {}

        embedding_base_url = (
            self.embedding_base_url
            or self._get_env_value(self.embedding_base_url_env)
            or self._resolve_base_url()
        )
        embedding_api_key = self._resolve_embedding_api_key(["OPENAI_API_KEY"])
        embedding_dimensions = self._infer_embedding_dimensions(
            self.embedding_model,
            agent_config.get("embedding_dim"),
        )

        self.memoryos = MemoryOSAdapter(
            state_path=self.memoryos_state_path,
            model=self.model,
            base_url=self._resolve_base_url(),
            api_key=self._resolve_llm_api_key(["OPENAI_API_KEY"]),
            embedding_model=self.embedding_model,
            embedding_provider=self.embedding_provider,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_dimensions=embedding_dimensions,
            retrieve_num=self.retrieve_num,
            short_term_capacity=agent_config.get("memoryos_short_term_capacity", 1),
            mid_term_capacity=agent_config.get("memoryos_mid_term_capacity", 2000),
            queue_capacity=agent_config.get("memoryos_queue_capacity", max(10, self.retrieve_num)),
            topic_similarity_threshold=agent_config.get("memoryos_topic_similarity_threshold", 0.6),
            heat_threshold=agent_config.get("memoryos_heat_threshold", 5.0),
            summary_chars=agent_config.get("memoryos_summary_chars", 384),
            keyword_limit=agent_config.get("memoryos_keyword_limit", 16),
            segment_threshold=agent_config.get("memoryos_segment_threshold", 0.1),
            page_threshold=agent_config.get("memoryos_page_threshold", 0.1),
            knowledge_threshold=agent_config.get("memoryos_knowledge_threshold", 0.1),
            llm_max_tokens=agent_config.get("memoryos_llm_max_tokens", 512),
        )
        if os.path.exists(self.memoryos_source_map_path):
            with open(self.memoryos_source_map_path, "r", encoding="utf-8") as f:
                self.memoryos_source_map = json.load(f)

    def _initialize_memtree_agent(self, agent_config, dataset_config):
        """Initialize MemTree with a local persistent state file."""
        from methods.memtree import MemTreeAdapter

        if self.model_provider not in {"openai", "azure_openai", "openai_compatible"}:
            raise NotImplementedError(
                "MemTree integration currently requires an OpenAI-compatible chat backend "
                "for the answer-generation and tree-aggregation steps."
            )

        self.retrieve_num = agent_config["retrieve_num"]
        self.chunk_size = agent_config["agent_chunk_size"]
        self.agent_start_time = time.time()
        self.memtree_marker_path = os.path.join(self.agent_save_to_folder, "memtree_ready.txt")
        self.memtree_state_path = os.path.join(self.agent_save_to_folder, "memtree_state.pkl")

        aggregate_api_key = self._get_env_value(self.api_key_env, ["OPENAI_API_KEY"])
        embedding_api_key = (
            self._get_env_value(self.embedding_api_key_env, ["OPENAI_API_KEY"])
            or aggregate_api_key
        )
        embedding_base_url = (
            self.embedding_base_url
            or self._get_env_value(self.embedding_base_url_env)
        )

        self.memtree = MemTreeAdapter(
            state_path=self.memtree_state_path,
            llm_model=agent_config.get("memtree_aggregate_model") or self.model,
            llm_provider=self.model_provider,
            llm_base_url=self._resolve_base_url(),
            llm_api_key=aggregate_api_key,
            llm_azure_endpoint=self.azure_endpoint,
            llm_azure_api_version=self.azure_api_version,
            llm_temperature=agent_config.get("memtree_aggregate_temperature", 0.0),
            summary_max_tokens=agent_config.get("memtree_summary_max_tokens", 256),
            embedding_model=(
                agent_config.get("memtree_embedding_model")
                or self.embedding_model
                or "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_provider=self.embedding_provider,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_azure_endpoint=self.embedding_azure_endpoint,
            embedding_azure_api_version=self.embedding_azure_api_version,
            retrieve_num=self.retrieve_num,
            base_threshold=agent_config.get("memtree_base_threshold", 0.4),
            rate=agent_config.get("memtree_rate", 0.5),
            max_depth=agent_config.get("memtree_max_depth", 15),
        )
        self.client = self._create_oai_client()

    def _initialize_cognee_agent(self, agent_config, dataset_config):
        """Initialize Cognee using the vendored official runtime configuration API."""
        self.context = ''
        self.chunks = []
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.agent_start_time = time.time()
        self.client = self._create_oai_client()
        self._ensure_explicit_embedding_config()
        self.cognee_ready_path = os.path.join(self.agent_save_to_folder, "cognee_ready.txt")
        self.cognee_system_dir = os.path.abspath(os.path.join(self.agent_save_to_folder, ".cognee_system"))
        self.cognee_data_dir = os.path.abspath(os.path.join(self.agent_save_to_folder, ".data_storage"))
        self.cognee_cache_dir = os.path.abspath(os.path.join(self.agent_save_to_folder, ".cognee_cache"))
        self.cognee_pending_datasets = set()
        os.makedirs(self.cognee_system_dir, exist_ok=True)
        os.makedirs(self.cognee_data_dir, exist_ok=True)
        os.makedirs(self.cognee_cache_dir, exist_ok=True)
        os.environ["COGNEE_SKIP_CONNECTION_TEST"] = (
            "true" if agent_config.get("cognee_skip_connection_test", True) else "false"
        )

        import cognee
        from cognee.base_config import get_base_config
        from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import create_embedding_engine
        from cognee.modules.search.types import SearchType

        cognee.config.system_root_directory(self.cognee_system_dir)
        cognee.config.data_root_directory(self.cognee_data_dir)
        get_base_config().cache_root_directory = self.cognee_cache_dir

        cognee_llm_model = self.model
        if self.model_provider == "openai_compatible":
            cognee.config.set_llm_provider("openai")
            cognee_llm_model = self._normalize_cognee_openai_compatible_model(cognee_llm_model)
        elif self.model_provider == "azure_openai":
            cognee.config.set_llm_provider("azure")
        elif self.model_provider:
            cognee.config.set_llm_provider(self.model_provider)
        cognee.config.set_llm_model(cognee_llm_model)
        api_key = self._resolve_llm_api_key(["OPENAI_API_KEY"])
        base_url = self._resolve_base_url()
        llm_api_key = api_key or ("EMPTY" if base_url else None)
        if llm_api_key:
            cognee.config.set_llm_api_key(llm_api_key)
        if base_url:
            cognee.config.set_llm_endpoint(base_url)

        cognee_llm_args = dict(agent_config.get("cognee_llm_args") or {})
        cognee_llm_streaming = agent_config.get("cognee_llm_streaming")
        if self._should_disable_qwen3_thinking():
            # Cognee issues its own LiteLLM requests, so Qwen3 compatibility
            # flags must be forwarded through Cognee's llm_args as well.
            cognee_llm_args.setdefault("enable_thinking", False)

            chat_template_kwargs = dict(cognee_llm_args.get("chat_template_kwargs") or {})
            chat_template_kwargs.setdefault("enable_thinking", False)
            cognee_llm_args["chat_template_kwargs"] = chat_template_kwargs

            extra_body = dict(cognee_llm_args.get("extra_body") or {})
            extra_body.setdefault("enable_thinking", False)
            extra_chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            extra_chat_template_kwargs.setdefault("enable_thinking", False)
            extra_body["chat_template_kwargs"] = extra_chat_template_kwargs
            cognee_llm_args["extra_body"] = extra_body

            if cognee_llm_streaming is None:
                cognee_llm_streaming = False

        if cognee_llm_args or cognee_llm_streaming is not None:
            cognee_llm_config = {"llm_args": cognee_llm_args}
            if cognee_llm_streaming is not None:
                cognee_llm_config["llm_streaming"] = bool(cognee_llm_streaming)
            cognee.config.set_llm_config(cognee_llm_config)

        embedding_model = self.embedding_model or agent_config.get("text_embedding")
        embedding_endpoint = (
            self.embedding_base_url
            or self._get_env_value(self.embedding_base_url_env)
            or base_url
        )
        embedding_api_key = self._resolve_embedding_api_key(["OPENAI_API_KEY"]) or api_key
        if embedding_endpoint and not embedding_api_key:
            embedding_api_key = "EMPTY"
        embedding_provider = (
            self.embedding_provider
            or ("openai_compatible" if embedding_endpoint else "openai")
        )
        if embedding_provider == "azure_openai":
            embedding_provider = "azure"

        if embedding_provider:
            cognee.config.set_embedding_provider(embedding_provider)
        if embedding_model:
            cognee.config.set_embedding_model(embedding_model)
        embedding_dimensions = self._infer_embedding_dimensions(
            embedding_model,
            agent_config.get("embedding_dim"),
        )
        if embedding_dimensions and hasattr(cognee.config, "set_embedding_dimensions"):
            cognee.config.set_embedding_dimensions(embedding_dimensions)
        if embedding_endpoint and hasattr(cognee.config, "set_embedding_endpoint"):
            cognee.config.set_embedding_endpoint(embedding_endpoint)
        if embedding_api_key and hasattr(cognee.config, "set_embedding_api_key"):
            cognee.config.set_embedding_api_key(embedding_api_key)
        if agent_config.get("cognee_vector_db_provider"):
            cognee.config.set_vector_db_provider(agent_config["cognee_vector_db_provider"])
        if agent_config.get("cognee_graph_provider"):
            cognee.config.set_graph_database_provider(agent_config["cognee_graph_provider"])
        create_embedding_engine.cache_clear()
        requested_search_type = str(agent_config.get("cognee_search_type", "CHUNKS") or "CHUNKS").strip().upper()
        allowed_search_types = {"CHUNKS", "CHUNKS_LEXICAL"}
        if requested_search_type not in allowed_search_types:
            raise ValueError(
                "Unsupported cognee_search_type={!r}. Supported values are: {}.".format(
                    requested_search_type,
                    ", ".join(sorted(allowed_search_types)),
                )
            )
        self.cognee_search_type = SearchType[requested_search_type]

    def _initialize_zep_agent(self, agent_config):
        from zep_cloud.client import Zep
        from methods.zep.zep import OpenAIAgent
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.context_id = -1

        zep_api_key = os.getenv("ZEP_API_KEY")
        zep_base_url = os.getenv("ZEP_API_URL")
        client_kwargs = {}
        if zep_api_key:
            client_kwargs["api_key"] = zep_api_key
        if zep_base_url:
            client_kwargs["base_url"] = (
                zep_base_url if zep_base_url.endswith("/api/v2") else f"{zep_base_url.rstrip('/')}/api/v2"
            )
        self.client = Zep(**client_kwargs)
        zep_provider = self._normalize_provider(
            agent_config.get("provider") or agent_config.get("service_name") or self.model_provider
        )
        self.oai_client = OpenAIAgent(
            model=self.model,
            provider=zep_provider,
            base_url=self.base_url,
            base_url_env=self.base_url_env,
            api_key_env=agent_config.get("api_key_env") or self.api_key_env,
            azure_endpoint=self.azure_endpoint,
            azure_api_version=self.azure_api_version,
            api_dict={
                "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            },
            temperature=self.temperature,
        )
        self.agent_start_time = time.time()

    def _initialize_memagent(self, agent_config):
        """Initialize MemAgent with recurrent memory configuration."""
        from methods.memagent import build_memagent
        self.context = ''
        self.memagent = build_memagent(agent_config)

        self.agent_start_time = time.time()

    def _initialize_memos_agent(self, agent_config, dataset_config):
        """Initialize MemOS (MemoryOS) agent with textual memory configuration."""
        # MemOS ships a vendored `deprecation.py` at its source root. Expose
        # its source tree as the `memos` namespace directly instead of adding
        # the whole root to sys.path, so other adapters (e.g. Cognee via
        # LanceDB) keep using the third-party `deprecation` package.
        MOS = _import_from_vendor_namespace(
            "memos.mem_os.main",
            "memos",
            "./methods/MemOS/source/src",
        ).MOS
        get_default = _import_from_vendor_namespace(
            "memos.mem_os.utils.default_config",
            "memos",
            "./methods/MemOS/source/src",
        ).get_default

        self.retrieve_num = agent_config.get('retrieve_num', 5)
        self.context = ''
        self.agent_start_time = time.time()
        self.memos_marker_path = os.path.join(self.agent_save_to_folder, "memos_ready.txt")
        self.memos_data_dir = os.path.join(self.agent_save_to_folder, "memos_data")
        self.memos_qdrant_dir = os.path.join(self.memos_data_dir, "qdrant")
        os.makedirs(self.memos_data_dir, exist_ok=True)
        os.makedirs(self.memos_qdrant_dir, exist_ok=True)

        # Resolve API credentials via the same logic as other agents
        api_key = self._get_env_value(self.api_key_env, ["OPENAI_API_KEY"])
        api_base = self._resolve_base_url() or "https://api.openai.com/v1"
        if not api_key:
            raise RuntimeError(
                "MemOS requires an API key. Set api_key_env to the env var name "
                "holding the key, or set the OPENAI_API_KEY environment variable."
            )

        text_mem_type = agent_config.get("text_mem_type", "general_text")
        # Include the per-context experiment folder (e.g. "exp_0") so that
        # each context gets an isolated MemOS user and avoids memory leaks
        # across different benchmark contexts.
        context_tag = os.path.basename(self.agent_save_to_folder)
        user_id = f"bench_{dataset_config['sub_dataset']}_{context_tag}"

        # Embedding config: prefer separate embedding endpoint if provided
        embedding_api_key = (
            self._get_env_value(self.embedding_api_key_env, ["OPENAI_API_KEY"])
            or api_key
        )
        embedding_api_base = (
            self.embedding_base_url
            or self._get_env_value(self.embedding_base_url_env)
            or api_base
        )
        embedder_model = (
            agent_config.get("embedder_model")
            or agent_config.get("embedding_model")
            or "text-embedding-3-large"
        )
        embedding_dimensions = self._infer_embedding_dimensions(
            embedder_model,
            agent_config.get("embedding_dim", agent_config.get("embedding_dimensions")),
        )

        # Increase embedding timeout for slow proxies
        os.environ.setdefault("MOS_EMBEDDER_TIMEOUT", str(agent_config.get("memos_embedder_timeout", 30)))

        # Build MOS config and cube via get_default helper
        kwargs = {
            "model_name": self.model,
            "temperature": self.temperature,
            "embedder_model": embedder_model,
            "embedder_api_key": embedding_api_key,
            "embedder_base_url": embedding_api_base,
            "qdrant_path": self.memos_qdrant_dir,
            "chunk_size": agent_config.get("chunk_size", 512),
            "chunk_overlap": agent_config.get("chunk_overlap", 128),
            "top_k": self.retrieve_num,
            "max_turns_window": agent_config.get("max_turns_window", 20),
            "enable_mem_scheduler": agent_config.get("enable_mem_scheduler", False),
            "tokenizer_or_token_counter": agent_config.get("tokenizer_encoding", "gpt2"),
        }
        for optional_key in ("cube_id", "collection_name", "memory_filename"):
            if agent_config.get(optional_key):
                kwargs[optional_key] = agent_config.get(optional_key)
        if embedding_dimensions is not None:
            kwargs["vector_dimension"] = embedding_dimensions
            kwargs["embedding_dimension"] = embedding_dimensions
        # Tree-text specific settings
        if text_mem_type == "tree_text":
            kwargs.update({
                "neo4j_uri": agent_config.get("neo4j_uri", "bolt://localhost:7687"),
                "neo4j_user": agent_config.get("neo4j_user", "neo4j"),
                "neo4j_password": agent_config.get("neo4j_password", "12345678"),
                "neo4j_db_name": agent_config.get("neo4j_db_name", "neo4j"),
                "use_multi_db": agent_config.get("use_multi_db", False),
            })

        # Build separate embedder config if the embedding endpoint differs from chat
        if embedding_api_base != api_base or embedding_api_key != api_key:
            # get_default uses the same key/base for both chat and embedding;
            # we override the embedder later via the cube config
            pass

        mos_config, default_cube = get_default(
            openai_api_key=api_key,
            openai_api_base=api_base,
            text_mem_type=text_mem_type,
            user_id=user_id,
            **kwargs,
        )

        if agent_config.get("pro_mode", False):
            mos_config.PRO_MODE = True

        self.memos = MOS(config=mos_config)
        self.memos.register_mem_cube(default_cube)
        self.memos_search_mode = agent_config.get("search_mode", "fast")
        self.memos_memorize_mode = agent_config.get("memorize_mode", "fine")

        # Suppress verbose MemOS and neo4j logging
        memos_log_level = agent_config.get("memos_log_level", "WARNING").upper()
        import logging as _logging
        _level = getattr(_logging, memos_log_level, _logging.WARNING)
        _logging.getLogger("memos").setLevel(_level)
        _logging.getLogger("neo4j").setLevel(_level)

        # Create an OAI client for the final answer generation step
        # using the same framework logic as other agents.
        self.client = self._create_oai_client()

    def _initialize_everos_agent(self, agent_config, dataset_config):
        """Initialize EverOS through its official HTTP API surface."""
        from methods.everos import EverOSAdapter

        self.retrieve_num = agent_config.get("retrieve_num", 5)
        self.chunk_size = agent_config.get(
            "agent_chunk_size", dataset_config.get("chunk_size", 0)
        )
        self.agent_start_time = time.time()
        self.everos_marker_path = os.path.join(self.agent_save_to_folder, "everos_ready.txt")

        everos_base_url = agent_config.get("everos_base_url")
        if not everos_base_url:
            raise RuntimeError(
                "EverOS integration requires 'everos_base_url' in the agent config, "
                "for example http://localhost:1995."
            )

        everos_api_key = self._get_env_value(agent_config.get("everos_api_key_env"))
        group_entropy = f"{self.agent_save_to_folder}|{time.time_ns()}"
        self.everos_group_id = EverOSAdapter.build_group_id(
            self.agent_name,
            self.sub_dataset,
            group_entropy,
        )
        self.everos_group_name = agent_config.get("everos_group_name") or (
            f"{self.agent_name}-{self.sub_dataset}"
        )

        self.everos = EverOSAdapter(
            base_url=everos_base_url,
            api_key=everos_api_key,
            group_id=self.everos_group_id,
            group_name=self.everos_group_name,
            scene=agent_config.get("everos_scene", "assistant"),
            default_timezone=agent_config.get("everos_default_timezone", "UTC"),
            retrieve_method=agent_config.get("everos_retrieve_method", "rrf"),
            memory_types=agent_config.get("everos_memory_types", ["episodic_memory"]),
            top_k=self.retrieve_num,
            timeout_seconds=agent_config.get("everos_timeout_seconds", 60),
            sync_mode=agent_config.get("everos_sync_mode", True),
            chunk_time_gap_minutes=agent_config.get("everos_chunk_time_gap_minutes", 360),
        )
        self.everos.prepare()
        self.client = self._create_oai_client()

    def _initialize_zep_local_agent(self, agent_config):
        from methods.zep_local import GraphitiLocalMemory

        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.context_id = -1
        self.zep_local_episode_counter = 0
        self.zep_local_namespace_prefix = agent_config.get("zep_local_namespace_prefix", "").strip()

        embedding_model_name = (
            agent_config.get("text_embedding")
            or self.embedding_model
            or agent_config.get("embedding_model")
        )
        embedding_base_url = (
            self.embedding_base_url
            or self._get_env_value(self.embedding_base_url_env)
            or agent_config.get("embedding_base_url")
        )
        embedding_api_key = self._resolve_embedding_api_key(["OPENAI_API_KEY"])
        embedding_dim = agent_config.get("embedding_dim")
        if not embedding_dim:
            if embedding_model_name and "Qwen3-Embedding-4B" in embedding_model_name:
                embedding_dim = 2560
            elif embedding_model_name == "text-embedding-3-large":
                embedding_dim = 3072
            else:
                embedding_dim = 1536

        self.zep_local = GraphitiLocalMemory(
            neo4j_uri=agent_config.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=agent_config.get("neo4j_user", "neo4j"),
            neo4j_password=agent_config.get("neo4j_password", "neo4jneo4j"),
            llm_model=self.model,
            llm_small_model=agent_config.get("zep_local_small_model") or self.model,
            llm_api_key=self._resolve_llm_api_key(["OPENAI_API_KEY"]),
            llm_base_url=self._resolve_base_url(),
            llm_temperature=agent_config.get("zep_local_llm_temperature", self.temperature),
            llm_max_tokens=agent_config.get("zep_local_llm_max_tokens", 512),
            episode_max_chars=agent_config.get("zep_local_episode_max_chars", 3000),
            embedding_model_name=embedding_model_name,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            embedding_dim=embedding_dim,
            answer_model=agent_config.get("zep_local_answer_model") or self.model,
            answer_api_key=(
                agent_config.get("zep_local_answer_api_key")
                or self._resolve_api_key(
                    explicit_env_name=agent_config.get("zep_local_answer_api_key_env") or self.api_key_env,
                    fallback_env_names=["OPENAI_API_KEY"],
                )
            ),
            answer_base_url=(
                agent_config.get("zep_local_answer_base_url")
                or self._resolve_base_url()
            ),
            answer_temperature=agent_config.get("zep_local_answer_temperature", 0.0),
            answer_max_tokens=agent_config.get("zep_local_answer_max_tokens", self.max_tokens),
        )
        self.agent_start_time = time.time()

    def _initialize_rag_agent(self, agent_config, dataset_config):
        """Initialize RAG agent with retrieval configuration."""
        self.context = ''
        self.chunks = []
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = dataset_config['chunk_size']
        self.context_len = 0
        self.context_id = -1

    def send_message(self, message, memorizing=False, query_id=None, context_id=None, eval_metadata=None):
        """
        Send a message to the agent for either memorization or querying.

        Args:
            message: The message content (context for memorization, query for answering)
            memorizing: Whether to memorize the message (True) or answer it (False)
            query_id: Unique identifier for the query
            context_id: Unique identifier for the context

        Returns:
            dict or str: Agent response with metadata (for queries) or confirmation (for memorization)
        """
        if memorizing:
            self._current_query_id = None
            self._current_context_id = context_id
            self._current_eval_metadata = None
            self._last_llm_trace = None
        else:
            self._current_query_id = query_id
            self._current_context_id = context_id
            self._current_eval_metadata = eval_metadata
            self._last_llm_trace = None

        message = self._normalize_message_payload(message, memorizing=memorizing)

        # Route to appropriate agent handler based on agent type
        if 'Long_context_agent' in self.agent_name:
            return self._handle_long_context_agent(message, memorizing)
        elif any(self._is_agent_type(agent_type) for agent_type in ["letta", "cognee", "mem0", "memtree", "memochat", "memoryos", "zep", "simplemem", "lightmem", "a_mem", "everos", "memagent", "MemOS"]):
            return self._handle_memory_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("rag"):
            return self._handle_rag_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _handle_long_context_agent(self, message, memorizing):
        """Handle message processing for long context agents."""
        if memorizing:
            # Add message to context memory
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:
            # Process query with context
            return self._query_long_context_agent(message)

    def _query_long_context_agent(self, message):
        """Process a query for long context agents."""
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        base_reserved_tokens = max(self.max_tokens + 2048, 4096)
        retry_reserves = [
            base_reserved_tokens,
            max(base_reserved_tokens, 6144),
            max(base_reserved_tokens, 8192),
            max(base_reserved_tokens, 12288),
        ]
        retry_reserves = list(dict.fromkeys(retry_reserves))

        # Query the model
        start_time = time.time()

        if self.model_provider in {"openai", "azure_openai", "openai_compatible"}:
            last_exception = None
            for reserved_tokens in retry_reserves:
                fitted_context = self._fit_text_to_prompt_token_limit(
                    self.context,
                    message,
                    system_message=system_message,
                    reserved_tokens=reserved_tokens,
                    truncation_strategy="tail",
                )
                full_message = "\n".join(
                    part for part in [fitted_context.strip(), message] if part and part.strip()
                )
                formatted_message = format_chat(message=full_message, system_message=system_message)
                request_kwargs = {
                    "model": self.model,
                    "messages": formatted_message,
                }
                if "o4" not in self.model:
                    request_kwargs["temperature"] = self.temperature
                    request_kwargs["max_tokens"] = self.max_tokens
                request_kwargs = self._prepare_openai_request_kwargs(request_kwargs)
                try:
                    response = self.client.chat.completions.create(**request_kwargs)
                    return self._format_openai_response(response, start_time)
                except Exception as exc:
                    if "maximum context length" not in str(exc).lower():
                        raise
                    last_exception = exc
            if last_exception is not None:
                raise last_exception

        elif self.model_provider == "anthropic":
            fitted_context = self._fit_text_to_prompt_token_limit(
                self.context,
                message,
                system_message=system_message,
                reserved_tokens=base_reserved_tokens,
                truncation_strategy="tail",
            )
            full_message = "\n".join(
                part for part in [fitted_context.strip(), message] if part and part.strip()
            )
            return self._query_claude(full_message, system_message, start_time)

        elif self.model_provider == "gemini":
            fitted_context = self._fit_text_to_prompt_token_limit(
                self.context,
                message,
                system_message=system_message,
                reserved_tokens=base_reserved_tokens,
                truncation_strategy="tail",
            )
            full_message = "\n".join(
                part for part in [fitted_context.strip(), message] if part and part.strip()
            )
            formatted_message = format_chat(message=full_message, system_message=system_message)
            return self._query_gemini(formatted_message, start_time)

        else:
            raise NotImplementedError(f"Model provider not supported: {self.model_provider}")

    def _truncate_context_if_needed(self, tokenizer):
        """Truncate context if it exceeds limits."""
        # Truncate context if it exceeds the context_max_length
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.context_max_length:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.context_max_length:])

        # Truncate if context exceeds the input_length_limit
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])

    def _format_openai_response(self, response, start_time):
        """Format OpenAI API response into standard output format."""
        return self._create_standard_response(
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            0,
            time.time() - start_time
        )

    def _fit_retrieved_contexts_to_token_limit(
        self,
        contexts,
        message,
        tokenizer,
        label_prefix="Memory",
        system_message="",
        reserved_tokens=1024,
    ):
        """Keep retrieved passages within the configured prompt budget."""
        if not contexts:
            return []

        def _encode(text):
            try:
                return tokenizer.encode(text, disallowed_special=())
            except TypeError:
                return tokenizer.encode(text)

        system_tokens = len(_encode(system_message or ""))
        prompt_budget_limit = self.input_length_limit
        if self.model_context_window:
            prompt_budget_limit = min(prompt_budget_limit, int(self.model_context_window))
        budget = max(256, prompt_budget_limit - system_tokens - reserved_tokens)

        def _decode(tokens):
            if hasattr(tokenizer, "decode"):
                return tokenizer.decode(tokens)
            return "".join(tokens)

        used_tokens = len(_encode(message))
        fitted_contexts = []

        for index, text in enumerate(contexts, start=1):
            prefix = f"{label_prefix} {index}:\n"
            prefix_tokens = len(_encode(prefix))
            text_tokens = _encode(text)
            remaining = budget - used_tokens - prefix_tokens

            if remaining <= 0:
                break

            if len(text_tokens) > remaining:
                truncated_text = _decode(text_tokens[:remaining]).strip()
                if truncated_text:
                    fitted_contexts.append(truncated_text)
                break

            fitted_contexts.append(text)
            used_tokens += prefix_tokens + len(text_tokens)

        if fitted_contexts:
            return fitted_contexts

        fallback_prefix_tokens = len(_encode(f"{label_prefix} 1:\n"))
        fallback_budget = max(0, budget - used_tokens - fallback_prefix_tokens)
        if fallback_budget <= 0:
            return []

        fallback_tokens = _encode(contexts[0])[:fallback_budget]
        fallback_text = _decode(fallback_tokens).strip()
        return [fallback_text] if fallback_text else []

    def _query_claude(self, message, system_message, start_time):
        """Query Claude model with proper formatting."""
        formatted_message = format_chat(message=message, system_message=system_message, include_system=False)
        response = self.client.messages.create(
            model=self.model,
            messages=formatted_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return self._create_standard_response(
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
            0,
            time.time() - start_time
        )

    def _query_gemini(self, formatted_message, start_time):
        """Query Gemini model with proper configuration."""
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=formatted_message[1]["content"],
            config=types.GenerateContentConfig(
                system_instruction=formatted_message[0]["content"],
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
        )
        return self._create_standard_response(
            response.text,
            response.usage_metadata.prompt_token_count,
            response.usage_metadata.candidates_token_count,
            0,
            time.time() - start_time
        )

    def _handle_memory_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for memory-based agents."""
        if self._is_agent_type("letta"):
            return self._handle_letta_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("cognee"):
            return self._handle_cognee_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("mem0"):
            return self._handle_mem0_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("memtree"):
            return self._handle_memtree_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("memochat"):
            return self._handle_memochat_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("memoryos"):
            return self._handle_memoryos_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("simplemem"):
            return self._handle_simplemem_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("lightmem"):
            return self._handle_lightmem_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("a_mem"):
            return self._handle_a_mem_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("everos"):
            return self._handle_everos_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("zep_local"):
            return self._handle_zep_local_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("zep"):
            return self._handle_zep_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("memagent"):
            return self._handle_memagent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("MemOS"):
            return self._handle_memos_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Memory agent type not supported: {self.agent_name}")

    def _handle_zep_local_agent(self, message, memorizing, query_id, context_id):
        namespace = _build_zep_local_namespace(
            context_id,
            self.sub_dataset,
            self.zep_local_namespace_prefix,
        )

        if self.context_id != context_id and memorizing:
            self.context_id = context_id
            self.zep_local_episode_counter = 0

        if memorizing:
            episode_bodies = self.zep_local.extract_episode_bodies_from_chunk(message)
            for episode_body in episode_bodies:
                self.zep_local_episode_counter += 1
                self.zep_local.add_memory_sync(
                    namespace=namespace,
                    content=episode_body,
                    name=f"{namespace}_episode_{self.zep_local_episode_counter}",
                    source_description=f"{self.sub_dataset} context {context_id}",
                )
            return "Memorized"

        memory_construction_time = time.time() - self.agent_start_time
        search_results = self.zep_local.search_sync(question=message, namespace=namespace)
        retrieved_source_id_groups = self._extract_locomo_source_id_groups_from_items(
            getattr(search_results, "episodes", []),
            candidate_keys=("content",),
        )
        response, retrieval_context = self.zep_local.answer_query_sync(
            question=message,
            namespace=namespace,
            results=search_results,
        )
        query_time_len = time.time() - self.agent_start_time - memory_construction_time

        self.context_id = context_id

        output = {
            "output": response,
            "input_len": len(self.tokenizer.encode(str(retrieval_context) + "\n" + message, disallowed_special=())),
            "output_len": len(self.tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
        self.agent_start_time = time.time()
        return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

    def _handle_letta_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Letta agents."""
        # Format message based on context
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
        else:
            formatted_message = message

        # Handle memory construction time for queries
        memory_construction_time = 0 if memorizing else time.time() - self.agent_start_time

        # Reload agent for queries
        if not memorizing:
            if os.path.exists(self.agent_save_to_folder):
                self._reload_letta_query_runtime()
            else:
                print(f"\n\nAgent {self.agent_name} not found in {self.agent_save_to_folder}\n\n")

        # Process based on Letta mode
        response = self._process_letta_message(formatted_message, memorizing, query_id, context_id)

        if memorizing:
            return "Memorized"

        # Create response for queries
        tokenizer = self.tokenizer
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response,
            len(tokenizer.encode(message, disallowed_special=())),
            len(tokenizer.encode(response, disallowed_special=())),
            memory_construction_time,
            query_time_len
        )
        self.agent_start_time = time.time()  # Reset time
        return output

    def _process_letta_message(self, formatted_message, memorizing, query_id, context_id):
        """Process message with Letta client based on mode."""
        from letta_client import Letta

        try:
            if self.letta_mode == 'insert':
                if memorizing:
                    self.client.server.passage_manager.insert_passage(
                        agent_state=self.agent_state,
                        agent_id=self.agent_state.id,
                        text=formatted_message,
                        actor=self.client.user,
                    )
                    # import ipdb; ipdb.set_trace()
                    return "Memorized"
                else:
                    response = self.client.send_message(
                        agent_id=self.agent_state.id,
                        message=formatted_message,
                        role='user')
                    return self._extract_letta_response_text(response)

            elif self.letta_mode == 'chat':
                response = self.client.send_message(
                    agent_id=self.agent_state.id,
                    message=formatted_message,
                    role='user')

                if memorizing:
                    return "Memorized"
                else:
                    return self._extract_letta_response_text(response)
            elif self.letta_mode == 'api':
                response = self.client.agents.messages.create(
                    agent_id=self.agent_state.id,
                    messages=[
                        {
                            "role": "user",
                            "content": formatted_message,
                        },
                    ],
                )
                print(f"\n\n\nresponse: {response}\n\n\n")
                return response.messages[-1].content
        except Exception as e:
            print(f"\n\n\nerror: {e}\n\n\n")
            return f"Error: {type(e).__name__}: {e}"

    def _extract_letta_response_text(self, response):
        """Robustly recover the final user-visible message from Letta responses."""
        message_list = getattr(response, "messages", None)
        if message_list is None and isinstance(response, (list, tuple)):
            message_list = response
        if message_list is None:
            return str(response)

        def _extract_from_tool_call(tool_call):
            if tool_call is None:
                return None
            if isinstance(tool_call, dict):
                function_obj = tool_call.get("function")
                function_name = tool_call.get("name")
                arguments = tool_call.get("arguments")
                if isinstance(function_obj, dict):
                    function_name = function_name or function_obj.get("name")
                    arguments = arguments or function_obj.get("arguments")
            else:
                function_obj = getattr(tool_call, "function", None)
                function_name = getattr(tool_call, "name", None) or getattr(function_obj, "name", None)
                arguments = getattr(tool_call, "arguments", None) or getattr(function_obj, "arguments", None)
            if function_name != "send_message" or not arguments:
                return None
            try:
                parsed_arguments = json.loads(arguments)
            except Exception:
                return None
            for key in ("message", "content", "assistant_message"):
                value = parsed_arguments.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        for item in reversed(list(message_list)):
            assistant_message = None
            tool_call = None
            if isinstance(item, dict):
                message_type = item.get("message_type")
                if message_type == "assistant_message":
                    assistant_message = item.get("content") or item.get("assistant_message")
                tool_call = item.get("tool_call") or item.get("function_call")
            else:
                message_type = getattr(item, "message_type", None)
                if message_type == "assistant_message":
                    assistant_message = getattr(item, "content", None) or getattr(item, "assistant_message", None)
                tool_call = getattr(item, "tool_call", None) or getattr(item, "function_call", None)
                if assistant_message is None:
                    assistant_message = getattr(item, "assistant_message", None)

            if isinstance(assistant_message, str) and assistant_message.strip():
                return assistant_message.strip()

            extracted_tool_message = _extract_from_tool_call(tool_call)
            if extracted_tool_message:
                return extracted_tool_message

            content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

        return ""

    def _handle_cognee_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Cognee agents."""
        import cognee
        import asyncio
        from cognee.modules.search.types import SearchType

        dataset_name = f'default_dataset_{self.sub_dataset}_context_{context_id}'

        if memorizing:
            # Add context to Cognee knowledge base
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))

            # Queue raw content during memorization; run cognify once before querying
            # so we do not rebuild the graph for every incoming benchmark chunk.
            asyncio.run(cognee.add(formatted_message, dataset_name=dataset_name))
            self.cognee_pending_datasets.add(dataset_name)

            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:
            if dataset_name in self.cognee_pending_datasets:
                asyncio.run(cognee.cognify(datasets=[dataset_name], chunk_size=self.chunk_size))
                self.cognee_pending_datasets.discard(dataset_name)
            memory_construction_time = time.time() - self.agent_start_time
            searched_results = asyncio.run(cognee.search(
                query_text=message,
                query_type=self.cognee_search_type,
                top_k=self.retrieve_num,
                datasets=[dataset_name],
                only_context=True,
            ))
            retrieved_source_id_groups = self._extract_locomo_source_id_groups_from_items(
                searched_results if isinstance(searched_results, (list, tuple)) else [searched_results]
            )
            raw_contexts = self._flatten_text_items(searched_results)
            cleaned_contexts = self._clean_retrieved_memory_contexts(raw_contexts)
            retrieved_contexts = self._fit_memories_for_answer(
                question=message,
                contexts=cleaned_contexts or raw_contexts,
                prompt_override=self.agent_config.get("cognee_memory_answer_prompt"),
            )
            memories_str = (
                "\n".join(f"- {result}" for result in retrieved_contexts)
                if retrieved_contexts
                else "(No retrieved memories found.)"
            )
            response_text, prompt_tokens, completion_tokens = self._generate_answer_from_memories(
                question=message,
                memories_text=memories_str,
                prompt_override=self.agent_config.get("cognee_memory_answer_prompt"),
            )
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            memory_retrieval_length = self._count_tokens(memories_str)
            output = self._create_standard_response(
                response_text,
                prompt_tokens + memory_retrieval_length,
                completion_tokens,
                memory_construction_time,
                query_time_len
            )
            self._maybe_save_longmemeval_backfill_debug(
                query_id=query_id,
                context_id=context_id,
                retrieved_paragraphs=raw_contexts,
                query_text=message,
                response_text=response_text,
                retrieved_source_id_groups=retrieved_source_id_groups,
            )
            self.agent_start_time = time.time()  # Reset time
            return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

    def _handle_memochat_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for the upstream-style MemoChat adapter."""
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(
                context=message,
                **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {})
            )
            self.memochat.add_chunk(formatted_message)
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"

        memory_construction_time = time.time() - self.agent_start_time
        retrieval = self.memochat.retrieve(message)
        retrieved_contexts = self._fit_memories_for_answer(
            question=message,
            contexts=retrieval.get("combined_texts", []),
            prompt_override=self.agent_config.get("memochat_memory_answer_prompt"),
        )
        memories_str = (
            "\n".join(retrieved_contexts)
            if retrieved_contexts
            else "(No retrieved memories found.)"
        )
        response_text, prompt_tokens, completion_tokens = self._generate_answer_from_memories(
            question=message,
            memories_text=memories_str,
            prompt_override=self.agent_config.get("memochat_memory_answer_prompt"),
        )
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        memory_retrieval_length = self._count_tokens(memories_str)
        output = self._create_standard_response(
            response_text,
            prompt_tokens + memory_retrieval_length,
            completion_tokens,
            memory_construction_time,
            query_time_len,
        )
        self._maybe_save_longmemeval_backfill_debug(
            query_id=query_id,
            context_id=context_id,
            retrieved_paragraphs=retrieval.get("combined_texts", []),
            query_text=message,
            response_text=response_text,
            extra_fields={
                "related_topics": retrieval.get("related_topics", []),
                "related_summaries": retrieval.get("related_summaries", []),
                "related_dialogs": retrieval.get("related_dialogs", []),
            },
        )
        self.agent_start_time = time.time()
        return output

    def _handle_memoryos_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for the upstream-style MemoryOS adapter."""
        if memorizing:
            chunk_source_ids = parse_locomo_source_ids(message)
            clean_message = self._prepare_memory_chunk_for_storage(message)
            self.memoryos.add_chunk(clean_message)
            self._remember_locomo_source_ids_for_text(
                self.memoryos_source_map,
                clean_message,
                chunk_source_ids,
            )
            self.context += "\n" + clean_message
            self.context = self.context.strip()
            return "Memorized"

        memory_construction_time = time.time() - self.agent_start_time
        retrieval = self.memoryos.retrieve(message)
        raw_contexts = retrieval.get("combined_texts", [])
        retrieved_source_id_groups = self._lookup_locomo_source_id_groups_by_texts(
            self.memoryos_source_map,
            [entry.get("user_input", "") for entry in retrieval.get("retrieval_queue", [])],
        )
        cleaned_contexts = self._clean_retrieved_memory_contexts(raw_contexts)
        retrieved_contexts = self._fit_memories_for_answer(
            question=message,
            contexts=cleaned_contexts or raw_contexts,
            prompt_override=self.agent_config.get("memoryos_memory_answer_prompt"),
        )
        memories_str = (
            "\n".join(retrieved_contexts)
            if retrieved_contexts
            else "(No retrieved memories found.)"
        )
        response_text, prompt_tokens, completion_tokens = self._generate_answer_from_memories(
            question=message,
            memories_text=memories_str,
            prompt_override=self.agent_config.get("memoryos_memory_answer_prompt"),
        )
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        memory_retrieval_length = self._count_tokens(memories_str)
        output = self._create_standard_response(
            response_text,
            prompt_tokens + memory_retrieval_length,
            completion_tokens,
            memory_construction_time,
            query_time_len,
        )
        self._maybe_save_longmemeval_backfill_debug(
            query_id=query_id,
            context_id=context_id,
            retrieved_paragraphs=raw_contexts,
            query_text=message,
            response_text=response_text,
            retrieved_source_id_groups=retrieved_source_id_groups,
            extra_fields={"retrieval_queue": retrieval.get("retrieval_queue", [])},
        )
        self.agent_start_time = time.time()
        return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

    def _handle_mem0_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Mem0 agents."""
        user_id = f'context_{context_id}_{self.sub_dataset}'
        if memorizing:
            system_message = get_template(self.sub_dataset, 'system', self.agent_name)
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            chunk_source_ids = parse_locomo_source_ids(formatted_message)

            # Pass only benchmark-derived content into mem0 so benchmark prompt edits
            # stay centralized in benchmark_templates.py.
            memory_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": formatted_message},
            ]

            vector_results = self.memory.add(
                memory_messages,
                user_id=user_id,
                infer=self.mem0_add_infer,
            )
            normalized_add_results = self._normalize_mem0_add_results(vector_results)
            if chunk_source_ids:
                for result in normalized_add_results:
                    memory_id = str(result.get("id", "")).strip()
                    if memory_id:
                        self.mem0_source_map[memory_id] = chunk_source_ids
            return "Memorized"
        else:
            # Retrieve relevant memories and generate response
            memory_construction_time = time.time() - self.agent_start_time
            relevant_memories = self.memory.search(query=message, user_id=user_id, limit=self.retrieve_num)
            normalized_search_results = self._normalize_mem0_search_results(relevant_memories)

            retrieved_source_id_groups = []
            raw_memory_texts = []
            for entry in normalized_search_results:
                memory_id = str(entry.get("id", "")).strip()
                source_ids = self.mem0_source_map.get(memory_id, [])
                if source_ids:
                    retrieved_source_id_groups.append(source_ids)
                else:
                    retrieved_source_id_groups.extend(
                        self._extract_locomo_source_id_groups_from_items(
                            [entry],
                            candidate_keys=("memory", "text", "content"),
                        )
                    )

                memory_text = (
                    entry.get("memory")
                    or entry.get("text")
                    or entry.get("content")
                    or ""
                )
                if isinstance(memory_text, str) and memory_text.strip():
                    raw_memory_texts.append(memory_text)

            cleaned_memory_texts = self._clean_retrieved_memory_contexts(raw_memory_texts)
            memories_for_prompt = cleaned_memory_texts or [strip_locomo_metadata(text) for text in raw_memory_texts]
            memories_str = "\n".join(f"- {text}" for text in memories_for_prompt) or "(No retrieved memories found.)"
            memory_answer_template = (
                self.agent_config.get("mem0_memory_answer_prompt")
                or get_template(self.sub_dataset, 'memory_answer', self.agent_name)
            )
            formatted_query = memory_answer_template.format(
                memories=memories_str or "(No retrieved memories found.)",
                question=message,
            )

            llm_messages = [
                {"role": "system", "content": get_template(self.sub_dataset, 'system', self.agent_name)},
                {"role": "user", "content": formatted_query + "\n\nCurrent Time: " + time.strftime("%Y-%m-%d %H:%M:%S")},
            ]
            request_kwargs = self._prepare_openai_request_kwargs(
                {
                    "model": self.model,
                    "messages": llm_messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
            )
            response = self.client.chat.completions.create(
                **request_kwargs
            )

            memory_retrieval_length = len(self.tokenizer.encode(memories_str, disallowed_special=()))
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            print(f"\nmemory_length: {memory_retrieval_length}\n")

            output = self._create_standard_response(
                response.choices[0].message.content,
                response.usage.prompt_tokens + memory_retrieval_length,
                response.usage.completion_tokens,
                memory_construction_time,
                query_time_len
            )
            self._maybe_save_longmemeval_backfill_debug(
                query_id=query_id,
                context_id=context_id,
                retrieved_paragraphs=raw_memory_texts,
                query_text=message,
                response_text=response.choices[0].message.content,
                retrieved_source_id_groups=retrieved_source_id_groups,
            )
            self.agent_start_time = time.time()  # Reset time
            return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

    def _handle_simplemem_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for SimpleMem agents."""
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(
                context=message,
                **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {})
            )
            self.simplemem.add_chunk(formatted_message, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
            return "Memorized"

        memory_construction_time = time.time() - self.agent_start_time
        retrieved_entries = self.simplemem.retrieve_entries(message)
        retrieved_source_id_groups = [
            entry.get("source_ids", [])
            for entry in retrieved_entries
            if entry.get("source_ids")
        ]
        retrieved_contexts = self._fit_memories_for_answer(
            question=message,
            contexts=[entry["text"] for entry in retrieved_entries if entry.get("text")],
            prompt_override=self.agent_config.get("simplemem_memory_answer_prompt"),
        )
        memories_str = (
            "\n".join(retrieved_contexts)
            if retrieved_contexts
            else "(No retrieved memories found.)"
        )
        response, prompt_tokens, completion_tokens = self._generate_answer_from_memories(
            question=message,
            memories_text=memories_str,
            prompt_override=self.agent_config.get("simplemem_memory_answer_prompt"),
        )
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response,
            prompt_tokens + self._count_tokens(memories_str),
            completion_tokens,
            memory_construction_time,
            query_time_len
        )
        self._maybe_save_longmemeval_backfill_debug(
            query_id=query_id,
            context_id=context_id,
            retrieved_paragraphs=[entry["text"] for entry in retrieved_entries if entry.get("text")],
            query_text=message,
            response_text=response,
            retrieved_source_id_groups=retrieved_source_id_groups,
        )
        self.agent_start_time = time.time()
        return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

    def _handle_lightmem_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for LightMem agents."""
        if memorizing:
            clean_message = self._prepare_memory_chunk_for_storage(message)
            stored_message = message if parse_locomo_source_ids(message) else clean_message
            self.lightmem.add_chunk(
                stored_message,
                timestamp=time.strftime("%Y/%m/%d (%a) %H:%M:%S"),
            )
            return "Memorized"

        memory_construction_time = time.time() - self.agent_start_time
        raw_contexts = self._flatten_text_items(self.lightmem.retrieve(message))
        retrieved_source_id_groups = self._extract_locomo_source_id_groups_from_texts(raw_contexts)
        cleaned_contexts = self._clean_retrieved_memory_contexts(raw_contexts)
        retrieved_contexts = self._fit_memories_for_answer(
            question=message,
            contexts=cleaned_contexts or raw_contexts,
            prompt_override=self.agent_config.get("lightmem_memory_answer_prompt"),
        )
        memories_str = (
            "\n".join(retrieved_contexts)
            if retrieved_contexts
            else "(No retrieved memories found.)"
        )
        response, prompt_tokens, completion_tokens = self._generate_answer_from_memories(
            question=message,
            memories_text=memories_str,
            prompt_override=self.agent_config.get("lightmem_memory_answer_prompt"),
        )
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response,
            prompt_tokens + self._count_tokens(memories_str),
            completion_tokens,
            memory_construction_time,
            query_time_len
        )
        self._maybe_save_longmemeval_backfill_debug(
            query_id=query_id,
            context_id=context_id,
            retrieved_paragraphs=raw_contexts,
            query_text=message,
            response_text=response,
            retrieved_source_id_groups=retrieved_source_id_groups,
        )
        self.agent_start_time = time.time()
        return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

    def _handle_a_mem_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for A-MEM agents."""
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(
                context=message,
                **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {})
            )
            self.a_mem.add_chunk(
                formatted_message,
                timestamp=time.strftime("%Y%m%d%H%M"),
            )
            return "Memorized"

        memory_construction_time = time.time() - self.agent_start_time
        retrieved_context, retrieved_source_id_groups = self.a_mem.retrieve_with_source_groups(message)
        response = self.a_mem.ask_with_retrieved_context(message, retrieved_context)
        tokenizer = self.tokenizer
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response,
            len(tokenizer.encode(retrieved_context + "\n" + message, disallowed_special=())),
            len(tokenizer.encode(response, disallowed_special=())),
            memory_construction_time,
            query_time_len
        )

        paragraphs = [p for p in retrieved_context.replace("\r\n", "\n").split("\n") if p.strip()]
        self._save_retrieval_debug_payload(
            {
                "retrieved_context_paragraphs": paragraphs,
                "response": response,
                "query": message,
                "retrieved_source_id_groups": retrieved_source_id_groups,
            },
            query_id,
            context_id,
        )

        self.agent_start_time = time.time()
        return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

    def _handle_memtree_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for MemTree agents."""
        if memorizing:
            chunk_source_ids = parse_locomo_source_ids(message)
            clean_message = self._prepare_memory_chunk_for_storage(message)
            self.memtree.add_chunk(clean_message, source_ids=chunk_source_ids)
            return "Memorized"

        memory_construction_time = time.time() - self.agent_start_time
        retrieved_entries = self.memtree.search_entries(message, top_k=self.retrieve_num)
        raw_retrieved_memories = [
            entry["content"] for entry in retrieved_entries if entry.get("content")
        ]
        retrieved_source_id_groups = [
            entry.get("source_ids", [])
            for entry in retrieved_entries
            if entry.get("source_ids")
        ]
        cleaned_retrieved_memories = self._clean_retrieved_memory_contexts(raw_retrieved_memories)
        fitted_memories = self._fit_memories_for_answer(
            question=message,
            contexts=cleaned_retrieved_memories or raw_retrieved_memories,
            prompt_override=self.agent_config.get("memtree_memory_answer_prompt"),
        )
        memories_str = "\n".join(f"- {entry}" for entry in fitted_memories) or "(No retrieved memories found.)"

        memory_answer_template = (
            self.agent_config.get("memtree_memory_answer_prompt")
            or get_template(self.sub_dataset, 'memory_answer', self.agent_name)
        )
        formatted_query = memory_answer_template.format(
            memories=memories_str,
            question=message,
        )

        llm_messages = [
            {"role": "system", "content": get_template(self.sub_dataset, 'system', self.agent_name)},
            {"role": "user", "content": formatted_query},
        ]
        request_kwargs = self._prepare_openai_request_kwargs(
            {
                "model": self.model,
                "messages": llm_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        response = self.client.chat.completions.create(**request_kwargs)
        response_text = response.choices[0].message.content or ""
        self._record_llm_trace(
            stage="memtree_answer",
            messages=llm_messages,
            response_text=response_text,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            extra={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )

        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response_text,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            memory_construction_time,
            query_time_len,
        )

        self._save_retrieval_debug_payload(
            {
                "retrieved_context_paragraphs": fitted_memories,
                "response": response.choices[0].message.content,
                "query": message,
                "retrieved_source_id_groups": retrieved_source_id_groups,
            },
            query_id,
            context_id,
        )

        self.agent_start_time = time.time()
        return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

    def _handle_everos_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for EverOS agents."""
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(
                context=message,
                **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {})
            )
            self.everos.add_chunk(formatted_message, role="user")
            return "Memorized"

        memory_construction_time = time.time() - self.agent_start_time
        search_result = self.everos.search(message, top_k=self.retrieve_num)
        retrieved_entries = [entry["text"] for entry in search_result["memory_entries"]]
        pending_entries = [entry["text"] for entry in search_result["pending_entries"][:self.retrieve_num]]
        combined_entries = retrieved_entries + pending_entries

        tokenizer = self.tokenizer
        memory_retrieval_length = len(tokenizer.encode(memories_str, disallowed_special=()))
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            memory_construction_time,
            query_time_len,
        )

        save_dir = resolve_results_artifact_path(
            self.artifact_root,
            "outputs",
            "rag_retrieved",
            self.agent_name,
            f"k_{self.retrieve_num}",
            self.sub_dataset,
            f"chunksize_{self.chunk_size}",
            f"query_{query_id}_context_{context_id}.json",
        )
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        with open(save_dir, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "retrieved_context_paragraphs": retrieved_memories,
                    "response": response.choices[0].message.content,
                    "query": message,
                    "everos_group_id": self.everos_group_id,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        self.agent_start_time = time.time()
        return output

    def _handle_everos_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for EverOS agents."""
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(
                context=message,
                **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {})
            )
            self.everos.add_chunk(formatted_message, role="user")
            return "Memorized"

        memory_construction_time = time.time() - self.agent_start_time
        search_result = self.everos.search(message, top_k=self.retrieve_num)
        retrieved_entries = [entry["text"] for entry in search_result["memory_entries"]]
        pending_entries = [entry["text"] for entry in search_result["pending_entries"][:self.retrieve_num]]
        combined_entries = retrieved_entries + pending_entries
        retrieved_source_id_groups = self._extract_locomo_source_id_groups_from_texts(combined_entries)

        tokenizer = self.tokenizer
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        fitted_entries = self._fit_retrieved_contexts_to_token_limit(
            self._clean_retrieved_memory_contexts(combined_entries) or combined_entries,
            message,
            tokenizer,
            label_prefix="EverOS Memory",
            system_message=system_message,
        )
        memories_str = "\n\n".join(f"- {entry}" for entry in fitted_entries) or "(No retrieved memories found.)"

        memory_answer_template = get_template(self.sub_dataset, 'memory_answer', self.agent_name)
        formatted_query = memory_answer_template.format(
            memories=memories_str,
            question=message,
        )
        llm_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": formatted_query},
        ]
        request_kwargs = self._prepare_openai_request_kwargs(
            {
                "model": self.model,
                "messages": llm_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        response = self.client.chat.completions.create(**request_kwargs)

        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            memory_construction_time,
            query_time_len,
        )

        self._save_retrieval_debug_payload(
            {
                "retrieved_context_paragraphs": fitted_entries,
                "retrieved_memory_count": len(retrieved_entries),
                "pending_message_count": len(pending_entries),
                "pending_messages": [entry["raw"] for entry in search_result["pending_entries"]],
                "raw_search_result": search_result["raw_result"],
                "response": response.choices[0].message.content,
                "query": message,
                "everos_group_id": self.everos_group_id,
                "retrieved_source_id_groups": retrieved_source_id_groups,
            },
            query_id,
            context_id,
        )

        self.agent_start_time = time.time()
        return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

    # Zep
    def _handle_zep_agent(self, message, memorizing, query_id, context_id):
        """Handle Zep processing."""
        from zep_cloud.types import Message
        from methods.zep.zep import compose_search_context, llm_response, get_retrieval_query, construct_messages

        def _safe_zep_call(step_name, func, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Zep call failed at '{step_name}' "
                    f"(user_id={user_id}, session_id={thread_id}): {exc}"
                ) from exc

        def _is_already_exists_error(exc):
            message = str(exc).lower()
            return (
                "already exists" in message
                or "duplicate" in message
                or "conflict" in message
                or "status_code: 409" in message
            )

        def _ensure_zep_resources():
            for step_name, func, kwargs in [
                ("user.add", self.client.user.add, {"user_id": user_id}),
                ("memory.add_session", self.client.memory.add_session, {"session_id": thread_id, "user_id": user_id}),
            ]:
                try:
                    func(**kwargs)
                except Exception as exc:
                    if not _is_already_exists_error(exc):
                        raise RuntimeError(
                            f"Zep resource initialization failed at '{step_name}' "
                            f"(user_id={user_id}, session_id={thread_id}): {exc}"
                        ) from exc

        # user id / session id / oai client
        user_id = f'user_{context_id}_{self.sub_dataset}'
        thread_id = f'thread_{context_id}_{self.sub_dataset}'

        # check the context id for user and session creation
        if self.context_id != context_id and memorizing:
            _ensure_zep_resources()
            self.context_id = context_id
        else:
            pass

        if memorizing:
            # graph add
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            content = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            _safe_zep_call(
                "graph.add",
                self.client.graph.add,
                type="text",
                data=content[:9998],
                user_id=user_id,
            )

            # # thread add
            messages = construct_messages(content, user_id)
            _safe_zep_call(
                "memory.add",
                self.client.memory.add,
                thread_id,
                messages=messages,
            )
            return "Memorized"
        else:
            memory_construction_time = time.time() - self.agent_start_time
            _ensure_zep_resources()

            # graph search
            retrieval_query = get_retrieval_query(message)
            print(f"\n\n\nretrieval_query: {retrieval_query}\n\n\n")

            edges_results = _safe_zep_call(
                "graph.search.edges",
                self.client.graph.search,
                query=retrieval_query[:399],
                scope='edges',
                limit=self.retrieve_num,
                user_id=user_id,
            ).edges
            node_results = _safe_zep_call(
                "graph.search.nodes",
                self.client.graph.search,
                query=retrieval_query[:399],
                scope='nodes',
                limit=self.retrieve_num,
                user_id=user_id,
            ).nodes
            episode_results = _safe_zep_call(
                "graph.search.episodes",
                self.client.graph.search,
                query=retrieval_query[:399],
                scope='episodes',
                limit=self.retrieve_num,
                user_id=user_id,
            ).episodes

            # print(f"\n\n\nepisode_results: {episode_results}\n\n\n")
            # print(f"\n\n\nedges_results: {edges_results}\n\n\n")
            # print(f"\n\n\nnode_results: {node_results}\n\n\n")

            # thread search / currently we do not use the thread info
            memory = _safe_zep_call(
                "memory.get",
                self.client.memory.get,
                thread_id,
            )
            context_block = memory.context

            # Prompt an LLM with relevant context
            retrieved_context = compose_search_context(edges_results, node_results, context_block, episode_results)
            retrieved_source_id_groups = self._extract_locomo_source_id_groups_from_items(
                episode_results,
                candidate_keys=("content",),
            )
            import asyncio
            cleaned_retrieved_context = strip_locomo_metadata(retrieved_context)
            response = asyncio.run(llm_response(self.oai_client, cleaned_retrieved_context, message))
            query_time_len = time.time() - self.agent_start_time - memory_construction_time

            output = self._create_standard_response(
                response,
                len(self.tokenizer.encode(cleaned_retrieved_context, disallowed_special=())),
                len(self.tokenizer.encode(response, disallowed_special=())),
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time

            # save the context
            paragraphs = [p for p in cleaned_retrieved_context.replace("\r\n", "\n").split("\n") if p.strip()]
            self._save_retrieval_debug_payload(
                {
                    "retrieved_context_paragraphs": paragraphs,
                    "response": response,
                    "query": message,
                    "retrieved_source_id_groups": retrieved_source_id_groups,
                },
                query_id,
                context_id,
            )

            return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)
    
    def _handle_memagent(self, message, memorizing, query_id, context_id):
        """Handle message processing for MemAgent (recurrent memory)."""
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(
                context=message,
                **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {})
            )
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:
            # MemAgent processes the full context recurrently at query time
            memory_construction_time = time.time() - self.agent_start_time
            start_time = time.time()
            response = self.memagent.answer(self.context, message)
            query_time_len = time.time() - start_time

            tokenizer = self.tokenizer
            output = self._create_standard_response(
                response,
                len(tokenizer.encode(self.context, disallowed_special=())),
                len(tokenizer.encode(response, disallowed_special=())),
                memory_construction_time,
                query_time_len,
            )
            self.agent_start_time = time.time()
            return output


    def _handle_memos_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for MemOS agents."""
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(
                context=message,
                **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {})
            )
            self.memos.add(memory_content=formatted_message, memorize_mode=self.memos_memorize_mode)
            return "Memorized"

        # Query phase: search MemOS memories, then answer via LLM
        memory_construction_time = time.time() - self.agent_start_time
        search_results = self.memos.search(message, top_k=self.retrieve_num, mode=self.memos_search_mode)

        # Extract memory strings from MOSSearchResult
        memories_list = []
        text_mem_results = search_results.get("text_mem", [])
        for cube_result in text_mem_results:
            for mem_item in cube_result.get("memories", []):
                mem_text = getattr(mem_item, "memory", None) or str(mem_item)
                memories_list.append(mem_text)
        retrieved_source_id_groups = self._extract_locomo_source_id_groups_from_texts(memories_list)
        cleaned_memories_list = self._clean_retrieved_memory_contexts(memories_list)
        candidate_memories = cleaned_memories_list or memories_list
        # Keep MemOS retrieval/ranking unchanged, but fit the final packed
        # evidence to the model's prompt budget before answer generation.
        fitted_memories = self._fit_memories_for_answer(
            question=message,
            contexts=candidate_memories,
            label_prefix="Memory",
        )

        memories_str = "\n".join(f"- {m}" for m in fitted_memories) if fitted_memories else "(No memories found.)"

        # Build answer prompt using benchmark template
        memory_answer_template = get_template(self.sub_dataset, 'memory_answer', self.agent_name)
        formatted_query = memory_answer_template.format(
            memories=memories_str,
            question=message,
        )
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        llm_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": formatted_query},
        ]
        request_kwargs = self._prepare_openai_request_kwargs(
            {
                "model": self.model,
                "messages": llm_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        response = self.client.chat.completions.create(**request_kwargs)

        tokenizer = self.tokenizer
        memory_retrieval_length = len(tokenizer.encode(memories_str, disallowed_special=()))
        query_time_len = time.time() - self.agent_start_time - memory_construction_time

        output = self._create_standard_response(
            response.choices[0].message.content,
            response.usage.prompt_tokens + memory_retrieval_length,
            response.usage.completion_tokens,
            memory_construction_time,
            query_time_len,
        )
        self.agent_start_time = time.time()
        return self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

    def _handle_rag_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for RAG agents."""
        if memorizing:
            # Add message to chunks and context
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            self.chunks.append(formatted_message)
            self.context_len = self.context_len + self.chunk_size

            # Truncate context if it exceeds limits
            if self.context_len > self.input_length_limit:
                self.chunks = self.chunks[1:]
                self.context_len = self.context_len - self.chunk_size
            return ''
        else:
            # Handle query processing for different RAG types
            return self._process_rag_query(message, query_id, context_id)

    def _process_rag_query(self, message, query_id, context_id):
        """Process query for RAG agents with different retrieval strategies."""

        # Truncate context if needed
        tokenizer = self.tokenizer
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])
        if self.context_len > self.input_length_limit:
            self.chunks = self.chunks[1:]
            self.context_len = self.context_len - self.chunk_size

        # Route to specific RAG implementation and get result
        rag_handlers = {
            "graph_rag": lambda: self._handle_graph_rag(message, context_id, tokenizer),
            "hippo_rag_v2_nv": lambda: self._handle_hippo_rag(message, context_id, tokenizer),
            "hippo_rag_v2_openai": lambda: self._handle_hippo_rag(message, context_id, tokenizer),
            "rag_bm25": lambda: self._handle_bm25_rag(message, context_id, tokenizer),
            "rag_contriever": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_large": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_small": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_qwen3_embedding_4b": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_raptor": lambda: self._handle_raptor_rag(message, context_id, tokenizer),
            "self_rag": lambda: self._handle_self_rag(message, context_id, tokenizer),
            "memo_rag": lambda: self._handle_memorag(message, context_id, tokenizer),
        }

        # Find matching handler
        handler = next((handler for agent_type, handler in rag_handlers.items() if self._is_agent_type(agent_type)), None)
        if not handler:
            raise NotImplementedError(f"RAG agent type not supported: {self.agent_name}")

        output = handler()

        if output.get("retrieval_context") and not output.get("retrieved_source_id_groups"):
            retrieval_context = output["retrieval_context"]
            retrieval_items = retrieval_context if isinstance(retrieval_context, list) else [retrieval_context]
            retrieved_source_id_groups = self._extract_locomo_source_id_groups_from_texts(retrieval_items)
            self._attach_locomo_recall_metadata(output, retrieved_source_id_groups)

        # Save the retrieved context as JSON (if the method provides it)
        if output.get("retrieval_context"):
            self._save_retrieval_debug_payload(
                output["retrieval_context"],
                query_id,
                context_id,
                extra_fields={
                    "query": message,
                    "response": output.get("output"),
                    "retrieved_source_id_groups": output.get("retrieved_source_id_groups"),
                },
            )

            # drop the retrieval_context
            output.pop("retrieval_context")

        return output

    def _handle_graph_rag(self, message, context_id, tokenizer):
        """Handle Graph RAG processing."""
        start_time = time.time()
        self._ensure_explicit_embedding_config()
        memory_construction_time = 0
        build_error = None

        # Build vectorstore if context changed
        if self.context_id != context_id:
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            try:
                from methods.graph_rag.graph_rag import GraphRAG
                self.graph_rag = GraphRAG(
                    temperature=self.temperature,
                    model_name=self.model,
                    retrieve_num=self.retrieve_num,
                    max_tokens=self.max_tokens,
                    internal_max_tokens=self.graph_rag_internal_max_tokens,
                    provider=self.model_provider,
                    base_url=self.base_url,
                    base_url_env=self.base_url_env,
                    api_key_env=self.api_key_env,
                    azure_endpoint=self.azure_endpoint,
                    azure_api_version=self.azure_api_version,
                    embedding_model=self.embedding_model,
                    embedding_provider=self.embedding_provider,
                    embedding_base_url=self.embedding_base_url,
                    embedding_base_url_env=self.embedding_base_url_env,
                    embedding_api_key_env=self.embedding_api_key_env,
                    embedding_azure_endpoint=self.embedding_azure_endpoint,
                    embedding_azure_api_version=self.embedding_azure_api_version,
                    tokenizer_model=self.tokenizer_model,
                    tokenizer_encoding=self.tokenizer_encoding,
                )
                self.graph_rag.process_documents(docs)
                memory_construction_time = time.time() - start_time
            except Exception as e:
                self.graph_rag = None
                build_error = e
                print(f"\n\n\n\nError: {e}\n\n\n\n")
            print(f"\n\nGraph RAG build vectorstore finished...\n\n")
        else:
            memory_construction_time = 0
            print(f"\n\nContext {context_id} already processed, skipping Graph RAG build vectorstore...\n\n")

        # Process query
        if build_error is not None:
            response = f"{build_error}"
            retrieval_context = "ERROR"
        else:
            try:
                response, retrieval_context = self.graph_rag.query(query=message)
            except Exception as e:
                response = f"{e}"
                retrieval_context = "ERROR"
                print(f"\n\n\n\nError: {e}\n\n\n\n")

        self.context_id = context_id

        print(f"\n\n\n\nResponse: {response}\n\n\n\n")
        if isinstance(response, str):
            response = response
        else:
            response = response.content
        query_time_len = time.time() - start_time - memory_construction_time

        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    def _handle_hippo_rag(self, message, context_id, tokenizer):
        """Handle HippoRAG processing."""
        start_time = time.time()

        if self.context_id != context_id:
            docs = self.chunks
            from methods.hipporag import HippoRAG
            if self.embedding_model:
                embedding_model_name = self.embedding_model
                embedding_label = embedding_model_name.replace("/", "_")
            elif any(agent_name in self.agent_name for agent_name in ["hippo_rag_v2_nv"]):
                embedding_model_name = 'nvidia/NV-Embed-v2'
                embedding_label = "NV-Embed-v2"
            elif any(agent_name in self.agent_name for agent_name in ["hippo_rag_v2_openai"]):
                embedding_model_name = 'text-embedding-ada-002'
                embedding_label = "OpenAIEmbedding"
            else:
                raise NotImplementedError(
                    "HippoRAG requires either an agent_name suffix with a known embedding backend "
                    "or an explicit embedding_model override in the agent config."
                )

            save_dir = resolve_results_artifact_path(
                self.artifact_root,
                "outputs",
                "rag_retrieved",
                embedding_label,
                self.sub_dataset,
                f"chunksize_{self.chunk_size}",
                f"context_id_{context_id}",
            )

            self.hipporag = HippoRAG(
                                save_dir=save_dir,
                                llm_model_name=self.model,
                                llm_base_url=self._resolve_base_url(),
                                llm_api_key=self._get_env_value(self.api_key_env, ["OPENAI_API_KEY"]),
                                embedding_model_name=embedding_model_name,
                                embedding_base_url=self.embedding_base_url or self._get_env_value(self.embedding_base_url_env),
                                embedding_api_key=self._get_env_value(self.embedding_api_key_env, ["OPENAI_API_KEY"]),
                                )
            self.hipporag.index(docs=docs)
            memory_construction_time = time.time() - start_time
            print(f"\n\nHippoRAG build vectorstore finished...\n\n")
        else:
            memory_construction_time = 0
            print(f"\n\nContext {context_id} already processed, skipping HippoRAG build vectorstore...\n\n")

        # Retrieve and answer
        queries = [message]
        retrieval_results, top_k_docs = self.hipporag.retrieve(queries=queries, num_to_retrieve=self.retrieve_num)

        qa_results = self.hipporag.rag_qa(retrieval_results)
        response = qa_results[0][0].answer

        retrieval_context = "\n\n".join([f"Passage {i+1}:\n{text}" for i, text in enumerate(top_k_docs)])
        query_time_len = time.time() - start_time - memory_construction_time

        self.context_id = context_id

        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    # RAG implementation methods
    def _handle_bm25_rag(self, message, context_id, tokenizer):
        """Handle BM25 RAG processing."""
        start_time = time.time()

        # Extract retrieval query from message
        retrieval_query = self._extract_retrieval_query(message)
        print(f"\n\n\n\nretrieval_query: {retrieval_query}\n\n\n\n")

        # Build vectorstore if context changed
        if self.context_id != context_id:
            from langchain_community.retrievers import BM25Retriever
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            print(f"\n\nBM25 build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping BM25 build vectorstore...\n\n")

        # Retrieve documents
        self.bm25_retriever.k = self.retrieve_num
        if hasattr(self.bm25_retriever, "invoke"):
            bm25_documents = self.bm25_retriever.invoke(retrieval_query)
        else:
            bm25_documents = self.bm25_retriever.get_relevant_documents(retrieval_query)
        raw_retrieval_context = [f"{doc.page_content}\n" for doc in bm25_documents]
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        retrieval_context = self._fit_retrieved_contexts_to_token_limit(
            self._clean_retrieved_memory_contexts(raw_retrieval_context) or raw_retrieval_context,
            message,
            tokenizer,
            system_message=system_message,
        )
        memory_construction_time = time.time() - start_time

        # Answer the query
        retrieval_memory_string = "\n".join([f"Memory {i+1}:\n{text}" for i, text in enumerate(retrieval_context)])

        # Format the message
        ask_llm_message = retrieval_memory_string + "\n" + message
        format_message = format_chat(message=ask_llm_message, system_message=system_message)

        # Generate response
        request_kwargs = self._prepare_openai_request_kwargs(
            {
                "model": self.model,
                "messages": format_message,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens if "gpt-4" in self.model else None,
            }
        )
        response = self._create_oai_client().chat.completions.create(**request_kwargs)

        query_time_len = time.time() - start_time - memory_construction_time
        self.context_id = context_id

        output = {
            "output": response.choices[0].message.content,
            "input_len": response.usage.prompt_tokens,
            "output_len": response.usage.completion_tokens,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
        return self._attach_locomo_recall_metadata(
            output,
            self._extract_locomo_source_id_groups_from_texts(raw_retrieval_context),
        )

    def _extract_retrieval_query(self, message):
        """Extract retrieval query from message using regex patterns."""
        patterns = [
            r"Now Answer the Question:\s*(.*)",
            r"Here is the conversation:\s*(.*)"
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.DOTALL)
            if match:
                return ''.join(match.groups())

        return message

    def _handle_embedding_rag(self, message, context_id, tokenizer):
        """Handle embedding-based RAG processing (Contriever, Text-embedding models)."""
        from methods.embedding_rag.embedding_retriever import TextRetriever, RAGSystem
        self._ensure_explicit_embedding_config()

        # Determine embedding model
        if any(agent_name in self.agent_name for agent_name in ["rag_contriever"]):
            embedding_model_name = "facebook/contriever"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_3_large"]):
            embedding_model_name = "text-embedding-3-large"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_3_small"]):
            embedding_model_name = "text-embedding-3-small"
        elif any(agent_name in self.agent_name for agent_name in ["rag_qwen3_embedding_4b"]):
            embedding_model_name = "Qwen/Qwen3-Embedding-4B"
        else:
            raise NotImplementedError

        if self.embedding_model:
            embedding_model_name = self.embedding_model

        # Build vectorstore if context changed
        if self.context_id != context_id:
            self.retriever = TextRetriever(
                embedding_model_name=embedding_model_name,
                provider=self.embedding_provider,
                base_url=self.embedding_base_url,
                base_url_env=self.embedding_base_url_env,
                api_key_env=self.embedding_api_key_env,
                azure_endpoint=self.embedding_azure_endpoint,
                azure_api_version=self.embedding_azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
            )
            self.retriever.build_vectorstore(self.chunks)
            print(f"\n\n{embedding_model_name} build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping {embedding_model_name} build vectorstore...\n\n")

        # Retrieve relevant passages and answer the query
        rag_system = RAGSystem(
            self.retriever,
            self.model,
            self.temperature,
            self.max_tokens,
            provider=self.model_provider,
            base_url=self.base_url,
            base_url_env=self.base_url_env,
            api_key_env=self.api_key_env,
            azure_endpoint=self.azure_endpoint,
            azure_api_version=self.azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
            qwen3_disable_thinking=bool(self.agent_config.get("qwen3_disable_thinking", True)),
            prompt_token_budget=self.input_length_limit,
            tokenizer=tokenizer,
        )
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        result = rag_system.answer_query(
            query=message,
            top_k=self.retrieve_num,
            system_message=system_message
        )
        retrieval_context = result['context_used']

        self.context_id = context_id

        return {
            "output": result["answer"],
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(result["answer"], disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }

    def _handle_raptor_rag(self, message, context_id, tokenizer):
        """Handle RAPTOR RAG processing."""
        self._ensure_explicit_embedding_config()
        # Build vectorstore if context changed
        if self.context_id != context_id:
            texts = self.chunks
            from methods.raptor.raptor import RAPTORMethod
            self.raptor_method = RAPTORMethod(
                texts,
                max_levels=3,
                model=self.model,
                temperature=self.temperature,
                provider=self.model_provider,
                base_url=self.base_url,
                base_url_env=self.base_url_env,
                api_key_env=self.api_key_env,
                azure_endpoint=self.azure_endpoint,
                azure_api_version=self.azure_api_version,
                embedding_model=self.embedding_model,
                embedding_provider=self.embedding_provider,
                embedding_base_url=self.embedding_base_url,
                embedding_base_url_env=self.embedding_base_url_env,
                embedding_api_key_env=self.embedding_api_key_env,
                embedding_azure_endpoint=self.embedding_azure_endpoint,
                embedding_azure_api_version=self.embedding_azure_api_version,
            )
            print(f"\n\nRaptor build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping Raptor build vectorstore...\n\n")

        # Retrieve relevant passages and answer the query
        result = self.raptor_method.run(query=message, k=self.retrieve_num)
        response = result['answer']
        retrieval_context = result['context_used']

        self.context_id = context_id

        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }

    def _handle_self_rag(self, message, context_id, tokenizer):
        """Handle Self-RAG processing."""
        from methods.self_rag.self_rag import SelfRAG
        start_time = time.time()
        self._ensure_explicit_embedding_config()

        # Build vectorstore if context changed
        if self.context_id != context_id:
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            self.self_rag = SelfRAG(
                documents=docs,
                temperature=self.temperature,
                top_k=self.retrieve_num,
                model=self.model,
                provider=self.model_provider,
                base_url=self.base_url,
                base_url_env=self.base_url_env,
                api_key_env=self.api_key_env,
                azure_endpoint=self.azure_endpoint,
                azure_api_version=self.azure_api_version,
                embedding_model=self.embedding_model,
                embedding_provider=self.embedding_provider,
                embedding_base_url=self.embedding_base_url,
                embedding_base_url_env=self.embedding_base_url_env,
                embedding_api_key_env=self.embedding_api_key_env,
                embedding_azure_endpoint=self.embedding_azure_endpoint,
                embedding_azure_api_version=self.embedding_azure_api_version,
            )
            print(f"\n\nSelf-RAG build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping Self-RAG build vectorstore...\n\n")

        # Process query
        try:
            response, retrieval_context_list, memory_construction_time, query_time_len = self.self_rag.run(query=message)
        except Exception as e:
            response = f"{e}"
            retrieval_context_list = ["ERROR"]
            memory_construction_time = 0
            query_time_len = 0
            print(f"\n\n\n\nError: {e}\n\n\n\n")

        # Prepare the context
        cleaned_context_list = self._clean_retrieved_memory_contexts(retrieval_context_list) or retrieval_context_list
        retrieval_context = "\n\n".join([f"Passage {i+1}:\n{text}"
                                        for i, text in enumerate(cleaned_context_list)])

        self.context_id = context_id

        output = {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
        return self._attach_locomo_recall_metadata(
            output,
            self._extract_locomo_source_id_groups_from_texts(retrieval_context_list),
        )

    # memorag
    def _handle_memorag(self, message, context_id, tokenizer):
        """Handle MemoRAG processing."""
        from methods.memorag import Agent, MemoRAG
        start_time = time.time()
        memory_construction_time = 0
        cache_context_save_dir = self._build_memorag_cache_dir(context_id)
        memorag_mem_model = self.agent_config.get(
            "memorag_mem_model_name_or_path",
            "TommyChien/memorag-qwen2-7b-inst",
        )
        memorag_ret_model = self.agent_config.get(
            "memorag_ret_model_name_or_path",
            "BAAI/bge-m3",
        )
        memorag_cache_dir = self.agent_config.get("memorag_cache_dir") or os.environ.get("MEMORAG_CACHE_DIR")
        memorag_beacon_ratio = self.agent_config.get("memorag_beacon_ratio", 4)
        memorag_load_in_4bit = self.agent_config.get("memorag_load_in_4bit", False)
        memorag_enable_flash_attn = self.agent_config.get("memorag_enable_flash_attn", True)
        memorag_save_cache = self.agent_config.get("memorag_save_cache", True)
        if memorag_cache_dir:
            memorag_cache_dir = os.path.abspath(memorag_cache_dir)
            os.makedirs(memorag_cache_dir, exist_ok=True)

        memorag_context_tokenizer = None
        if self.agent_config.get("memorag_max_context_tokens"):
            try:
                memorag_context_tokenizer = AutoTokenizer.from_pretrained(
                    memorag_mem_model,
                    cache_dir=memorag_cache_dir,
                    trust_remote_code=True,
                )
            except Exception as exc:
                print(f"MemoRAG tokenizer load failed, falling back to {self.tokenizer_encoding or 'cl100k_base'}: {exc}")

        # build rag agent
        if self.context_id != context_id:
            # Map provider → Agent source + api_dict
            if self.model_provider == "azure_openai":
                source = "azure"
                api_dict = {
                    "endpoint":    self.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    "api_version": self.azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
                    "api_key":     self._get_env_value(self.api_key_env, ["AZURE_OPENAI_API_KEY"]),
                }
            elif self.model_provider == "openai_compatible":
                # reuses the "deepseek" branch in Agent, which accepts base_url + api_key
                source = "deepseek"
                api_dict = {
                    "base_url": self._resolve_base_url(),
                    "api_key":  self._get_env_value(self.api_key_env, ["OPENAI_API_KEY"]),
                }
            else:  # openai
                source = "openai"
                api_dict = {}
            gen_model = Agent(model=self.model, source=source, temperature=self.temperature, api_dict=api_dict)
            self.MemoRAG = MemoRAG(
                mem_model_name_or_path=memorag_mem_model,
                ret_model_name_or_path=memorag_ret_model,
                customized_gen_model=gen_model,
                ret_hit=self.retrieve_num,
                retrieval_chunk_size=self.chunk_size,
                cache_dir=memorag_cache_dir,
                beacon_ratio=memorag_beacon_ratio,
                load_in_4bit=memorag_load_in_4bit,
                enable_flash_attn=memorag_enable_flash_attn,
            )
            # Use the loaded context / memorize the context for question answering
            context = self._prepare_memorag_context(
                " ".join(self.chunks),
                token_counter=memorag_context_tokenizer,
            )
            ## load the context from the cache
            if os.path.exists(f'{cache_context_save_dir}/memory.bin'):
                self.MemoRAG.load(cache_context_save_dir, print_stats=True)
            else:
                save_dir = cache_context_save_dir if memorag_save_cache else None
                self.MemoRAG.memorize(context, save_dir=save_dir, print_stats=True)
            memory_construction_time = time.time() - start_time
            print(f"Finish memorizing, time cost {memory_construction_time}")
        else:
            print(f"\n\nContext {context_id} already processed, skipping MemoRAG build vectorstore...\n\n")

        # Retrieve and answer
        if self.sub_dataset == "infbench_sum_eng_shots2":
            response, retrieval_context = self.MemoRAG(query=message, task_type="summarize", max_new_tokens=self.max_tokens)
        else:
            response, retrieval_context = self.MemoRAG(query=message, task_type="memorag", max_new_tokens=self.max_tokens)

        query_time_len = time.time() - start_time - memory_construction_time

        self.context_id = context_id

        return {
            "output": response,
            "input_len": len(tokenizer.encode(str(retrieval_context) + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    def save_agent(self):
        """Save agent state to disk for persistence."""
        # Currently only implemented for Letta agents
        if not self._is_agent_type("letta") and not self._is_agent_type("mem0") and not self._is_agent_type("cognee") and not self._is_agent_type("memtree") and not self._is_agent_type("memochat") and not self._is_agent_type("memoryos") and not self._is_agent_type("zep") and not self._is_agent_type("simplemem") and not self._is_agent_type("lightmem") and not self._is_agent_type("a_mem") and not self._is_agent_type("everos") and not self._is_agent_type("MemOS"):
            print("\n\n Agent not saved (not implemented for this agent type) \n\n")
            return

        if self._is_agent_type("letta"):
            agent_save_folder = self.agent_save_to_folder
            os.makedirs(agent_save_folder, exist_ok=True)
            if "api" not in self.agent_name:
                runtime_db_path = self.letta_runtime_db_path
                if not os.path.exists(runtime_db_path):
                    print(f"Letta runtime database not found at {runtime_db_path}; skipping snapshot.")

            # Save the agent ID for future loading
            with open(f"{agent_save_folder}/agent_id.txt", "w") as f:
                f.write(self.agent_state.id)
            self._snapshot_letta_query_baseline()
        elif self._is_agent_type("zep_local"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            # Communities are optional in zep_local answer construction. On
            # larger graphs this synchronous post-processing can hang long
            # enough to block the whole benchmark context, so we skip it in
            # the online save path and allow queries to proceed immediately.
            skipped_path = os.path.join(
                self.agent_save_to_folder, "community_build_skipped.txt"
            )
            with open(skipped_path, "w", encoding="utf-8") as f:
                f.write(
                    "Skipped build_communities_sync during online benchmark run.\n"
                )
                f.write(
                    "Reason: community build is optional for answering and can "
                    "block large-graph contexts.\n"
                )
            with open(f"{self.agent_save_to_folder}/messages.txt", "w") as f:
                f.write("agent finished memorization")
        elif self._is_agent_type("zep"):
            # save the message that agent has processed
            messages = "agent finished memorization"
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/messages.txt", "w") as f:
                f.write(messages)
        elif self._is_agent_type("mem0"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(self.mem0_source_map_path, "w", encoding="utf-8") as f:
                json.dump(self.mem0_source_map, f, ensure_ascii=False, indent=2)
        elif self._is_agent_type("simplemem"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            self.simplemem.finalize()
            with open(self.simplemem_source_map_path, "w", encoding="utf-8") as f:
                json.dump(self.simplemem.entry_source_map, f, ensure_ascii=False, indent=2)
            with open(self.simplemem_marker_path, "w") as f:
                f.write("ready")
        elif self._is_agent_type("lightmem"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            self.lightmem.finalize()
            with open(self.lightmem_marker_path, "w") as f:
                f.write("ready")
        elif self._is_agent_type("a_mem"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            self.a_mem.save()
            with open(self.a_mem_marker_path, "w") as f:
                f.write("ready")
        elif self._is_agent_type("cognee"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(self.cognee_ready_path, "w") as f:
                f.write("ready")
        elif self._is_agent_type("memtree"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            self.memtree.save()
            with open(self.memtree_marker_path, "w") as f:
                f.write("ready")
        elif self._is_agent_type("everos"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(self.everos_marker_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "group_id": self.everos_group_id,
                        "group_name": self.everos_group_name,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        elif self._is_agent_type("memochat"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            self.memochat.save()
            with open(self.memochat_marker_path, "w") as f:
                f.write("ready")
        elif self._is_agent_type("memoryos"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            self.memoryos.save()
            with open(self.memoryos_source_map_path, "w", encoding="utf-8") as f:
                json.dump(self.memoryos_source_map, f, ensure_ascii=False, indent=2)
            with open(self.memoryos_marker_path, "w") as f:
                f.write("ready")
        elif self._is_agent_type("MemOS"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(self.memos_marker_path, "w") as f:
                f.write("ready")

        print("\n\n Agent saved...\n\n")

    def load_agent(self):
        """Load agent state from disk."""
        agent_save_folder = self.agent_save_to_folder
        assert os.path.exists(agent_save_folder), f"Folder {agent_save_folder} does not exist."

        if not self._is_agent_type("letta") and not self._is_agent_type("mem0") and not self._is_agent_type("cognee") and not self._is_agent_type("memtree") and not self._is_agent_type("memochat") and not self._is_agent_type("memoryos") and not self._is_agent_type("zep") and not self._is_agent_type("simplemem") and not self._is_agent_type("lightmem") and not self._is_agent_type("a_mem") and not self._is_agent_type("everos") and not self._is_agent_type("MemOS"):
            print("\n\nAgent loading not implemented for this agent type\n\n")
            return None

        if self._is_agent_type("letta"):
            # Load agent ID and find the corresponding agent state
            with open(f"{agent_save_folder}/agent_id.txt", "r") as f:
                agent_id = f.read()

            if "api" in self.agent_name:
                self.agent_state = self.client.agents.retrieve(agent_id)
            else:
                # Find the agent state with the matching ID
                for agent_state in self.client.list_agents():
                    if agent_state.id == agent_id:
                        self.agent_state = agent_state
                        break
        elif self._is_agent_type("zep_local"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/messages.txt", "r") as f:
                _ = f.read()
        elif self._is_agent_type("zep"):
            # load the message that agent has processed
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/messages.txt", "r") as f:
                messages = f.read()
        elif self._is_agent_type("mem0"):
            self._require_locomo_provenance_sidecar(self.mem0_source_map_path, "mem0")
            if os.path.exists(self.mem0_source_map_path):
                with open(self.mem0_source_map_path, "r", encoding="utf-8") as f:
                    self.mem0_source_map = json.load(f)
        elif self._is_agent_type("simplemem"):
            if not os.path.exists(self.simplemem_marker_path):
                raise FileNotFoundError(
                    f"SimpleMem marker not found at {self.simplemem_marker_path}"
                )
            self._require_locomo_provenance_sidecar(self.simplemem_source_map_path, "SimpleMem")
        elif self._is_agent_type("lightmem"):
            if not os.path.exists(self.lightmem_marker_path):
                raise FileNotFoundError(
                    f"LightMem marker not found at {self.lightmem_marker_path}"
                )
        elif self._is_agent_type("a_mem"):
            if not os.path.exists(self.a_mem_marker_path):
                raise FileNotFoundError(
                    f"A-MEM marker not found at {self.a_mem_marker_path}"
                )
            self.a_mem.load()
        elif self._is_agent_type("cognee"):
            if not os.path.exists(self.cognee_ready_path):
                raise FileNotFoundError(
                    f"Cognee marker not found at {self.cognee_ready_path}"
                )
        elif self._is_agent_type("memtree"):
            if not os.path.exists(self.memtree_marker_path):
                raise FileNotFoundError(
                    f"MemTree marker not found at {self.memtree_marker_path}"
                )
            self.memtree.load()
            if (
                self.sub_dataset == "locomo_qa"
                and self.memtree.memory_count() > 0
                and not any(self.memtree.node_source_ids.values())
            ):
                raise RuntimeError(
                    f"MemTree saved state at {self.agent_save_to_folder} is missing "
                    "LoCoMo provenance metadata. This cache predates the current "
                    "recall tracking. Rebuild the agent state with --force before "
                    "running LoCoMo recall evaluation."
                )
        elif self._is_agent_type("everos"):
            if not os.path.exists(self.everos_marker_path):
                raise FileNotFoundError(
                    f"EverOS marker not found at {self.everos_marker_path}"
                )
            with open(self.everos_marker_path, "r", encoding="utf-8") as f:
                _ = json.load(f)
        elif self._is_agent_type("memochat"):
            if not os.path.exists(self.memochat_marker_path):
                raise FileNotFoundError(
                    f"MemoChat marker not found at {self.memochat_marker_path}"
                )
            self.memochat.load()
        elif self._is_agent_type("memoryos"):
            if not os.path.exists(self.memoryos_marker_path):
                raise FileNotFoundError(
                    f"MemoryOS marker not found at {self.memoryos_marker_path}"
                )
            self._require_locomo_provenance_sidecar(self.memoryos_source_map_path, "MemoryOS")
            self.memoryos.load()
            if os.path.exists(self.memoryos_source_map_path):
                with open(self.memoryos_source_map_path, "r", encoding="utf-8") as f:
                    self.memoryos_source_map = json.load(f)
        elif self._is_agent_type("MemOS"):
            if not os.path.exists(self.memos_marker_path):
                raise FileNotFoundError(
                    f"MemOS marker not found at {self.memos_marker_path}"
                )

        print("\n\n Agent loaded successfully...\n\n")
