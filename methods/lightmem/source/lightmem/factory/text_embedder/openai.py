from openai import OpenAI
from typing import Optional, List, Union
import os
import httpx
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig

try:
    from utils.provider_utils import ApproximateTokenizer, load_local_hf_tokenizer
except Exception:  # pragma: no cover - LightMem can be imported outside the repository
    ApproximateTokenizer = None
    load_local_hf_tokenizer = None


class _CharacterTokenizer:
    def encode(self, text, disallowed_special=()):
        del disallowed_special
        return list(text or "")

    def decode(self, tokens):
        return "".join(tokens)


def _infer_embedding_token_limit(model_name: str) -> int:
    normalized = str(model_name or "").strip().lower()
    if "qwen3-embedding" in normalized:
        return 8192
    if "text-embedding-3" in normalized:
        return 8192
    return 8192


class TextEmbedderOpenAI:
    def __init__(self, config: Optional[BaseTextEmbedderConfig] = None):
        self.config = config
        self.model = getattr(config, "model", None) or "text-embedding-3-small"        
        http_client = httpx.Client(verify=False)
        api_key = self.config.api_key 
        base_url = self.config.openai_base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
        self.total_calls = 0
        self.total_tokens = 0
        model_kwargs = getattr(self.config, "model_kwargs", None) or {}
        configured_limit = model_kwargs.get("max_input_tokens") if isinstance(model_kwargs, dict) else None
        self._max_input_tokens = int(configured_limit or _infer_embedding_token_limit(self.model))
        self._safe_input_tokens = max(256, self._max_input_tokens - 128)
        if load_local_hf_tokenizer is not None:
            fallback_tokenizer = ApproximateTokenizer() if ApproximateTokenizer is not None else _CharacterTokenizer()
            self._tokenizer = load_local_hf_tokenizer(self.model) or fallback_tokenizer
        elif ApproximateTokenizer is not None:
            self._tokenizer = ApproximateTokenizer()
        else:
            self._tokenizer = _CharacterTokenizer()

    @classmethod
    def from_config(cls, config: BaseTextEmbedderConfig):
        return cls(config)

    def _fit_to_token_budget(self, text: str) -> str:
        if not text:
            return text

        tokens = self._tokenizer.encode(text, disallowed_special=())
        if len(tokens) <= self._safe_input_tokens:
            return text

        head = self._safe_input_tokens // 2
        tail = self._safe_input_tokens - head
        fitted_tokens = tokens[:head] + tokens[-tail:]
        return self._tokenizer.decode(fitted_tokens)

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        def preprocess(t):
            normalized = str(t).replace("\n", " ")
            return self._fit_to_token_budget(normalized)

        api_params = {"model": self.config.model}
        model_name = str(getattr(self.config, "model", "") or "")
        if self.config.embedding_dims and "Qwen3-Embedding-4B" not in model_name:
            api_params["dimensions"] = self.config.embedding_dims

        if isinstance(text, list):
            if len(text) == 0:
                return []
            inputs = [preprocess(x) for x in text]
            resp = self.client.embeddings.create(input=inputs, **api_params)
            self.total_calls += 1
            self.total_tokens += resp.usage.total_tokens
            return [item.embedding for item in resp.data]
        else:
            preprocessed = preprocess(text)
            resp = self.client.embeddings.create(input=[preprocessed], **api_params)
            self.total_calls += 1
            self.total_tokens += resp.usage.total_tokens
            return resp.data[0].embedding
        
    def get_stats(self):
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
        }
