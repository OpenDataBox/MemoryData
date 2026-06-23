"""Reusable runtime helpers for benchmark-native memory adapters."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Optional

import httpx
import numpy as np
from openai import OpenAI
from utils.provider_utils import ApproximateTokenizer, load_local_hf_tokenizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional path
    SentenceTransformer = None


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had",
    "has", "have", "he", "her", "hers", "him", "his", "if", "in", "into", "is", "it", "its",
    "of", "on", "or", "our", "ours", "she", "that", "the", "their", "them", "there", "they",
    "this", "to", "was", "were", "will", "with", "you", "your", "yours",
}

_CONTEXT_LIMIT_ERROR_MARKERS = (
    "maximum context length",
    "context length",
    "input_tokens",
    "requested 0 output tokens",
)


def _extract_openai_message_text(message: Any) -> str:
    """Best-effort extraction across string/list/reasoning payload variants."""
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                text_parts.append(item.strip())
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())
        if text_parts:
            return "\n".join(text_parts)

    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content.strip()

    if isinstance(message, dict):
        dict_content = message.get("content")
        if isinstance(dict_content, str) and dict_content.strip():
            return dict_content.strip()
        if isinstance(dict_content, list):
            text_parts = []
            for item in dict_content:
                if isinstance(item, str) and item.strip():
                    text_parts.append(item.strip())
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("content")
                    if isinstance(text_value, str) and text_value.strip():
                        text_parts.append(text_value.strip())
            if text_parts:
                return "\n".join(text_parts)
        reasoning_value = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning_value, str) and reasoning_value.strip():
            return reasoning_value.strip()

    return ""


def _should_disable_qwen3_thinking(model_name: Optional[str]) -> bool:
    return "qwen3" in str(model_name or "").strip().lower()


def _infer_model_context_window(model_name: Optional[str], default: int = 32768) -> int:
    normalized = str(model_name or "").strip().lower()
    if "qwen3-embedding" in normalized:
        return 8192
    if "text-embedding-3" in normalized:
        return 8192
    if "qwen3" in normalized or "qwen2.5" in normalized:
        return 32768
    return default


def normalize_embedding(vector: Any) -> list[float]:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return float(np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))


def extract_keywords(text: str, limit: int) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
    filtered = [token for token in tokens if token not in STOPWORDS]
    if not filtered:
        return []
    counts = Counter(filtered)
    return [token for token, _ in counts.most_common(limit)]


def keyword_overlap(query_keywords: set[str], candidate_keywords: list[str]) -> float:
    if not query_keywords or not candidate_keywords:
        return 0.0
    candidate_set = set(candidate_keywords)
    overlap = query_keywords & candidate_set
    return len(overlap) / max(len(query_keywords), len(candidate_set), 1)


def build_summary(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized

    sentence_break = normalized.rfind(". ", 0, max_chars)
    if sentence_break >= max_chars // 2:
        return normalized[: sentence_break + 1].strip()
    return normalized[:max_chars].rstrip() + "..."


def load_json_from_model_output(text: str) -> Optional[Any]:
    if not text:
        return None

    candidates = [text.strip()]
    fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(match.strip() for match in fence_matches if match.strip())

    bracket_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if bracket_match:
        candidates.append(bracket_match.group(1).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


class EmbeddingBackend:
    """Minimal embedding abstraction over OpenAI-compatible APIs or sentence-transformers."""

    def __init__(
        self,
        *,
        model: str,
        provider: Optional[str],
        base_url: Optional[str],
        api_key: Optional[str],
        embedding_dimensions: Optional[int] = None,
    ) -> None:
        self.model = model
        self.provider = (provider or "").lower()
        self.base_url = base_url
        self.api_key = api_key or "EMPTY"
        self.embedding_dimensions = embedding_dimensions
        self._hash_dimensions = embedding_dimensions or 1024

        self._client = None
        self._local_model = None
        self._use_hashing = self.provider == "hashing"
        self._tokenizer = load_local_hf_tokenizer(model) or ApproximateTokenizer()
        self._max_input_tokens = _infer_model_context_window(model, default=8192)
        self._use_openai = (
            self.provider in {"openai", "openai_compatible", "azure", "azure_openai"}
            or bool(self.base_url)
            or self.model.startswith("text-embedding-")
        )

        if self._use_hashing:
            return
        if self._use_openai:
            resolved_key = self.api_key if self.api_key else ("EMPTY" if self.base_url else None)
            self._client = OpenAI(
                api_key=resolved_key,
                base_url=self.base_url,
                http_client=httpx.Client(trust_env=False),
            )
        else:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence_transformers is required for local embeddings but is not installed."
                )
            self._local_model = SentenceTransformer(self.model)

    def _hash_embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self._hash_dimensions, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            return vector.tolist()

        for token in tokens:
            vector[hash(token) % self._hash_dimensions] += 1.0
        for left, right in zip(tokens, tokens[1:]):
            vector[hash(f"{left}::{right}") % self._hash_dimensions] += 0.5
        return normalize_embedding(vector)

    def _fit_text_to_token_budget(self, text: str, max_input_tokens: int) -> str:
        if not text:
            return ""

        tokens = self._tokenizer.encode(text, disallowed_special=())
        if len(tokens) <= max_input_tokens:
            return text

        if max_input_tokens <= 16:
            return self._tokenizer.decode(tokens[:max_input_tokens])

        head = max_input_tokens // 2
        tail = max_input_tokens - head
        fitted_tokens = tokens[:head] + tokens[-tail:]
        return self._tokenizer.decode(fitted_tokens)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._use_hashing:
            return [self._hash_embed_text(text) for text in texts]

        fitted_texts = [
            self._fit_text_to_token_budget(text, max(256, self._max_input_tokens - 32))
            for text in texts
        ]

        if self._use_openai:
            kwargs = {
                "model": self.model,
                "input": fitted_texts,
            }
            if self.embedding_dimensions and self.model.startswith("text-embedding-3-"):
                kwargs["dimensions"] = self.embedding_dimensions
            response = self._client.embeddings.create(**kwargs)
            return [normalize_embedding(item.embedding) for item in response.data]

        embeddings = self._local_model.encode(
            fitted_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [np.asarray(embedding, dtype=np.float32).tolist() for embedding in embeddings]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class ChatBackend:
    """Small OpenAI-compatible chat helper used by local runtimes."""

    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = None
        self._tokenizer = load_local_hf_tokenizer(model) or ApproximateTokenizer()
        self._context_window = _infer_model_context_window(model, default=32768)

        if not model:
            return
        if not base_url and not api_key:
            return

        resolved_key = api_key or ("EMPTY" if base_url else None)
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            http_client=httpx.Client(trust_env=False),
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(
        self,
        *,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        if not self._client:
            raise RuntimeError("Chat backend is not configured.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": self._fit_prompt_to_budget(prompt, system=system, max_tokens=max_tokens),
            }
        )
        response = self._create_with_retries(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _extract_openai_message_text(response.choices[0].message)

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        if not self._client:
            raise RuntimeError("Chat backend is not configured.")

        fitted_messages = []
        for index, message in enumerate(messages):
            copied_message = dict(message)
            if index == len(messages) - 1 and copied_message.get("role") == "user":
                system_text = "\n".join(
                    item.get("content", "")
                    for item in messages[:-1]
                    if item.get("role") == "system"
                )
                copied_message["content"] = self._fit_prompt_to_budget(
                    str(copied_message.get("content", "")),
                    system=system_text,
                    max_tokens=max_tokens,
                )
            fitted_messages.append(copied_message)

        response = self._create_with_retries(
            messages=fitted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _extract_openai_message_text(response.choices[0].message)

    def _fit_prompt_to_budget(self, prompt: str, *, system: Optional[str], max_tokens: int) -> str:
        if not prompt:
            return ""

        reserve_tokens = max(64, max_tokens or 0)
        system_tokens = len(self._tokenizer.encode(system or "", disallowed_special=()))
        max_prompt_tokens = max(256, self._context_window - reserve_tokens - system_tokens - 32)
        prompt_tokens = self._tokenizer.encode(prompt, disallowed_special=())
        if len(prompt_tokens) <= max_prompt_tokens:
            return prompt

        if max_prompt_tokens <= 32:
            return self._tokenizer.decode(prompt_tokens[-max_prompt_tokens:])

        head = max_prompt_tokens // 2
        tail = max_prompt_tokens - head
        fitted_tokens = prompt_tokens[:head] + prompt_tokens[-tail:]
        return self._tokenizer.decode(fitted_tokens)

    def _create_with_retries(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ):
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if _should_disable_qwen3_thinking(self.model):
            request_kwargs["extra_body"] = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }

        current_messages = [dict(message) for message in messages]
        for _attempt in range(3):
            try:
                request_kwargs["messages"] = current_messages
                return self._client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                error_text = str(exc).lower()
                if any(marker in error_text for marker in _CONTEXT_LIMIT_ERROR_MARKERS):
                    user_index = max(
                        (idx for idx, message in enumerate(current_messages) if message.get("role") == "user"),
                        default=None,
                    )
                    if user_index is None:
                        raise
                    current_prompt = str(current_messages[user_index].get("content", ""))
                    shortened_prompt = self._fit_prompt_to_budget(
                        current_prompt,
                        system="\n".join(
                            message.get("content", "")
                            for message in current_messages
                            if message.get("role") == "system"
                        ),
                        max_tokens=max(32, int(max_tokens * 0.8)),
                    )
                    if shortened_prompt == current_prompt:
                        raise
                    current_messages[user_index]["content"] = shortened_prompt
                    continue
                raise
