import os
import re

from abc import ABC, abstractmethod
from functools import lru_cache

from memos.configs.embedder import BaseEmbedderConfig

try:
    from utils.provider_utils import ApproximateTokenizer, load_local_hf_tokenizer
except Exception:
    ApproximateTokenizer = None
    load_local_hf_tokenizer = None


class _LocalApproximateTokenizer:
    def encode(self, text, disallowed_special=()):
        del disallowed_special
        return list(text or "")

    def decode(self, tokens):
        return "".join(tokens)


@lru_cache(maxsize=8)
def _get_embedding_tokenizer(model_name: str | None):
    normalized_model_name = str(model_name or "").strip()

    if load_local_hf_tokenizer is not None and normalized_model_name:
        tokenizer = load_local_hf_tokenizer(normalized_model_name)
        if tokenizer is not None:
            return tokenizer

    try:
        import tiktoken

        if normalized_model_name:
            try:
                return tiktoken.encoding_for_model(normalized_model_name)
            except Exception:
                pass
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        if ApproximateTokenizer is not None:
            return ApproximateTokenizer()
        return _LocalApproximateTokenizer()


def _count_tokens_for_embedding(text: str, model_name: str | None = None) -> int:
    """
    Count tokens in text for embedding truncation.
    Uses the target embedding model tokenizer when available.

    Args:
        text: Text to count tokens for.
        model_name: Embedding model name for tokenizer lookup.

    Returns:
        Number of tokens.
    """
    if not text:
        return 0

    try:
        tokenizer = _get_embedding_tokenizer(model_name)
        return len(tokenizer.encode(text or "", disallowed_special=()))
    except Exception:
        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        zh = len(zh_chars)
        rest = len(text) - zh
        return zh + max(1, rest // 4)


def _truncate_text_to_tokens(text: str, max_tokens: int, model_name: str | None = None) -> str:
    """
    Truncate text to fit within max_tokens limit.
    Uses binary search to find the optimal truncation point.

    Args:
        text: Text to truncate.
        max_tokens: Maximum number of tokens allowed.
        model_name: Embedding model name for tokenizer lookup.

    Returns:
        Truncated text.
    """
    if not text or max_tokens is None or max_tokens <= 0:
        return text

    current_tokens = _count_tokens_for_embedding(text, model_name=model_name)
    if current_tokens <= max_tokens:
        return text

    # Binary search for the right truncation point
    low, high = 0, len(text)
    best_text = ""

    while low < high:
        mid = (low + high + 1) // 2  # Use +1 to avoid infinite loop
        truncated = text[:mid]
        tokens = _count_tokens_for_embedding(truncated, model_name=model_name)

        if tokens <= max_tokens:
            best_text = truncated
            low = mid
        else:
            high = mid - 1

    return best_text if best_text else text[:1]  # Fallback to at least one character


class BaseEmbedder(ABC):
    """Base class for all Embedding models."""

    @abstractmethod
    def __init__(self, config: BaseEmbedderConfig):
        """Initialize the embedding model with the given configuration."""
        self.config = config

    def _truncate_texts(self, texts: list[str], approx_char_per_token=1.0) -> (list)[str]:
        """
        Truncate texts to fit within max_tokens limit if configured.

        Args:
            texts: List of texts to truncate.

        Returns:
            List of truncated texts.
        """
        del approx_char_per_token
        if not hasattr(self, "config") or self.config.max_tokens is None:
            return texts
        max_tokens = int(self.config.max_tokens)
        model_name = getattr(self.config, "model_name_or_path", None)
        safety_margin = int(os.getenv("MEMOS_EMBEDDER_TOKEN_SAFETY_MARGIN", "256"))
        effective_budget = max(1, max_tokens - safety_margin)

        return [
            _truncate_text_to_tokens(text, effective_budget, model_name=model_name)
            for text in texts
        ]

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the given texts."""
