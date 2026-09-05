"""
Embedding model factory for hybrid router.

Supports:
- HuggingFace embedding models (sentence-transformers)
- OpenAI compatible embedding models (any OpenAI-compatible API)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed text(s) into vectors.

        Args:
            texts: Single text or list of texts to embed.

        Returns:
            Numpy array of embeddings. Shape: (n_texts, embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    """HuggingFace embedding model using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize HuggingFace embedding model.

        Args:
            model_name: HuggingFace model name or local path.
            device: Device to run model on ('cuda', 'cpu', or None for auto).
            normalize_embeddings: Whether to normalize embeddings to unit length.
            batch_size: Batch size for encoding.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for HuggingFace embeddings. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        logger.info(f"Loading HuggingFace embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded (dim={self._embedding_dim})")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed text(s) using sentence-transformers."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.array(embeddings)

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI compatible embedding model."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        normalize_embeddings: bool = True,
        batch_size: int = 100,
    ):
        """
        Initialize OpenAI compatible embedding model.

        Args:
            api_key: API key for the embedding service.
            model: Embedding model name.
            base_url: API base URL (for OpenAI compatible services).
            normalize_embeddings: Whether to normalize embeddings.
            batch_size: Maximum batch size for API calls.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install openai"
            )

        self.model = model
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._embedding_dim: Optional[int] = None

        logger.info(f"Initializing OpenAI embedding model: {model}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed text(s) using OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            response = self.client.embeddings.create(input=batch, model=self.model)

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings)

        # Store embedding dimension on first call
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]

        # Normalize if requested
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        if self._embedding_dim is None:
            # Probe with a test embedding
            test_embedding = self.embed("test")
            self._embedding_dim = test_embedding.shape[1]
        return self._embedding_dim


class EmbeddingModelFactory:
    """Factory for creating embedding models."""

    SUPPORTED_PROVIDERS = ["huggingface", "openai"]

    @staticmethod
    def create(
        provider: str,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseEmbeddingModel:
        """
        Create an embedding model instance.

        Args:
            provider: Provider type ('huggingface' or 'openai').
            model_name: Model name/identifier.
            config: Additional configuration for the model.

        Returns:
            BaseEmbeddingModel instance.

        Raises:
            ValueError: If provider is not supported.
        """
        config = config or {}
        provider = provider.lower()

        if provider == "huggingface":
            model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            return HuggingFaceEmbeddingModel(
                model_name=model_name,
                device=config.get("device"),
                normalize_embeddings=config.get("normalize_embeddings", True),
                batch_size=config.get("batch_size", 32),
            )

        elif provider == "openai":
            if "api_key" not in config:
                raise ValueError("api_key is required for OpenAI embedding model")

            model_name = model_name or "text-embedding-3-small"
            return OpenAIEmbeddingModel(
                api_key=config["api_key"],
                model=model_name,
                base_url=config.get("base_url", "https://api.openai.com/v1"),
                normalize_embeddings=config.get("normalize_embeddings", True),
                batch_size=config.get("batch_size", 100),
            )

        else:
            raise ValueError(
                f"Unsupported embedding provider: {provider}. "
                f"Supported: {EmbeddingModelFactory.SUPPORTED_PROVIDERS}"
            )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between vectors.

    Args:
        a: Query vector(s). Shape: (embedding_dim,) or (n_queries, embedding_dim)
        b: Document vectors. Shape: (n_docs, embedding_dim)

    Returns:
        Similarity scores. Shape: (n_queries, n_docs) or (n_docs,) if single query.
    """
    # Ensure 2D
    if a.ndim == 1:
        a = a.reshape(1, -1)
        squeeze_result = True
    else:
        squeeze_result = False

    # Normalize
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-10)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-10)

    # Compute similarity
    similarities = np.dot(a_norm, b_norm.T)

    if squeeze_result:
        return similarities.squeeze(0)
    return similarities

