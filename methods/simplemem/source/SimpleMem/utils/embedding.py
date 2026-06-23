"""
Embedding utilities - Generate vector embeddings via an OpenAI-compatible
embeddings service.
"""
from typing import List
import numpy as np
from openai import OpenAI
from .. import config


class EmbeddingModel:
    """
    Embedding model backed by an OpenAI-compatible embeddings endpoint.
    """
    def __init__(self, model_name: str = None, use_optimization: bool = True):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_optimization = use_optimization
        self.model_type = "openai_compatible_embedding"
        self.supports_query_prompt = False
        self.base_url = config.EMBEDDING_BASE_URL
        self.api_key = config.EMBEDDING_API_KEY

        if not self.api_key:
            raise RuntimeError(
                "SimpleMem embedding requires SIMPLEMEM_EMBEDDING_API_KEY or OPENAI_API_KEY."
            )

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)
        self.dimension = self._resolve_dimension(config.EMBEDDING_DIMENSION)

        print(f"Using OpenAI-compatible embedding model: {self.model_name}")
        if self.base_url:
            print(f"Embedding base URL: {self.base_url}")
        print(f"Embedding dimension: {self.dimension}")

    def _resolve_dimension(self, configured_dimension: int) -> int:
        """Resolve embedding dimensionality before the LanceDB schema is created."""
        if configured_dimension:
            return int(configured_dimension)
        if "Qwen3-Embedding-4B" in self.model_name:
            return 2560
        if self.model_name == "text-embedding-3-large":
            return 3072
        if self.model_name in {"text-embedding-3-small", "text-embedding-ada-002"}:
            return 1536
        return self._probe_dimension()

    def _probe_dimension(self) -> int:
        """Probe the service once when the dimension is not declared explicitly."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=["SimpleMem embedding dimension probe"],
        )
        if not getattr(response, "data", None):
            raise RuntimeError("Embedding service returned no data during dimension probe.")
        return len(response.data[0].embedding)

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode list of texts to vectors
        
        Args:
        - texts: List of texts to encode
        - is_query: Whether these are query texts (for Qwen3 prompt optimization)
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._encode_standard(texts)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode single text
        
        Args:
        - text: Text to encode
        - is_query: Whether this is a query text (for Qwen3 prompt optimization)
        """
        return self.encode([text], is_query=is_query)[0]
    
    def encode_query(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries.
        """
        return self.encode(queries, is_query=True)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents.
        """
        return self.encode(documents, is_query=False)

    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the configured embeddings API and L2-normalize the result."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        if not getattr(response, "data", None):
            raise RuntimeError("Embedding service returned no vectors.")

        embeddings = [
            np.asarray(item.embedding, dtype=np.float32)
            for item in sorted(response.data, key=lambda item: item.index)
        ]
        matrix = np.vstack(embeddings)
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {matrix.shape[1]}."
            )

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
