from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class TextEmbedder:
    _SBERT_MODEL_MAP = {
        "mpnet": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "qaminilm": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    }

    def __init__(
        self,
        backend: str = "contriever",
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self.backend = backend
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim: Optional[int] = None

        if backend == "contriever":
            self.model = AutoModel.from_pretrained("facebook/contriever").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
            self.model.eval()
        elif backend in self._SBERT_MODEL_MAP:
            self.model = SentenceTransformer(self._SBERT_MODEL_MAP[backend], device=self.device)
            self.tokenizer = None
        else:
            raise ValueError(
                f"Unsupported embedder '{backend}'. "
                f"Choose one of {['contriever', 'mpnet', 'minilm', 'qaminilm']}."
            )

    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            if self.embedding_dim is None:
                return torch.empty((0, 0), dtype=torch.float32)
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        if self.backend == "contriever":
            vectors = self._encode_contriever(texts)
        else:
            vectors = self._encode_sbert(texts)
        self.embedding_dim = vectors.shape[1]
        return vectors

    def _encode_sbert(self, texts: List[str]) -> torch.Tensor:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if not isinstance(vectors, torch.Tensor):
            vectors = torch.tensor(vectors)
        return vectors.detach().cpu().float()

    def _encode_contriever(self, texts: List[str]) -> torch.Tensor:
        all_vecs = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs).last_hidden_state
                pooled = self._mean_pooling(outputs, inputs["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                all_vecs.append(pooled.cpu())
        return torch.cat(all_vecs, dim=0).float()

    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
