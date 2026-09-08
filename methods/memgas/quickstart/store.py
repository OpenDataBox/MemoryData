from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .schemas import MemoryRecord


class MemoryStore:
    EMBED_KEYS = ("session", "turn", "summary", "keyword", "hybrid")

    def __init__(self, storage_dir: str) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.storage_dir / "memory_state.pt"

        self.records: List[MemoryRecord] = []
        self.embeddings: Dict[str, torch.Tensor] = {}
        self._id_to_index: Dict[str, int] = {}
        self.load()

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: MemoryRecord, vectors: Dict[str, torch.Tensor]) -> None:
        for key in self.EMBED_KEYS:
            if key not in vectors:
                raise KeyError(f"Missing embedding key: {key}")

        row_index = len(self.records)
        self.records.append(record)
        self._id_to_index[record.memory_id] = row_index
        self._append_vectors(vectors)

    def update(self, memory_id: str, record: MemoryRecord, vectors: Dict[str, torch.Tensor]) -> None:
        row_index = self.get_index(memory_id)
        self.records[row_index] = record
        for key in self.EMBED_KEYS:
            row = self._to_row(vectors[key]).squeeze(0)
            self.embeddings[key][row_index] = row

    def get_index(self, memory_id: str) -> int:
        if memory_id not in self._id_to_index:
            raise KeyError(f"memory_id '{memory_id}' not found")
        return self._id_to_index[memory_id]

    def filter_indices(self, conversation_id: Optional[str] = None) -> List[int]:
        if conversation_id is None:
            return list(range(len(self.records)))
        return [
            idx
            for idx, record in enumerate(self.records)
            if record.conversation_id == conversation_id
        ]

    def slice_embeddings(self, indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        return {key: value[list(indices)] for key, value in self.embeddings.items()}

    def save(self) -> None:
        payload = {
            "version": 1,
            "records": [record.to_dict() for record in self.records],
            "embeddings": {k: v.cpu() for k, v in self.embeddings.items()},
        }
        torch.save(payload, self.state_path)

    def load(self) -> None:
        if not self.state_path.exists():
            return
        payload = torch.load(self.state_path, map_location="cpu")
        self.records = [MemoryRecord.from_dict(item) for item in payload.get("records", [])]
        self.embeddings = {
            key: value.float().cpu() for key, value in payload.get("embeddings", {}).items()
        }
        self._id_to_index = {
            record.memory_id: idx for idx, record in enumerate(self.records)
        }

    def _append_vectors(self, vectors: Dict[str, torch.Tensor]) -> None:
        for key in self.EMBED_KEYS:
            row = self._to_row(vectors[key])
            if key not in self.embeddings:
                self.embeddings[key] = row
            else:
                self.embeddings[key] = torch.cat([self.embeddings[key], row], dim=0)

    @staticmethod
    def _to_row(vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim == 1:
            vector = vector.unsqueeze(0)
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError(f"Expected vector with shape [D] or [1, D], got {tuple(vector.shape)}")
        return vector.detach().cpu().float()
