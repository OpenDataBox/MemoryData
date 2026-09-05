from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryRecord:
    memory_id: str
    conversation_id: str
    session: List[str]
    summary: str
    keywords: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryRecord":
        return cls(**payload)


@dataclass
class RetrievalHit:
    memory_id: str
    conversation_id: str
    score: float
    session: List[str]
    summary: str
    keywords: List[str]
    metadata: Dict[str, Any]
    granularity_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
