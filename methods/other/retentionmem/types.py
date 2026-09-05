"""Public, JSON-serializable contracts. Benchmark labels never enter Event."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Event:
    id: str
    text: str
    timestamp: str = ""
    actor: str = "user"
    kind: Literal["dialogue", "observation", "tool", "trajectory"] = "dialogue"
    session: str = ""
    source_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.id or not self.text.strip():
            raise ValueError("Event id and nonempty text are required")
        if self.kind not in {"dialogue", "observation", "tool", "trajectory"}:
            raise ValueError(f"Unknown event kind: {self.kind}")
        object.__setattr__(self, "source_ids", tuple(self.source_ids))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SourceSpan:
    event_id: str
    start: int
    end: int

    def extract(self, events: dict[str, Event]) -> str:
        event = events[self.event_id]
        if not (0 <= self.start < self.end <= len(event.text)):
            raise ValueError(f"Invalid source offsets: {self}")
        return event.text[self.start:self.end]


@dataclass(frozen=True)
class Probe:
    id: str
    question: str
    answer: str
    spans: tuple[SourceSpan, ...]
    scope: Literal["current", "historical", "atemporal", "action"] = "atemporal"

    def __post_init__(self):
        if not self.id or not self.question.strip() or not self.answer.strip() or not self.spans:
            raise ValueError("A probe requires a question, reference answer and anchored source spans")
        if self.scope not in {"current", "historical", "atemporal", "action"}:
            raise ValueError(f"Unknown probe scope: {self.scope}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "spans": tuple(SourceSpan(**s) for s in value["spans"])})


@dataclass(frozen=True)
class Candidate:
    summary: str
    probes: tuple[Probe, ...]
    style: str = "structured"


@dataclass(frozen=True)
class MemoryUnit:
    id: str
    text: str
    event_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    version: int = 0
    residual_spans: tuple[SourceSpan, ...] = ()


@dataclass
class MaintenanceResult:
    action: str
    reason: str
    region: list[str]
    retained_source_ids: list[str]
    before_tokens: int
    after_tokens: int
    summary_tokens: int = 0
    residual_tokens: int = 0
    predicted_risk: float | None = None
    budget_met: bool = True
    residual_spans: list[dict] = field(default_factory=list)
    costs: dict = field(default_factory=dict)
    version: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RetrievedMemory:
    id: str
    text: str
    source_ids: tuple[str, ...]
    tokens: int
    score: float
    version: int
    truncated: bool = False
    raw_fallback: bool = False
