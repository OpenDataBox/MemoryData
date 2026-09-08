# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUntypedBaseClass=false, reportUnknownMemberType=false, reportGeneralTypeIssues=false, reportAssignmentType=false
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

try:
    from models.memory_entry import MemoryEntry
except Exception:

    class MemoryEntry(BaseModel):
        """Fallback MemoryEntry definition for type checking and tooling."""

        entry_id: str
        lossless_restatement: str
        keywords: list[str]
        timestamp: Optional[str]
        location: Optional[str]
        persons: list[str]
        entities: list[str]
        topic: Optional[str]


class SessionStatus(str, Enum):
    """Lifecycle status for a memory session record."""

    active = "active"
    completed = "completed"
    failed = "failed"


class EventKind(str, Enum):
    """Kinds of events captured during a session."""

    message = "message"
    tool_use = "tool_use"
    file_change = "file_change"
    note = "note"
    system = "system"


class ObservationType(str, Enum):
    """Semantic observation categories extracted from sessions."""

    decision = "decision"
    bugfix = "bugfix"
    feature = "feature"
    refactor = "refactor"
    discovery = "discovery"
    change = "change"


class RedactionLevel(str, Enum):
    """Redaction levels for event payloads."""

    none = "none"
    partial = "partial"
    full = "full"


class SessionRecord(BaseModel):
    """Represents a conversation session persisted in SQLite."""

    id: Optional[int] = None
    tenant_id: str = "default"
    content_session_id: str
    memory_session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project: str
    user_prompt: Optional[str] = None
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: SessionStatus
    metadata_json: Optional[str] = None


class SessionEvent(BaseModel):
    """Represents a single event during a session timeline."""

    event_id: Optional[int] = None
    memory_session_id: str
    timestamp: datetime
    kind: EventKind
    title: Optional[str] = None
    payload_json: Optional[str] = None
    redaction_level: RedactionLevel = RedactionLevel.none


class CrossObservation(BaseModel):
    """An observation extracted from a session for cross-session memory."""

    obs_id: Optional[int] = None
    memory_session_id: str
    timestamp: datetime
    type: ObservationType
    title: str
    subtitle: Optional[str] = None
    facts_json: Optional[str] = None
    narrative: Optional[str] = None
    concepts_json: Optional[str] = None
    files_json: Optional[str] = None
    vector_ref: Optional[str] = None


class SessionSummary(BaseModel):
    """Summary generated when a session ends."""

    summary_id: Optional[int] = None
    memory_session_id: str
    timestamp: datetime
    request: Optional[str] = None
    investigated: Optional[str] = None
    learned: Optional[str] = None
    completed: Optional[str] = None
    next_steps: Optional[str] = None
    vector_ref: Optional[str] = None


class MemoryLink(BaseModel):
    """Traceability mapping from vectors back to source evidence."""

    link_id: Optional[int] = None
    memory_entry_id: str
    source_kind: str
    source_id: int
    score: float
    timestamp: datetime


class CrossMemoryEntry(MemoryEntry):
    """Memory entry with cross-session provenance fields."""

    tenant_id: str
    memory_session_id: str
    source_kind: str
    source_id: Optional[int] = None
    importance: float = Field(0.5, ge=0.0, le=1.0)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    superseded_by: Optional[str] = None


class ContextBundle(BaseModel):
    """Payload injected at session start with relevant cross-session context."""

    session_summaries: list[SessionSummary] = Field(default_factory=list)
    timeline_observations: list[CrossObservation] = Field(default_factory=list)
    memory_entries: list[CrossMemoryEntry] = Field(default_factory=list)
    total_tokens_estimate: int = 0

    def render(self, max_tokens: int, style: str = "summary") -> str:
        """Render the bundle into a string capped by a token estimate."""

        def estimate_tokens(text: str) -> int:
            return len(text.split())

        lines: list[str] = []
        token_count = 0

        def try_add(line: str) -> None:
            nonlocal token_count
            if not line:
                return
            next_tokens = estimate_tokens(line)
            if token_count + next_tokens > max_tokens:
                return
            lines.append(line)
            token_count += next_tokens

        if self.session_summaries:
            try_add("Session summaries:")
            for summary in self.session_summaries:
                text = (
                    summary.completed
                    or summary.learned
                    or summary.investigated
                    or summary.request
                    or "Summary available."
                )
                try_add(f"- {text}")

        if self.timeline_observations:
            try_add("Timeline observations:")
            for observation in self.timeline_observations:
                detail = observation.subtitle or observation.narrative or ""
                line = f"- {observation.title}"
                if detail:
                    line = f"{line}: {detail}"
                try_add(line)

        if self.memory_entries:
            try_add("Memory entries:")
            for entry in self.memory_entries:
                line = f"- {entry.lossless_restatement}"
                try_add(line)

        if not lines:
            return ""

        if style == "summary":
            return "\n".join(lines)

        return "\n".join(lines)


class FinalizationReport(BaseModel):
    """Report returned when a session finishes."""

    memory_session_id: str
    observations_count: int
    summary_generated: bool
    entries_stored: int
    consolidation_triggered: bool


class ConsolidationRun(BaseModel):
    """Record of a consolidation run and its policy/statistics."""

    run_id: Optional[int] = None
    tenant_id: str
    timestamp: datetime
    policy_json: Optional[str] = None
    stats_json: Optional[str] = None
