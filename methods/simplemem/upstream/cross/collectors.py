# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportGeneralTypeIssues=false, reportUndefinedVariable=false
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING, Union, cast

if TYPE_CHECKING:
    from enum import Enum
    from pydantic import BaseModel

    class EventKind(str, Enum):
        """Fallback EventKind definition for type checking."""

        message = "message"
        tool_use = "tool_use"
        file_change = "file_change"
        note = "note"
        system = "system"

    class RedactionLevel(str, Enum):
        """Fallback RedactionLevel definition for type checking."""

        none = "none"
        partial = "partial"
        full = "full"

    class SessionEvent(BaseModel):
        """Fallback SessionEvent definition for type checking."""

        memory_session_id: str
        timestamp: datetime
        kind: EventKind
        title: Optional[str] = None
        payload_json: Optional[str] = None
        redaction_level: RedactionLevel = RedactionLevel.none

    class ObservationType(str, Enum):
        decision = "decision"
        bugfix = "bugfix"
        feature = "feature"
        refactor = "refactor"
        discovery = "discovery"
        change = "change"

    class CrossObservation(BaseModel):
        memory_session_id: str
        timestamp: datetime
        type: ObservationType
        title: str
        narrative: Optional[str] = None
else:
    from cross.types import (
        CrossObservation,
        EventKind,
        ObservationType,
        RedactionLevel,
        SessionEvent,
    )


if TYPE_CHECKING:
    from models.memory_entry import Dialogue as Dialogue
else:
    try:
        from models.memory_entry import Dialogue
    except Exception:
        from pydantic import BaseModel

        class Dialogue(BaseModel):
            """Fallback Dialogue definition for type checking."""

            dialogue_id: int
            speaker: str
            content: str
            timestamp: Optional[str] = None


JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, list["JSONValue"], dict[str, "JSONValue"]]
JSONDict = dict[str, JSONValue]


def _event_kind(value: str) -> EventKind:
    try:
        return EventKind(value)
    except Exception:
        return cast(EventKind, value)


def _redaction_level(value: str) -> RedactionLevel:
    try:
        return RedactionLevel(value)
    except Exception:
        return cast(RedactionLevel, value)


EVENT_KIND_MESSAGE = _event_kind("message")
EVENT_KIND_TOOL_USE = _event_kind("tool_use")
EVENT_KIND_FILE_CHANGE = _event_kind("file_change")
EVENT_KIND_NOTE = _event_kind("note")

REDACTION_NONE = _redaction_level("none")
REDACTION_PARTIAL = _redaction_level("partial")
REDACTION_FULL = _redaction_level("full")


class RedactionFilter:
    """Filters sensitive content from event payloads before storage.

    Detects and redacts:
    - API keys (patterns like sk-*, key-*, token-*, etc.)
    - Credentials (password fields, auth headers)
    - File paths with sensitive names (.env, credentials, secrets)
    - Base64-encoded tokens
    """

    SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"(sk-[a-zA-Z0-9]{20,})"), "[REDACTED_API_KEY]"),
        (re.compile(r"(key-[a-zA-Z0-9]{20,})"), "[REDACTED_KEY]"),
        (
            re.compile(r"(token[\"\s:=]+[\"\']?[a-zA-Z0-9_\-\.]{20,}[\"\']?)"),
            "[REDACTED_TOKEN]",
        ),
        (
            re.compile(r"(password[\"\s:=]+[\"\']?[^\s\"\']{4,}[\"\']?)"),
            "[REDACTED_PASSWORD]",
        ),
        (re.compile(r"(Bearer\s+[a-zA-Z0-9_\-\.]+)"), "Bearer [REDACTED]"),
        (
            re.compile(r"(Authorization[\"\s:]+[\"\']?[^\s\"\']+[\"\']?)"),
            "Authorization: [REDACTED]",
        ),
        (
            re.compile(r"([A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"),
            "[REDACTED_JWT]",
        ),
        (
            re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/])"),
            "[REDACTED_BASE64]",
        ),
    ]

    _SENSITIVE_FILE_PATTERN: re.Pattern[str] = re.compile(
        r"(\.env|credentials|secret|secrets|password|token|\.pem|\.key|\.p12|\.pfx|id_rsa|id_dsa|\.npmrc|\.aws|\.gcp|\.azure)",
        re.IGNORECASE,
    )

    def redact(self, text: str) -> tuple[str, RedactionLevel]:
        """Apply redaction patterns to text.

        Returns:
            Tuple of (redacted_text, redaction_level)
        """

        if not text:
            return "", RedactionLevel.none

        redacted = text
        redaction_level = RedactionLevel.none

        for pattern, replacement in self.SENSITIVE_PATTERNS:
            if pattern.search(redacted):
                redacted = pattern.sub(replacement, redacted)
                redaction_level = RedactionLevel.partial

        return redacted, redaction_level

    def should_redact_file(self, filepath: str) -> bool:
        """Check if a file path suggests sensitive content."""

        if not filepath:
            return False
        return bool(self._SENSITIVE_FILE_PATTERN.search(filepath))


class EventCollector:
    """Collects and buffers session events during a conversation.

    Maintains an in-memory buffer of events for the current session,
    applies redaction, and provides methods to extract observations.
    """

    def __init__(
        self,
        redaction_filter: Optional[RedactionFilter] = None,
        tool_output_max_length: int = 2000,
        memory_session_id: Optional[str] = None,
    ):
        self._events: list[SessionEvent] = []
        self._redaction_filter: RedactionFilter = redaction_filter or RedactionFilter()
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._lock: threading.Lock = threading.Lock()
        self._tool_output_max_length: int = tool_output_max_length
        # For compatibility with session_manager.py which passes memory_session_id to constructor
        self.memory_session_id: Optional[str] = memory_session_id

    def record_message(
        self,
        memory_session_id: str,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
    ) -> SessionEvent:
        """Record a user/assistant message.

        Creates a SessionEvent with kind=EventKind.message.
        """

        safe_content, redaction_level = self._redact_text(content)
        payload: JSONDict = {
            "role": role,
            "content": safe_content,
        }
        return self._record_event(
            memory_session_id=memory_session_id,
            kind=EventKind.message,
            title=role,
            payload=payload,
            redaction_level=redaction_level,
            timestamp=timestamp,
        )

    def record_tool_use(
        self,
        memory_session_id: str,
        tool_name: str,
        tool_input: str,
        tool_output: str,
        timestamp: Optional[datetime] = None,
        files_read: Optional[list[str]] = None,
        files_modified: Optional[list[str]] = None,
    ) -> SessionEvent:
        """Record a tool usage event.

        Creates a SessionEvent with kind=EventKind.tool_use.
        The payload includes tool_name, redacted input/output, and file lists.
        """

        safe_input, input_level = self._redact_text(tool_input)
        safe_output, output_level = self._redact_text(tool_output)
        safe_output = self._truncate(safe_output, self._tool_output_max_length)

        files_read = files_read or []
        files_modified = files_modified or []
        safe_files_read, files_read_level = self._redact_file_list(files_read)
        safe_files_modified, files_modified_level = self._redact_file_list(
            files_modified
        )
        safe_files_read_values: list[JSONValue] = [path for path in safe_files_read]
        safe_files_modified_values: list[JSONValue] = [
            path for path in safe_files_modified
        ]

        redaction_level = self._max_redaction_level(
            [input_level, output_level, files_read_level, files_modified_level]
        )

        payload: JSONDict = {
            "tool_name": tool_name,
            "tool_input": safe_input,
            "tool_output": safe_output,
            "files_read": safe_files_read_values,
            "files_modified": safe_files_modified_values,
        }

        return self._record_event(
            memory_session_id=memory_session_id,
            kind=EventKind.tool_use,
            title=tool_name,
            payload=payload,
            redaction_level=redaction_level,
            timestamp=timestamp,
        )

    def record_file_change(
        self,
        memory_session_id: str,
        filepath: str,
        change_type: str = "modified",
        timestamp: Optional[datetime] = None,
    ) -> SessionEvent:
        """Record a file change event."""

        redaction_level = RedactionLevel.none
        safe_path = filepath
        if self._redaction_filter.should_redact_file(filepath):
            safe_path = "[REDACTED_PATH]"
            redaction_level = RedactionLevel.full

        payload: JSONDict = {
            "filepath": safe_path,
            "change_type": change_type,
        }

        return self._record_event(
            memory_session_id=memory_session_id,
            kind=EventKind.file_change,
            title=change_type,
            payload=payload,
            redaction_level=redaction_level,
            timestamp=timestamp,
        )

    def record_note(
        self,
        memory_session_id: str,
        note: str,
        timestamp: Optional[datetime] = None,
    ) -> SessionEvent:
        """Record a freeform note event."""

        safe_note, redaction_level = self._redact_text(note)
        payload: JSONDict = {
            "note": safe_note,
        }

        return self._record_event(
            memory_session_id=memory_session_id,
            kind=EventKind.note,
            title="note",
            payload=payload,
            redaction_level=redaction_level,
            timestamp=timestamp,
        )

    def get_events(self, kinds: Optional[list[EventKind]] = None) -> list[SessionEvent]:
        """Get collected events, optionally filtered by kind."""

        with self._lock:
            events = list(self._events)

        if kinds:
            events = [event for event in events if event.kind in kinds]

        return [event for event in events if self._is_valuable(event)]

    def get_tool_events(self) -> list[SessionEvent]:
        """Get only tool usage events (most valuable for memory)."""

        return self.get_events(kinds=[EventKind.tool_use])

    def clear(self):
        """Clear all buffered events."""

        with self._lock:
            self._events.clear()

    @property
    def event_count(self) -> int:
        return len(self._events)

    def add_event(
        self,
        kind: EventKind,
        title: Optional[str] = None,
        payload: Optional[dict[str, object]] = None,
    ) -> SessionEvent:
        """Add an event by kind (compatibility API for session_manager.py)."""
        session_id = self.memory_session_id or ""
        payload_dict: JSONDict = cast(JSONDict, dict(payload) if payload else {})
        return self._record_event(
            memory_session_id=session_id,
            kind=kind,
            title=title,
            payload=payload_dict,
            redaction_level=RedactionLevel.none,
            timestamp=None,
        )

    def flush(self) -> list[SessionEvent]:
        """Return all buffered events and clear the buffer."""
        events = self.get_events()
        self.clear()
        return events

    def _record_event(
        self,
        memory_session_id: str,
        kind: EventKind,
        title: Optional[str],
        payload: JSONDict,
        redaction_level: RedactionLevel,
        timestamp: Optional[datetime],
    ) -> SessionEvent:
        event = SessionEvent(
            memory_session_id=memory_session_id,
            timestamp=timestamp or datetime.now(timezone.utc),
            kind=kind,
            title=title,
            payload_json=self._serialize_payload(payload),
            redaction_level=redaction_level,
        )

        with self._lock:
            self._events.append(event)

        return event

    def _serialize_payload(self, payload: JSONDict) -> str:
        try:
            return json.dumps(payload, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            self._logger.warning("Failed to serialize payload: %s", exc)
            fallback: JSONDict = {"raw": str(payload)}
            return json.dumps(fallback, ensure_ascii=True)

    def _normalize_text(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=True)
        except (TypeError, ValueError):
            return str(value)

    def _redact_text(self, value: object) -> tuple[str, RedactionLevel]:
        text = self._normalize_text(value)
        if not text:
            return "", RedactionLevel.none

        if text[:1] in {"{", "["}:
            redacted, level = self._redact_json_payload(text)
            if redacted is not None:
                return redacted, level

        return self._redaction_filter.redact(text)

    def _redact_json_payload(self, text: str) -> tuple[Optional[str], RedactionLevel]:
        try:
            data = cast(JSONValue, json.loads(text))
        except json.JSONDecodeError:
            return None, RedactionLevel.none

        redaction_level = RedactionLevel.none

        def redact_value(value: JSONValue) -> JSONValue:
            nonlocal redaction_level
            if isinstance(value, str):
                redacted, level = self._redaction_filter.redact(value)
                redaction_level = self._max_redaction_level([redaction_level, level])
                return redacted
            if isinstance(value, list):
                return [redact_value(item) for item in value]
            if isinstance(value, dict):
                return {str(key): redact_value(item) for key, item in value.items()}
            return value

        redacted_data = redact_value(data)
        return json.dumps(redacted_data, ensure_ascii=True), redaction_level

    def _truncate(self, text: str, max_length: int) -> str:
        if max_length <= 0:
            return ""
        if len(text) <= max_length:
            return text
        return f"{text[:max_length]}..."

    def _redact_file_list(self, files: list[str]) -> tuple[list[str], RedactionLevel]:
        redaction_level = RedactionLevel.none
        safe_files: list[str] = []
        for path in files:
            if self._redaction_filter.should_redact_file(path):
                safe_files.append("[REDACTED_PATH]")
                redaction_level = self._max_redaction_level(
                    [redaction_level, RedactionLevel.full]
                )
            else:
                safe_files.append(path)
        return safe_files, redaction_level

    def _is_valuable(self, event: SessionEvent) -> bool:
        if event.kind == EventKind.message:
            payload = _safe_load_payload(event.payload_json)
            content = str(payload.get("content", "")).strip()
            return bool(content)
        if event.kind == EventKind.tool_use:
            payload = _safe_load_payload(event.payload_json)
            tool_name = str(payload.get("tool_name", "")).strip()
            return bool(tool_name)
        if event.kind == EventKind.file_change:
            payload = _safe_load_payload(event.payload_json)
            filepath = str(payload.get("filepath", "")).strip()
            return bool(filepath)
        if event.kind == EventKind.note:
            payload = _safe_load_payload(event.payload_json)
            note = str(payload.get("note", "")).strip()
            return bool(note)
        return True

    def _max_redaction_level(self, levels: list[RedactionLevel]) -> RedactionLevel:
        if RedactionLevel.full in levels:
            return RedactionLevel.full
        if RedactionLevel.partial in levels:
            return RedactionLevel.partial
        return RedactionLevel.none


class ObservationExtractor:
    """Extracts structured observations from collected events.

    Uses SimpleMem's MemoryBuilder internally by converting events
    into Dialogue objects that the builder can process.
    """

    def events_to_dialogues(self, events: list[SessionEvent]) -> list[Dialogue]:
        """Convert session events to SimpleMem Dialogue objects.

        Maps events to the format MemoryBuilder expects:
        - Messages become speaker/content dialogues
        - Tool uses become "Agent: Used tool X -> result Y" dialogues
        - File changes become "System: File X was modified" dialogues

        Returns:
            List of Dialogue objects (from models.memory_entry)
        """

        dialogues: list[Dialogue] = []
        for index, event in enumerate(events, start=1):
            payload = _safe_load_payload(event.payload_json)
            speaker, content = self._event_to_dialogue_content(event, payload)
            if not content:
                continue
            timestamp = event.timestamp.astimezone(timezone.utc).isoformat()
            dialogues.append(
                Dialogue(
                    dialogue_id=index,
                    speaker=speaker,
                    content=content,
                    timestamp=timestamp,
                )
            )
        return dialogues

    def extract_tool_summary(self, tool_events: list[SessionEvent]) -> list[JSONDict]:
        """Extract a summary of tool usage from events.

        Returns list of dicts with:
        - tool_name: str
        - input_summary: str (first 200 chars)
        - output_summary: str (first 500 chars)
        - files_read: list[str]
        - files_modified: list[str]
        - timestamp: str
        """

        summaries: list[JSONDict] = []
        for event in tool_events:
            payload = _safe_load_payload(event.payload_json)
            tool_name = str(payload.get("tool_name", ""))
            tool_input = str(payload.get("tool_input", ""))
            tool_output = str(payload.get("tool_output", ""))
            summary: JSONDict = {
                "tool_name": tool_name,
                "input_summary": tool_input[:200],
                "output_summary": tool_output[:500],
                "files_read": payload.get("files_read", []) or [],
                "files_modified": payload.get("files_modified", []) or [],
                "timestamp": event.timestamp.astimezone(timezone.utc).isoformat(),
            }
            summaries.append(summary)
        return summaries

    def estimate_session_value(self, events: list[SessionEvent]) -> float:
        """Estimate the value/importance of a session based on events.

        Higher value for sessions with:
        - More tool usage (especially file modifications)
        - Longer conversations
        - More diverse event types
        - Code changes

        Returns:
            Float between 0.0 and 1.0
        """

        if not events:
            return 0.0

        message_count = sum(1 for event in events if event.kind == EventKind.message)
        tool_count = sum(1 for event in events if event.kind == EventKind.tool_use)
        file_change_count = sum(
            1 for event in events if event.kind == EventKind.file_change
        )
        note_count = sum(1 for event in events if event.kind == EventKind.note)

        modified_file_hits = 0
        for event in events:
            if event.kind != EventKind.tool_use:
                continue
            payload = _safe_load_payload(event.payload_json)
            modified = payload.get("files_modified", []) or []
            if isinstance(modified, list):
                modified_file_hits += len(
                    [item for item in modified if isinstance(item, str)]
                )
            elif isinstance(modified, str):
                modified_file_hits += 1

        value = 0.1
        value += min(message_count * 0.02, 0.3)
        value += min(tool_count * 0.08, 0.5)
        value += min(file_change_count * 0.12, 0.5)
        value += min(modified_file_hits * 0.05, 0.4)
        if (
            sum(
                bool(count)
                for count in [message_count, tool_count, file_change_count, note_count]
            )
            >= 3
        ):
            value += 0.1

        return max(0.0, min(1.0, value))

    _KIND_TO_OBS_TYPE: dict[str, ObservationType] = {
        "tool_use": ObservationType.change,
        "file_change": ObservationType.change,
        "message": ObservationType.discovery,
        "note": ObservationType.discovery,
        "system": ObservationType.discovery,
    }

    def extract_from_events(
        self,
        events: list[SessionEvent],
        memory_session_id: str,
    ) -> list[CrossObservation]:
        """Extract CrossObservation objects from events (compatibility API)."""
        observations: list[CrossObservation] = []
        for event in events:
            title = event.title
            if not title:
                continue
            kind_value = (
                event.kind.value if hasattr(event.kind, "value") else str(event.kind)
            )
            obs_type = self._KIND_TO_OBS_TYPE.get(kind_value, ObservationType.discovery)
            payload = _safe_load_payload(event.payload_json)
            narrative: Optional[str] = None
            if payload:
                content = (
                    payload.get("content")
                    or payload.get("output")
                    or payload.get("note")
                )
                if content:
                    narrative = (
                        str(content)[:500] if len(str(content)) > 500 else str(content)
                    )
            observations.append(
                CrossObservation(
                    memory_session_id=memory_session_id,
                    timestamp=event.timestamp,
                    type=obs_type,
                    title=title,
                    narrative=narrative,
                )
            )
        return observations

    def _event_to_dialogue_content(
        self, event: SessionEvent, payload: JSONDict
    ) -> tuple[str, str]:
        if event.kind == EventKind.message:
            speaker = str(payload.get("role", "Speaker")) or "Speaker"
            content = str(payload.get("content", "")).strip()
            return speaker, content

        if event.kind == EventKind.tool_use:
            tool_name = str(payload.get("tool_name", "tool")).strip() or "tool"
            tool_input = str(payload.get("tool_input", "")).strip()
            tool_output = str(payload.get("tool_output", "")).strip()
            content_parts = [f"Used tool {tool_name}."]
            if tool_input:
                content_parts.append(f"Input: {tool_input}")
            if tool_output:
                content_parts.append(f"Output: {tool_output}")
            return "Agent", " ".join(content_parts).strip()

        if event.kind == EventKind.file_change:
            filepath = str(payload.get("filepath", "file")).strip() or "file"
            change_type = (
                str(payload.get("change_type", "modified")).strip() or "modified"
            )
            return "System", f"File {filepath} was {change_type}."

        if event.kind == EventKind.note:
            note = str(payload.get("note", "")).strip()
            return "System", note

        return "System", ""


def create_collector(enable_redaction: bool = True) -> EventCollector:
    """Create an EventCollector with optional redaction."""

    redaction_filter = RedactionFilter() if enable_redaction else None
    return EventCollector(redaction_filter=redaction_filter)


def collect_tool_event(
    memory_session_id: str,
    tool_name: str,
    tool_input: str,
    tool_output: str,
    **kwargs: object,
) -> SessionEvent:
    """Convenience function to create a single tool event."""

    collector = create_collector(enable_redaction=True)
    timestamp_value = kwargs.get("timestamp")
    files_read_value = kwargs.get("files_read")
    files_modified_value = kwargs.get("files_modified")

    timestamp = timestamp_value if isinstance(timestamp_value, datetime) else None
    files_read = (
        [item for item in files_read_value if isinstance(item, str)]
        if isinstance(files_read_value, list)
        else None
    )
    files_modified = (
        [item for item in files_modified_value if isinstance(item, str)]
        if isinstance(files_modified_value, list)
        else None
    )

    return collector.record_tool_use(
        memory_session_id=memory_session_id,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        timestamp=timestamp,
        files_read=files_read,
        files_modified=files_modified,
    )


def _safe_load_payload(payload_json: Optional[str]) -> JSONDict:
    if not payload_json:
        return {}
    try:
        data = cast(JSONValue, json.loads(payload_json))
        if isinstance(data, dict):
            return {str(key): value for key, value in data.items()}
        return {"value": data}
    except json.JSONDecodeError:
        return {"raw": payload_json}
