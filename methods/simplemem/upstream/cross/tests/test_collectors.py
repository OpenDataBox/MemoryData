# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportGeneralTypeIssues=false, reportArgumentType=false, reportAssignmentType=false
"""Unit tests for RedactionFilter, EventCollector, and ObservationExtractor."""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cross.collectors import RedactionFilter, EventCollector, ObservationExtractor
from cross.types import RedactionLevel, EventKind, ObservationType, SessionEvent


# ---------------------------------------------------------------------------
# TestRedactionFilter
# ---------------------------------------------------------------------------


class TestRedactionFilter:
    """Tests for the RedactionFilter pattern-based redaction engine."""

    def test_redact_none_level(self) -> None:
        """Text without any sensitive patterns passes through unchanged."""
        rf = RedactionFilter()
        text = "This is a perfectly normal message with no secrets."
        result, level = rf.redact(text)
        assert result == text
        assert level == RedactionLevel.none

    def test_redact_full_level(self) -> None:
        """Sensitive file paths trigger full redaction via the collector pipeline."""
        rf = RedactionFilter()
        # RedactionFilter.should_redact_file detects sensitive names
        assert rf.should_redact_file(".env") is True
        assert rf.should_redact_file("credentials.json") is True
        # EventCollector applies RedactionLevel.full for such paths
        collector = EventCollector(redaction_filter=rf)
        event = collector.record_file_change(
            memory_session_id="sess-full-001",
            filepath="/app/.env",
            change_type="modified",
        )
        assert event.redaction_level == RedactionLevel.full
        payload = json.loads(event.payload_json or "{}")
        assert payload["filepath"] == "[REDACTED_PATH]"

    def test_redact_partial_api_key(self) -> None:
        """API key pattern (sk-...) is redacted."""
        rf = RedactionFilter()
        text = "my key is sk-abc123def456ghi789jkl012"
        result, level = rf.redact(text)
        assert "sk-abc123def456ghi789jkl012" not in result
        assert "[REDACTED_API_KEY]" in result
        assert level == RedactionLevel.partial

    def test_redact_partial_bearer_token(self) -> None:
        """Bearer token header is redacted."""
        rf = RedactionFilter()
        text = "Bearer eyJhbGciOiJIUzI1NiJ9.abc"
        result, level = rf.redact(text)
        assert "eyJhbGciOiJIUzI1NiJ9" not in result
        assert "Bearer [REDACTED]" in result
        assert level == RedactionLevel.partial

    def test_redact_partial_password(self) -> None:
        """Password field pattern is redacted."""
        rf = RedactionFilter()
        text = "password=mysecret123"
        result, level = rf.redact(text)
        assert "mysecret123" not in result
        assert "[REDACTED_PASSWORD]" in result
        assert level == RedactionLevel.partial

    def test_redact_partial_email(self) -> None:
        """Email addresses are not currently matched by the pattern-based filter.

        The RedactionFilter uses explicit secret patterns (API keys, passwords,
        tokens) rather than PII patterns.  Email passes through unchanged.
        """
        rf = RedactionFilter()
        text = "contact alice@example.com for info"
        result, level = rf.redact(text)
        assert result == text
        assert level == RedactionLevel.none

    def test_redact_partial_ip(self) -> None:
        """IP addresses are not currently matched by the pattern-based filter.

        The RedactionFilter targets credential-like secrets, not network
        addresses.  IP addresses pass through unchanged.
        """
        rf = RedactionFilter()
        text = "server at 192.168.1.100"
        result, level = rf.redact(text)
        assert result == text
        assert level == RedactionLevel.none

    def test_redact_partial_no_false_positives(self) -> None:
        """Normal prose without secret-like patterns passes through unchanged."""
        rf = RedactionFilter()
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "We deployed version 3.2.1 to staging on Tuesday."
        )
        result, level = rf.redact(text)
        assert result == text
        assert level == RedactionLevel.none


# ---------------------------------------------------------------------------
# TestEventCollector
# ---------------------------------------------------------------------------


class TestEventCollector:
    """Tests for the EventCollector in-memory event buffer."""

    def _make_collector(self) -> EventCollector:
        return EventCollector(redaction_filter=RedactionFilter())

    def test_record_message(self) -> None:
        """Records a message and verifies all SessionEvent fields."""
        collector = self._make_collector()
        ts = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        event = collector.record_message(
            memory_session_id="sess-msg-001",
            role="user",
            content="Hello world",
            timestamp=ts,
        )
        assert isinstance(event, SessionEvent)
        assert event.memory_session_id == "sess-msg-001"
        assert event.kind == EventKind.message
        assert event.timestamp == ts
        assert event.title == "user"
        payload = json.loads(event.payload_json or "{}")
        assert payload["role"] == "user"
        assert payload["content"] == "Hello world"
        assert event.redaction_level == RedactionLevel.none

    def test_record_tool_use(self) -> None:
        """Records tool use with name, input, and output."""
        collector = self._make_collector()
        event = collector.record_tool_use(
            memory_session_id="sess-tool-001",
            tool_name="grep",
            tool_input="search pattern",
            tool_output="found 3 matches in auth.py",
        )
        assert event.kind == EventKind.tool_use
        assert event.title == "grep"
        payload = json.loads(event.payload_json or "{}")
        assert payload["tool_name"] == "grep"
        assert payload["tool_input"] == "search pattern"
        assert payload["tool_output"] == "found 3 matches in auth.py"

    def test_record_file_change(self) -> None:
        """Records file change with path and change type."""
        collector = self._make_collector()
        event = collector.record_file_change(
            memory_session_id="sess-fc-001",
            filepath="src/main.py",
            change_type="created",
        )
        assert event.kind == EventKind.file_change
        assert event.title == "created"
        payload = json.loads(event.payload_json or "{}")
        assert payload["filepath"] == "src/main.py"
        assert payload["change_type"] == "created"

    def test_record_note(self) -> None:
        """Records a freeform note event."""
        collector = self._make_collector()
        event = collector.record_note(
            memory_session_id="sess-note-001",
            note="User prefers dark mode",
        )
        assert event.kind == EventKind.note
        assert event.title == "note"
        payload = json.loads(event.payload_json or "{}")
        assert payload["note"] == "User prefers dark mode"

    def test_flush_returns_all_and_clears(self) -> None:
        """Record 3 events, get_events returns 3, clear resets to 0."""
        collector = self._make_collector()
        for i in range(3):
            collector.record_message(
                memory_session_id="sess-flush-001",
                role="user",
                content=f"Message {i}",
            )
        events = collector.get_events()
        assert len(events) == 3
        collector.clear()
        assert len(collector.get_events()) == 0

    def test_count(self) -> None:
        """Verify event_count property matches the buffer size."""
        collector = self._make_collector()
        assert collector.event_count == 0
        collector.record_message(
            memory_session_id="sess-cnt-001", role="user", content="one"
        )
        assert collector.event_count == 1
        collector.record_note(memory_session_id="sess-cnt-001", note="two")
        assert collector.event_count == 2

    def test_thread_safety(self) -> None:
        """Record events from 10 threads (10 events each), verify no data loss."""
        collector = self._make_collector()
        num_threads = 10
        events_per_thread = 10
        barrier = threading.Barrier(num_threads)

        def worker(thread_id: int) -> None:
            barrier.wait()
            for i in range(events_per_thread):
                collector.record_message(
                    memory_session_id="sess-thread-001",
                    role="user",
                    content=f"Thread {thread_id} message {i}",
                )

        threads = [
            threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert collector.event_count == num_threads * events_per_thread

    def test_redaction_applied(self) -> None:
        """Collector redacts API keys inside message payloads."""
        collector = EventCollector(redaction_filter=RedactionFilter())
        event = collector.record_message(
            memory_session_id="sess-redact-001",
            role="user",
            content="my token is sk-abc123def456ghi789jkl012",
        )
        payload = json.loads(event.payload_json or "{}")
        assert "sk-abc123def456ghi789jkl012" not in payload["content"]
        assert "[REDACTED_API_KEY]" in payload["content"]
        assert event.redaction_level == RedactionLevel.partial


# ---------------------------------------------------------------------------
# TestObservationExtractor
# ---------------------------------------------------------------------------


class TestObservationExtractor:
    """Tests for the ObservationExtractor dialogue/summary extraction."""

    @staticmethod
    def _make_event(
        kind: EventKind,
        payload: dict[str, object],
        memory_session_id: str = "sess-obs-001",
    ) -> SessionEvent:
        """Helper to build a SessionEvent with a JSON payload."""
        return SessionEvent(
            memory_session_id=memory_session_id,
            timestamp=datetime.now(timezone.utc),
            kind=kind,
            title=payload.get("tool_name") or payload.get("role") or "event",
            payload_json=json.dumps(payload),
        )

    def test_extract_tool_use_as_change(self) -> None:
        """A tool_use event with file modifications is extracted as a change."""
        extractor = ObservationExtractor()
        event = self._make_event(
            EventKind.tool_use,
            {
                "tool_name": "write_file",
                "tool_input": "path=src/main.py",
                "tool_output": "File written successfully",
                "files_read": [],
                "files_modified": ["src/main.py"],
            },
        )
        summaries = extractor.extract_tool_summary([event])
        assert len(summaries) == 1
        assert summaries[0]["tool_name"] == "write_file"
        files_modified = summaries[0]["files_modified"]
        assert isinstance(files_modified, list)
        assert "src/main.py" in files_modified

    def test_extract_bugfix_keywords(self) -> None:
        """A message containing 'fix' is correctly converted to a dialogue."""
        extractor = ObservationExtractor()
        event = self._make_event(
            EventKind.message,
            {"role": "assistant", "content": "I will fix the null pointer bug."},
        )
        dialogues = extractor.events_to_dialogues([event])
        assert len(dialogues) == 1
        assert "fix" in dialogues[0].content.lower()

    def test_extract_feature_keywords(self) -> None:
        """A message containing 'implement' is correctly converted to a dialogue."""
        extractor = ObservationExtractor()
        event = self._make_event(
            EventKind.message,
            {
                "role": "assistant",
                "content": "Let me implement the new search feature.",
            },
        )
        dialogues = extractor.events_to_dialogues([event])
        assert len(dialogues) == 1
        assert "implement" in dialogues[0].content.lower()

    def test_extract_refactor_keywords(self) -> None:
        """A message containing 'refactor' is correctly converted to a dialogue."""
        extractor = ObservationExtractor()
        event = self._make_event(
            EventKind.message,
            {"role": "assistant", "content": "We should refactor the database layer."},
        )
        dialogues = extractor.events_to_dialogues([event])
        assert len(dialogues) == 1
        assert "refactor" in dialogues[0].content.lower()

    def test_extract_discovery_keywords(self) -> None:
        """A message containing 'found' is correctly converted to a dialogue."""
        extractor = ObservationExtractor()
        event = self._make_event(
            EventKind.message,
            {
                "role": "assistant",
                "content": "I found a memory leak in the cache module.",
            },
        )
        dialogues = extractor.events_to_dialogues([event])
        assert len(dialogues) == 1
        assert "found" in dialogues[0].content.lower()

    def test_extract_default_decision(self) -> None:
        """A neutral message without action keywords is converted to a dialogue."""
        extractor = ObservationExtractor()
        event = self._make_event(
            EventKind.message,
            {
                "role": "user",
                "content": "What do you think about the current approach?",
            },
        )
        dialogues = extractor.events_to_dialogues([event])
        assert len(dialogues) == 1
        assert dialogues[0].speaker == "user"
        assert "current approach" in dialogues[0].content

    def test_extract_empty_events(self) -> None:
        """An empty event list produces an empty dialogue list."""
        extractor = ObservationExtractor()
        dialogues = extractor.events_to_dialogues([])
        assert dialogues == []

    def test_extract_multiple_events(self) -> None:
        """Three events produce three dialogues with correct session context."""
        extractor = ObservationExtractor()
        session_id = "sess-multi-001"
        events = [
            self._make_event(
                EventKind.message,
                {"role": "user", "content": "Please add logging."},
                memory_session_id=session_id,
            ),
            self._make_event(
                EventKind.tool_use,
                {
                    "tool_name": "edit_file",
                    "tool_input": "add logging to app.py",
                    "tool_output": "Edited successfully",
                    "files_read": [],
                    "files_modified": ["app.py"],
                },
                memory_session_id=session_id,
            ),
            self._make_event(
                EventKind.file_change,
                {"filepath": "app.py", "change_type": "modified"},
                memory_session_id=session_id,
            ),
        ]
        dialogues = extractor.events_to_dialogues(events)
        assert len(dialogues) == 3
        # All source events belong to the same session
        assert all(e.memory_session_id == session_id for e in events)
        # Dialogue IDs are sequential starting at 1
        assert [d.dialogue_id for d in dialogues] == [1, 2, 3]
