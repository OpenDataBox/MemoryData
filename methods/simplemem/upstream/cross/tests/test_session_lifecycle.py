# pyright: reportMissingImports=false
"""Unit tests for SessionManager lifecycle operations."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure cross.* imports resolve from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cross.session_manager import SessionManager
from cross.storage_sqlite import SQLiteStorage
from cross.types import (
    CrossObservation,
    EventKind,
    FinalizationReport,
    ObservationType,
    SessionStatus,
)


# ---------------------------------------------------------------------------
# Stub EventCollector / ObservationExtractor compatible with SessionManager
# ---------------------------------------------------------------------------


class _StubCollectedEvent:
    """In-memory event representation matching the fallback stub API."""

    __slots__ = ("kind", "title", "payload", "timestamp")

    def __init__(
        self,
        kind: EventKind,
        title: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.kind = kind
        self.title = title
        self.payload = payload
        self.timestamp = datetime.now(timezone.utc)


class _StubEventCollector:
    """Stub collector exposing the ``add_event`` / ``flush`` API that
    ``SessionManager.record_event`` and ``finalize_session`` expect.

    The real ``cross.collectors.EventCollector`` has a different interface
    (``record_message``, ``record_tool_use``, etc.).  This stub matches
    the fallback class defined inside ``session_manager.py`` so the
    SessionManager code path exercises correctly.
    """

    def __init__(self, memory_session_id: str) -> None:
        self.memory_session_id = memory_session_id
        self._events: List[_StubCollectedEvent] = []

    def add_event(
        self,
        kind: EventKind,
        title: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> _StubCollectedEvent:
        event = _StubCollectedEvent(kind=kind, title=title, payload=payload)
        self._events.append(event)
        return event

    def flush(self) -> List[_StubCollectedEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    @property
    def event_count(self) -> int:
        return len(self._events)


class _StubObservationExtractor:
    """Stub matching the fallback ``ObservationExtractor`` interface used by
    ``SessionManager.finalize_session``.  The real ``cross.collectors``
    class has a different API (``events_to_dialogues``, etc.).
    """

    _KIND_TO_OBS_TYPE: Dict[str, ObservationType] = {
        "tool_use": ObservationType.change,
        "file_change": ObservationType.change,
        "message": ObservationType.discovery,
        "note": ObservationType.discovery,
        "system": ObservationType.discovery,
    }

    def extract_from_events(
        self,
        events: List[Any],
        memory_session_id: str,
    ) -> List[CrossObservation]:
        observations: List[CrossObservation] = []
        for event in events:
            title = getattr(event, "title", None)
            if not title:
                continue
            kind_value = (
                event.kind.value if hasattr(event.kind, "value") else str(event.kind)
            )
            obs_type = self._KIND_TO_OBS_TYPE.get(kind_value, ObservationType.discovery)
            observations.append(
                CrossObservation(
                    memory_session_id=memory_session_id,
                    timestamp=getattr(event, "timestamp", datetime.now(timezone.utc)),
                    type=obs_type,
                    title=title,
                )
            )
        return observations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TENANT = "test-tenant"
PROJECT = "test-project"
CONTENT_SID = "content-sess-lifecycle"
USER_PROMPT = "Implement feature X"

_COLLECTOR_PATCH = "cross.session_manager.EventCollector"
_EXTRACTOR_PATCH = "cross.session_manager.ObservationExtractor"


def _make_manager(
    tmp_path: Path,
    *,
    simplemem: object | None = None,
    db_name: str = "test.db",
) -> tuple[SessionManager, SQLiteStorage, MagicMock]:
    """Create a SessionManager backed by a real SQLiteStorage + mock vector store."""
    storage = SQLiteStorage(str(tmp_path / db_name))
    vector_store = MagicMock()
    vector_store.semantic_search.return_value = []
    vector_store.add_entries.return_value = None
    mgr = SessionManager(storage, vector_store, simplemem=simplemem)
    return mgr, storage, vector_store


# ---------------------------------------------------------------------------
# TestSessionManager
# ---------------------------------------------------------------------------


@patch(_EXTRACTOR_PATCH, _StubObservationExtractor)
@patch(_COLLECTOR_PATCH, _StubEventCollector)
class TestSessionManager:
    """Tests covering the full session lifecycle via SessionManager."""

    # -- start_session -----------------------------------------------------

    def test_start_session(self, tmp_path: Path) -> None:
        """start_session returns a SessionRecord with the expected fields."""
        mgr, _, _ = _make_manager(tmp_path)

        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id=CONTENT_SID,
            project=PROJECT,
            user_prompt=USER_PROMPT,
        )

        assert session is not None
        assert session.tenant_id == TENANT
        assert session.content_session_id == CONTENT_SID
        assert session.project == PROJECT
        assert session.user_prompt == USER_PROMPT
        assert session.status == SessionStatus.active
        assert session.memory_session_id  # non-empty UUID string

    def test_start_session_creates_collector(self, tmp_path: Path) -> None:
        """After start_session the internal _collectors dict has an entry."""
        mgr, _, _ = _make_manager(tmp_path)

        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id=CONTENT_SID,
            project=PROJECT,
        )

        assert session.memory_session_id in mgr._collectors

    # -- record_message ----------------------------------------------------

    def test_record_message(self, tmp_path: Path) -> None:
        """record_message persists a message event to SQLite."""
        mgr, storage, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-msg",
            project=PROJECT,
        )
        mid = session.memory_session_id

        event_id = mgr.record_message(mid, content="Hello world", role="user")

        assert isinstance(event_id, int)
        assert event_id > 0
        events = storage.get_events_for_session(mid)
        assert len(events) == 1
        assert events[0].kind == EventKind.message
        assert events[0].title == "user message"

    # -- record_tool_use ---------------------------------------------------

    def test_record_tool_use(self, tmp_path: Path) -> None:
        """record_tool_use persists a tool_use event to SQLite."""
        mgr, storage, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-tool",
            project=PROJECT,
        )
        mid = session.memory_session_id

        event_id = mgr.record_tool_use(
            mid,
            tool_name="grep",
            tool_input="pattern",
            tool_output="match found",
        )

        assert isinstance(event_id, int)
        assert event_id > 0
        events = storage.get_events_for_session(mid)
        assert len(events) == 1
        assert events[0].kind == EventKind.tool_use
        assert events[0].title == "tool:grep"

    # -- record_event (generic) --------------------------------------------

    def test_record_event_generic(self, tmp_path: Path) -> None:
        """record_event with arbitrary kind/title/payload stores correctly."""
        mgr, storage, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-generic",
            project=PROJECT,
        )
        mid = session.memory_session_id

        event_id = mgr.record_event(
            mid,
            kind=EventKind.note,
            title="design decision",
            payload={"reason": "simplicity"},
        )

        assert isinstance(event_id, int)
        events = storage.get_events_for_session(mid)
        assert len(events) == 1
        assert events[0].kind == EventKind.note
        assert events[0].title == "design decision"

    # -- finalize_session --------------------------------------------------

    def test_finalize_session(self, tmp_path: Path) -> None:
        """Finalize after recording 3 events returns a valid FinalizationReport."""
        mgr, _, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-finalize",
            project=PROJECT,
            user_prompt=USER_PROMPT,
        )
        mid = session.memory_session_id

        mgr.record_message(mid, content="Hello", role="user")
        mgr.record_tool_use(mid, "read_file", "main.py", "contents...")
        mgr.record_event(mid, kind=EventKind.note, title="refactor plan")

        report = mgr.finalize_session(mid)

        assert isinstance(report, FinalizationReport)
        assert report.memory_session_id == mid
        assert report.observations_count >= 0
        assert isinstance(report.summary_generated, bool)
        assert isinstance(report.entries_stored, int)
        assert isinstance(report.consolidation_triggered, bool)

    def test_finalize_extracts_observations(self, tmp_path: Path) -> None:
        """Finalization extracts observations and stores them in SQLite."""
        mgr, storage, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-obs",
            project=PROJECT,
            user_prompt="fix bug",
        )
        mid = session.memory_session_id

        mgr.record_message(mid, content="investigating issue", role="user")
        mgr.record_tool_use(mid, "grep", "error", "found error in line 42")

        report = mgr.finalize_session(mid)

        # Observations should have been stored
        observations = storage.get_observations_for_session(mid)
        assert report.observations_count == len(observations)
        assert report.observations_count > 0

    def test_finalize_generates_summary(self, tmp_path: Path) -> None:
        """Finalization generates and stores a summary in SQLite."""
        mgr, storage, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-summary",
            project=PROJECT,
            user_prompt="add logging",
        )
        mid = session.memory_session_id

        mgr.record_message(mid, content="adding log statements", role="assistant")

        report = mgr.finalize_session(mid)

        assert report.summary_generated is True
        summary = storage.get_summary_for_session(mid)
        assert summary is not None
        assert summary.request is not None
        assert summary.completed is not None

    def test_finalize_unknown_session(self, tmp_path: Path) -> None:
        """Finalizing a non-existent session returns an empty report."""
        mgr, _, _ = _make_manager(tmp_path)

        report = mgr.finalize_session("non-existent-session-id")

        assert isinstance(report, FinalizationReport)
        assert report.memory_session_id == "non-existent-session-id"
        assert report.observations_count == 0
        assert report.summary_generated is False
        assert report.entries_stored == 0

    # -- end_session -------------------------------------------------------

    def test_end_session(self, tmp_path: Path) -> None:
        """end_session with completed status updates the session record."""
        mgr, storage, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-end",
            project=PROJECT,
        )
        mid = session.memory_session_id

        mgr.end_session(mid, status=SessionStatus.completed)

        updated = storage.get_session_by_memory_id(mid)
        assert updated is not None
        assert updated.status == SessionStatus.completed
        assert updated.ended_at is not None

    def test_end_session_failed(self, tmp_path: Path) -> None:
        """end_session with failed status marks the session as failed."""
        mgr, storage, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-fail",
            project=PROJECT,
        )
        mid = session.memory_session_id

        mgr.end_session(mid, status=SessionStatus.failed)

        updated = storage.get_session_by_memory_id(mid)
        assert updated is not None
        assert updated.status == SessionStatus.failed

    # -- query helpers -----------------------------------------------------

    def test_get_session(self, tmp_path: Path) -> None:
        """get_session round-trips the session record via memory_session_id."""
        mgr, _, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-get",
            project=PROJECT,
            user_prompt="query test",
        )
        mid = session.memory_session_id

        retrieved = mgr.get_session(mid)

        assert retrieved is not None
        assert retrieved.memory_session_id == mid
        assert retrieved.tenant_id == TENANT
        assert retrieved.project == PROJECT
        assert retrieved.user_prompt == "query test"

    def test_get_events(self, tmp_path: Path) -> None:
        """get_events returns the correct list after recording events."""
        mgr, _, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-events",
            project=PROJECT,
        )
        mid = session.memory_session_id

        mgr.record_message(mid, content="msg1", role="user")
        mgr.record_message(mid, content="msg2", role="assistant")
        mgr.record_tool_use(mid, "ls", ".", "file1 file2")

        events = mgr.get_events(mid)

        assert len(events) == 3
        kinds = [e.kind for e in events]
        assert EventKind.message in kinds
        assert EventKind.tool_use in kinds

    def test_get_observations(self, tmp_path: Path) -> None:
        """get_observations returns observations stored during finalization."""
        mgr, _, _ = _make_manager(tmp_path)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-get-obs",
            project=PROJECT,
        )
        mid = session.memory_session_id

        mgr.record_message(mid, content="discovered a bug", role="user")
        mgr.record_tool_use(mid, "grep", "error", "found it")
        mgr.finalize_session(mid)

        observations = mgr.get_observations(mid)

        assert isinstance(observations, list)
        assert len(observations) > 0
        # Each observation should have required fields
        for obs in observations:
            assert obs.memory_session_id == mid
            assert obs.title

    # -- SimpleMem integration ---------------------------------------------

    def test_finalize_with_simplemem(self, tmp_path: Path) -> None:
        """When simplemem is provided, finalization calls add_dialogues + finalize."""
        mock_sm = MagicMock()
        mock_sm.add_dialogues.return_value = None
        mock_sm.finalize.return_value = []

        mgr, _, _ = _make_manager(tmp_path, simplemem=mock_sm)
        session = mgr.start_session(
            tenant_id=TENANT,
            content_session_id="cs-simplemem",
            project=PROJECT,
            user_prompt="test simplemem integration",
        )
        mid = session.memory_session_id

        mgr.record_message(mid, content="Hello from user", role="user")
        mgr.record_message(mid, content="Hello from assistant", role="assistant")

        report = mgr.finalize_session(mid)

        assert isinstance(report, FinalizationReport)
        mock_sm.add_dialogues.assert_called_once()
        mock_sm.finalize.assert_called_once()

        dialogues_arg = mock_sm.add_dialogues.call_args[0][0]
        assert len(dialogues_arg) >= 2
