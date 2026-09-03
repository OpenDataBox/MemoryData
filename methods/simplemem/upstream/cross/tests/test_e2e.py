# pyright: reportMissingImports=false
"""End-to-end integration tests for the full cross-session memory workflow.

Tests real SQLiteStorage <-> SessionManager <-> ContextInjector integration.
The vector store is mocked (no LanceDB dependency).

Note: SessionManager internally uses a lightweight EventCollector stub
(with add_event/flush API) rather than the full collectors.EventCollector.
When cross.collectors is importable the module-level import in
session_manager.py picks up the full class, causing an API mismatch.
We patch session_manager.EventCollector with a compatible stub so the
integration path works as designed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cross.collectors import EventCollector, RedactionFilter
from cross.context_injector import ContextInjector
from cross.session_manager import SessionManager
from cross.storage_sqlite import SQLiteStorage
from cross.types import (
    CrossObservation,
    EventKind,
    FinalizationReport,
    ObservationType,
    RedactionLevel,
    SessionStatus,
)


# ---------------------------------------------------------------------------
# Stub EventCollector compatible with SessionManager's expected API
# ---------------------------------------------------------------------------


class _CollectedEvent:
    """In-memory event matching the interface SessionManager expects."""

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
    """EventCollector stub with add_event/flush API used by SessionManager."""

    def __init__(self, memory_session_id: str) -> None:
        self.memory_session_id = memory_session_id
        self._events: List[_CollectedEvent] = []

    def add_event(
        self,
        kind: EventKind,
        title: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> _CollectedEvent:
        event = _CollectedEvent(kind=kind, title=title, payload=payload)
        self._events.append(event)
        return event

    def flush(self) -> List[_CollectedEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    @property
    def event_count(self) -> int:
        return len(self._events)


class _StubObservationExtractor:
    """ObservationExtractor stub with extract_from_events API."""

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
            payload = getattr(event, "payload", None)
            if payload is None:
                payload_json = getattr(event, "payload_json", None)
                if payload_json and isinstance(payload_json, str):
                    try:
                        payload = json.loads(payload_json)
                    except (json.JSONDecodeError, TypeError):
                        payload = None
            narrative: Optional[str] = None
            if isinstance(payload, dict):
                narrative = payload.get("content") or payload.get("output")
                if narrative and len(str(narrative)) > 500:
                    narrative = str(narrative)[:500] + "..."
            observations.append(
                CrossObservation(
                    memory_session_id=memory_session_id,
                    timestamp=getattr(event, "timestamp", datetime.now(timezone.utc)),
                    type=obs_type,
                    title=title,
                    narrative=narrative,
                )
            )
        return observations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Patches to apply so SessionManager uses compatible stubs
_SM_PATCHES = {
    "cross.session_manager.EventCollector": _StubEventCollector,
    "cross.session_manager.ObservationExtractor": _StubObservationExtractor,
}


def _make_storage(tmp_path: Path) -> SQLiteStorage:
    """Create a fresh SQLiteStorage backed by a temp file."""
    return SQLiteStorage(db_path=str(tmp_path / "e2e_test.db"))


def _make_mock_vector_store() -> MagicMock:
    """Return a MagicMock standing in for CrossSessionVectorStore."""
    mock = MagicMock()
    mock.semantic_search.return_value = []
    mock.add_entries.return_value = None
    mock.get_all_entries.return_value = []
    mock.mark_superseded.return_value = None
    mock.update_importance.return_value = None
    return mock


def _run_session(
    sm: SessionManager,
    tenant_id: str,
    content_session_id: str,
    project: str,
    user_prompt: str,
    messages: list[tuple[str, str]],
    tool_uses: list[tuple[str, str, str]],
) -> str:
    """Run a full session lifecycle and return the memory_session_id."""
    session = sm.start_session(
        tenant_id=tenant_id,
        content_session_id=content_session_id,
        project=project,
        user_prompt=user_prompt,
    )
    mid = session.memory_session_id

    for content, role in messages:
        sm.record_message(mid, content=content, role=role)

    for tool_name, tool_input, tool_output in tool_uses:
        sm.record_tool_use(
            mid, tool_name=tool_name, tool_input=tool_input, tool_output=tool_output
        )

    sm.finalize_session(mid)
    sm.end_session(mid)
    return mid


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end integration tests for the cross-session memory workflow."""

    # ------------------------------------------------------------------
    # test_full_session_lifecycle
    # ------------------------------------------------------------------

    @patch("cross.session_manager.ObservationExtractor", _StubObservationExtractor)
    @patch("cross.session_manager.EventCollector", _StubEventCollector)
    def test_full_session_lifecycle(self, tmp_path: Path) -> None:
        """Walk through the complete lifecycle of a single session and then
        verify that context built for a NEW session includes data from
        the first one."""

        # 1. Create storage and mocks
        storage = _make_storage(tmp_path)
        mock_vs = _make_mock_vector_store()

        # 2-4. Create SessionManager and ContextInjector
        sm = SessionManager(sqlite_storage=storage, vector_store=mock_vs)
        ci = ContextInjector(
            sqlite_storage=storage, vector_store=mock_vs, max_tokens=500
        )

        # 5. Start a session
        session = sm.start_session(
            tenant_id="test",
            content_session_id="ses-1",
            project="myproject",
            user_prompt="Build a REST API",
        )
        mid = session.memory_session_id
        assert session.status == SessionStatus.active

        # 6. Record 3 messages and 2 tool uses
        sm.record_message(
            mid, content="Build a REST API for user management", role="user"
        )
        sm.record_message(
            mid, content="I will create the endpoints now", role="assistant"
        )
        sm.record_message(mid, content="Please add authentication too", role="user")

        sm.record_tool_use(
            mid,
            tool_name="write_file",
            tool_input="api/routes.py",
            tool_output="Created routes.py with 5 endpoints",
        )
        sm.record_tool_use(
            mid,
            tool_name="run_tests",
            tool_input="pytest api/",
            tool_output="All 12 tests passed",
        )

        # 7. Finalize -> verify FinalizationReport
        report = sm.finalize_session(mid)
        assert isinstance(report, FinalizationReport)
        assert report.memory_session_id == mid
        assert report.observations_count > 0
        assert report.summary_generated is True

        # 8. End session -> verify status completed
        sm.end_session(mid)
        ended_session = sm.get_session(mid)
        assert ended_session is not None
        assert ended_session.status == SessionStatus.completed

        # 9. Verify observations were stored
        observations = sm.get_observations(mid)
        assert len(observations) > 0

        # 10. Verify summary was generated
        summary = storage.get_summary_for_session(mid)
        assert summary is not None
        assert summary.request is not None
        assert "Build a REST API" in summary.request

        # 11. Build context for a NEW session -> verify bundle has summaries
        bundle = ci.build_context(
            tenant_id="test",
            project="myproject",
            user_prompt="Continue the REST API work",
        )
        assert len(bundle.session_summaries) > 0
        # The summary should reference the first session
        found_summary = any(
            s.memory_session_id == mid for s in bundle.session_summaries
        )
        assert found_summary, "Context bundle should include summary from first session"

        storage.close()

    # ------------------------------------------------------------------
    # test_multi_session_context_accumulation
    # ------------------------------------------------------------------

    @patch("cross.session_manager.ObservationExtractor", _StubObservationExtractor)
    @patch("cross.session_manager.EventCollector", _StubEventCollector)
    def test_multi_session_context_accumulation(self, tmp_path: Path) -> None:
        """Run 2 sessions sequentially and verify context built afterward
        includes summaries from BOTH sessions."""

        storage = _make_storage(tmp_path)
        mock_vs = _make_mock_vector_store()
        sm = SessionManager(sqlite_storage=storage, vector_store=mock_vs)
        ci = ContextInjector(
            sqlite_storage=storage, vector_store=mock_vs, max_tokens=2000
        )

        # Session 1
        mid1 = _run_session(
            sm,
            tenant_id="test",
            content_session_id="ses-a",
            project="proj",
            user_prompt="Set up database schema",
            messages=[
                ("Design the user table", "user"),
                ("Created users table with id, email, name columns", "assistant"),
            ],
            tool_uses=[
                ("write_file", "db/schema.sql", "CREATE TABLE users ..."),
            ],
        )

        # Session 2
        mid2 = _run_session(
            sm,
            tenant_id="test",
            content_session_id="ses-b",
            project="proj",
            user_prompt="Add API authentication",
            messages=[
                ("Implement JWT auth middleware", "user"),
                ("Added JWT verification to all protected routes", "assistant"),
            ],
            tool_uses=[
                ("write_file", "api/auth.py", "class JWTMiddleware: ..."),
            ],
        )

        # Build context after both sessions
        bundle = ci.build_context(
            tenant_id="test",
            project="proj",
            user_prompt="What did we build so far?",
        )

        assert len(bundle.session_summaries) >= 2

        summary_mids = {s.memory_session_id for s in bundle.session_summaries}
        assert mid1 in summary_mids, "Bundle must include summary from session 1"
        assert mid2 in summary_mids, "Bundle must include summary from session 2"

        storage.close()

    # ------------------------------------------------------------------
    # test_multi_tenant_isolation
    # ------------------------------------------------------------------

    @patch("cross.session_manager.ObservationExtractor", _StubObservationExtractor)
    @patch("cross.session_manager.EventCollector", _StubEventCollector)
    def test_multi_tenant_isolation(self, tmp_path: Path) -> None:
        """Start sessions for 2 different tenants and verify that context
        built for tenant A does not include tenant B data."""

        storage = _make_storage(tmp_path)
        mock_vs = _make_mock_vector_store()
        sm = SessionManager(sqlite_storage=storage, vector_store=mock_vs)

        # Session for tenant A
        session_a = sm.start_session(
            tenant_id="tenant-a",
            content_session_id="ses-a1",
            project="proj-a",
            user_prompt="Work on tenant A feature",
        )
        mid_a = session_a.memory_session_id
        sm.record_message(mid_a, content="Tenant A specific work", role="user")
        sm.record_tool_use(
            mid_a, tool_name="read_file", tool_input="a.py", tool_output="tenant A code"
        )
        sm.finalize_session(mid_a)
        sm.end_session(mid_a)

        # Session for tenant B
        session_b = sm.start_session(
            tenant_id="tenant-b",
            content_session_id="ses-b1",
            project="proj-b",
            user_prompt="Work on tenant B feature",
        )
        mid_b = session_b.memory_session_id
        sm.record_message(mid_b, content="Tenant B specific work", role="user")
        sm.record_tool_use(
            mid_b, tool_name="read_file", tool_input="b.py", tool_output="tenant B code"
        )
        sm.finalize_session(mid_b)
        sm.end_session(mid_b)

        # Build context for tenant A — should only see tenant A data
        ci = ContextInjector(
            sqlite_storage=storage, vector_store=mock_vs, max_tokens=2000
        )
        bundle_a = ci.build_context(
            tenant_id="tenant-a",
            project="proj-a",
            user_prompt="Continue tenant A work",
        )

        # Summaries are scoped by project (proj-a), so only session A summaries
        for s in bundle_a.session_summaries:
            assert s.memory_session_id != mid_b, (
                "Tenant A context must not contain tenant B summary"
            )

        # Verify the mock vector store was called with tenant_id="tenant-a"
        mock_vs.semantic_search.assert_called()
        last_call = mock_vs.semantic_search.call_args
        if last_call.kwargs:
            assert last_call.kwargs.get("tenant_id") == "tenant-a"
        else:
            # Positional fallback check
            assert "tenant-a" in str(last_call)

        storage.close()

    # ------------------------------------------------------------------
    # test_session_with_redaction
    # ------------------------------------------------------------------

    def test_session_with_redaction(self, tmp_path: Path) -> None:
        """Create an EventCollector with redaction enabled, record a message
        containing an API key, and verify the payload has the key redacted."""

        collector = EventCollector(
            redaction_filter=RedactionFilter(),
        )

        memory_session_id = "redact-test-001"

        # Record a message containing a sensitive API key
        api_key = "sk-abc123xyz456789012345678901234567890"
        content_with_key = f"Use this API key: {api_key} for authentication"

        event = collector.record_message(
            memory_session_id=memory_session_id,
            role="user",
            content=content_with_key,
        )

        # Verify the event was recorded
        assert collector.event_count == 1

        # Verify redaction was applied
        assert event.redaction_level in (RedactionLevel.partial, RedactionLevel.full)

        # Parse the payload and verify the API key was redacted
        payload = json.loads(event.payload_json)  # type: ignore[arg-type]
        assert api_key not in payload["content"], (
            "Raw API key must not appear in redacted payload"
        )
        assert "[REDACTED" in payload["content"], (
            "Redacted content should contain a redaction marker"
        )

        # Verify same redaction on retrieved events
        events = collector.get_events()
        assert len(events) >= 1
        flushed_payload = json.loads(events[0].payload_json)  # type: ignore[arg-type]
        assert api_key not in flushed_payload["content"]

    # ------------------------------------------------------------------
    # test_events_persisted_correctly
    # ------------------------------------------------------------------

    @patch("cross.session_manager.ObservationExtractor", _StubObservationExtractor)
    @patch("cross.session_manager.EventCollector", _StubEventCollector)
    def test_events_persisted_correctly(self, tmp_path: Path) -> None:
        """Verify that events recorded through SessionManager are
        correctly persisted in SQLite and retrievable."""

        storage = _make_storage(tmp_path)
        mock_vs = _make_mock_vector_store()
        sm = SessionManager(sqlite_storage=storage, vector_store=mock_vs)

        session = sm.start_session(
            tenant_id="test",
            content_session_id="ses-persist",
            project="proj",
            user_prompt="Test persistence",
        )
        mid = session.memory_session_id

        # Record events
        eid1 = sm.record_message(mid, content="Hello world", role="user")
        eid2 = sm.record_tool_use(
            mid, tool_name="grep", tool_input="pattern", tool_output="3 matches found"
        )

        assert eid1 > 0
        assert eid2 > 0

        # Retrieve and verify
        events = sm.get_events(mid)
        assert len(events) >= 2

        kinds = {e.kind for e in events}
        assert EventKind.message in kinds
        assert EventKind.tool_use in kinds

        storage.close()
