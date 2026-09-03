# pyright: reportMissingImports=false
"""Unit tests for SQLiteStorage — cross-session memory persistence layer.

Each test creates its own SQLiteStorage backed by a temporary SQLite database
(via pytest ``tmp_path``).  No mocking — these are real integration tests for
the storage layer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cross.storage_sqlite import SQLiteStorage
from cross.types import EventKind, ObservationType, SessionStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage(tmp_path: Path) -> SQLiteStorage:
    """Create a fresh SQLiteStorage backed by a temp file."""
    return SQLiteStorage(db_path=str(tmp_path / "test.db"))


def _create_session(
    storage: SQLiteStorage,
    content_id: str = "cs-1",
    project: str = "proj",
    tenant_id: str = "default",
    user_prompt: str | None = "do something",
):
    """Shortcut to create a session with sensible defaults."""
    return storage.create_session(
        tenant_id=tenant_id,
        content_session_id=content_id,
        project=project,
        user_prompt=user_prompt,
    )


# ---------------------------------------------------------------------------
# TestSQLiteStorage
# ---------------------------------------------------------------------------


class TestSQLiteStorage:
    """Integration tests for SQLiteStorage using real temporary SQLite databases."""

    # -- Sessions ----------------------------------------------------------

    def test_create_session(self, tmp_path: Path) -> None:
        """Create a session and verify it is retrievable."""
        storage = _make_storage(tmp_path)
        session = _create_session(storage)

        assert session is not None
        assert session.content_session_id == "cs-1"
        assert session.project == "proj"
        assert session.status == SessionStatus.active
        assert session.memory_session_id  # non-empty UUID string
        assert session.id is not None and session.id > 0

        # Retrieve by content_session_id
        fetched = storage.get_session_by_content_id("cs-1")
        assert fetched is not None
        assert fetched.memory_session_id == session.memory_session_id
        assert fetched.id == session.id

        # Retrieve by memory_session_id
        fetched2 = storage.get_session_by_memory_id(session.memory_session_id)
        assert fetched2 is not None
        assert fetched2.content_session_id == "cs-1"

    def test_create_session_idempotent(self, tmp_path: Path) -> None:
        """Creating the same content_session_id twice returns the same record."""
        storage = _make_storage(tmp_path)
        s1 = _create_session(storage, content_id="dup-1")
        s2 = _create_session(storage, content_id="dup-1")

        # INSERT OR IGNORE means the second call is a no-op insert; both
        # return the row looked up by content_session_id.
        assert s1.memory_session_id == s2.memory_session_id
        assert s1.id == s2.id

    def test_update_session_status(self, tmp_path: Path) -> None:
        """Start active, update to completed, verify ended_at is set."""
        storage = _make_storage(tmp_path)
        session = _create_session(storage)
        assert session.status == SessionStatus.active

        storage.update_session_status(
            session.memory_session_id, SessionStatus.completed
        )

        updated = storage.get_session_by_memory_id(session.memory_session_id)
        assert updated is not None
        assert updated.status == SessionStatus.completed
        assert updated.ended_at is not None

    # -- Events ------------------------------------------------------------

    def test_add_event(self, tmp_path: Path) -> None:
        """Add events and verify they are retrievable in chronological order."""
        storage = _make_storage(tmp_path)
        session = _create_session(storage)
        mid = session.memory_session_id

        eid1 = storage.add_event(mid, EventKind.message, title="hello")
        time.sleep(0.02)
        eid2 = storage.add_event(mid, EventKind.tool_use, title="run lint")
        time.sleep(0.02)
        eid3 = storage.add_event(
            mid,
            EventKind.file_change,
            title="edit main.py",
            payload_json={"file": "main.py", "action": "edit"},
        )

        assert eid1 > 0
        assert eid2 > eid1
        assert eid3 > eid2

        events = storage.get_events_for_session(mid)
        assert len(events) == 3
        assert events[0].title == "hello"
        assert events[0].kind == EventKind.message
        assert events[1].title == "run lint"
        assert events[1].kind == EventKind.tool_use
        assert events[2].title == "edit main.py"
        assert events[2].kind == EventKind.file_change

    def test_get_events_for_session(self, tmp_path: Path) -> None:
        """Add events to two sessions and verify isolation."""
        storage = _make_storage(tmp_path)
        s1 = _create_session(storage, content_id="iso-1")
        s2 = _create_session(storage, content_id="iso-2")

        storage.add_event(s1.memory_session_id, EventKind.note, title="s1-ev")
        storage.add_event(s2.memory_session_id, EventKind.note, title="s2-ev-a")
        storage.add_event(s2.memory_session_id, EventKind.system, title="s2-ev-b")

        ev1 = storage.get_events_for_session(s1.memory_session_id)
        ev2 = storage.get_events_for_session(s2.memory_session_id)

        assert len(ev1) == 1
        assert ev1[0].title == "s1-ev"

        assert len(ev2) == 2
        titles = {e.title for e in ev2}
        assert titles == {"s2-ev-a", "s2-ev-b"}

    # -- Observations ------------------------------------------------------

    def test_store_observation(self, tmp_path: Path) -> None:
        """Store an observation and retrieve it."""
        storage = _make_storage(tmp_path)
        session = _create_session(storage)
        mid = session.memory_session_id

        obs_id = storage.store_observation(
            memory_session_id=mid,
            type=ObservationType.bugfix,
            title="Fixed null pointer",
            subtitle="in parser module",
            narrative="Discovered a null check was missing",
        )
        assert obs_id > 0

        observations = storage.get_observations_for_session(mid)
        assert len(observations) == 1

        obs = observations[0]
        assert obs.obs_id == obs_id
        assert obs.title == "Fixed null pointer"
        assert obs.subtitle == "in parser module"
        assert obs.type == ObservationType.bugfix
        assert obs.narrative == "Discovered a null check was missing"
        assert obs.memory_session_id == mid

    def test_get_observations_for_session(self, tmp_path: Path) -> None:
        """Verify session-scoped observation retrieval."""
        storage = _make_storage(tmp_path)
        s1 = _create_session(storage, content_id="obs-s1")
        s2 = _create_session(storage, content_id="obs-s2")

        storage.store_observation(
            s1.memory_session_id, ObservationType.feature, "feat A"
        )
        storage.store_observation(
            s2.memory_session_id, ObservationType.decision, "decision B"
        )
        storage.store_observation(
            s2.memory_session_id, ObservationType.refactor, "refactor C"
        )

        obs_s1 = storage.get_observations_for_session(s1.memory_session_id)
        obs_s2 = storage.get_observations_for_session(s2.memory_session_id)

        assert len(obs_s1) == 1
        assert obs_s1[0].title == "feat A"

        assert len(obs_s2) == 2
        titles = {o.title for o in obs_s2}
        assert titles == {"decision B", "refactor C"}

    def test_get_recent_observations(self, tmp_path: Path) -> None:
        """Store observations across sessions and verify project filtering."""
        storage = _make_storage(tmp_path)

        s1 = _create_session(storage, content_id="ro-1", project="myproj")
        s2 = _create_session(storage, content_id="ro-2", project="myproj")
        s3 = _create_session(storage, content_id="ro-3", project="other")

        storage.store_observation(s1.memory_session_id, ObservationType.bugfix, "fix 1")
        time.sleep(0.02)
        storage.store_observation(
            s2.memory_session_id, ObservationType.feature, "feat 2"
        )
        time.sleep(0.02)
        storage.store_observation(
            s3.memory_session_id, ObservationType.discovery, "disc 3"
        )

        # Only myproj observations
        recent = storage.get_recent_observations("myproj", limit=10)
        assert len(recent) == 2
        titles = {o.title for o in recent}
        assert titles == {"fix 1", "feat 2"}

        # Filter by type
        bugfixes = storage.get_recent_observations(
            "myproj", limit=10, types=[ObservationType.bugfix]
        )
        assert len(bugfixes) == 1
        assert bugfixes[0].title == "fix 1"

        # Other project
        other_obs = storage.get_recent_observations("other", limit=10)
        assert len(other_obs) == 1
        assert other_obs[0].title == "disc 3"

    # -- Summaries ---------------------------------------------------------

    def test_store_summary(self, tmp_path: Path) -> None:
        """Store a summary and verify all fields round-trip."""
        storage = _make_storage(tmp_path)
        session = _create_session(storage)
        mid = session.memory_session_id

        sid = storage.store_summary(
            memory_session_id=mid,
            request="build login page",
            investigated="auth flows",
            learned="OAuth2 best practices",
            completed="login form with validation",
            next_steps="add MFA support",
        )
        assert sid > 0

        summary = storage.get_summary_for_session(mid)
        assert summary is not None
        assert summary.summary_id == sid
        assert summary.memory_session_id == mid
        assert summary.request == "build login page"
        assert summary.investigated == "auth flows"
        assert summary.learned == "OAuth2 best practices"
        assert summary.completed == "login form with validation"
        assert summary.next_steps == "add MFA support"
        assert summary.timestamp is not None

    def test_get_recent_summaries(self, tmp_path: Path) -> None:
        """Store 3 summaries, verify limit and DESC ordering."""
        storage = _make_storage(tmp_path)

        # Create 3 sessions in "alpha" project with summaries
        for i in range(3):
            s = _create_session(storage, content_id=f"sum-{i}", project="alpha")
            time.sleep(0.02)
            storage.store_summary(
                memory_session_id=s.memory_session_id,
                request=f"task-{i}",
                completed=f"done-{i}",
            )
            time.sleep(0.02)

        # One session in a different project
        other = _create_session(storage, content_id="sum-other", project="beta")
        storage.store_summary(
            memory_session_id=other.memory_session_id,
            request="other-task",
        )

        # Limit to 2 — most recent first
        recent = storage.get_recent_summaries("alpha", limit=2)
        assert len(recent) == 2
        assert recent[0].request == "task-2"
        assert recent[1].request == "task-1"

        # All alpha summaries
        all_alpha = storage.get_recent_summaries("alpha", limit=10)
        assert len(all_alpha) == 3

        # Beta project
        beta = storage.get_recent_summaries("beta", limit=10)
        assert len(beta) == 1
        assert beta[0].request == "other-task"

    # -- Consolidation -----------------------------------------------------

    def test_record_consolidation_run(self, tmp_path: Path) -> None:
        """Record a consolidation run and verify it persists."""
        storage = _make_storage(tmp_path)

        run_id = storage.record_consolidation_run(
            tenant_id="default",
            policy_json={"max_age_days": 30},
            stats_json={"merged": 5, "pruned": 2},
        )
        assert run_id > 0

        runs = storage.get_recent_consolidation_runs("default", limit=5)
        assert len(runs) == 1
        assert runs[0].run_id == run_id
        assert runs[0].tenant_id == "default"
        assert runs[0].timestamp is not None

        # Second run
        time.sleep(0.02)
        run_id2 = storage.record_consolidation_run(
            tenant_id="default",
            stats_json={"merged": 3},
        )
        assert run_id2 > run_id

        runs2 = storage.get_recent_consolidation_runs("default", limit=5)
        assert len(runs2) == 2
        # Most recent first
        assert runs2[0].run_id == run_id2

    # -- Stats -------------------------------------------------------------

    def test_get_stats(self, tmp_path: Path) -> None:
        """Create sessions/events/observations/summaries and verify stat counts."""
        storage = _make_storage(tmp_path)

        s1 = _create_session(storage, content_id="st-1", project="p")
        s2 = _create_session(storage, content_id="st-2", project="p")

        storage.add_event(s1.memory_session_id, EventKind.message, title="msg")
        storage.add_event(s1.memory_session_id, EventKind.note, title="note")
        storage.add_event(s2.memory_session_id, EventKind.system, title="sys")

        storage.store_observation(s1.memory_session_id, ObservationType.bugfix, "obs1")

        storage.store_summary(memory_session_id=s1.memory_session_id, request="r1")
        storage.store_summary(memory_session_id=s2.memory_session_id, request="r2")

        stats = storage.get_stats()
        assert stats["sessions"] == 2
        assert stats["events"] == 3
        assert stats["observations"] == 1
        assert stats["summaries"] == 2

        # Filter by tenant
        stats_tenant = storage.get_stats(tenant_id="default")
        assert stats_tenant["sessions"] == 2

        # Filter by project
        stats_proj = storage.get_stats(project="p")
        assert stats_proj["sessions"] == 2

        # Non-existent project yields zeroes
        stats_empty = storage.get_stats(project="nonexistent")
        assert stats_empty["sessions"] == 0
        assert stats_empty["events"] == 0
        assert stats_empty["observations"] == 0
        assert stats_empty["summaries"] == 0

    # -- WAL mode ----------------------------------------------------------

    def test_wal_mode(self, tmp_path: Path) -> None:
        """Verify WAL journal mode is enabled on the connection."""
        storage = _make_storage(tmp_path)
        cursor = storage.conn.execute("PRAGMA journal_mode")
        row = cursor.fetchone()
        assert row is not None
        mode = row[0]
        assert mode.lower() == "wal"

    # -- Close & reopen ----------------------------------------------------

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        """Close DB, reopen at the same path, and verify data persists."""
        db_path = str(tmp_path / "reopen.db")

        # --- Phase 1: write data ---
        storage = SQLiteStorage(db_path=db_path)
        session = storage.create_session(
            tenant_id="default",
            content_session_id="persist-1",
            project="proj",
        )
        mid = session.memory_session_id

        storage.add_event(mid, EventKind.message, title="persisted event")
        storage.store_observation(
            mid, ObservationType.discovery, "persisted observation"
        )
        storage.store_summary(
            memory_session_id=mid,
            request="persist test",
            completed="verified persistence",
        )
        storage.close()

        # --- Phase 2: reopen and verify ---
        storage2 = SQLiteStorage(db_path=db_path)

        fetched = storage2.get_session_by_content_id("persist-1")
        assert fetched is not None
        assert fetched.memory_session_id == mid

        events = storage2.get_events_for_session(mid)
        assert len(events) == 1
        assert events[0].title == "persisted event"

        observations = storage2.get_observations_for_session(mid)
        assert len(observations) == 1
        assert observations[0].title == "persisted observation"

        summary = storage2.get_summary_for_session(mid)
        assert summary is not None
        assert summary.request == "persist test"
        assert summary.completed == "verified persistence"

        storage2.close()
