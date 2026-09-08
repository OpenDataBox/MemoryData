# pyright: reportMissingImports=false
"""Shared pytest fixtures for cross-session memory tests."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure cross.* imports resolve from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cross.types import (
    CrossMemoryEntry,
    CrossObservation,
    EventKind,
    ObservationType,
    RedactionLevel,
    SessionEvent,
    SessionRecord,
    SessionStatus,
    SessionSummary,
)


@pytest.fixture
def tmp_sqlite_path(tmp_path: Path) -> Path:
    """Return a path for a temporary SQLite database file."""
    return tmp_path / "test_cross_memory.db"


@pytest.fixture
def sqlite_storage(tmp_sqlite_path: Path):
    """Create a real SQLiteStorage instance backed by a temporary database."""
    from cross.storage_sqlite import SQLiteStorage

    storage = SQLiteStorage(db_path=str(tmp_sqlite_path))
    yield storage
    storage.close()


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Return a MagicMock spec'd to CrossSessionVectorStore with safe defaults."""
    mock = MagicMock()
    mock.semantic_search.return_value = []
    mock.add_entries.return_value = None
    mock.get_all_entries.return_value = []
    mock.mark_superseded.return_value = None
    mock.update_importance.return_value = None
    return mock


@pytest.fixture
def mock_simplemem() -> MagicMock:
    """Return a MagicMock standing in for SimpleMemSystem."""
    mock = MagicMock()
    mock.add_dialogues.return_value = None
    mock.add_dialogue.return_value = None
    mock.finalize.return_value = []
    mock.get_all_memories.return_value = []
    return mock


@pytest.fixture
def sample_session_record() -> SessionRecord:
    """Return a SessionRecord populated with test data."""
    return SessionRecord(
        tenant_id="test-tenant",
        content_session_id="content-sess-001",
        project="test-project",
        user_prompt="Implement feature X",
        started_at=datetime.now(timezone.utc),
        status=SessionStatus.active,
    )


@pytest.fixture
def sample_events() -> list[SessionEvent]:
    """Return a list of three SessionEvent objects covering different kinds."""
    now = datetime.now(timezone.utc)
    return [
        SessionEvent(
            memory_session_id="mem-sess-001",
            timestamp=now,
            kind=EventKind.message,
            title="User asked about auth",
            payload_json='{"role": "user", "text": "How does auth work?"}',
        ),
        SessionEvent(
            memory_session_id="mem-sess-001",
            timestamp=now,
            kind=EventKind.tool_use,
            title="Ran grep for auth module",
            payload_json='{"tool": "grep", "query": "auth"}',
        ),
        SessionEvent(
            memory_session_id="mem-sess-001",
            timestamp=now,
            kind=EventKind.file_change,
            title="Modified auth.py",
            payload_json='{"file": "auth.py", "action": "edit"}',
        ),
    ]
