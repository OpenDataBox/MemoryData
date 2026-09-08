# pyright: reportMissingImports=false
"""Unit tests for cross.types module."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cross.types import (
    ContextBundle,
    ConsolidationRun,
    CrossMemoryEntry,
    CrossObservation,
    EventKind,
    FinalizationReport,
    MemoryLink,
    ObservationType,
    RedactionLevel,
    SessionEvent,
    SessionRecord,
    SessionStatus,
    SessionSummary,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestSessionStatus:
    def test_enum_values(self):
        assert SessionStatus.active.value == "active"
        assert SessionStatus.completed.value == "completed"
        assert SessionStatus.failed.value == "failed"

    def test_string_serialization(self):
        assert str(SessionStatus.active) == "SessionStatus.active"
        assert SessionStatus.active == "active"

    def test_from_value(self):
        assert SessionStatus("active") is SessionStatus.active


class TestEventKind:
    def test_all_values(self):
        expected = {"message", "tool_use", "file_change", "note", "system"}
        actual = {e.value for e in EventKind}
        assert actual == expected

    def test_count(self):
        assert len(EventKind) == 5


class TestObservationType:
    def test_all_values(self):
        expected = {"decision", "bugfix", "feature", "refactor", "discovery", "change"}
        actual = {e.value for e in ObservationType}
        assert actual == expected

    def test_count(self):
        assert len(ObservationType) == 6


class TestRedactionLevel:
    def test_all_values(self):
        expected = {"none", "partial", "full"}
        actual = {e.value for e in RedactionLevel}
        assert actual == expected

    def test_count(self):
        assert len(RedactionLevel) == 3


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestSessionRecord:
    def test_creation_with_defaults(self):
        rec = SessionRecord(
            content_session_id="cs-1",
            project="proj",
            started_at=datetime.now(timezone.utc),
            status=SessionStatus.active,
        )
        assert rec.tenant_id == "default"
        assert rec.id is None
        assert rec.ended_at is None
        assert rec.metadata_json is None
        assert rec.user_prompt is None

    def test_memory_session_id_auto_generation(self):
        rec = SessionRecord(
            content_session_id="cs-1",
            project="proj",
            started_at=datetime.now(timezone.utc),
            status=SessionStatus.completed,
        )
        # Should be a valid uuid4 string
        parsed = uuid.UUID(rec.memory_session_id, version=4)
        assert str(parsed) == rec.memory_session_id

    def test_all_fields(self):
        now = datetime.now(timezone.utc)
        rec = SessionRecord(
            id=42,
            tenant_id="tenant-a",
            content_session_id="cs-2",
            memory_session_id="custom-mem-id",
            project="big-project",
            user_prompt="Do the thing",
            started_at=now,
            ended_at=now,
            status=SessionStatus.failed,
            metadata_json='{"key": "value"}',
        )
        assert rec.id == 42
        assert rec.tenant_id == "tenant-a"
        assert rec.content_session_id == "cs-2"
        assert rec.memory_session_id == "custom-mem-id"
        assert rec.project == "big-project"
        assert rec.user_prompt == "Do the thing"
        assert rec.started_at == now
        assert rec.ended_at == now
        assert rec.status == SessionStatus.failed
        assert json.loads(rec.metadata_json or "")["key"] == "value"


class TestSessionEvent:
    def test_creation(self):
        now = datetime.now(timezone.utc)
        evt = SessionEvent(
            memory_session_id="ms-1",
            timestamp=now,
            kind=EventKind.message,
        )
        assert evt.event_id is None
        assert evt.title is None
        assert evt.payload_json is None
        assert evt.redaction_level == RedactionLevel.none

    def test_payload_json_serialization(self):
        payload = {"tool": "grep", "args": ["foo"]}
        evt = SessionEvent(
            memory_session_id="ms-1",
            timestamp=datetime.now(timezone.utc),
            kind=EventKind.tool_use,
            payload_json=json.dumps(payload),
        )
        parsed = json.loads(evt.payload_json or "")
        assert parsed["tool"] == "grep"
        assert parsed["args"] == ["foo"]


class TestCrossObservation:
    def test_creation_required_fields(self):
        now = datetime.now(timezone.utc)
        obs = CrossObservation(
            memory_session_id="ms-1",
            timestamp=now,
            type=ObservationType.decision,
            title="Chose REST over GraphQL",
        )
        assert obs.obs_id is None
        assert obs.subtitle is None
        assert obs.facts_json is None
        assert obs.narrative is None
        assert obs.concepts_json is None
        assert obs.files_json is None
        assert obs.vector_ref is None

    def test_creation_with_optional_fields(self):
        now = datetime.now(timezone.utc)
        obs = CrossObservation(
            memory_session_id="ms-2",
            timestamp=now,
            type=ObservationType.bugfix,
            title="Fixed null pointer in parser",
            subtitle="parser.py line 42",
            narrative="The parser crashed on empty input.",
            files_json='["parser.py"]',
        )
        assert obs.subtitle == "parser.py line 42"
        assert obs.narrative == "The parser crashed on empty input."
        assert json.loads(obs.files_json or "") == ["parser.py"]


class TestSessionSummary:
    def test_creation(self):
        now = datetime.now(timezone.utc)
        summary = SessionSummary(
            memory_session_id="ms-1",
            timestamp=now,
            request="Build auth module",
            investigated="Looked at OAuth2 libraries",
            learned="Flask-Login integrates cleanly",
            completed="Basic login/logout flow",
            next_steps="Add token refresh",
        )
        assert summary.summary_id is None
        assert summary.request == "Build auth module"
        assert summary.completed == "Basic login/logout flow"
        assert summary.vector_ref is None


class TestMemoryLink:
    def test_creation(self):
        now = datetime.now(timezone.utc)
        link = MemoryLink(
            memory_entry_id="entry-1",
            source_kind="observation",
            source_id=7,
            score=0.85,
            timestamp=now,
        )
        assert link.link_id is None
        assert link.memory_entry_id == "entry-1"
        assert link.source_kind == "observation"
        assert link.source_id == 7
        assert link.score == pytest.approx(0.85)

    def test_score_validation(self):
        """Score is a float; very large/small values are accepted by the model."""
        now = datetime.now(timezone.utc)
        link = MemoryLink(
            memory_entry_id="e-2",
            source_kind="summary",
            source_id=1,
            score=-0.5,
            timestamp=now,
        )
        assert link.score == pytest.approx(-0.5)


class TestCrossMemoryEntry:
    def _make_entry(self, **overrides) -> CrossMemoryEntry:
        defaults = dict(
            entry_id="cme-1",
            lossless_restatement="User prefers dark mode",
            keywords=["preference", "dark-mode"],
            timestamp=None,
            location=None,
            persons=[],
            entities=[],
            topic="settings",
            tenant_id="t1",
            memory_session_id="ms-1",
            source_kind="observation",
        )
        defaults.update(overrides)
        return CrossMemoryEntry(**defaults)

    def test_inherits_memory_entry_fields(self):
        entry = self._make_entry()
        assert entry.lossless_restatement == "User prefers dark mode"
        assert entry.keywords == ["preference", "dark-mode"]
        assert entry.entry_id == "cme-1"

    def test_provenance_fields(self):
        entry = self._make_entry()
        assert entry.tenant_id == "t1"
        assert entry.memory_session_id == "ms-1"
        assert entry.source_kind == "observation"
        assert entry.source_id is None

    def test_importance_default(self):
        entry = self._make_entry()
        assert entry.importance == pytest.approx(0.5)

    def test_importance_bounds(self):
        entry = self._make_entry(importance=0.0)
        assert entry.importance == pytest.approx(0.0)
        entry = self._make_entry(importance=1.0)
        assert entry.importance == pytest.approx(1.0)

    def test_importance_out_of_bounds_rejected(self):
        with pytest.raises(Exception):
            self._make_entry(importance=1.5)
        with pytest.raises(Exception):
            self._make_entry(importance=-0.1)

    def test_superseded_and_validity(self):
        now = datetime.now(timezone.utc)
        entry = self._make_entry(
            valid_from=now,
            valid_to=now,
            superseded_by="cme-2",
        )
        assert entry.valid_from == now
        assert entry.valid_to == now
        assert entry.superseded_by == "cme-2"


class TestContextBundle:
    def test_creation_with_empty_lists(self):
        bundle = ContextBundle()
        assert bundle.session_summaries == []
        assert bundle.timeline_observations == []
        assert bundle.memory_entries == []
        assert bundle.total_tokens_estimate == 0

    def test_render_empty_returns_empty_string(self):
        bundle = ContextBundle()
        assert bundle.render(max_tokens=100) == ""

    def test_render_with_summaries(self):
        now = datetime.now(timezone.utc)
        bundle = ContextBundle(
            session_summaries=[
                SessionSummary(
                    memory_session_id="ms-1",
                    timestamp=now,
                    completed="Built auth flow",
                ),
            ],
        )
        rendered = bundle.render(max_tokens=500)
        assert "Session summaries:" in rendered
        assert "Built auth flow" in rendered

    def test_render_respects_max_tokens(self):
        now = datetime.now(timezone.utc)
        entries = [
            CrossMemoryEntry(
                entry_id=f"e-{i}",
                lossless_restatement=f"Long memory entry number {i} " + "x " * 50,
                keywords=["test"],
                timestamp=None,
                location=None,
                persons=[],
                entities=[],
                topic="test",
                tenant_id="t",
                memory_session_id="ms",
                source_kind="obs",
            )
            for i in range(20)
        ]
        bundle = ContextBundle(memory_entries=entries)
        short = bundle.render(max_tokens=10)
        full = bundle.render(max_tokens=100000)
        assert len(short) < len(full)


class TestFinalizationReport:
    def test_creation_with_all_fields(self):
        report = FinalizationReport(
            memory_session_id="ms-1",
            observations_count=5,
            summary_generated=True,
            entries_stored=12,
            consolidation_triggered=False,
        )
        assert report.memory_session_id == "ms-1"
        assert report.observations_count == 5
        assert report.summary_generated is True
        assert report.entries_stored == 12
        assert report.consolidation_triggered is False


class TestConsolidationRun:
    def test_creation(self):
        now = datetime.now(timezone.utc)
        run = ConsolidationRun(
            tenant_id="t1",
            timestamp=now,
            policy_json='{"max_age_days": 30}',
            stats_json='{"merged": 4}',
        )
        assert run.run_id is None
        assert run.tenant_id == "t1"
        assert run.timestamp == now
        assert json.loads(run.policy_json or "")["max_age_days"] == 30
        assert json.loads(run.stats_json or "")["merged"] == 4
