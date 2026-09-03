# pyright: reportMissingImports=false
"""Unit tests for cross.consolidation module."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cross.consolidation import (
    ConsolidationPolicy,
    ConsolidationResult,
    ConsolidationWorker,
    run_consolidation,
)
from cross.types import CrossMemoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str = "e1",
    importance: float = 0.5,
    valid_from: datetime | None = None,
    superseded_by: str | None = None,
    lossless_restatement: str = "some fact",
    tenant_id: str = "t1",
) -> CrossMemoryEntry:
    """Create a CrossMemoryEntry with sensible defaults for testing."""
    return CrossMemoryEntry(
        entry_id=entry_id,
        lossless_restatement=lossless_restatement,
        keywords=["test"],
        timestamp=None,
        location=None,
        persons=[],
        entities=[],
        topic=None,
        tenant_id=tenant_id,
        memory_session_id="ses-1",
        source_kind="observation",
        importance=importance,
        valid_from=valid_from or datetime.now(timezone.utc),
        superseded_by=superseded_by,
    )


def _make_storage_mocks():
    """Return (sqlite_storage, vector_store) MagicMock pair."""
    sqlite_storage = MagicMock()
    vector_store = MagicMock()
    vector_store.get_all_entries = MagicMock(return_value=[])
    vector_store.update_importance = MagicMock()
    vector_store.mark_superseded = MagicMock()
    vector_store.embedding_model = MagicMock()
    return sqlite_storage, vector_store


# ---------------------------------------------------------------------------
# TestConsolidationPolicy
# ---------------------------------------------------------------------------


class TestConsolidationPolicy:
    def test_default_values(self):
        policy = ConsolidationPolicy()
        assert policy.max_age_days == 90
        assert policy.decay_factor == 0.9
        assert policy.merge_similarity_threshold == 0.95
        assert policy.min_importance == 0.05
        assert policy.max_entries_per_run == 1000

    def test_custom_values(self):
        policy = ConsolidationPolicy(
            max_age_days=30,
            decay_factor=0.8,
            merge_similarity_threshold=0.85,
            min_importance=0.1,
            max_entries_per_run=500,
        )
        assert policy.max_age_days == 30
        assert policy.decay_factor == 0.8
        assert policy.merge_similarity_threshold == 0.85
        assert policy.min_importance == 0.1
        assert policy.max_entries_per_run == 500


# ---------------------------------------------------------------------------
# TestConsolidationResult
# ---------------------------------------------------------------------------


class TestConsolidationResult:
    def test_creation(self):
        result = ConsolidationResult(
            decayed_count=5,
            merged_count=3,
            pruned_count=2,
            duration_seconds=1.234,
        )
        assert result.decayed_count == 5
        assert result.merged_count == 3
        assert result.pruned_count == 2
        assert result.duration_seconds == 1.234

    def test_defaults(self):
        result = ConsolidationResult()
        assert result.decayed_count == 0
        assert result.merged_count == 0
        assert result.pruned_count == 0
        assert result.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# TestConsolidationWorker
# ---------------------------------------------------------------------------


class TestConsolidationWorker:
    def test_run_empty(self):
        """No entries in vector store -> result with all zeros."""
        sqlite_storage, vector_store = _make_storage_mocks()
        vector_store.get_all_entries.return_value = []

        worker = ConsolidationWorker(sqlite_storage, vector_store)
        result = worker.run("t1")

        assert result.decayed_count == 0
        assert result.merged_count == 0
        assert result.pruned_count == 0
        assert result.duration_seconds >= 0.0
        sqlite_storage.record_consolidation_run.assert_called_once()

    def test_decay_old_entries(self):
        """Entries older than max_age_days should have their importance decayed."""
        sqlite_storage, vector_store = _make_storage_mocks()

        old_date = datetime.now(timezone.utc) - timedelta(days=120)
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)

        old_entry = _make_entry(entry_id="old-1", importance=0.8, valid_from=old_date)
        recent_entry = _make_entry(
            entry_id="new-1", importance=0.8, valid_from=recent_date
        )

        vector_store.get_all_entries.return_value = [old_entry, recent_entry]
        # Merge phase needs embeddings — provide dummy ones so it doesn't error
        vector_store.embedding_model.encode_documents.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        policy = ConsolidationPolicy(max_age_days=90, decay_factor=0.9)
        worker = ConsolidationWorker(sqlite_storage, vector_store, policy=policy)
        result = worker.run("t1")

        assert result.decayed_count == 1
        # update_importance should have been called for the old entry
        vector_store.update_importance.assert_called_once()
        call_args = vector_store.update_importance.call_args
        assert call_args[0][0] == "old-1"
        expected_importance = 0.8 * 0.9
        assert abs(call_args[0][1] - expected_importance) < 1e-9

    def test_prune_low_importance(self):
        """Entries with importance below min_importance should be pruned."""
        sqlite_storage, vector_store = _make_storage_mocks()

        low_entry = _make_entry(entry_id="low-1", importance=0.02)
        ok_entry = _make_entry(entry_id="ok-1", importance=0.5)

        vector_store.get_all_entries.return_value = [low_entry, ok_entry]
        # Provide embeddings with low similarity so merge phase does nothing
        vector_store.embedding_model.encode_documents.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        policy = ConsolidationPolicy(min_importance=0.05)
        worker = ConsolidationWorker(sqlite_storage, vector_store, policy=policy)
        result = worker.run("t1")

        assert result.pruned_count == 1
        vector_store.mark_superseded.assert_any_call("low-1", "__pruned__")

    def test_merge_similar_entries(self):
        """Two entries with high cosine similarity should be merged."""
        sqlite_storage, vector_store = _make_storage_mocks()

        # Two entries; second has lower importance -> it should be superseded
        entry_a = _make_entry(
            entry_id="a", importance=0.8, lossless_restatement="fact A"
        )
        entry_b = _make_entry(
            entry_id="b", importance=0.3, lossless_restatement="fact B"
        )

        vector_store.get_all_entries.return_value = [entry_a, entry_b]

        # Create nearly identical unit vectors (cosine sim ~ 0.9998)
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.9999, 0.01, 0.0])
        vec_b = vec_b / np.linalg.norm(vec_b)  # normalize
        vectors = np.array([vec_a, vec_b])

        vector_store.embedding_model.encode_documents.return_value = vectors

        policy = ConsolidationPolicy(merge_similarity_threshold=0.95)
        worker = ConsolidationWorker(sqlite_storage, vector_store, policy=policy)
        result = worker.run("t1")

        assert result.merged_count == 1
        # The lower-importance entry (b) should be superseded by (a)
        vector_store.mark_superseded.assert_any_call("b", "a")

    def test_no_merge_below_threshold(self):
        """Two entries with low cosine similarity should NOT be merged."""
        sqlite_storage, vector_store = _make_storage_mocks()

        entry_a = _make_entry(
            entry_id="a", importance=0.8, lossless_restatement="fact A"
        )
        entry_b = _make_entry(
            entry_id="b", importance=0.3, lossless_restatement="fact B"
        )

        vector_store.get_all_entries.return_value = [entry_a, entry_b]

        # Orthogonal vectors (cosine sim = 0.0)
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        vector_store.embedding_model.encode_documents.return_value = vectors

        policy = ConsolidationPolicy(merge_similarity_threshold=0.95)
        worker = ConsolidationWorker(sqlite_storage, vector_store, policy=policy)
        result = worker.run("t1")

        assert result.merged_count == 0
        # mark_superseded should NOT have been called for merge
        # (might be called for prune if importance < threshold, so check specifically)
        for call in vector_store.mark_superseded.call_args_list:
            # No call should merge b into a or vice versa
            assert call[0] != ("b", "a")
            assert call[0] != ("a", "b")

    def test_run_records_consolidation(self):
        """Verify sqlite_storage.record_consolidation_run is called."""
        sqlite_storage, vector_store = _make_storage_mocks()
        vector_store.get_all_entries.return_value = []

        worker = ConsolidationWorker(sqlite_storage, vector_store)
        worker.run("t1")

        sqlite_storage.record_consolidation_run.assert_called_once()
        call_kwargs = sqlite_storage.record_consolidation_run.call_args
        # Verify tenant_id is passed
        assert call_kwargs[1]["tenant_id"] == "t1"
        # Verify policy_json and stats_json are dicts
        assert isinstance(call_kwargs[1]["policy_json"], dict)
        assert isinstance(call_kwargs[1]["stats_json"], dict)

    def test_run_consolidation_convenience(self):
        """The run_consolidation() convenience function delegates correctly."""
        sqlite_storage, vector_store = _make_storage_mocks()
        vector_store.get_all_entries.return_value = []

        result = run_consolidation(
            sqlite_storage=sqlite_storage,
            vector_store=vector_store,
            tenant_id="t1",
        )

        assert isinstance(result, ConsolidationResult)
        assert result.decayed_count == 0
        assert result.merged_count == 0
        assert result.pruned_count == 0
        sqlite_storage.record_consolidation_run.assert_called_once()

    def test_run_consolidation_convenience_with_policy(self):
        """Convenience function forwards custom policy."""
        sqlite_storage, vector_store = _make_storage_mocks()
        vector_store.get_all_entries.return_value = []

        policy = ConsolidationPolicy(max_age_days=7, decay_factor=0.5)
        result = run_consolidation(
            sqlite_storage=sqlite_storage,
            vector_store=vector_store,
            tenant_id="t1",
            policy=policy,
        )

        assert isinstance(result, ConsolidationResult)
        call_kwargs = sqlite_storage.record_consolidation_run.call_args[1]
        assert call_kwargs["policy_json"]["max_age_days"] == 7
        assert call_kwargs["policy_json"]["decay_factor"] == 0.5

    def test_superseded_entries_filtered_out(self):
        """Entries already superseded should be excluded from processing."""
        sqlite_storage, vector_store = _make_storage_mocks()

        active_1 = _make_entry(entry_id="active-1", importance=0.5)
        active_2 = _make_entry(entry_id="active-2", importance=0.6)
        superseded = _make_entry(
            entry_id="old-1", importance=0.5, superseded_by="active-1"
        )

        vector_store.get_all_entries.return_value = [active_1, active_2, superseded]
        # Two active entries -> merge phase calls encode_documents
        # Use orthogonal vectors so nothing merges
        vector_store.embedding_model.encode_documents.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        worker = ConsolidationWorker(sqlite_storage, vector_store)
        result = worker.run("t1")

        # encode_documents should only receive the 2 active entries' texts
        call_args = vector_store.embedding_model.encode_documents.call_args[0][0]
        assert len(call_args) == 2
        assert result.merged_count == 0
