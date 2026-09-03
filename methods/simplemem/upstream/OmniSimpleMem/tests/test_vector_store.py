"""
Tests for VectorStore (numpy fallback mode -- no FAISS required).
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from omni_memory.core.config import EmbeddingConfig
from omni_memory.storage.vector_store import VectorStore


DIM = 8  # small dimension for tests


@pytest.fixture
def store(tmp_path):
    """Create a VectorStore using numpy backend."""
    cfg = EmbeddingConfig(embedding_dim=DIM)
    return VectorStore(
        storage_path=str(tmp_path / "vectors"),
        config=cfg,
        use_faiss=False,  # force numpy fallback
    )


def _random_vec(dim=DIM, seed=None):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v.tolist()


def _unit_vec(index, dim=DIM):
    """Create a unit vector with 1.0 at the given index."""
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v.tolist()


# ---------------------------------------------------------------------------
# Basic add / search
# ---------------------------------------------------------------------------

class TestVectorStoreBasic:
    def test_empty_store_search(self, store):
        results = store.search(_random_vec(), top_k=5)
        assert results == []

    def test_add_single_and_search(self, store):
        vec = _random_vec(seed=42)
        store.add("mau_001", vec)
        assert store.size() == 1

        results = store.search(vec, top_k=1)
        assert len(results) >= 1
        assert results[0][0] == "mau_001"
        # Self-similarity should be close to 1.0
        assert results[0][1] > 0.99

    def test_add_multiple_and_rank(self, store):
        # Add two orthogonal-ish vectors
        v1 = _unit_vec(0)
        v2 = _unit_vec(1)
        store.add("mau_a", v1)
        store.add("mau_b", v2)

        # Query with v1 should rank mau_a first
        results = store.search(v1, top_k=2)
        assert results[0][0] == "mau_a"

    def test_size(self, store):
        assert store.size() == 0
        store.add("m1", _random_vec(seed=1))
        store.add("m2", _random_vec(seed=2))
        assert store.size() == 2

    def test_count(self, store):
        store.add("m1", _random_vec(seed=1))
        assert store.count() >= 1


# ---------------------------------------------------------------------------
# Cosine similarity correctness
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self, store):
        vec = _random_vec(seed=10)
        store.add("mau_same", vec)
        results = store.search(vec, top_k=1)
        assert abs(results[0][1] - 1.0) < 1e-5

    def test_orthogonal_vectors_score_zero(self, store):
        v1 = _unit_vec(0)
        v2 = _unit_vec(1)
        store.add("mau_orth", v1)
        results = store.search(v2, top_k=1)
        # Cosine similarity of orthogonal vectors ~ 0
        assert abs(results[0][1]) < 1e-5

    def test_opposite_vectors_negative(self, store):
        vec = [1.0] * DIM
        neg = [-1.0] * DIM
        store.add("mau_pos", vec)
        results = store.search(neg, top_k=1)
        # Opposite vectors should have negative similarity
        assert results[0][1] < 0


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

class TestBatchOperations:
    def test_add_batch(self, store):
        items = [(f"mau_{i}", _random_vec(seed=i)) for i in range(10)]
        store.add_batch(items)
        assert store.size() == 10

    def test_search_batch(self, store):
        items = [(f"mau_{i}", _random_vec(seed=i)) for i in range(5)]
        store.add_batch(items)
        queries = [_random_vec(seed=0), _random_vec(seed=1)]
        results = store.search_batch(queries, top_k=3)
        assert len(results) == 2
        for r in results:
            assert len(r) <= 3


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        cfg = EmbeddingConfig(embedding_dim=DIM)
        path = str(tmp_path / "persist_vectors")

        # Create, add, save
        s1 = VectorStore(storage_path=path, config=cfg, use_faiss=False)
        s1.add("mau_p1", _random_vec(seed=100))
        s1.add("mau_p2", _random_vec(seed=200))
        s1.save()

        # Reload
        s2 = VectorStore(storage_path=path, config=cfg, use_faiss=False)
        assert s2.size() == 2
        results = s2.search(_random_vec(seed=100), top_k=1)
        assert results[0][0] == "mau_p1"


# ---------------------------------------------------------------------------
# Delete and rebuild
# ---------------------------------------------------------------------------

class TestDeleteRebuild:
    def test_delete_marks_none(self, store):
        store.add("mau_del", _random_vec(seed=1))
        assert store.delete("mau_del") is True
        assert store.delete("nonexistent") is False

    def test_rebuild_index(self, store):
        store.add("old_1", _random_vec(seed=1))
        store.add("old_2", _random_vec(seed=2))
        assert store.size() == 2

        new_items = [("new_1", _random_vec(seed=10))]
        store.rebuild_index(new_items)
        assert store.size() == 1

    def test_get_embedding(self, store):
        vec = _random_vec(seed=42)
        store.add("mau_emb", vec)
        retrieved = store.get_embedding("mau_emb")
        assert retrieved is not None
        # Should be normalized version of original
        assert retrieved.shape == (DIM,)

    def test_get_embedding_missing(self, store):
        assert store.get_embedding("nonexistent") is None
