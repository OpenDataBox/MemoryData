"""
Tests for BM25Store keyword search.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Check if rank_bm25 is available
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from omni_memory.retrieval.bm25_store import BM25Store, _tokenize


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_punctuation(self):
        tokens = _tokenize("Hello, World! How's it going?")
        # Single-char tokens are dropped (len >= 2)
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_short_tokens_dropped(self):
        tokens = _tokenize("I a am ok")
        # "I" and "a" are length 1, dropped
        assert "am" in tokens
        assert "ok" in tokens


# ---------------------------------------------------------------------------
# Fake MAU for building the BM25 index
# ---------------------------------------------------------------------------

@dataclass
class FakeMAUMetadata:
    tags: List[str] = field(default_factory=list)
    persons: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    location: Optional[str] = None
    topic: Optional[str] = None


@dataclass
class FakeMAU:
    id: str = "mau_fake"
    summary: str = ""
    details: Optional[Dict[str, Any]] = None
    status: str = "ACTIVE"
    metadata: FakeMAUMetadata = field(default_factory=FakeMAUMetadata)


class FakeMAUStore:
    """Minimal MAU store that supports iter_all() and get()."""

    def __init__(self, maus: List[FakeMAU]):
        self._maus = {m.id: m for m in maus}

    def iter_all(self):
        return iter(self._maus.values())

    def get(self, mau_id):
        return self._maus.get(mau_id)


# ---------------------------------------------------------------------------
# BM25Store tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_BM25, reason="rank_bm25 not installed")
class TestBM25Store:
    @pytest.fixture
    def populated_store(self):
        maus = [
            FakeMAU(id="mau_1", summary="The quick brown fox jumps over the lazy dog"),
            FakeMAU(id="mau_2", summary="Machine learning and deep learning with Python"),
            FakeMAU(id="mau_3", summary="The weather today is sunny and warm"),
            FakeMAU(id="mau_4", summary="Albert Einstein developed the theory of relativity"),
            FakeMAU(id="mau_5", summary="Python programming language is versatile"),
        ]
        fake_store = FakeMAUStore(maus)
        bm25 = BM25Store()
        count = bm25.build_from_mau_store(fake_store)
        assert count == 5
        return bm25

    def test_build_index(self, populated_store):
        assert populated_store.is_available is True
        assert populated_store.doc_count == 5

    def test_search_relevant_results(self, populated_store):
        results = populated_store.search("Python programming", top_k=3)
        assert len(results) > 0
        ids = [r[0] for r in results]
        # mau_2 or mau_5 mention Python
        assert "mau_2" in ids or "mau_5" in ids

    def test_search_no_match(self, populated_store):
        results = populated_store.search("xyznonexistentword", top_k=3)
        assert results == []

    def test_search_empty_query(self, populated_store):
        results = populated_store.search("", top_k=3)
        assert results == []

    def test_empty_index(self):
        bm25 = BM25Store()
        assert bm25.is_available is False
        assert bm25.doc_count == 0
        results = bm25.search("anything", top_k=5)
        assert results == []

    def test_archived_maus_skipped(self):
        maus = [
            FakeMAU(id="active_1", summary="Active document about cats and dogs"),
            FakeMAU(id="active_2", summary="Another document about fish and birds"),
            FakeMAU(id="active_3", summary="Third document about cars and trains"),
            FakeMAU(id="archived_1", summary="Archived document about cats", status="ARCHIVED"),
        ]
        fake_store = FakeMAUStore(maus)
        bm25 = BM25Store()
        count = bm25.build_from_mau_store(fake_store)
        # Only 3 active MAUs indexed (archived_1 skipped)
        assert count == 3
        # Verify archived ID is not in the index
        assert "archived_1" not in bm25._mau_ids

    def test_keyword_matching_scores_descending(self, populated_store):
        results = populated_store.search("fox dog lazy", top_k=5)
        if len(results) >= 2:
            # Scores should be descending
            for i in range(len(results) - 1):
                assert results[i][1] >= results[i + 1][1]

    def test_enriched_metadata_indexed(self):
        """Verify that enriched metadata fields are tokenized into the corpus."""
        maus = [
            FakeMAU(
                id="mau_meta",
                summary="A meeting",
                metadata=FakeMAUMetadata(
                    keywords=["quarterly", "review"],
                    persons=["Alice"],
                    location="conference room",
                ),
            ),
        ]
        fake_store = FakeMAUStore(maus)
        bm25 = BM25Store()
        bm25.build_from_mau_store(fake_store)
        # Verify the enriched metadata tokens are in the corpus
        assert "alice" in bm25._corpus[0]
        assert "quarterly" in bm25._corpus[0]
        assert "conference" in bm25._corpus[0]
