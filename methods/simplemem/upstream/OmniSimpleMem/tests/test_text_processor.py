"""
Tests for TextProcessor.

All LLM and embedding calls are mocked so no API keys are needed.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock

from omni_memory.core.config import OmniMemoryConfig
from omni_memory.processors.text_processor import TextProcessor
from omni_memory.triggers.base import TriggerDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 384


def _make_processor(**kwargs):
    """Create a TextProcessor with mocked cold storage."""
    config = OmniMemoryConfig()
    cold_storage = MagicMock()
    return TextProcessor(config=config, cold_storage=cold_storage, **kwargs)


# ---------------------------------------------------------------------------
# Redundancy detection
# ---------------------------------------------------------------------------

class TestRedundancyDetection:
    def test_first_text_is_accepted(self):
        proc = _make_processor()
        result = proc._check_redundancy("This is brand new text")
        assert result.decision == TriggerDecision.ACCEPT

    def test_identical_text_is_rejected(self):
        proc = _make_processor()
        proc._recent_texts.append("the quick brown fox jumps over the lazy dog")
        result = proc._check_redundancy("the quick brown fox jumps over the lazy dog")
        assert result.decision == TriggerDecision.REJECT

    def test_novel_text_is_accepted(self):
        proc = _make_processor()
        proc._recent_texts.append("the quick brown fox jumps over the lazy dog")
        result = proc._check_redundancy("quantum physics explains particle behavior")
        assert result.decision == TriggerDecision.ACCEPT

    def test_partially_overlapping_text(self):
        proc = _make_processor()
        proc._recent_texts.append("the quick brown fox jumps over the lazy dog near the river")
        # ~50% overlap
        result = proc._check_redundancy(
            "the quick brown fox went to the market and bought some apples"
        )
        assert result.decision in (TriggerDecision.ACCEPT, TriggerDecision.UNCERTAIN)


# ---------------------------------------------------------------------------
# Text length validation
# ---------------------------------------------------------------------------

class TestLengthValidation:
    @patch("omni_memory.processors.text_processor.TextProcessor.generate_embedding", return_value=FAKE_EMBEDDING)
    @patch("omni_memory.processors.text_processor.TextProcessor.generate_summary", side_effect=lambda t: t[:200])
    def test_short_text_rejected(self, mock_summary, mock_embed):
        proc = _make_processor(min_length=10)
        result = proc.process("hi")
        assert result.success is False
        assert result.skipped is True

    @patch("omni_memory.processors.text_processor.TextProcessor.generate_embedding", return_value=FAKE_EMBEDDING)
    @patch("omni_memory.processors.text_processor.TextProcessor.generate_summary", side_effect=lambda t: t[:200])
    def test_short_text_accepted_with_force(self, mock_summary, mock_embed):
        proc = _make_processor(min_length=10)
        result = proc.process("hi", force=True)
        assert result.success is True
        assert result.mau is not None

    @patch("omni_memory.processors.text_processor.TextProcessor.generate_embedding", return_value=FAKE_EMBEDDING)
    @patch("omni_memory.processors.text_processor.TextProcessor.generate_summary", side_effect=lambda t: t[:200])
    def test_long_text_truncated(self, mock_summary, mock_embed):
        proc = _make_processor(max_length=50)
        long_text = "x" * 200
        result = proc.process(long_text)
        assert result.success is True
        # The details should contain truncated text
        assert result.mau.details["full_text"].endswith("...")

    @patch("omni_memory.processors.text_processor.TextProcessor.generate_embedding", return_value=FAKE_EMBEDDING)
    @patch("omni_memory.processors.text_processor.TextProcessor.generate_summary", side_effect=lambda t: t[:200])
    def test_normal_text_accepted(self, mock_summary, mock_embed):
        proc = _make_processor(min_length=5)
        result = proc.process("This is a perfectly normal piece of text for testing.")
        assert result.success is True
        assert result.mau is not None
        assert result.mau.modality_type.value == "text"


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

class TestSummaryGeneration:
    def test_short_text_used_as_summary(self):
        proc = _make_processor()
        summary = proc.generate_summary("Short text under 200 chars")
        assert summary == "Short text under 200 chars"

    @patch.object(TextProcessor, "_call_llm", return_value="A concise summary.")
    def test_long_text_calls_llm(self, mock_llm):
        proc = _make_processor()
        long_text = "word " * 100  # well over 200 chars
        summary = proc.generate_summary(long_text)
        assert summary == "A concise summary."
        mock_llm.assert_called_once()

    @patch.object(TextProcessor, "_call_llm", return_value="")
    def test_llm_failure_returns_truncated(self, mock_llm):
        proc = _make_processor()
        long_text = "word " * 100
        summary = proc.generate_summary(long_text)
        # Falls back to first 200 chars
        assert len(summary) <= 200


# ---------------------------------------------------------------------------
# Full process pipeline (mocked embedding)
# ---------------------------------------------------------------------------

class TestProcessPipeline:
    @patch("omni_memory.processors.text_processor.TextProcessor.generate_embedding", return_value=FAKE_EMBEDDING)
    @patch("omni_memory.processors.text_processor.TextProcessor.generate_summary", side_effect=lambda t: t[:200])
    def test_process_creates_mau(self, mock_summary, mock_embed):
        proc = _make_processor()
        result = proc.process(
            "A detailed document about artificial intelligence and machine learning.",
            session_id="test_session",
        )
        assert result.success is True
        mau = result.mau
        assert mau is not None
        assert mau.embedding == FAKE_EMBEDDING
        assert mau.metadata.session_id == "test_session"

    @patch("omni_memory.processors.text_processor.TextProcessor.generate_embedding", return_value=[])
    @patch("omni_memory.processors.text_processor.TextProcessor.generate_summary", side_effect=lambda t: t[:200])
    def test_process_fails_on_empty_embedding(self, mock_summary, mock_embed):
        proc = _make_processor()
        result = proc.process("Some valid text content for testing purposes.")
        assert result.success is False
        assert result.error is not None

    def test_reset_clears_recent_texts(self):
        proc = _make_processor()
        proc._recent_texts = ["a", "b", "c"]
        proc.reset()
        assert proc._recent_texts == []
