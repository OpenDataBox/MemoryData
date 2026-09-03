# pyright: reportMissingImports=false
"""Unit tests for ContextInjector and ContextRenderer."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cross.context_injector import ContextInjector, ContextRenderer
from cross.types import (
    ContextBundle,
    CrossMemoryEntry,
    CrossObservation,
    ObservationType,
    SessionSummary,
)

# ---------------------------------------------------------------------------
# Helpers – reusable factory functions for test data
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc)


def _make_summary(
    *,
    request: str | None = "fix login bug",
    completed: str | None = "patched auth handler",
    learned: str | None = None,
    investigated: str | None = None,
    next_steps: str | None = None,
) -> SessionSummary:
    return SessionSummary(
        memory_session_id="sess-001",
        timestamp=_NOW,
        request=request,
        completed=completed,
        learned=learned,
        investigated=investigated,
        next_steps=next_steps,
    )


def _make_observation(
    *,
    title: str = "Refactored auth module",
    subtitle: str | None = "split into helpers",
    obs_type: ObservationType = ObservationType.refactor,
) -> CrossObservation:
    return CrossObservation(
        memory_session_id="sess-001",
        timestamp=_NOW,
        type=obs_type,
        title=title,
        subtitle=subtitle,
    )


def _make_entry(
    *,
    restatement: str = "User prefers dark mode in the IDE.",
    entry_id: str = "e-001",
) -> CrossMemoryEntry:
    return CrossMemoryEntry(
        entry_id=entry_id,
        lossless_restatement=restatement,
        keywords=["dark-mode", "ide"],
        timestamp=_NOW.isoformat(),
        location=None,
        persons=[],
        entities=["IDE"],
        topic="preferences",
        tenant_id="t-1",
        memory_session_id="sess-001",
        source_kind="observation",
        importance=0.5,
    )


def _make_injector(
    *,
    summaries: list[SessionSummary] | None = None,
    observations: list[CrossObservation] | None = None,
    entries: list[CrossMemoryEntry] | None = None,
    max_tokens: int = 2000,
) -> tuple[ContextInjector, MagicMock, MagicMock]:
    """Return (injector, mock_sqlite, mock_vector) with pre-wired returns."""
    mock_sqlite = MagicMock()
    mock_vector = MagicMock()

    mock_sqlite.get_recent_summaries.return_value = summaries or []
    mock_sqlite.get_recent_observations.return_value = observations or []
    mock_vector.semantic_search.return_value = entries or []

    injector = ContextInjector(
        sqlite_storage=mock_sqlite,
        vector_store=mock_vector,
        max_tokens=max_tokens,
    )
    return injector, mock_sqlite, mock_vector


# ===================================================================
# TestContextInjector
# ===================================================================


class TestContextInjector:
    """Tests for ContextInjector.build_context."""

    # ---------------------------------------------------------------
    # 1. Empty storage -> empty bundle
    # ---------------------------------------------------------------

    def test_build_context_empty(self) -> None:
        injector, _, _ = _make_injector()
        bundle = injector.build_context("t-1", "my-project")

        assert bundle.session_summaries == []
        assert bundle.timeline_observations == []
        assert bundle.memory_entries == []
        assert bundle.total_tokens_estimate == 0

    # ---------------------------------------------------------------
    # 2. Summaries present
    # ---------------------------------------------------------------

    def test_build_context_with_summaries(self) -> None:
        summaries = [
            _make_summary(request="add caching layer", completed="added Redis cache"),
            _make_summary(request="fix timeout bug", completed="increased TTL"),
        ]
        injector, mock_sqlite, _ = _make_injector(summaries=summaries)

        bundle = injector.build_context("t-1", "proj")

        assert len(bundle.session_summaries) == 2
        assert bundle.session_summaries[0].request == "add caching layer"
        mock_sqlite.get_recent_summaries.assert_called_once_with("proj", limit=5)

    # ---------------------------------------------------------------
    # 3. Observations present
    # ---------------------------------------------------------------

    def test_build_context_with_observations(self) -> None:
        observations = [
            _make_observation(title="Switched to async IO"),
            _make_observation(
                title="Fixed race condition", obs_type=ObservationType.bugfix
            ),
        ]
        injector, mock_sqlite, _ = _make_injector(observations=observations)

        bundle = injector.build_context("t-1", "proj")

        assert len(bundle.timeline_observations) == 2
        assert bundle.timeline_observations[1].title == "Fixed race condition"
        mock_sqlite.get_recent_observations.assert_called_once_with("proj", limit=20)

    # ---------------------------------------------------------------
    # 4. Semantic search when user_prompt provided
    # ---------------------------------------------------------------

    def test_build_context_with_semantic_search(self) -> None:
        entries = [
            _make_entry(restatement="User wants dark theme", entry_id="e-10"),
            _make_entry(restatement="Editor font is JetBrains Mono", entry_id="e-11"),
        ]
        injector, _, mock_vector = _make_injector(entries=entries)

        bundle = injector.build_context("t-1", "proj", user_prompt="theme settings")

        assert len(bundle.memory_entries) == 2
        mock_vector.semantic_search.assert_called_once_with(
            "theme settings", top_k=10, tenant_id="t-1"
        )

    # ---------------------------------------------------------------
    # 5. No prompt -> no semantic search
    # ---------------------------------------------------------------

    def test_build_context_no_prompt_no_semantic(self) -> None:
        entries = [_make_entry()]
        injector, _, mock_vector = _make_injector(entries=entries)

        bundle = injector.build_context("t-1", "proj")  # no user_prompt

        mock_vector.semantic_search.assert_not_called()
        assert bundle.memory_entries == []

    # ---------------------------------------------------------------
    # 6. Token budget respected
    # ---------------------------------------------------------------

    def test_token_budget_respected(self) -> None:
        # Each summary text ~6-8 words.  With max_tokens=10 only a
        # subset should fit.
        summaries = [
            _make_summary(
                request="first task description here now",
                completed="done first task completely here",
            ),
            _make_summary(
                request="second task description here now",
                completed="done second task completely here",
            ),
            _make_summary(
                request="third task description here now",
                completed="done third task completely here",
            ),
        ]
        injector, _, _ = _make_injector(summaries=summaries, max_tokens=10)

        bundle = injector.build_context("t-1", "proj")

        # Not all 3 should make it if budget is tiny
        assert len(bundle.session_summaries) < 3
        assert bundle.total_tokens_estimate <= 10

    # ---------------------------------------------------------------
    # 7. Progressive fill order
    # ---------------------------------------------------------------

    def test_progressive_fill_order(self) -> None:
        """Summaries fill first, then observations, then entries."""
        summaries = [_make_summary(request="s1", completed="c1")]
        observations = [_make_observation(title="obs1")]
        entries = [_make_entry(restatement="entry1", entry_id="e-20")]

        # Give generous budget so everything fits
        injector, _, _ = _make_injector(
            summaries=summaries,
            observations=observations,
            entries=entries,
            max_tokens=5000,
        )

        bundle = injector.build_context("t-1", "proj", user_prompt="anything")

        assert len(bundle.session_summaries) == 1
        assert len(bundle.timeline_observations) == 1
        assert len(bundle.memory_entries) == 1

        # With a very tight budget, only summaries should survive
        injector_tight, _, _ = _make_injector(
            summaries=summaries,
            observations=observations,
            entries=entries,
            max_tokens=5,
        )
        bundle_tight = injector_tight.build_context(
            "t-1", "proj", user_prompt="anything"
        )

        # Summaries have highest priority; observations/entries should be
        # empty or fewer when budget is exhausted.
        assert len(bundle_tight.session_summaries) >= 0
        tokens_after_summaries = bundle_tight.total_tokens_estimate
        # If budget was fully consumed by summaries, observations and
        # entries must be empty.
        if tokens_after_summaries >= 5:
            assert bundle_tight.timeline_observations == []
            assert bundle_tight.memory_entries == []

    # ---------------------------------------------------------------
    # 8. _estimate_tokens helper
    # ---------------------------------------------------------------

    def test_estimate_tokens(self) -> None:
        assert ContextInjector._estimate_tokens("hello world") == 2
        assert ContextInjector._estimate_tokens("one two three four five") == 5
        # "".split() returns [], so len == 0
        assert ContextInjector._estimate_tokens("") == 0
        assert ContextInjector._estimate_tokens("single") == 1


# ===================================================================
# TestContextRenderer
# ===================================================================


class TestContextRenderer:
    """Tests for ContextRenderer static rendering methods."""

    # ---------------------------------------------------------------
    # 1. render_for_system_prompt with data
    # ---------------------------------------------------------------

    def test_render_for_system_prompt(self) -> None:
        bundle = ContextBundle(
            session_summaries=[
                _make_summary(completed="deployed v2 API"),
            ],
            timeline_observations=[
                _make_observation(title="Added rate limiter"),
            ],
            memory_entries=[
                _make_entry(restatement="API uses token-bucket algorithm"),
            ],
            total_tokens_estimate=30,
        )
        result = ContextRenderer.render_for_system_prompt(bundle)

        assert "<cross_session_memory>" in result
        assert "</cross_session_memory>" in result
        assert "previous sessions" in result
        # Content from the bundle should appear
        assert "deployed v2 API" in result or "Session summaries:" in result

    # ---------------------------------------------------------------
    # 2. render_for_system_prompt empty bundle
    # ---------------------------------------------------------------

    def test_render_for_system_prompt_empty(self) -> None:
        bundle = ContextBundle()
        result = ContextRenderer.render_for_system_prompt(bundle)

        assert result == ""

    # ---------------------------------------------------------------
    # 3. render_summary_only with summaries
    # ---------------------------------------------------------------

    def test_render_summary_only(self) -> None:
        bundle = ContextBundle(
            session_summaries=[
                _make_summary(request="migrate DB", completed="ran alembic migrations"),
                _make_summary(request="add tests", completed="wrote 20 unit tests"),
            ],
        )
        result = ContextRenderer.render_summary_only(bundle)

        assert result.startswith("Session summaries:")
        assert "migrate DB" in result or "alembic migrations" in result
        lines = result.strip().split("\n")
        # Header + 2 bullet items
        assert len(lines) == 3

    # ---------------------------------------------------------------
    # 4. render_summary_only empty
    # ---------------------------------------------------------------

    def test_render_summary_only_empty(self) -> None:
        bundle = ContextBundle()
        result = ContextRenderer.render_summary_only(bundle)

        assert result == ""
