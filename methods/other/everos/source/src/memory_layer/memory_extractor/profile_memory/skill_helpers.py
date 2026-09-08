"""Skill normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from memory_layer.memory_extractor.profile_memory.value_helpers import (
    extract_values_with_evidence,
    merge_value_with_evidences_lists,
    merge_value_with_evidences_lists_keep_highest_level,
)


def normalize_skills_with_evidence(
    raw_value: Any,
    *,
    field_name: str,
    valid_conversation_ids: Optional[Set[str]] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    if not raw_value:
        return None

    return extract_values_with_evidence(
        raw_value,
        field_name=field_name,
        valid_conversation_ids=valid_conversation_ids,
        conversation_date_map=conversation_date_map,
    )


def merge_skill_lists(
    existing: Optional[List[Dict[str, Any]]],
    incoming: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Merge two skill lists using the shared value merge helper."""
    return merge_value_with_evidences_lists(existing, incoming)


def merge_skill_lists_keep_highest_level(
    *sources: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Merge multiple skill lists while keeping the highest level for each skill.

    This function is designed for merging skills across multiple groups where
    we want to preserve the highest level achieved for each skill.

    Args:
        *sources: Variable number of skill lists to merge

    Returns:
        Merged skill list with highest levels preserved, or None if all sources are empty
    """
    return merge_value_with_evidences_lists_keep_highest_level(*sources)


__all__ = [
    "normalize_skills_with_evidence",
    "merge_skill_lists",
    "merge_skill_lists_keep_highest_level",
]
