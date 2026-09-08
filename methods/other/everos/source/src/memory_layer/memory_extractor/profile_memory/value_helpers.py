"""Value normalization helpers for profile memories."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from core.observation.logger import get_logger

from memory_layer.memory_extractor.profile_memory.evidence_utils import (
    conversation_id_from_evidence,
    ensure_str_list,
    format_evidence_entry,
)

logger = get_logger(__name__)


# Level priority mapping for comparison (higher number = higher level)
LEVEL_PRIORITY = {
    # Empty/missing
    "": 0,
    # Low level
    "低级": 1,
    "low": 1,
    "basic": 1,
    "beginner": 1,
    "初级": 1,
    "familiar": 1,
    "weak": 1,
    # Medium level
    "中级": 2,
    "medium": 2,
    "intermediate": 2,
    # High level
    "高级": 3,
    "high": 3,
    "advanced": 3,
    "strong": 3,
    "专家": 3,
    "expert": 3,
}


def _normalize_level(level: Any) -> str:
    """Normalize level value to lowercase string for comparison."""
    if level is None or level == "":
        return ""
    level_str = level.strip() if isinstance(level, str) else str(level).strip()
    return level_str.lower()


def _get_level_priority(level: Any) -> int:
    """Get the priority value for a given level."""
    normalized = _normalize_level(level)
    return LEVEL_PRIORITY.get(normalized, 0)


def _compare_levels(level1: Any, level2: Any) -> str:
    """Compare two levels and return the higher one (original casing preserved)."""
    if not level1 and not level2:
        return ""
    if not level1:
        return level2.strip() if isinstance(level2, str) else str(level2).strip()
    if not level2:
        return level1.strip() if isinstance(level1, str) else str(level1).strip()

    priority1 = _get_level_priority(level1)
    priority2 = _get_level_priority(level2)

    if priority1 >= priority2:
        return level1.strip() if isinstance(level1, str) else str(level1).strip()
    else:
        return level2.strip() if isinstance(level2, str) else str(level2).strip()


def merge_value_with_evidences_lists_keep_highest_level(
    *sources: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """
    Merge multiple value/evidence lists while keeping the highest level for each value.

    This function is designed for fields that have a 'level' attribute (e.g., skills,
    motivation_system, fear_system, etc.). For each unique value across all source lists,
    it keeps the highest level found and merges all evidences.

    Args:
        *sources: Variable number of source lists to merge

    Returns:
        Merged list with highest levels preserved, or None if all sources are empty

    Example:
        >>> source1 = [{"value": "Python", "level": "high", "evidences": ["ev1"]}],
        >>> source2 = [{"value": "Python", "level": "medium", "evidences": ["ev2"]}]
        >>> merge_value_with_evidences_lists_keep_highest_level(source1, source2)
        [{"value": "Python", "level": "high", "evidences": ["ev1", "ev2"]}]
    """
    if not sources:
        return None

    merged_map: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for source in sources:
        if not source:
            continue
        for item in source:
            if not isinstance(item, dict) or not item:
                continue

            value = item.get("value", "")
            evidences = item.get("evidences", [])
            level_value = item.get("level", "")

            if not value:
                continue

            value_key = value.strip() if isinstance(value, str) else str(value).strip()
            if not value_key:
                continue

            # Initialize if first time seeing this value
            if value_key not in merged_map:
                order.append(value_key)
                merged_map[value_key] = {"evidences": [], "level": ""}

            # Merge evidences (deduplicate)
            if evidences:
                for ev in evidences:
                    if ev and ev not in merged_map[value_key]["evidences"]:
                        merged_map[value_key]["evidences"].append(ev)

            # Keep the highest level
            current_level = merged_map[value_key]["level"]
            new_level = _compare_levels(current_level, level_value)
            merged_map[value_key]["level"] = new_level

    if not order:
        return None

    # Build final result list
    merged_list: List[Dict[str, Any]] = []
    for val in order:
        entry = {"value": val, "evidences": merged_map[val]["evidences"]}
        level = merged_map[val].get("level", "")
        if level:
            entry["level"] = level
        merged_list.append(entry)

    return merged_list


def merge_value_with_evidences_lists(
    existing: Optional[List[Dict[str, Any]]], incoming: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """Merge two value/evidence lists while deduplicating evidences."""
    if not existing and not incoming:
        return None

    merged_map: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def add_from(source: Optional[List[Dict[str, Any]]]) -> None:
        if not source:
            return
        for item in source:
            if not isinstance(item, dict) or not item:
                continue
            value = item.get("value", "")
            evidences = item.get("evidences", [])
            level_value = item.get("level", "")
            if not value:
                continue
            value_key = value.strip() if isinstance(value, str) else str(value).strip()
            if not value_key:
                continue
            if value_key not in merged_map:
                order.append(value_key)
                merged_map[value_key] = {"evidences": [], "level": ""}
            if evidences:
                for ev in evidences:
                    if ev and ev not in merged_map[value_key]["evidences"]:
                        merged_map[value_key]["evidences"].append(ev)
            level_str = (
                level_value.strip()
                if isinstance(level_value, str)
                else str(level_value).strip() if level_value is not None else ""
            )
            if level_str:
                merged_map[value_key]["level"] = level_str

    add_from(existing)
    add_from(incoming)

    if not order:
        return None

    merged_list: List[Dict[str, Any]] = []
    for val in order:
        entry = {"value": val, "evidences": merged_map[val]["evidences"]}
        level = merged_map[val].get("level", "")
        if level:
            entry["level"] = level
        merged_list.append(entry)
    return merged_list


def extract_values_with_evidence(
    raw_value: Any,
    *,
    field_name: str,
    valid_conversation_ids: Optional[Set[str]] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Extract value/evidence pairs from heterogeneous LLM responses."""
    result: List[Dict[str, Any]] = []
    seen_values: Dict[str, Dict[str, Any]] = {}

    def add_entry(key: Any, evidence_list: Any, level: Any = None) -> None:
        if key is None:
            return
        value_str = key.strip() if isinstance(key, str) else str(key).strip()
        if not value_str:
            return

        evidence_items = ensure_str_list(evidence_list)
        formatted_evidences: List[str] = []
        for item in evidence_items:
            formatted = format_evidence_entry(
                item, conversation_date_map=conversation_date_map
            )
            if not formatted:
                continue
            if valid_conversation_ids is not None:
                conversation_id = conversation_id_from_evidence(formatted)
                if not conversation_id or conversation_id not in valid_conversation_ids:
                    logger.debug(
                        "LLM Generated Unknown Conversation ID %s", conversation_id
                    )
                    continue
            formatted_evidences.append(formatted)

        level_str = (
            level.strip()
            if isinstance(level, str)
            else str(level).strip() if level is not None and level != "" else ""
        )

        if value_str not in seen_values:
            if not formatted_evidences:
                logger.info(
                    "LLM returned value %s for field %s without evidences",
                    value_str,
                    field_name,
                )
            seen_values[value_str] = {"evidences": [], "level": ""}

        entry = seen_values[value_str]
        if level_str:
            entry["level"] = level_str
        for ev in formatted_evidences:
            if ev not in entry["evidences"]:
                entry["evidences"].append(ev)

    if isinstance(raw_value, dict):
        if "value" in raw_value:
            add_entry(
                raw_value.get("value"),
                raw_value.get("evidences"),
                raw_value.get("level"),
            )
        else:
            for key, evidence_list in raw_value.items():
                if key == "evidences":
                    continue
                add_entry(key, evidence_list, raw_value.get("level"))
    elif isinstance(raw_value, list):
        for entry in raw_value:
            if isinstance(entry, dict):
                if "value" in entry:
                    add_entry(
                        entry.get("value"), entry.get("evidences"), entry.get("level")
                    )
                else:
                    evidence_source = (
                        entry.get("evidences") if "evidences" in entry else None
                    )
                    processed = False
                    for key, evidence_list in entry.items():
                        if key == "evidences":
                            continue
                        add_entry(
                            key,
                            (
                                evidence_list
                                if evidence_source is None
                                else evidence_source
                            ),
                            entry.get("level"),
                        )
                        processed = True
                    if not processed:
                        add_entry(entry, None, entry.get("level"))
            elif isinstance(entry, str):
                add_entry(entry, None)
            elif entry is not None:
                add_entry(entry, None)
    elif raw_value is not None:
        add_entry(raw_value, None)

    for value, stored in seen_values.items():
        entry = {"value": value, "evidences": stored.get("evidences", [])}
        level_value = stored.get("level", "")
        if level_value:
            entry["level"] = level_value
        result.append(entry)

    return result or None


__all__ = [
    "merge_value_with_evidences_lists",
    "merge_value_with_evidences_lists_keep_highest_level",
    "extract_values_with_evidence",
]
