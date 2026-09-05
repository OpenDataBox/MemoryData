"""Evidence utilities shared across profile normalization helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from core.observation.logger import get_logger

logger = get_logger(__name__)

ALLOWED_OPINION_TENDENCY_TYPES = {
    "stance",
    "suggestion",
    "his own opinion",
}


def ensure_str_list(value: Any) -> List[str]:
    """Convert arbitrary values into a deduplicated list of stripped strings."""
    if not value:
        return []
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            if item is None:
                continue
            text = item.strip() if isinstance(item, str) else str(item).strip()
            if text and text not in result:
                result.append(text)
        return result
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    text = str(value).strip()
    return [text] if text else []


def filter_opinion_tendency_by_type(entries: Any) -> Any:
    """Filter opinion_tendency items, keeping only allowed type values."""
    if not isinstance(entries, list) or not entries:
        return entries

    filtered: List[Dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        raw_type = item.get("type")
        if raw_type is None:
            continue
        normalized_type = str(raw_type).strip().lower()
        if normalized_type in ALLOWED_OPINION_TENDENCY_TYPES:
            filtered.append(item)
        else:
            logger.info(
                "Removing opinion_tendency item type %s, content=%s, evidences=%s",
                raw_type,
                item.get("value"),
                item.get("evidences"),
            )
    return filtered


def format_evidence_entry(
    value: Any,
    *,
    conversation_date_map: Optional[Dict[str, str]],
) -> Optional[str]:
    """Format evidence entries to include the appropriate date prefix."""
    if value is None:
        return None
    item_str = value.strip() if isinstance(value, str) else str(value).strip()
    if not item_str:
        return None
    if "|" in item_str:
        return item_str

    conversation_id = conversation_id_from_evidence(item_str)
    if conversation_id:
        normalized_key = conversation_id
    elif "conversation_id" in item_str:
        normalized_key = item_str.split("conversation_id:")[-1].strip("[] ") or item_str
    else:
        normalized_key = item_str

    evidence_date: Optional[str] = None
    if conversation_id and conversation_date_map:
        evidence_date = conversation_date_map.get(conversation_id)
    if evidence_date:
        return f"{evidence_date}|{normalized_key}"
    return normalized_key


def conversation_id_from_evidence(evidence: Any) -> Optional[str]:
    """Extract the conversation identifier from a formatted evidence entry."""
    if not isinstance(evidence, str):
        return None
    entry = evidence.strip()
    if not entry:
        return None
    if "|" in entry:
        entry = entry.split("|")[-1].strip()
    if "conversation_id:" in entry:
        entry = entry.split("conversation_id:")[-1]
    return entry.strip("[] ") or None


def _strip_evidences_for_identifier(value: Any) -> Any:
    """Remove evidences recursively for comparison purposes."""
    if isinstance(value, dict):
        return {
            key: _strip_evidences_for_identifier(val)
            for key, val in value.items()
            if key != "evidences"
        }
    if isinstance(value, list):
        return [_strip_evidences_for_identifier(item) for item in value]
    return value


def _build_item_identifier(item: Dict[str, Any]) -> Optional[str]:
    """Generate a structural signature for matching list entries."""
    if not isinstance(item, dict):
        return None
    stripped = _strip_evidences_for_identifier(item)
    if not stripped:
        return None
    try:
        return json.dumps(stripped, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return None


def _find_matching_item(
    items: List[Any],
    completed_item: Any,
) -> Optional[Any]:
    """Locate the list item corresponding to the completed entry."""
    if not isinstance(completed_item, dict):
        return None

    identifier = _build_item_identifier(completed_item)
    if identifier:
        for candidate in items:
            if isinstance(candidate, dict) and _build_item_identifier(candidate) == identifier:
                return candidate

    value_keys = (
        "value",
        "skill",
        "project_id",
        "project_name",
        "user_id",
        "name",
        "title",
    )
    for key in value_keys:
        candidate_value = completed_item.get(key)
        if candidate_value is None or candidate_value == "":
            continue
        normalized_candidate = str(candidate_value).strip()
        if not normalized_candidate:
            continue
        for candidate in items:
            if not isinstance(candidate, dict):
                continue
            existing_value = candidate.get(key)
            if existing_value is None or existing_value == "":
                continue
            if str(existing_value).strip() == normalized_candidate:
                return candidate

    return None


def _format_and_validate_evidences(
    evidences: Any,
    *,
    valid_conversation_ids: Optional[Set[str]],
    conversation_date_map: Optional[Dict[str, str]],
) -> List[str]:
    """Format evidences into the expected YYYY-MM-DD|conversation_id structure."""
    formatted: List[str] = []
    for evidence in ensure_str_list(evidences):
        candidate = evidence.strip()
        if not candidate:
            continue
        conversation_id = conversation_id_from_evidence(candidate) or candidate
        if (
            valid_conversation_ids is not None
            and conversation_id
            and conversation_id not in valid_conversation_ids
        ):
            logger.warning(
                "Evidence completion produced unknown conversation ID %s",
                conversation_id,
            )
            continue
        formatted_entry = format_evidence_entry(
            conversation_id,
            conversation_date_map=conversation_date_map,
        )
        if formatted_entry and formatted_entry not in formatted:
            formatted.append(formatted_entry)
    return formatted


def merge_evidences_recursive(
    original: Any,
    completed: Any,
    *,
    valid_conversation_ids: Optional[Set[str]],
    conversation_date_map: Optional[Dict[str, str]],
    path: str = "user_profile",
) -> None:
    """Recursively merge evidences from the completed payload into the original."""
    if isinstance(original, dict) and isinstance(completed, dict):
        if "evidences" in completed and isinstance(completed["evidences"], list):
            formatted = _format_and_validate_evidences(
                completed["evidences"],
                valid_conversation_ids=valid_conversation_ids,
                conversation_date_map=conversation_date_map,
            )
            if formatted:
                original["evidences"] = formatted
                logger.info(
                    "Added %d evidence(s) to path: %s",
                    len(formatted),
                    path,
                )
        for key, value in completed.items():
            if key == "evidences":
                continue
            if key in original:
                merge_evidences_recursive(
                    original[key],
                    value,
                    valid_conversation_ids=valid_conversation_ids,
                    conversation_date_map=conversation_date_map,
                    path=f"{path}.{key}",
                )
        return

    if isinstance(original, list) and isinstance(completed, list):
        for idx, completed_item in enumerate(completed):
            target_item = _find_matching_item(original, completed_item)
            if target_item is None:
                continue
            target_idx = original.index(target_item)
            merge_evidences_recursive(
                target_item,
                completed_item,
                valid_conversation_ids=valid_conversation_ids,
                conversation_date_map=conversation_date_map,
                path=f"{path}[{target_idx}]",
            )

def remove_entries_without_evidence(payload: Any, *, path: str = "user_profile") -> Any:
    """
    Recursively remove entries that lack evidences after completion.

    Args:
        payload: Arbitrary profile payload structure.
        path: Logical path for debugging output.

    Returns:
        The sanitized payload. Returns None when a branch should be removed.
    """
    if isinstance(payload, dict):
        for key in list(payload.keys()):
            if key == "evidences":
                continue
            cleaned = remove_entries_without_evidence(
                payload[key], path=f"{path}.{key}"
            )
            if cleaned is None:
                payload.pop(key, None)
            else:
                payload[key] = cleaned

        if "evidences" in payload:
            normalized = ensure_str_list(payload["evidences"])
            if not normalized:
                logger.debug("Removing entry at %s due to empty evidences", path)
                return None
            payload["evidences"] = normalized

        if not payload:
            return None
        return payload

    if isinstance(payload, list):
        sanitized: List[Any] = []
        for index, item in enumerate(payload):
            cleaned = remove_entries_without_evidence(item, path=f"{path}[{index}]")
            if cleaned is None:
                continue
            sanitized.append(cleaned)
        return sanitized

    return payload

__all__ = [
    "ensure_str_list",
    "filter_opinion_tendency_by_type",
    "format_evidence_entry",
    "conversation_id_from_evidence",
    "merge_evidences_recursive",
]
