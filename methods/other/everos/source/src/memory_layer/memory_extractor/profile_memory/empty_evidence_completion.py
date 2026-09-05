"""Evidence completion utilities for profile memory extraction."""

from __future__ import annotations

from typing import AbstractSet, Any, Callable, Dict, List, Optional, Set, Tuple

from core.observation.logger import get_logger

from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.profile_memory.conversation import build_evidence_completion_prompt
from memory_layer.memory_extractor.profile_memory.data_normalize import merge_evidences_recursive
from memory_layer.memory_extractor.profile_memory.evidence_utils import conversation_id_from_evidence, ensure_str_list, format_evidence_entry

logger = get_logger(__name__)

PROJECTS_PARTICIPATED_NESTED_FIELDS = {
    "subtasks",
    "user_objective",
    "contributions",
    "user_concerns",
}


def _has_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float, bool)):
        return bool(value)
    if isinstance(value, list):
        return any(_has_non_empty_value(item) for item in value)
    if isinstance(value, dict):
        return any(_has_non_empty_value(val) for val in value.values())
    return True


def _project_nested_item_has_missing_evidence(item: Any) -> bool:
    if item is None:
        return False
    if isinstance(item, dict):
        evidences = item.get("evidences")
        if evidences:
            return False
        if "value" in item:
            return _has_non_empty_value(item.get("value"))
        return any(
            _project_nested_item_has_missing_evidence(val) for val in item.values()
        )
    if isinstance(item, list):
        return any(_project_nested_item_has_missing_evidence(val) for val in item)
    if isinstance(item, str):
        return bool(item.strip())
    return bool(item)


async def complete_missing_evidences(
    profiles: List[Dict[str, Any]],
    *,
    conversation_lines: List[str],
    valid_conversation_ids: Optional[Set[str]],
    conversation_participants_map: Optional[Dict[str, Optional[AbstractSet[str]]]],
    conversation_date_map: Optional[Dict[str, str]],
    llm_provider: Optional[LLMProvider],
    parse_payload: Callable[[str], Any],
) -> None:
    """Supplement missing evidences for a batch of profiles using the LLM."""
    if not profiles or not llm_provider:
        return

    conversation_text = "\n".join(conversation_lines)
    completion_targets: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        if valid_conversation_ids is not None:
            _remove_invalid_evidences(
                profile,
                valid_conversation_ids=valid_conversation_ids,
                conversation_participants_map=conversation_participants_map,
            )
        payload = _extract_missing_evidences_payload(profile)
        if payload:
            completion_targets.append((profile, payload))

    if not completion_targets:
        return

    batch_payload = [payload for _, payload in completion_targets]
    prompt = build_evidence_completion_prompt(conversation_text, batch_payload)

    extraction_attempts = 2
    response_text: Optional[str] = None
    parsed_payload: Optional[Any] = None

    for attempt in range(extraction_attempts):
        try:
            response_text = await llm_provider.generate(prompt, temperature=0.2)
            parsed_payload = parse_payload(response_text)
            break
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Evidence completion batch failed (attempt %s/%s): %s",
                attempt + 1,
                extraction_attempts,
                exc,
            )
            if response_text:
                logger.warning(
                    "Evidence completion response preview (attempt %s): %s",
                    attempt + 1,
                    response_text[:800],
                )

            if attempt < extraction_attempts - 1:
                response_text = None
                parsed_payload = None
                continue

            repair_prompt = (
                "The input string is in json format, but has syntax errors. Please fix the syntax errors and output only the correctly formatted json ```json {}```. "
                "Do not include any additional explanations or notes.\n Original string:\n"
                + (response_text or "")
            )
            try:
                response_text = await llm_provider.generate(
                    repair_prompt, temperature=0
                )
                parsed_payload = parse_payload(response_text)
                break
            except Exception as repair_exc:  # pylint: disable=broad-except
                logger.error(
                    "Evidence completion repair attempt failed: %s", repair_exc
                )
                if response_text:
                    logger.error(
                        "Evidence completion repair response preview: %s",
                        response_text[:500],
                    )
                return

    if parsed_payload is None:
        return

    if isinstance(parsed_payload, dict):
        completed_profiles = parsed_payload.get("user_profiles")
        if not isinstance(completed_profiles, list):
            completed_profiles = parsed_payload.get("user_profile")
            if isinstance(completed_profiles, dict):
                completed_profiles = [completed_profiles]
    elif isinstance(parsed_payload, list):
        completed_profiles = parsed_payload
    else:
        completed_profiles = None

    if not completed_profiles:
        logger.warning("Evidence completion batch returned empty payload")
        return

    completed_map: Dict[str, Dict[str, Any]] = {}
    for item in completed_profiles:
        if not isinstance(item, dict):
            continue
        user_id = str(item.get("user_id", "")).strip()
        if not user_id:
            continue
        if valid_conversation_ids is not None:
            _remove_invalid_evidences(
                item,
                valid_conversation_ids=valid_conversation_ids,
                conversation_participants_map=conversation_participants_map,
            )
        _format_evidences_with_dates(item, conversation_date_map=conversation_date_map)
        completed_map[user_id] = item

    for original_profile, payload in completion_targets:
        user_id = str(payload.get("user_id", "")).strip()
        completed_profile = completed_map.get(user_id)
        if not completed_profile:
            logger.warning(
                "Evidence completion response missing profile for user %s",
                user_id or "<unknown>",
            )
            continue

        _merge_completed_evidences(
            original_profile,
            completed_profile,
            valid_conversation_ids=valid_conversation_ids,
            conversation_date_map=conversation_date_map,
        )


def _extract_missing_evidences_payload(
    profile: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Return a minimal payload containing only entries lacking evidences."""
    if not profile:
        return None

    user_id = profile.get("user_id")
    if not user_id:
        return None

    evidence_fields = {
        "hard_skills",
        "soft_skills",
        "motivation_system",
        "fear_system",
        "value_system",
        "humor_use",
        "colloquialism",
        "way_of_decision_making",
        "personality",
        "projects_participated",
        "user_goal",
        "role_responsibility",
        "work_responsibility",
        "working_habit_preference",
        "interests",
        "opinion_tendency",
        "tendency",
    }

    payload: Dict[str, Any] = {"user_id": user_id}
    if profile.get("user_name"):
        payload["user_name"] = profile["user_name"]

    for field in evidence_fields:
        pruned = _prune_missing_evidences(profile.get(field), root_field=field)
        if pruned:
            payload[field] = pruned

    extra_keys = [key for key in payload.keys() if key not in {"user_id", "user_name"}]
    return payload if extra_keys else None


def _remove_invalid_evidences(
    profile: Dict[str, Any],
    *,
    valid_conversation_ids: Optional[Set[str]],
    conversation_participants_map: Optional[Dict[str, Optional[AbstractSet[str]]]],
) -> None:
    """Remove evidences that reference unknown conversation IDs or mismatched participants."""
    if not profile or valid_conversation_ids is None:
        return

    user_id = (
        str(profile.get("user_id", "")).strip() if isinstance(profile, dict) else ""
    )

    def sanitize(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "evidences":
                    evidences_list = ensure_str_list(value)
                    filtered: List[str] = []
                    for text in evidences_list:
                        if not text:
                            continue
                        conversation_id = conversation_id_from_evidence(text) or text
                        should_remove = False
                        if conversation_id not in valid_conversation_ids:
                            should_remove = True
                            logger.debug(
                                "Removing hallucinated evidence %s for user %s",
                                conversation_id or "<unknown>",
                                user_id or "<unknown>",
                            )
                        elif conversation_participants_map is not None:
                            participants = conversation_participants_map.get(
                                conversation_id
                            )
                            if participants is None:
                                should_remove = True
                                logger.debug(
                                    "Removing evidence %s for user %s: participants not available",
                                    conversation_id or "<unknown>",
                                    user_id or "<unknown>",
                                )
                            elif user_id not in participants:
                                should_remove = True
                                logger.debug(
                                    "Removing evidence %s for user %s not in participants",
                                    conversation_id or "<unknown>",
                                    user_id or "<unknown>",
                                )
                        if not should_remove and text not in filtered:
                            filtered.append(text)
                    node[key] = filtered
                else:
                    sanitize(value)
        elif isinstance(node, list):
            for item in node:
                sanitize(item)

    sanitize(profile)


def _format_evidences_with_dates(
    profile: Dict[str, Any], *, conversation_date_map: Optional[Dict[str, str]]
) -> None:
    """Format all evidence entries to include conversation dates when available."""
    if not profile:
        return

    def format_node(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "evidences":
                    evidences_list = ensure_str_list(value)
                    formatted: List[str] = []
                    seen: Set[str] = set()
                    for evidence in evidences_list:
                        if not evidence:
                            continue
                        conversation_id = (
                            conversation_id_from_evidence(evidence) or evidence
                        )
                        formatted_entry = format_evidence_entry(
                            conversation_id, conversation_date_map=conversation_date_map
                        )
                        if formatted_entry and formatted_entry not in seen:
                            formatted.append(formatted_entry)
                            seen.add(formatted_entry)
                    node[key] = formatted
                else:
                    format_node(value)
        elif isinstance(node, list):
            for item in node:
                format_node(item)

    format_node(profile)


def _prune_missing_evidences(
    value: Any, *, root_field: Optional[str] = None, parent_key: Optional[str] = None
) -> Optional[Any]:
    """Recursively retain only segments where evidences are absent."""
    if isinstance(value, dict):
        if "evidences" in value:
            evidences = value.get("evidences")
            if evidences:
                return None
            result = {}
            for key, val in value.items():
                if key == "evidences":
                    result[key] = []
                else:
                    result[key] = val
            return result

        result_dict: Dict[str, Any] = {}
        for key, item in value.items():
            pruned_item = _prune_missing_evidences(
                item, root_field=root_field, parent_key=key
            )
            if pruned_item is not None:
                result_dict[key] = pruned_item
        if not result_dict:
            return None

        if root_field == "projects_participated" and any(
            key in value for key in PROJECTS_PARTICIPATED_NESTED_FIELDS
        ):
            has_pending_nested = any(
                _project_nested_item_has_missing_evidence(result_dict.get(sub_key))
                for sub_key in PROJECTS_PARTICIPATED_NESTED_FIELDS
            )
            if not has_pending_nested:
                return None

        return result_dict

    if isinstance(value, list):
        result_list: List[Any] = []
        for item in value:
            pruned_item = _prune_missing_evidences(
                item, root_field=root_field, parent_key=parent_key
            )
            if pruned_item is not None:
                if (
                    root_field == "projects_participated"
                    and parent_key in PROJECTS_PARTICIPATED_NESTED_FIELDS
                ):
                    if not _project_nested_item_has_missing_evidence(pruned_item):
                        continue
                result_list.append(pruned_item)
        return result_list or None

    return value


def _merge_completed_evidences(
    original_profile: Dict[str, Any],
    completed_profile: Dict[str, Any],
    *,
    valid_conversation_ids: Optional[Set[str]],
    conversation_date_map: Optional[Dict[str, str]],
) -> None:
    """Overlay completed evidences back onto the original profile."""
    if not original_profile or not completed_profile:
        return

    merge_evidences_recursive(
        original_profile,
        completed_profile,
        valid_conversation_ids=valid_conversation_ids,
        conversation_date_map=conversation_date_map,
    )
