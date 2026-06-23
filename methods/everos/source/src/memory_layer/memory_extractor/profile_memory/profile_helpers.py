"""Profile-level helpers: payload conversion, accumulation, and merging."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from core.observation.logger import get_logger

from api_specs.memory_types import BaseMemory, MemoryType, RawDataType
from memory_layer.memory_extractor.profile_memory.project_helpers import (
    convert_projects_to_dataclass,
    merge_projects_participated,
    project_to_dict,
)
from memory_layer.memory_extractor.profile_memory.skill_helpers import merge_skill_lists, normalize_skills_with_evidence
from memory_layer.memory_extractor.profile_memory.types import ProfileMemory
from memory_layer.memory_extractor.profile_memory.value_helpers import (
    extract_values_with_evidence,
    merge_value_with_evidences_lists,
)

logger = get_logger(__name__)


def remove_evidences_from_profile(profile_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively remove evidence fields to keep prompts concise."""

    def strip_content(content: Any) -> Any:
        if isinstance(content, dict):
            return {
                key: strip_content(value)
                for key, value in content.items()
                if key != "evidences"
            }
        if isinstance(content, list):
            return [strip_content(item) for item in content]
        return content

    result: Dict[str, Any] = {}
    for key, value in profile_obj.items():
        if key in {"evidences", "output_reasoning"}:
            continue
        result[key] = strip_content(value)
    return result


def accumulate_old_memory_entry(
    memory: BaseMemory, participants_profile_list: List[Dict[str, Any]]
) -> None:
    """Convert legacy BaseMemory objects into prompt-ready dictionaries."""
    try:
        if memory.memory_type != MemoryType.PROFILE:
            return

        profile_obj: Dict[str, Any] = {"user_id": memory.user_id}

        if getattr(memory, "user_name", None):
            profile_obj["user_name"] = memory.user_name

        hard_skills = getattr(memory, "hard_skills", None)
        if hard_skills:
            profile_obj["hard_skills"] = hard_skills

        soft_skills = getattr(memory, "soft_skills", None)
        if soft_skills:
            profile_obj["soft_skills"] = soft_skills

        for field_name in (
            "motivation_system",
            "fear_system",
            "value_system",
            "humor_use",
            "colloquialism",
        ):
            value = getattr(memory, field_name, None)
            if value:
                profile_obj[field_name] = value

        for field_name in (
            "way_of_decision_making",
            "personality",
            "user_goal",
            "work_responsibility",
            "working_habit_preference",
            "interests",
            "tendency",
        ):
            value = getattr(memory, field_name, None)
            if value:
                profile_obj[field_name] = value

        projects = getattr(memory, "projects_participated", None)
        if projects:
            project_payload = [
                project_to_dict(project) for project in projects if project is not None
            ]
            if project_payload:
                profile_obj["projects_participated"] = project_payload

        if len(profile_obj) > 1:
            participants_profile_list.append(profile_obj)

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to extract old memory entry: %s", exc)


def profile_payload_to_memory(
    profile_data: Dict[str, Any],
    *,
    group_id: str,
    project_data: Optional[Dict[str, Any]] = None,
    valid_conversation_ids: Optional[Set[str]] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> Optional[ProfileMemory]:
    """Convert LLM payloads into ProfileMemory instances."""
    if not isinstance(profile_data, dict):
        return None

    extracted_user_id = str(profile_data.get("user_id", "")).strip()
    extracted_user_name = profile_data.get("user_name", "")
    if not extracted_user_id:
        logger.debug(
            "LLM generated user %s has no user_id, skipping", extracted_user_name
        )
        return None

    hard_skills = normalize_skills_with_evidence(
        profile_data.get("hard_skills"),
        field_name="hard_skills",
        valid_conversation_ids=valid_conversation_ids,
        conversation_date_map=conversation_date_map,
    )

    soft_skills = normalize_skills_with_evidence(
        profile_data.get("soft_skills"),
        field_name="soft_skills",
        valid_conversation_ids=valid_conversation_ids,
        conversation_date_map=conversation_date_map,
    )
    output_reasoning_raw = profile_data.get("output_reasoning")
    output_reasoning: Optional[str] = None
    if output_reasoning_raw is not None:
        output_reasoning = str(output_reasoning_raw).strip() or None

    def extract(field: str) -> Optional[List[Dict[str, Any]]]:
        return extract_values_with_evidence(
            profile_data.get(field),
            field_name=field,
            valid_conversation_ids=valid_conversation_ids,
            conversation_date_map=conversation_date_map,
        )

    motivation_values = extract("motivation_system")
    fear_values = extract("fear_system")
    value_system_values = extract("value_system")
    humor_values = extract("humor_use")
    colloquialism_values = extract("colloquialism")

    work_responsibility_values = extract_values_with_evidence(
        profile_data.get("role_responsibility"),
        field_name="role_responsibility",
        valid_conversation_ids=valid_conversation_ids,
        conversation_date_map=conversation_date_map,
    )
    user_goal_values = extract("user_goal")
    working_habit_values = extract("working_habit_preference")
    interests_values = extract("interests")

    tendency_source = profile_data.get("opinion_tendency")
    tendency_field_name = "opinion_tendency"
    if tendency_source is None:
        tendency_source = profile_data.get("tendency")
        tendency_field_name = "tendency"
    tendency_values = extract_values_with_evidence(
        tendency_source,
        field_name=tendency_field_name,
        valid_conversation_ids=valid_conversation_ids,
        conversation_date_map=conversation_date_map,
    )

    personality_values = extract("personality")
    way_of_decision_values = extract("way_of_decision_making")

    if project_data is not None:
        projects_participated = convert_projects_to_dataclass(
            project_data.get("projects_participated", []),
            valid_conversation_ids=valid_conversation_ids,
            conversation_date_map=conversation_date_map,
        )
    else:
        projects_participated = convert_projects_to_dataclass(
            profile_data.get("projects_participated", []),
            valid_conversation_ids=valid_conversation_ids,
            conversation_date_map=conversation_date_map,
        )

    if not (
        hard_skills
        or soft_skills
        or output_reasoning
        or motivation_values
        or fear_values
        or value_system_values
        or humor_values
        or colloquialism_values
        or way_of_decision_values
        or personality_values
        or projects_participated
        or user_goal_values
        or working_habit_values
        or interests_values
        or tendency_values
        or work_responsibility_values
    ):
        return None

    return ProfileMemory(
        memory_type=MemoryType.PROFILE,
        user_id=extracted_user_id,
        timestamp="",
        ori_event_id_list=[],
        group_id=group_id,
        user_name=extracted_user_name,
        hard_skills=hard_skills or None,
        soft_skills=soft_skills or None,
        output_reasoning=output_reasoning,
        motivation_system=motivation_values or None,
        fear_system=fear_values or None,
        value_system=value_system_values or None,
        humor_use=humor_values or None,
        colloquialism=colloquialism_values or None,
        way_of_decision_making=way_of_decision_values or None,
        personality=personality_values or None,
        projects_participated=projects_participated or None,
        user_goal=user_goal_values or None,
        work_responsibility=work_responsibility_values,
        working_habit_preference=working_habit_values or None,
        interests=interests_values or None,
        tendency=tendency_values or None,
        type=RawDataType.CONVERSATION,
    )


def merge_single_profile(
    existing: ProfileMemory, new: ProfileMemory, *, group_id: str
) -> ProfileMemory:
    """Merge two ProfileMemory objects with the same user id."""
    merged_hard_skills = merge_skill_lists(existing.hard_skills, new.hard_skills)
    merged_soft_skills = merge_skill_lists(existing.soft_skills, new.soft_skills)

    merged_value_fields = _merge_value_fields(
        existing,
        new,
        field_names=(
            "motivation_system",
            "fear_system",
            "value_system",
            "humor_use",
            "colloquialism",
            "way_of_decision_making",
            "personality",
            "user_goal",
            "work_responsibility",
            "working_habit_preference",
            "interests",
            "tendency",
        ),
    )

    merged_projects = merge_projects_participated(
        existing.projects_participated, new.projects_participated
    )

    output_reasoning = (
        new.output_reasoning
        if new.output_reasoning is not None
        else existing.output_reasoning
    )

    return ProfileMemory(
        memory_type=MemoryType.PROFILE,
        user_id=existing.user_id,
        timestamp=new.timestamp or existing.timestamp,
        ori_event_id_list=new.ori_event_id_list or existing.ori_event_id_list,
        user_name=new.user_name or existing.user_name,
        group_id=group_id or new.group_id or existing.group_id,
        hard_skills=merged_hard_skills,
        soft_skills=merged_soft_skills,
        output_reasoning=output_reasoning,
        motivation_system=merged_value_fields.get("motivation_system"),
        fear_system=merged_value_fields.get("fear_system"),
        value_system=merged_value_fields.get("value_system"),
        humor_use=merged_value_fields.get("humor_use"),
        colloquialism=merged_value_fields.get("colloquialism"),
        way_of_decision_making=merged_value_fields.get("way_of_decision_making"),
        personality=merged_value_fields.get("personality"),
        projects_participated=merged_projects or None,
        user_goal=merged_value_fields.get("user_goal"),
        work_responsibility=merged_value_fields.get("work_responsibility"),
        working_habit_preference=merged_value_fields.get("working_habit_preference"),
        interests=merged_value_fields.get("interests"),
        tendency=merged_value_fields.get("tendency"),
        type=RawDataType.CONVERSATION,
    )


def merge_profiles(
    profile_memories: Iterable[ProfileMemory],
    participants_profile_list: Iterable[Dict[str, Any]],
    *,
    group_id: str,
    valid_conversation_ids: Optional[Set[str]] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> List[ProfileMemory]:
    """Merge extracted profiles with existing participant profiles."""
    merged_dict: Dict[str, ProfileMemory] = {}

    for participant_profile in participants_profile_list:
        user_id = participant_profile.get("user_id")
        if not user_id:
            continue

        profile_memory = ProfileMemory(
            memory_type=MemoryType.PROFILE,
            user_id=user_id,
            timestamp="",
            ori_event_id_list=[],
            group_id=group_id,
            user_name=participant_profile.get("user_name"),
            hard_skills=normalize_skills_with_evidence(
                participant_profile.get("hard_skills"),
                field_name="hard_skills",
                valid_conversation_ids=None,
                conversation_date_map=None,
            )
            or None,
            soft_skills=normalize_skills_with_evidence(
                participant_profile.get("soft_skills"),
                field_name="soft_skills",
                valid_conversation_ids=None,
                conversation_date_map=None,
            )
            or None,
            motivation_system=participant_profile.get("motivation_system"),
            fear_system=participant_profile.get("fear_system"),
            value_system=participant_profile.get("value_system"),
            humor_use=participant_profile.get("humor_use"),
            colloquialism=participant_profile.get("colloquialism"),
            way_of_decision_making=participant_profile.get("way_of_decision_making"),
            personality=participant_profile.get("personality"),
            projects_participated=convert_projects_to_dataclass(
                participant_profile.get("projects_participated", []),
                valid_conversation_ids=None,
                conversation_date_map=None,
            )
            or None,
            user_goal=participant_profile.get("user_goal"),
            work_responsibility=participant_profile.get("work_responsibility"),
            working_habit_preference=participant_profile.get(
                "working_habit_preference"
            ),
            interests=participant_profile.get("interests"),
            tendency=participant_profile.get("tendency"),
            type=RawDataType.CONVERSATION,
        )
        merged_dict[user_id] = profile_memory

    for new_profile in profile_memories:
        user_id = new_profile.user_id
        if user_id in merged_dict:
            existing_profile = merged_dict[user_id]
            merged_dict[user_id] = merge_single_profile(
                existing_profile, new_profile, group_id=group_id
            )
        else:
            merged_dict[user_id] = new_profile

    return list(merged_dict.values())


def _merge_value_fields(
    existing: ProfileMemory, new: ProfileMemory, *, field_names: Iterable[str]
) -> Dict[str, Optional[List[Dict[str, Any]]]]:
    """Merge multiple value-based fields and return a mapping."""
    merged: Dict[str, Optional[List[Dict[str, Any]]]] = {}
    for field in field_names:
        merged[field] = merge_value_with_evidences_lists(
            getattr(existing, field, None), getattr(new, field, None)
        )
    return merged


__all__ = [
    "remove_evidences_from_profile",
    "accumulate_old_memory_entry",
    "profile_payload_to_memory",
    "merge_single_profile",
    "merge_profiles",
]
