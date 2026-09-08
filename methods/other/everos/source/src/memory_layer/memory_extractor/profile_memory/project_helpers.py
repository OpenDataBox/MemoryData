"""Project-related normalization helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set

from core.observation.logger import get_logger

from memory_layer.memory_extractor.profile_memory.types import ProjectInfo
from memory_layer.memory_extractor.profile_memory.value_helpers import (
    extract_values_with_evidence,
    merge_value_with_evidences_lists,
)

logger = get_logger(__name__)


def project_to_dict(project: ProjectInfo | Dict[str, Any]) -> Dict[str, Any]:
    """Serialize ProjectInfo for prompt payloads."""
    if isinstance(project, ProjectInfo):
        return {
            "project_id": project.project_id,
            "project_name": project.project_name,
            "entry_date": project.entry_date,
            "subtasks": project.subtasks or [],
            "user_objective": project.user_objective or [],
            "contributions": project.contributions or [],
            "user_concerns": project.user_concerns or [],
        }
    return {
        "project_id": project.get("project_id", ""),
        "project_name": project.get("project_name", ""),
        "entry_date": project.get("entry_date", ""),
        "subtasks": project.get("subtasks", []),
        "user_objective": project.get("user_objective", []),
        "contributions": project.get("contributions", []),
        "user_concerns": project.get("user_concerns", []),
    }


def convert_projects_to_dataclass(
    projects_data: Iterable[Any],
    *,
    valid_conversation_ids: Optional[Set[str]] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> List[ProjectInfo]:
    """Convert project payloads into ProjectInfo dataclasses."""
    projects: List[ProjectInfo] = []
    for project_data in projects_data:
        if isinstance(project_data, ProjectInfo):
            projects.append(
                ProjectInfo(
                    project_id=project_data.project_id,
                    project_name=project_data.project_name,
                    entry_date=project_data.entry_date,
                    subtasks=(
                        list(project_data.subtasks) if project_data.subtasks else None
                    ),
                    user_objective=(
                        list(project_data.user_objective)
                        if project_data.user_objective
                        else None
                    ),
                    contributions=(
                        list(project_data.contributions)
                        if project_data.contributions
                        else None
                    ),
                    user_concerns=(
                        list(project_data.user_concerns)
                        if project_data.user_concerns
                        else None
                    ),
                )
            )
            continue
        if not isinstance(project_data, dict):
            continue

        project_id = str(project_data.get("project_id") or "").strip()
        project_name = str(project_data.get("project_name") or "").strip()
        entry_date = _normalize_entry_date(project_data.get("entry_date"))

        subtasks = _normalize_project_field(
            project_data.get("subtasks"),
            field_name="subtasks",
            valid_conversation_ids=valid_conversation_ids,
            conversation_date_map=conversation_date_map,
        )
        user_objective = _normalize_project_field(
            project_data.get("user_objective"),
            field_name="user_objective",
            valid_conversation_ids=valid_conversation_ids,
            conversation_date_map=conversation_date_map,
        )
        contributions = _normalize_project_field(
            project_data.get("contributions"),
            field_name="contributions",
            valid_conversation_ids=valid_conversation_ids,
            conversation_date_map=conversation_date_map,
        )
        user_concerns = _normalize_project_field(
            project_data.get("user_concerns"),
            field_name="user_concerns",
            valid_conversation_ids=valid_conversation_ids,
            conversation_date_map=conversation_date_map,
        )

        projects.append(
            ProjectInfo(
                project_id=project_id,
                project_name=project_name,
                entry_date=entry_date,
                subtasks=subtasks or None,
                user_objective=user_objective or None,
                contributions=contributions or None,
                user_concerns=user_concerns or None,
            )
        )
    return projects


def merge_projects_participated(
    existing_projects: Optional[List[ProjectInfo]],
    incoming_projects: Optional[List[ProjectInfo]],
) -> List[ProjectInfo]:
    """Merge project participation lists, deduplicating by project id/name."""

    def clone_project(project: ProjectInfo) -> ProjectInfo:
        return ProjectInfo(
            project_id=project.project_id,
            project_name=project.project_name,
            entry_date=project.entry_date,
            subtasks=list(project.subtasks) if project.subtasks else None,
            user_objective=(
                list(project.user_objective) if project.user_objective else None
            ),
            contributions=(
                list(project.contributions) if project.contributions else None
            ),
            user_concerns=(
                list(project.user_concerns) if project.user_concerns else None
            ),
        )

    merged_projects: List[ProjectInfo] = [
        clone_project(project) for project in existing_projects or []
    ]

    for project in incoming_projects or []:
        match: Optional[ProjectInfo] = None
        for existing_project in merged_projects:
            if project.project_id and existing_project.project_id:
                if project.project_id == existing_project.project_id:
                    match = existing_project
                    break
            elif project.project_name and existing_project.project_name:
                if project.project_name == existing_project.project_name:
                    match = existing_project
                    break

        if match:
            match.entry_date = match.entry_date or project.entry_date
            match.subtasks = merge_value_with_evidences_lists(
                match.subtasks, project.subtasks
            )
            match.user_objective = merge_value_with_evidences_lists(
                match.user_objective, project.user_objective
            )
            match.contributions = merge_value_with_evidences_lists(
                match.contributions, project.contributions
            )
            match.user_concerns = merge_value_with_evidences_lists(
                match.user_concerns, project.user_concerns
            )
        else:
            merged_projects.append(clone_project(project))

    return merged_projects


def _normalize_project_field(
    value: Any,
    *,
    field_name: str,
    valid_conversation_ids: Optional[Set[str]],
    conversation_date_map: Optional[Dict[str, str]],
) -> List[Dict[str, Any]]:
    if isinstance(value, list) and value:
        return (
            extract_values_with_evidence(
                value,
                field_name=field_name,
                valid_conversation_ids=valid_conversation_ids,
                conversation_date_map=conversation_date_map,
            )
            or []
        )
    return []


def _normalize_entry_date(value: Any) -> str:
    """Return a YYYY-MM-DD date string or empty string when invalid."""
    if value is None:
        return ""
    entry_date = str(value).strip()
    if not entry_date:
        return ""
    try:
        datetime.strptime(entry_date, "%Y-%m-%d")
    except ValueError:
        logger.debug("Invalid entry_date `%s`; resetting to empty", value)
        return ""
    return entry_date


def filter_project_items_by_type(
    projects_participated: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """
    Filter subtasks and contributions in projects_participated by type.

    For subtasks: only keep items with type='taskbyhimself'
    For contributions: only keep items with type='result'

    Args:
        projects_participated: List of project dictionaries

    Returns:
        Filtered list of projects with only relevant types in subtasks and contributions
    """
    if not projects_participated:
        return projects_participated

    filtered_projects = []
    for project in projects_participated:
        if not isinstance(project, dict):
            filtered_projects.append(project)
            continue

        filtered_project = project.copy()
        project_id = project.get("project_id", "")
        project_name = project.get("project_name", "")

        # Filter subtasks - only keep type='taskbyhimself'
        subtasks = project.get("subtasks")
        if isinstance(subtasks, list) and subtasks:
            filtered_subtasks = []
            for item in subtasks:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "taskbyhimself":
                        filtered_subtasks.append(item)
                    else:
                        # Log removed items
                        logger.info(
                            "Removing subtask from '%s'(%s): value=%s, type=%s, evidences=%s",
                            project_name,
                            project_id,
                            item.get("value"),
                            item_type,
                            item.get("evidences"),
                        )
            filtered_project["subtasks"] = (
                filtered_subtasks if filtered_subtasks else None
            )

        # Filter contributions - only keep type='result'
        contributions = project.get("contributions")
        if isinstance(contributions, list) and contributions:
            filtered_contributions = []
            for item in contributions:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "result":
                        filtered_contributions.append(item)
                    else:
                        # Log removed items
                        logger.info(
                            "Removing contribution from '%s'(%s): value=%s, type=%s, evidences=%s",
                            project_name,
                            project_id,
                            item.get("value"),
                            item_type,
                            item.get("evidences"),
                        )
            filtered_project["contributions"] = (
                filtered_contributions if filtered_contributions else None
            )

        filtered_projects.append(filtered_project)

    return filtered_projects


__all__ = [
    "project_to_dict",
    "convert_projects_to_dataclass",
    "merge_projects_participated",
    "filter_project_items_by_type",
]
