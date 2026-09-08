"""Dataclasses and type definitions for profile memory extraction."""

from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, List, Optional

from api_specs.memory_types import BaseMemory, MemoryType, MemCell
from memory_layer.memory_extractor.base_memory_extractor import MemoryExtractRequest


@dataclass
class ProjectInfo:
    """Project participation information."""

    project_id: str
    project_name: str
    entry_date: str
    subtasks: Optional[List[Dict[str, Any]]] = None
    user_objective: Optional[List[Dict[str, Any]]] = None
    contributions: Optional[List[Dict[str, Any]]] = None
    user_concerns: Optional[List[Dict[str, Any]]] = None


@dataclass
class ImportanceEvidence:
    """Aggregated evidence indicating user importance within a group."""

    user_id: str
    group_id: str
    speak_count: int = 0
    refer_count: int = 0
    conversation_count: int = 0


@dataclass
class GroupImportanceEvidence:
    """Group-level importance assessment for a user."""

    group_id: str
    evidence_list: List[ImportanceEvidence]
    is_important: bool


@dataclass
class ProfileMemory(BaseMemory):
    """
    Profile memory result class.

    Contains user profile information extracted from conversations.
    All list attributes now contain dicts with 'value' and 'evidences' fields.
    """

    user_name: Optional[str] = None

    # Skills: [{"value": "Python", "level": "高级", "evidences": ["2024-01-01|conv_123"]}]
    # Legacy format: [{"skill": "Python", "level": "高级", "evidences": ["..."]}]
    hard_skills: Optional[List[Dict[str, Any]]] = None
    soft_skills: Optional[List[Dict[str, Any]]] = None

    output_reasoning: Optional[str] = None

    # Other attributes: [{"value": "xxx", "evidences": ["2024-01-01|conv_123"]}]
    way_of_decision_making: Optional[List[Dict[str, Any]]] = None
    personality: Optional[List[Dict[str, Any]]] = None
    projects_participated: Optional[List[ProjectInfo]] = None
    user_goal: Optional[List[Dict[str, Any]]] = None
    work_responsibility: Optional[List[Dict[str, Any]]] = None
    working_habit_preference: Optional[List[Dict[str, Any]]] = None
    interests: Optional[List[Dict[str, Any]]] = None
    tendency: Optional[List[Dict[str, Any]]] = None

    # Motivational attributes: [{"value": "achievement", "level": "high", "evidences": ["2024-01-01|conv_123"]}]
    motivation_system: Optional[List[Dict[str, Any]]] = None
    fear_system: Optional[List[Dict[str, Any]]] = None
    value_system: Optional[List[Dict[str, Any]]] = None
    humor_use: Optional[List[Dict[str, Any]]] = None
    colloquialism: Optional[List[Dict[str, Any]]] = None

    group_importance_evidence: Optional[GroupImportanceEvidence] = None

    def __post_init__(self) -> None:
        """Ensure the memory type is set to PROFILE."""
        self.memory_type = MemoryType.PROFILE

    def to_dict(self) -> Dict[str, Any]:
        """Override to_dict() to include all fields of ProfileMemory"""
        # First get the base class fields
        base_dict = super().to_dict()

        # Add ProfileMemory specific fields
        base_dict.update(
            {
                "user_name": self.user_name,
                "hard_skills": self.hard_skills,
                "soft_skills": self.soft_skills,
                "output_reasoning": self.output_reasoning,
                "way_of_decision_making": self.way_of_decision_making,
                "personality": self.personality,
                "projects_participated": (
                    [
                        asdict(p) if is_dataclass(p) else p
                        for p in (self.projects_participated or [])
                    ]
                    if self.projects_participated
                    else None
                ),
                "user_goal": self.user_goal,
                "work_responsibility": self.work_responsibility,
                "working_habit_preference": self.working_habit_preference,
                "interests": self.interests,
                "tendency": self.tendency,
                "motivation_system": self.motivation_system,
                "fear_system": self.fear_system,
                "value_system": self.value_system,
                "humor_use": self.humor_use,
                "colloquialism": self.colloquialism,
                "group_importance_evidence": (
                    asdict(self.group_importance_evidence)
                    if self.group_importance_evidence and is_dataclass(self.group_importance_evidence)
                    else self.group_importance_evidence
                ),
            }
        )

        return base_dict


@dataclass
class ProfileMemoryExtractRequest(MemoryExtractRequest):
    """
    Request payload used by ProfileMemoryExtractor.

    Profile extraction needs to process multiple MemCells (from clustering), thus overriding the base class's single memcell,
    using memcell_list and user_id_list
    """

    # Override base class field, set to None (Profile does not use single memcell)
    memcell: Optional[MemCell] = None

    # Profile specific fields
    memcell_list: List[MemCell] = None
    user_id_list: Optional[List[str]] = None

    def __post_init__(self):
        # Ensure memcell_list is not None
        if self.memcell_list is None:
            self.memcell_list = []
