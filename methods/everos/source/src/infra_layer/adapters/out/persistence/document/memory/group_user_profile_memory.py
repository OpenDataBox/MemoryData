from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase


class GroupUserProfileMemory(DocumentBase, AuditBase):
    """
    Core memory document model

    Unified storage for user's basic information, personal profile, and preference settings.
    A single document contains data of all three memory types.

    All profile fields now use the embedded evidences format:
    - Skills: [{"value": "Python", "level": "Advanced", "evidences": ["2024-01-01|conv_123"]}]
    - Legacy format: [{"skill": "Python", "level": "Advanced", "evidences": ["..."]}] (automatically converted)
    - Other attributes: [{"value": "xxx", "evidences": ["2024-01-01|conv_123"]}]
    """

    user_id: Indexed(str) = Field(..., description="User ID")
    group_id: Indexed(str) = Field(..., description="Group ID")

    # ==================== Version control fields ====================
    version: Optional[str] = Field(
        default=None, description="Version number, used for version management"
    )
    is_latest: Optional[bool] = Field(
        default=True, description="Whether it is the latest version, default is True"
    )

    user_name: Optional[str] = Field(default=None, description="User name")

    # ==================== Profile fields ====================
    # Skill field - Format: [{"value": "Python", "level": "Advanced", "evidences": ["id1"]}]
    # Legacy format: [{"skill": "Python", "level": "Advanced", "evidences": ["..."]}] (automatically converted)
    hard_skills: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Hard skills, such as SQL, Python, product design, etc., along with proficiency levels, including evidences",
    )
    soft_skills: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Soft skills, such as communication, teamwork, emotional intelligence, etc., including evidences",
    )
    output_reasoning: Optional[str] = Field(
        default=None, description="Reasoning explanation for this output"
    )
    motivation_system: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Motivation system, containing value/level/evidences"
    )
    fear_system: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Fear system, containing value/level/evidences"
    )
    value_system: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Value system, containing value/level/evidences"
    )
    humor_use: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Humor usage style, containing value/level/evidences"
    )
    colloquialism: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Preferred catchphrases, containing value/level/evidences",
    )

    # Other profile fields - Format: [{"value": "xxx", "evidences": ["id1"]}]
    personality: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="User personality, including evidences"
    )
    projects_participated: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Information about participated projects"
    )
    user_goal: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="User goals, including evidences"
    )
    work_responsibility: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Work responsibilities, including evidences"
    )
    working_habit_preference: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Work habit preferences, including evidences"
    )
    interests: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Hobbies and interests, including evidences"
    )
    tendency: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="User preference tendencies, including evidences"
    )
    way_of_decision_making: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Decision-making style, including evidences"
    )

    group_importance_evidence: Optional[Dict[str, Any]] = Field(
        default=None, description="Evidence of group importance"
    )

    model_config = ConfigDict(
        collection="group_core_profile_memory",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "group_id": "group_12345",
                "personality": "Introverted but good at communication, enjoys deep thinking",
                "hard_skills": [{"Python": "Advanced"}],
                "working_habit_preference": ["Remote work", "Flexible hours"],
                "user_goal": ["Become a technical expert", "Improve leadership"],
                "extend": {"priority": "high"},
            }
        },
        extra="allow",
    )

    class Settings:
        """Beanie settings"""

        name = "group_core_profile_memory"
        indexes = [
            # Composite unique index on user_id, group_id, and version
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("version", ASCENDING),
                ],
                unique=True,
                name="idx_user_id_group_id_version_unique",
            ),
            # Index for querying the latest version by user_id
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("is_latest", ASCENDING),
                ],
                name="idx_user_id_group_id_is_latest",
            ),
            # Index for querying the latest version by group_id (supports get_by_group_id method)
            IndexModel(
                [("group_id", ASCENDING), ("is_latest", ASCENDING)],
                name="idx_group_id_is_latest",
            ),
            # Indexes for audit fields
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
