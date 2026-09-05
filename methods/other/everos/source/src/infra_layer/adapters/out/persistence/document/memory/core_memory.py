from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase


class CoreMemory(DocumentBase, AuditBase):
    """
    Core memory document model

    Unified storage for user's basic information, personal profile, and preference settings.
    A single document contains data of all three memory types.
    """

    user_id: Indexed(str) = Field(..., description="User ID")

    # ==================== Version control fields ====================
    version: Optional[str] = Field(
        default=None, description="Version number, used for version management"
    )
    is_latest: Optional[bool] = Field(
        default=True, description="Whether it is the latest version, default is True"
    )

    # ==================== BaseMemory fields ====================
    # Basic information fields
    user_name: Optional[str] = Field(default=None, description="User name")
    gender: Optional[str] = Field(default=None, description="Gender")
    position: Optional[str] = Field(default=None, description="Position")
    supervisor_user_id: Optional[str] = Field(
        default=None, description="Supervisor User ID"
    )
    team_members: Optional[List[str]] = Field(
        default=None, description="Team members' User IDs"
    )
    okr: Optional[List[Dict[str, str]]] = Field(default=None, description="OKR metrics")
    base_location: Optional[str] = Field(default=None, description="Base location")
    hiredate: Optional[str] = Field(default=None, description="Hire date")
    age: Optional[int] = Field(default=None, description="Age")
    department: Optional[str] = Field(default=None, description="Department")

    # ==================== Profile fields ====================
    # Personal profile fields - all fields now use the embedded evidences format
    # Skill format: [{"value": "Python", "level": "Advanced", "evidences": ["2024-01-01|conv_123"]}]
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
        default=None, description="Reasoning explanation for the current output result"
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
        description="Preferred slang or catchphrases, containing value/level/evidences",
    )

    # Other attributes format: [{"value": "xxx", "evidences": ["2024-01-01|conv_123"]}]
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
        default=None, description="User choice preferences, including evidences"
    )
    way_of_decision_making: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Decision-making style, including evidences"
    )

    group_importance_evidence: Optional[Dict[str, Any]] = Field(
        default=None, description="Group importance evidence"
    )

    # ==================== General fields ====================
    extend: Optional[Dict[str, Any]] = Field(
        default=None, description="Reserved extension field"
    )

    model_config = ConfigDict(
        collection="core_memories",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "user_name": "Zhang San",
                "gender": "Male",
                "position": "Senior Engineer",
                "department": "Technology Department",
                "age": 30,
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

        name = "core_memories"
        indexes = [
            # Unique compound index on user_id and version
            IndexModel(
                [("user_id", ASCENDING), ("version", ASCENDING)],
                unique=True,
                name="idx_user_id_version_unique",
            ),
            # Index on is_latest field (for fast querying of latest version)
            IndexModel(
                [("user_id", ASCENDING), ("is_latest", ASCENDING)],
                name="idx_user_id_is_latest",
            ),
            # Audit field indexes
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
