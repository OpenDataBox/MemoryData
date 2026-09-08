from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from core.oxm.mongo.audit_base import AuditBase


class BehaviorHistory(DocumentBase, AuditBase):
    """
    Behavior history document model

    Records various user behaviors, including chat, email, file operations, etc.
    """

    # Composite primary key
    user_id: Indexed(str) = Field(
        ..., description="User ID, part of composite primary key"
    )
    timestamp: Indexed(datetime) = Field(
        ...,
        description="Timestamp when behavior occurred, part of composite primary key",
    )

    # Behavior information
    behavior_type: List[str] = Field(
        ...,
        description="List of behavior types (chat, follow-up, Smart-Reply, Vote, file, Email, link-doc, etc.)",
    )
    event_id: Optional[str] = Field(
        default=None, description="Associated memory unit ID (if exists)"
    )
    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata: conversation details, original email content, etc.",
    )

    # Generic fields
    extend: Optional[Dict[str, Any]] = Field(
        default=None, description="Reserved extension field"
    )

    model_config = ConfigDict(
        collection="behavior_histories",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "user_id": "user_001",
                "timestamp": datetime(2021, 1, 1, 0, 0, 0),
                "behavior_type": ["chat", "follow-up"],
                "event_id": "evt_001",
                "meta": {
                    "conversation_id": "conv_001",
                    "message_count": 5,
                    "duration_minutes": 15,
                    "topics": ["Technical discussion", "Project planning"],
                },
                "extend": {"priority": "high", "location": "office"},
            }
        },
        extra="allow",
    )

    class Settings:
        """Beanie settings"""

        name = "behavior_histories"
        indexes = [
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("behavior_type", ASCENDING),
                    ("timestamp", ASCENDING),
                ],
                name="idx_user_type_timestamp",
            ),
            IndexModel([("event_id", ASCENDING)], name="idx_event_id"),
        ]
        validate_on_save = True
        use_state_management = True
