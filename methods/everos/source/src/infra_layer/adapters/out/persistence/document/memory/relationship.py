from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase


class Relationship(DocumentBase, AuditBase):
    """
    Relationship document model

    Describes relationships between entities, supporting multiple relationship types and detailed information.
    """

    # Composite primary key
    source_entity_id: Indexed(str) = Field(
        ..., description="Source entity ID, part of composite primary key"
    )
    target_entity_id: Indexed(str) = Field(
        ..., description="Target entity ID, part of composite primary key"
    )

    # Relationship information
    relationship: List[Dict[str, str]] = Field(
        ...,
        description="List of relationships, each containing fields such as type, content, detail",
    )

    # General fields
    extend: Optional[Dict[str, Any]] = Field(
        default=None, description="Reserved extension field"
    )

    model_config = ConfigDict(
        collection="relationships",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "source_entity_id": "entity_001",
                "target_entity_id": "entity_002",
                "relationship": [
                    {
                        "type": "Interpersonal relationship",
                        "content": "Project collaboration",
                        "detail": "Collaborated on the e-commerce platform refactoring project",
                    },
                    {
                        "type": "Work relationship",
                        "content": "Superior-subordinate",
                        "detail": "Zhang San is responsible for guiding Li Si's technical work",
                    },
                ],
                "extend": {"strength": "strong", "context": "work environment"},
            }
        },
        extra="allow",
    )

    class Settings:
        """Beanie Settings"""

        name = "relationships"
        indexes = [
            IndexModel(
                [("source_entity_id", ASCENDING), ("target_entity_id", ASCENDING)],
                unique=True,
                name="idx_source_target_unique",
            ),
            IndexModel(
                [("target_entity_id", ASCENDING), ("source_entity_id", ASCENDING)],
                unique=True,
                name="idx_target_source_unique",
            ),
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
