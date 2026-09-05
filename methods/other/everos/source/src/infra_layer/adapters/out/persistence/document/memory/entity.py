from datetime import datetime
from typing import List, Optional, Dict, Any
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from beanie import PydanticObjectId
from core.oxm.mongo.audit_base import AuditBase


class Entity(DocumentBase, AuditBase):
    """
    Entity document model

    Stores entity information extracted from episodic memory, including people, projects, organizations, etc.
    """

    # Basic information
    name: str = Field(..., description="Entity name")
    type: str = Field(
        ..., description="Entity type (Project, Person, Organization, etc.)"
    )
    aliases: Optional[List[str]] = Field(default=None, description="Associated aliases")

    # Common fields
    extend: Optional[Dict[str, Any]] = Field(
        default=None, description="Reserved extension field"
    )

    model_config = ConfigDict(
        collection="entities",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "name": "Zhang San",
                "type": "Person",
                "aliases": ["Xiao Zhang", "Engineer Zhang", "zhangsan"],
                "extend": {
                    "department": "Technology Department",
                    "level": "Senior Engineer",
                },
            }
        },
        extra="allow",
    )

    @property
    def entity_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie settings"""

        name = "entities"
        indexes = [
            # Note: entity_id maps to the _id field, MongoDB automatically creates a primary key index on _id
            IndexModel([("aliases", ASCENDING)], name="idx_aliases", sparse=True),
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
