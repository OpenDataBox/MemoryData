from typing import Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field
from core.oxm.mongo.audit_base import AuditBase
from pymongo import IndexModel, ASCENDING, DESCENDING


class GlobalUserProfile(DocumentBase, AuditBase):
    """
    Global user profile document model

    Stores global user profile information
    """

    # Composite primary key
    user_id: Indexed(str) = Field(..., description="User ID")

    # Profile content (stored in JSON format)
    profile_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User profile data (including role, skills, preferences, personality, etc.)",
    )

    custom_profile_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom user profile data (including role, skills, preferences, personality, etc.)",
    )

    # Metadata
    confidence: float = Field(default=0.0, description="Profile confidence score (0-1)")

    memcell_count: int = Field(
        default=0, description="Number of MemCells involved in extraction"
    )

    class Settings:
        """Beanie settings"""

        name = "global_user_profiles"
        indexes = [
            IndexModel([("user_id", ASCENDING)], name="idx_user_id"),
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
