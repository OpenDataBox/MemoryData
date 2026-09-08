"""Conversation metadata resource DTOs.

This module contains DTOs related to conversation metadata CRUD operations:
- Create (POST /api/v1/memories/conversation-meta)
- Get (GET /api/v1/memories/conversation-meta)
- Patch (PATCH /api/v1/memories/conversation-meta)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator, SkipValidation

from api_specs.dtos.base import BaseApiResponse
from api_specs.memory_models import MessageSenderRole
from common_utils.datetime_utils import get_timezone


# =============================================================================
# Common Types
# =============================================================================


class UserDetail(BaseModel):
    """User details

    Structure for the value of ConversationMetaRequest.user_details
    """

    full_name: Optional[str] = Field(
        default=None, description="User full name", examples=["John Smith"]
    )
    role: Optional[str] = Field(
        default=None,
        description="""User type role, used to identify if this user is a human or AI.
Enum values from MessageSenderRole:
- user: Human user
- assistant: AI assistant/bot""",
        examples=["user", "assistant"],
    )
    custom_role: Optional[str] = Field(
        default=None,
        description="User's job/position role (e.g. developer, designer, manager)",
        examples=["developer"],
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional information",
        examples=[{"department": "Engineering"}],
    )

    @model_validator(mode="after")
    def validate_role(self):
        """Validate that role is a valid MessageSenderRole value"""
        if self.role is not None and not MessageSenderRole.is_valid(self.role):
            raise ValueError(
                f"Invalid role '{self.role}'. Must be one of: {[r.value for r in MessageSenderRole]}"
            )
        return self


# =============================================================================
# Internal Request (Business Layer)
# =============================================================================


class ConversationMetaRequest(BaseModel):
    """Conversation metadata request (internal use)"""

    scene: str  # Scene identifier
    scene_desc: Dict[
        str, Any
    ]  # Scene description, usually contains fields like description
    name: str  # Conversation name
    group_id: str  # Group ID
    created_at: str  # Creation time, ISO format string
    description: Optional[str] = None  # Conversation description
    default_timezone: Optional[str] = Field(
        default_factory=lambda: get_timezone().key
    )  # Default timezone
    user_details: Dict[str, UserDetail] = Field(
        default_factory=dict
    )  # User details, key is dynamic (e.g., user_001, robot_001), value structure is fixed
    tags: List[str] = Field(default_factory=list)  # List of tags


# =============================================================================
# Create DTOs (POST /api/v1/memories/conversation-meta)
# =============================================================================


class ConversationMetaCreateRequest(BaseModel):
    """
    Create conversation metadata request body

    Used for POST /api/v1/memories/conversation-meta endpoint
    """

    scene: str = Field(
        ...,
        description="""Scene identifier, enum values from ScenarioType:
- group_chat: work/group chat scenario, suitable for group conversations such as multi-person collaboration and project discussions
- assistant: assistant scenario, suitable for one-on-one AI assistant conversations""",
        examples=["group_chat"],
    )
    scene_desc: Dict[str, Any] = Field(
        ...,
        description="Scene description object, can include fields like description",
        examples=[
            {
                "description": "Project discussion group chat",
                "type": "project_discussion",
            }
        ],
    )
    name: str = Field(
        ..., description="Conversation name", examples=["Project Discussion Group"]
    )
    description: Optional[str] = Field(
        default=None,
        description="Conversation description",
        examples=["Technical discussion for new feature development"],
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Group unique identifier. When null/not provided, represents default settings for this scene.",
        examples=["group_123", None],
    )
    created_at: str = Field(
        ...,
        description="Conversation creation time (ISO 8601 format)",
        examples=["2025-01-15T10:00:00+00:00"],
    )
    default_timezone: Optional[str] = Field(
        default=None, description="Default timezone", examples=["UTC"]
    )
    user_details: Optional[Dict[str, UserDetail]] = Field(
        default=None,
        description="Participant details, key is user ID, value is user detail object",
        examples=[
            {
                "user_001": {
                    "full_name": "John Smith",
                    "role": "user",
                    "custom_role": "developer",
                    "extra": {"department": "Engineering"},
                },
                "bot_001": {
                    "full_name": "AI Assistant",
                    "role": "assistant",
                    "custom_role": "assistant",
                    "extra": {"type": "ai"},
                },
            }
        ],
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Tag list", examples=[["work", "technical"]]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "With group_id",
                    "value": {
                        "scene": "group_chat",
                        "scene_desc": {
                            "description": "Project discussion group chat",
                            "type": "project_discussion",
                        },
                        "name": "Project Discussion Group",
                        "description": "Technical discussion for new feature development",
                        "group_id": "group_123",
                        "created_at": "2025-01-15T10:00:00+00:00",
                        "default_timezone": "UTC",
                        "user_details": {
                            "user_001": {
                                "full_name": "John Smith",
                                "role": "user",
                                "custom_role": "developer",
                                "extra": {"department": "Engineering"},
                            },
                            "bot_001": {
                                "full_name": "AI Assistant",
                                "role": "assistant",
                            },
                        },
                        "tags": ["work", "technical"],
                    },
                },
                {
                    "summary": "Default config (group_id is null)",
                    "value": {
                        "scene": "group_chat",
                        "scene_desc": {
                            "description": "Default conversation meta config"
                        },
                        "name": "Default Group Chat Settings",
                        "description": "Default settings for group_chat scene",
                        "group_id": None,
                        "created_at": "2025-01-15T10:00:00+00:00",
                        "default_timezone": "UTC",
                        "tags": ["default"],
                    },
                },
            ]
        }
    }


# =============================================================================
# Get DTOs (GET /api/v1/memories/conversation-meta)
# =============================================================================


class ConversationMetaGetRequest(BaseModel):
    """
    Get conversation metadata request parameters

    Used for GET /api/v1/memories/conversation-meta endpoint
    """

    group_id: Optional[str] = Field(
        default=None,
        description="Group ID to look up. If not found, will automatically fallback to default config (group_id=null). If not provided, returns default config directly.",
        examples=["group_123"],
    )

    model_config = {"json_schema_extra": {"example": {"group_id": "group_123"}}}


class ConversationMetaResponse(BaseModel):
    """
    Conversation metadata response DTO (result data)

    Used for GET /api/v1/memories/conversation-meta response
    """

    id: str = Field(..., description="Document ID")
    group_id: Optional[str] = Field(
        default=None, description="Group ID (null for default config)"
    )
    scene: str = Field(..., description="Scene identifier")
    scene_desc: Optional[Dict[str, Any]] = Field(
        default=None, description="Scene description"
    )
    name: str = Field(..., description="Conversation name")
    description: Optional[str] = Field(
        default=None, description="Conversation description"
    )
    conversation_created_at: str = Field(..., description="Conversation creation time")
    default_timezone: Optional[str] = Field(
        default=None, description="Default timezone"
    )
    user_details: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="User details"
    )
    tags: List[str] = Field(default_factory=list, description="Tags")
    is_default: bool = Field(
        default=False, description="Whether this is the default config"
    )
    created_at: Optional[str] = Field(default=None, description="Record creation time")
    updated_at: Optional[str] = Field(default=None, description="Record update time")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "group_id": "group_123",
                "scene": "group_chat",
                "scene_desc": {"description": "Project discussion group chat"},
                "name": "Project Discussion",
                "description": "Technical discussion group",
                "conversation_created_at": "2025-01-15T10:00:00+00:00",
                "default_timezone": "UTC",
                "user_details": {
                    "user_001": {
                        "full_name": "John",
                        "role": "user",
                        "custom_role": "developer",
                    },
                    "bot_001": {"full_name": "AI Assistant", "role": "assistant"},
                },
                "tags": ["work", "tech"],
                "is_default": False,
                "created_at": "2025-01-15T10:00:00+00:00",
                "updated_at": "2025-01-15T10:00:00+00:00",
            }
        }
    }


class GetConversationMetaResponse(BaseApiResponse[ConversationMetaResponse]):
    """Get conversation metadata API response

    Response for GET /api/v1/memories/conversation-meta endpoint.
    """

    result: ConversationMetaResponse = Field(description="Conversation metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Found by group_id",
                    "value": {
                        "status": "ok",
                        "message": "Conversation metadata retrieved successfully",
                        "result": {
                            "id": "507f1f77bcf86cd799439011",
                            "group_id": "group_123",
                            "scene": "group_chat",
                            "scene_desc": {
                                "description": "Project discussion group chat"
                            },
                            "name": "Project Discussion",
                            "conversation_created_at": "2025-01-15T10:00:00+00:00",
                            "is_default": False,
                        },
                    },
                },
                {
                    "summary": "Fallback to default config",
                    "value": {
                        "status": "ok",
                        "message": "Using default config",
                        "result": {
                            "id": "507f1f77bcf86cd799439012",
                            "group_id": None,
                            "scene": "group_chat",
                            "scene_desc": {
                                "description": "Default conversation meta config"
                            },
                            "name": "Default Settings",
                            "conversation_created_at": "2025-01-15T10:00:00+00:00",
                            "is_default": True,
                        },
                    },
                },
            ]
        }
    }


class SaveConversationMetaResponse(BaseApiResponse[ConversationMetaResponse]):
    """Save conversation metadata API response

    Response for POST /api/v1/memories/conversation-meta endpoint.
    """

    result: ConversationMetaResponse = Field(description="Saved conversation metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "message": "Conversation metadata saved successfully",
                "result": {
                    "id": "507f1f77bcf86cd799439011",
                    "group_id": "group_123",
                    "scene": "group_chat",
                    "scene_desc": {"description": "Project discussion group chat"},
                    "name": "Project Discussion",
                    "conversation_created_at": "2025-01-15T10:00:00+00:00",
                    "is_default": False,
                },
            }
        }
    }


# =============================================================================
# Patch DTOs (PATCH /api/v1/memories/conversation-meta)
# =============================================================================


class ConversationMetaPatchRequest(BaseModel):
    """
    Partial update conversation metadata request body

    Used for PATCH /api/v1/memories/conversation-meta endpoint
    """

    group_id: Optional[str] = Field(
        default=None,
        description="Group ID to update. When null, updates the default config.",
        examples=["group_123", None],
    )
    name: Optional[str] = Field(
        default=None,
        description="New conversation name",
        examples=["New Conversation Name"],
    )
    description: Optional[str] = Field(
        default=None,
        description="New conversation description",
        examples=["Updated description"],
    )
    scene_desc: Optional[Dict[str, Any]] = Field(
        default=None,
        description="New scene description",
        examples=[{"description": "Project discussion group chat"}],
    )
    tags: Optional[List[str]] = Field(
        default=None, description="New tag list", examples=[["tag1", "tag2"]]
    )
    user_details: Optional[Dict[str, UserDetail]] = Field(
        default=None,
        description="New user details (will completely replace existing user_details)",
        examples=[
            {
                "user_001": {
                    "full_name": "John Smith",
                    "role": "user",
                    "custom_role": "lead",
                }
            }
        ],
    )
    default_timezone: Optional[str] = Field(
        default=None, description="New default timezone", examples=["Asia/Shanghai"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Update by group_id",
                    "value": {
                        "group_id": "group_123",
                        "name": "New Conversation Name",
                        "tags": ["updated", "tags"],
                    },
                },
                {
                    "summary": "Update default config (group_id is null)",
                    "value": {"group_id": None, "name": "Updated Default Settings"},
                },
            ]
        }
    }


class PatchConversationMetaResult(BaseModel):
    """Patch conversation metadata result data"""

    id: str = Field(..., description="Document ID")
    group_id: Optional[str] = Field(
        default=None, description="Group ID (null for default config)"
    )
    scene: Optional[str] = Field(default=None, description="Scene identifier")
    name: Optional[str] = Field(default=None, description="Conversation name")
    updated_fields: List[str] = Field(
        default_factory=list, description="List of updated field names"
    )
    updated_at: Optional[str] = Field(default=None, description="Record update time")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "group_id": "group_123",
                "scene": "group_chat",
                "name": "New conversation name",
                "updated_fields": ["name", "tags"],
                "updated_at": "2025-01-15T10:30:00+00:00",
            }
        }
    }


class PatchConversationMetaResponse(BaseApiResponse[PatchConversationMetaResult]):
    """Patch conversation metadata API response

    Response for PATCH /api/v1/memories/conversation-meta endpoint.
    """

    result: PatchConversationMetaResult = Field(
        description="Patch result with updated fields"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "message": "Conversation metadata updated successfully, updated 2 fields",
                "result": {
                    "id": "507f1f77bcf86cd799439011",
                    "group_id": "group_123",
                    "scene": "group_chat",
                    "name": "New conversation name",
                    "updated_fields": ["name", "tags"],
                    "updated_at": "2025-01-15T10:30:00+00:00",
                },
            }
        }
    }
