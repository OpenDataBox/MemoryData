"""DTO (Data Transfer Object) types for API specifications.

This package organizes DTOs by resource type:
- base: Common base types (BaseApiResponse)
- memory: Memory resource DTOs (memorize, fetch, search, delete)
- conversation_meta: Conversation metadata resource DTOs
"""

# Base types
from api_specs.dtos.base import BaseApiResponse, T

# Memory resource DTOs
from api_specs.dtos.memory import (
    # Raw data
    RawData,
    # Memorize
    MemorizeRequest,
    MemorizeMessageRequest,
    MemorizeResult,
    MemorizeResponse,
    # Fetch
    FetchMemRequest,
    FetchMemResponse,
    FetchMemoriesResponse,
    # Search/Retrieve
    RetrieveMemRequest,
    PendingMessage,
    RetrieveMemResponse,
    SearchMemoriesResponse,
    # Delete
    DeleteMemoriesRequest,
    DeleteMemoriesResult,
    DeleteMemoriesResponse,
)

# Conversation metadata resource DTOs
from api_specs.dtos.conversation_meta import (
    # Common types
    UserDetail,
    # Internal request
    ConversationMetaRequest,
    # Create
    ConversationMetaCreateRequest,
    # Get
    ConversationMetaGetRequest,
    ConversationMetaResponse,
    GetConversationMetaResponse,
    SaveConversationMetaResponse,
    # Patch
    ConversationMetaPatchRequest,
    PatchConversationMetaResult,
    PatchConversationMetaResponse,
)

__all__ = [
    # Base
    "BaseApiResponse",
    "T",
    # Memory - Raw data
    "RawData",
    # Memory - Memorize
    "MemorizeRequest",
    "MemorizeMessageRequest",
    "MemorizeResult",
    "MemorizeResponse",
    # Memory - Fetch
    "FetchMemRequest",
    "FetchMemResponse",
    "FetchMemoriesResponse",
    # Memory - Search/Retrieve
    "RetrieveMemRequest",
    "PendingMessage",
    "RetrieveMemResponse",
    "SearchMemoriesResponse",
    # Memory - Delete
    "DeleteMemoriesRequest",
    "DeleteMemoriesResult",
    "DeleteMemoriesResponse",
    # Conversation metadata
    "UserDetail",
    "ConversationMetaRequest",
    "ConversationMetaCreateRequest",
    "ConversationMetaGetRequest",
    "ConversationMetaResponse",
    "GetConversationMetaResponse",
    "SaveConversationMetaResponse",
    "ConversationMetaPatchRequest",
    "PatchConversationMetaResult",
    "PatchConversationMetaResponse",
]
