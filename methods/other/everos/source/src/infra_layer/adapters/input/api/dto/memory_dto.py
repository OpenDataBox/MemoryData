# -*- coding: utf-8 -*-
"""
Memory API DTO

Request and response data transfer objects for Memory API.
These models are re-exported from api_specs.dtos for backward compatibility.
"""

# Re-export from api_specs.dtos
from api_specs.dtos import (
    # Base API Response
    BaseApiResponse,
    # Command DTOs
    MemorizeMessageRequest,
    DeleteMemoriesRequest as DeleteMemoriesRequestDTO,
    # Request DTOs
    FetchMemRequest,
    RetrieveMemRequest,
    UserDetail,
    ConversationMetaCreateRequest,
    ConversationMetaGetRequest,
    ConversationMetaPatchRequest,
    # Response DTOs (result data)
    FetchMemResponse,
    RetrieveMemResponse,
    ConversationMetaResponse,
    PatchConversationMetaResult,
    DeleteMemoriesResult,
    MemorizeResult,
    # API Response wrappers
    MemorizeResponse,
    FetchMemoriesResponse,
    SearchMemoriesResponse,
    GetConversationMetaResponse,
    SaveConversationMetaResponse,
    PatchConversationMetaResponse,
    DeleteMemoriesResponse,
)

# Backward compatibility aliases
FetchMemoriesParams = FetchMemRequest
SearchMemoriesRequest = RetrieveMemRequest
UserDetailRequest = UserDetail
DeleteMemoriesRequest = DeleteMemoriesRequestDTO

__all__ = [
    # Base Response
    "BaseApiResponse",
    # Command DTOs
    "MemorizeMessageRequest",
    "DeleteMemoriesRequest",
    "DeleteMemoriesRequestDTO",
    # Query DTOs (Requests)
    "FetchMemRequest",
    "RetrieveMemRequest",
    "UserDetail",
    "ConversationMetaCreateRequest",
    "ConversationMetaGetRequest",
    "ConversationMetaPatchRequest",
    # Response DTOs (result data)
    "FetchMemResponse",
    "RetrieveMemResponse",
    "ConversationMetaResponse",
    "PatchConversationMetaResult",
    "DeleteMemoriesResult",
    "MemorizeResult",
    # API Response wrappers
    "MemorizeResponse",
    "FetchMemoriesResponse",
    "SearchMemoriesResponse",
    "GetConversationMetaResponse",
    "SaveConversationMetaResponse",
    "PatchConversationMetaResponse",
    "DeleteMemoriesResponse",
    # Backward compatibility aliases
    "FetchMemoriesParams",
    "SearchMemoriesRequest",
    "UserDetailRequest",
]
