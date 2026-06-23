"""Base DTO types for API responses.

This module contains common base types used across all API endpoints.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field


# Generic type for API response result
T = TypeVar("T")


class BaseApiResponse(BaseModel, Generic[T]):
    """Base API response wrapper

    Unified response format for all API endpoints.
    """

    status: str = Field(
        default="ok", description="Response status", examples=["ok", "failed"]
    )
    message: str = Field(
        default="", description="Response message", examples=["Operation successful"]
    )
    result: T = Field(description="Response result data")

    model_config = {"arbitrary_types_allowed": True}
