# -*- coding: utf-8 -*-
"""
Global User Profile DTO

Data transfer objects for global user profile API.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CustomProfileData(BaseModel):
    """
    Custom profile data structure

    Currently only supports initial_profile field.
    """

    initial_profile: List[str] = Field(
        ...,
        description="List of profile sentences describing the user",
        examples=[
            [
                "User is a software engineer",
                "User is proficient in Python programming",
                "User is interested in AI technology",
            ]
        ],
    )


class UpsertCustomProfileRequest(BaseModel):
    """
    Upsert custom profile request

    Request body for upserting custom profile data.
    Will merge with existing data, overlapping fields will be overwritten by input.
    """

    user_id: str = Field(..., description="User ID")
    custom_profile_data: CustomProfileData = Field(
        ..., description="Custom profile data to upsert"
    )


class UpsertCustomProfileResponse(BaseModel):
    """
    Upsert custom profile response

    Response for upsert custom profile API.
    """

    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Dict[str, Any]] = Field(
        default=None, description="Created/updated profile data"
    )
    message: Optional[str] = Field(default=None, description="Message")


class GetGlobalUserProfileResponse(BaseModel):
    """
    Get global user profile response

    Response for get global user profile API.
    """

    success: bool = Field(..., description="Whether the query was successful")
    found: bool = Field(default=False, description="Whether the profile was found")
    data: Optional[Dict[str, Any]] = Field(
        default=None, description="Global user profile data"
    )
    message: Optional[str] = Field(default=None, description="Message")


class DeleteGlobalUserProfileResponse(BaseModel):
    """
    Delete global user profile response

    Response for delete global user profile API.
    """

    success: bool = Field(..., description="Whether the operation was successful")
    deleted_count: int = Field(default=0, description="Number of deleted records")
    message: Optional[str] = Field(default=None, description="Message")
