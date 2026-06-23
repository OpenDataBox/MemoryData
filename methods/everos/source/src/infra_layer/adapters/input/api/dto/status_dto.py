# -*- coding: utf-8 -*-
"""
Request Status DTO

Data transfer objects for request status API.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RequestStatusResponse(BaseModel):
    """
    Request status response

    Contains detailed status information of the request.
    """

    success: bool = Field(..., description="Whether the query was successful")
    found: bool = Field(
        default=False, description="Whether the request status was found"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None, description="Request status data"
    )
    message: Optional[str] = Field(default=None, description="Message")


class RequestStatusData(BaseModel):
    """
    Request status data model

    Request status information stored in Redis.
    """

    organization_id: str = Field(..., description="Organization ID")
    space_id: str = Field(..., description="Space ID")
    request_id: str = Field(..., description="Request ID")
    status: str = Field(..., description="Request status (start/success/failed)")
    url: Optional[str] = Field(default=None, description="Request URL")
    method: Optional[str] = Field(default=None, description="HTTP method")
    http_code: Optional[int] = Field(default=None, description="HTTP status code")
    time_ms: Optional[int] = Field(
        default=None, description="Request duration (milliseconds)"
    )
    error_message: Optional[str] = Field(default=None, description="Error message")
    start_time: Optional[int] = Field(
        default=None, description="Start timestamp (milliseconds)"
    )
    end_time: Optional[int] = Field(
        default=None, description="End timestamp (milliseconds)"
    )
    ttl_seconds: Optional[int] = Field(
        default=None, description="Remaining TTL (seconds)"
    )
