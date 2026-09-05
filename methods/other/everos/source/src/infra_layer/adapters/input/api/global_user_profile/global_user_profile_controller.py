# -*- coding: utf-8 -*-
"""
Global user profile controller

Provides API for managing global user profiles.
Mainly used for upserting custom profile data for users.
"""

from fastapi import HTTPException

from core.di.decorators import component
from core.interface.controller.base_controller import BaseController, post
from core.observation.logger import get_logger
from service.global_user_profile_service import GlobalUserProfileService
from infra_layer.adapters.input.api.dto.global_user_profile_dto import (
    UpsertCustomProfileRequest,
    UpsertCustomProfileResponse,
)


logger = get_logger(__name__)


@component(name="globalUserProfileController")
class GlobalUserProfileController(BaseController):
    """
    Global user profile controller

    Provides API for managing global user profiles,
    currently supports upserting custom profile data for users.
    """

    def __init__(self, global_user_profile_service: GlobalUserProfileService):
        """
        Initialize the controller

        Args:
            global_user_profile_service: Global user profile service (via dependency injection)
        """
        super().__init__(
            prefix="/api/v1/global-user-profile",
            tags=["Global User Profile"],
            default_auth="none",  # Adjust authentication strategy as needed
        )
        self.global_user_profile_service = global_user_profile_service
        logger.info("GlobalUserProfileController initialized")

    @post(
        "/custom",
        response_model=UpsertCustomProfileResponse,
        summary="Upsert custom profile",
        description="""
        Upsert custom profile data for a user

        ## Function description:
        - Create or update custom profile data for a specific user
        - Will merge with existing custom_profile_data, overlapping fields will be overwritten by input
        - Currently only supports initial_profile field in custom_profile_data

        ## Request body:
        - user_id: User ID (required)
        - custom_profile_data: Custom profile data object (required)
          - initial_profile: List of profile sentences (required)

        ## Example request:
        ```json
        {
            "user_id": "user_123",
            "custom_profile_data": {
                "initial_profile": [
                    "User is a software engineer",
                    "User is proficient in Python programming",
                    "User is interested in AI technology"
                ]
            }
        }
        ```

        ## API path:
        POST /api/v1/global-user-profile/custom
        """,
        responses={
            200: {
                "description": "Successfully upserted custom profile",
                "content": {
                    "application/json": {
                        "example": {
                            "success": True,
                            "data": {
                                "id": "507f1f77bcf86cd799439011",
                                "user_id": "user_123",
                                "profile_data": None,
                                "custom_profile_data": {
                                    "initial_profile": [
                                        "User is a software engineer",
                                        "User is proficient in Python programming",
                                        "User is interested in AI technology",
                                    ]
                                },
                                "confidence": 0.0,
                                "memcell_count": 0,
                                "created_at": "2024-01-15T10:30:00+00:00",
                                "updated_at": "2024-01-15T10:30:00+00:00",
                            },
                            "message": None,
                        }
                    }
                },
            },
            400: {
                "description": "Parameter error",
                "content": {
                    "application/json": {"example": {"detail": "user_id is required"}}
                },
            },
        },
    )
    async def upsert_custom_profile(
        self, request: UpsertCustomProfileRequest
    ) -> UpsertCustomProfileResponse:
        """
        Upsert custom profile data for a user

        Args:
            request: UpsertCustomProfileRequest containing user_id and custom_profile_data

        Returns:
            UpsertCustomProfileResponse with the created/updated profile data
        """
        # Parameter validation
        if not request.user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        if (
            not request.custom_profile_data
            or not request.custom_profile_data.initial_profile
        ):
            raise HTTPException(
                status_code=400,
                detail="custom_profile_data.initial_profile is required and cannot be empty",
            )

        try:
            # Convert CustomProfileData to dict for service
            custom_profile_data_dict = request.custom_profile_data.model_dump()

            # Call service to upsert custom profile
            data = await self.global_user_profile_service.upsert_custom_profile(
                user_id=request.user_id, custom_profile_data=custom_profile_data_dict
            )

            if data is None:
                return UpsertCustomProfileResponse(
                    success=False, data=None, message="Failed to upsert custom profile"
                )

            return UpsertCustomProfileResponse(success=True, data=data, message=None)

        except Exception as e:
            logger.error(
                "Exception when upserting custom profile: user_id=%s, error=%s",
                request.user_id,
                str(e),
                exc_info=True,
            )
            return UpsertCustomProfileResponse(
                success=False,
                data=None,
                message=f"Failed to upsert custom profile: {str(e)}",
            )
