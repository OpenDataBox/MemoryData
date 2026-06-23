# -*- coding: utf-8 -*-
"""
Global user profile service

Provides business logic for global user profile operations.
"""

import logging
from typing import Optional, Dict, Any

from core.di import service
from core.di.utils import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.global_user_profile_raw_repository import (
    GlobalUserProfileRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.global_user_profile import (
    GlobalUserProfile,
)

logger = logging.getLogger(__name__)


@service("global_user_profile_service")
class GlobalUserProfileService:
    """
    Global user profile service

    Provides:
    - Upsert custom profile for a user (merge with existing data)
    - Get global user profile
    - Delete global user profile
    """

    def __init__(self):
        """Initialize service"""
        self._repository: Optional[GlobalUserProfileRawRepository] = None

    def _get_repository(self) -> GlobalUserProfileRawRepository:
        """Get repository (lazy loading)"""
        if self._repository is None:
            self._repository = get_bean_by_type(GlobalUserProfileRawRepository)
        return self._repository

    async def upsert_custom_profile(
        self, user_id: str, custom_profile_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Upsert custom profile data for a user

        Will merge with existing custom_profile_data, overlapping fields
        will be overwritten by input data.

        Args:
            user_id: User ID
            custom_profile_data: Custom profile data dict to merge

        Returns:
            Dict with profile info or None if failed
        """
        repo = self._get_repository()

        # First, try to get existing profile
        existing = await repo.get_by_user_id(user_id)

        # Merge custom_profile_data: existing data + new data (new overwrites existing)
        merged_custom_profile_data: Dict[str, Any] = {}
        if existing and existing.custom_profile_data:
            merged_custom_profile_data = dict(existing.custom_profile_data)

        # Merge: input data overwrites existing
        merged_custom_profile_data.update(custom_profile_data)

        result = await repo.upsert_custom_profile(
            user_id=user_id, custom_profile_data=merged_custom_profile_data
        )

        if not result:
            logger.error("Failed to upsert custom profile: user_id=%s", user_id)
            return None

        logger.info("Upserted custom profile: user_id=%s", user_id)

        return self._to_response_dict(result)

    async def get_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get global user profile by user_id

        Args:
            user_id: User ID

        Returns:
            Dict with profile info or None if not found
        """
        repo = self._get_repository()
        profile = await repo.get_by_user_id(user_id)

        if not profile:
            logger.debug("Global user profile not found: user_id=%s", user_id)
            return None

        return self._to_response_dict(profile)

    async def delete_by_user_id(self, user_id: str) -> int:
        """
        Delete global user profile by user_id

        Args:
            user_id: User ID

        Returns:
            Number of deleted records
        """
        repo = self._get_repository()
        return await repo.delete_by_user_id(user_id)

    def _to_response_dict(self, profile: GlobalUserProfile) -> Dict[str, Any]:
        """
        Convert GlobalUserProfile to response dict

        Args:
            profile: GlobalUserProfile document

        Returns:
            Dict representation
        """
        return {
            "id": str(profile.id) if profile.id else None,
            "user_id": profile.user_id,
            "profile_data": profile.profile_data,
            "custom_profile_data": profile.custom_profile_data,
            "confidence": profile.confidence,
            "memcell_count": profile.memcell_count,
            "created_at": (
                profile.created_at.isoformat() if profile.created_at else None
            ),
            "updated_at": (
                profile.updated_at.isoformat() if profile.updated_at else None
            ),
        }
