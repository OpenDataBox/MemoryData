# -*- coding: utf-8 -*-
"""
GlobalUserProfile native CRUD repository

Global user profile data access layer based on Beanie ODM.
"""

from typing import Optional, Dict, Any
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository

from infra_layer.adapters.out.persistence.document.memory.global_user_profile import (
    GlobalUserProfile,
)

logger = get_logger(__name__)


@repository("global_user_profile_raw_repository", primary=True)
class GlobalUserProfileRawRepository(BaseRepository[GlobalUserProfile]):
    """
    GlobalUserProfile native CRUD repository

    Provides:
    - CRUD operations based on user_id
    - Custom profile data management
    """

    def __init__(self):
        super().__init__(GlobalUserProfile)

    # ==================== Query methods ====================

    async def get_by_user_id(self, user_id: str) -> Optional[GlobalUserProfile]:
        """
        Get global user profile by user_id

        Args:
            user_id: User ID

        Returns:
            GlobalUserProfile or None if not found
        """
        try:
            return await self.model.find_one(GlobalUserProfile.user_id == user_id)
        except Exception as e:
            logger.error(
                f"Failed to retrieve global user profile: user_id={user_id}, error={e}"
            )
            return None

    # ==================== Create/Update methods ====================

    async def upsert(
        self,
        user_id: str,
        profile_data: Optional[Dict[str, Any]] = None,
        custom_profile_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[GlobalUserProfile]:
        """
        Create or update global user profile

        Args:
            user_id: User ID
            profile_data: Profile data dict (optional)
            custom_profile_data: Custom profile data dict (optional)

        Returns:
            GlobalUserProfile or None if failed
        """
        try:
            existing = await self.get_by_user_id(user_id)

            if existing:
                # Update existing profile
                if profile_data is not None:
                    existing.profile_data = profile_data
                if custom_profile_data is not None:
                    existing.custom_profile_data = custom_profile_data

                await existing.save()
                logger.debug(f"Updated global user profile: user_id={user_id}")
                return existing
            else:
                # Create new profile (confidence and memcell_count use DB default values)
                global_user_profile = GlobalUserProfile(
                    user_id=user_id,
                    profile_data=profile_data,
                    custom_profile_data=custom_profile_data,
                )
                await global_user_profile.insert()
                logger.info(f"Created global user profile: user_id={user_id}")
                return global_user_profile
        except Exception as e:
            logger.error(
                f"Failed to save global user profile: user_id={user_id}, error={e}"
            )
            return None

    async def upsert_custom_profile(
        self, user_id: str, custom_profile_data: Dict[str, Any]
    ) -> Optional[GlobalUserProfile]:
        """
        Upsert custom profile data for a user

        Args:
            user_id: User ID
            custom_profile_data: Custom profile data dict (already merged by service layer)

        Returns:
            GlobalUserProfile or None if failed
        """
        try:
            existing = await self.get_by_user_id(user_id)

            if existing:
                # Update existing custom profile data
                existing.custom_profile_data = custom_profile_data
                await existing.save()
                logger.debug(f"Updated custom profile: user_id={user_id}")
                return existing
            else:
                # Create new profile with custom data (confidence and memcell_count use DB default values)
                global_user_profile = GlobalUserProfile(
                    user_id=user_id,
                    profile_data=None,  # profile_data is empty for new custom profiles
                    custom_profile_data=custom_profile_data,
                )
                await global_user_profile.insert()
                logger.info(f"Created custom profile: user_id={user_id}")
                return global_user_profile
        except Exception as e:
            logger.error(
                f"Failed to upsert custom profile: user_id={user_id}, error={e}"
            )
            return None

    # ==================== Delete methods ====================

    async def delete_by_user_id(self, user_id: str) -> int:
        """
        Delete global user profile by user_id

        Args:
            user_id: User ID

        Returns:
            Number of deleted records
        """
        try:
            result = await self.model.find(
                GlobalUserProfile.user_id == user_id
            ).delete()

            count = result.deleted_count if result else 0
            logger.info(
                f"Deleted global user profile: user_id={user_id}, count={count}"
            )
            return count
        except Exception as e:
            logger.error(
                f"Failed to delete global user profile: user_id={user_id}, error={e}"
            )
            return 0
