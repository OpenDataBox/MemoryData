"""ProfileManager - Pure computation component for profile extraction.

This module provides pure computation logic for extracting user profiles
from memcells. Storage is managed by the caller, not by ProfileManager itself.

Design:
- ProfileManager is a pure computation component
- Input: memcells + old_profiles
- Output: new_profiles
- Caller is responsible for loading/saving profiles
"""

import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path

from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.profile_memory_extractor import (
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
)
from memory_layer.memory_extractor.profile_memory_life import (
    ProfileMemoryLifeExtractor,
    ProfileMemoryLifeExtractRequest,
    ProfileMemoryLife,
)
from memory_layer.profile_manager.config import ProfileManagerConfig, ScenarioType
from core.observation.logger import get_logger

logger = get_logger(__name__)


class ProfileManager:
    """Pure computation component for profile extraction.

    ProfileManager extracts user profiles from memcells using LLM.
    It does NOT handle storage - the caller is responsible for loading
    old profiles and saving new profiles.

    Usage:
        ```python
        profile_mgr = ProfileManager(llm_provider, config)

        # Caller loads old profiles
        old_profiles = await storage.get_all_profiles()

        # Pure computation - extract profiles
        new_profiles = await profile_mgr.extract_profiles(
            memcells=memcell_list,
            old_profiles=list(old_profiles.values()),
            user_id_list=["user1", "user2"],
        )

        # Caller saves new profiles
        for profile in new_profiles:
            await storage.save_profile(profile.user_id, profile)
        ```
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[ProfileManagerConfig] = None,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
    ):
        """Initialize ProfileManager.

        Args:
            llm_provider: LLM provider for profile extraction
            config: Manager configuration (uses defaults if None)
            group_id: Group/conversation identifier
            group_name: Group/conversation name
        """
        self.llm_provider = llm_provider
        self.config = config or ProfileManagerConfig()
        self.group_id = group_id or "default"
        self.group_name = group_name

        # Initialize profile extractor
        self._profile_extractor = ProfileMemoryExtractor(llm_provider=llm_provider)
        self._profile_extractor_life = ProfileMemoryLifeExtractor(
            llm_provider=llm_provider
        )
        # Statistics
        self._stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
        }

    async def extract_profiles(
        self,
        memcells: List[Any],
        old_profiles: Optional[List[Any]] = None,
        user_id_list: Optional[List[str]] = None,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
    ) -> List[Any]:
        """Extract profiles from memcells - pure computation.

        This method performs profile extraction without any storage operations.
        The caller is responsible for loading old profiles and saving new profiles.

        Args:
            memcells: List of memcells to extract profiles from
            old_profiles: Existing profiles for incremental merging (optional)
            user_id_list: List of user IDs to extract profiles for (optional)
            group_id: Override group_id (optional, uses instance default)
            group_name: Override group_name (optional, uses instance default)

        Returns:
            List of extracted ProfileMemory objects
        """
        self._stats["total_extractions"] += 1

        if not memcells:
            logger.warning("No memcells provided for profile extraction")
            return []

        # Use provided or instance values
        gid = group_id or self.group_id
        gname = group_name or self.group_name

        # Limit batch size
        if len(memcells) > self.config.batch_size:
            logger.warning(
                f"Got {len(memcells)} memcells, limiting to {self.config.batch_size} most recent"
            )
            memcells = memcells[-self.config.batch_size :]

        # Build extraction request
        request = ProfileMemoryExtractRequest(
            memcell_list=memcells,
            user_id_list=user_id_list or [],
            group_id=gid,
            group_name=gname,
            old_memory_list=old_profiles if old_profiles else None,
        )

        # Extract profiles with retry logic
        for attempt in range(self.config.max_retries):
            try:
                logger.info(
                    f"Extracting profiles (scenario: {self.config.scenario.value})..."
                )

                if self.config.scenario == ScenarioType.ASSISTANT:
                    result = await self._profile_extractor.extract_profile_companion(
                        request
                    )
                else:
                    result = await self._profile_extractor.extract_memory(request)

                if not result:
                    logger.warning("Profile extraction returned empty result")
                    return []

                self._stats["successful_extractions"] += 1
                logger.info(f"Extracted {len(result)} profiles")

                return result

            except Exception as e:
                logger.warning(
                    f"Profile extraction attempt {attempt + 1}/{self.config.max_retries} failed: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    self._stats["failed_extractions"] += 1
                    logger.error("All profile extraction attempts failed")
                    raise

        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return dict(self._stats)

    def _extract_context_from_memcell(self, memcell: Any) -> Dict[str, Any]:
        """Extract context from MemCell for LLM.

        Supports both MemCell objects and dict representations.

        Returns:
            Context dict with id, created_at, summary, original_data
        """
        if isinstance(memcell, dict):
            # Dict format (from JSON)
            event_id = str(memcell.get("event_id", "") or memcell.get("id", ""))
            created_at = memcell.get("timestamp") or memcell.get("created_at")
            summary = memcell.get("summary", "")
            original_data = memcell.get("original_data", [])
        else:
            # MemCell object
            event_id = (
                str(memcell.event_id)
                if hasattr(memcell, 'event_id') and memcell.event_id
                else ""
            )
            created_at = memcell.timestamp if hasattr(memcell, 'timestamp') else None
            summary = memcell.summary if hasattr(memcell, 'summary') else ""
            original_data = (
                memcell.original_data if hasattr(memcell, 'original_data') else []
            )

        return {
            "id": event_id,
            "created_at": created_at,
            "summary": summary,
            "original_data": original_data,
        }

    # =========================================================================
    # Life Profile Extraction - Explicit info + Implicit traits
    # =========================================================================

    async def extract_profiles_life(
        self,
        memcells: List[Any],
        old_profiles: Optional[List[Any]] = None,
        user_id_list: Optional[List[str]] = None,
        group_id: Optional[str] = None,
        max_items: int = 25,
    ) -> List[ProfileMemoryLife]:
        """Life Profile Extraction - Explicit Information + Implicit Traits (batch multi-user).

        The LLM will see 3 types of information:
        1. old_profile - Current user profile (each entry contains evidence + sources)
        2. cluster_memcells - MemCells from the same cluster (for context reference)
        3. new_memcell - The latest MemCell (last in the list)

        Note: Referenced_episodes are not needed, since each evidence in the profile already explains "why it exists."

        Note: This method only works in the ASSISTANT scenario.

        Args:
            memcells: List of MemCells (last one is new_memcell, others are cluster context)
            old_profiles: List of existing profiles (for incremental updates)
            user_id_list: List of user IDs to extract profiles for
            group_id: Group ID (optional)
            max_items: Maximum number of profile items

        Returns:
            List of ProfileMemoryLife objects; empty list if not in ASSISTANT scenario
        """

        self._stats["total_extractions"] += 1
        # Life Profile only works in ASSISTANT scenario
        if self.config.scenario != ScenarioType.ASSISTANT:
            logger.error(
                f"extract_profiles_life only works in ASSISTANT scenario, "
                f"current scenario: {self.config.scenario.value}"
            )
            return []

        if not memcells:
            logger.error("No memcells provided for Life profile extraction")
            return []

        if not user_id_list:
            logger.error("No user_id_list provided for Life profile extraction")
            return []

        # Last memcell is new_memcell, others are cluster context
        new_memcell = memcells[-1]
        cluster_memcells = memcells[:-1] if len(memcells) > 1 else []

        # Convert memcells to episode dicts for LLM
        new_context = self._extract_context_from_memcell(new_memcell)
        cluster_contexts = [
            self._extract_context_from_memcell(mc) for mc in cluster_memcells
        ]

        # Convert old_profiles list to dict by user_id
        old_profiles_dict: Dict[str, ProfileMemoryLife] = {}
        logger.info(f"[LifeProfile] Processing {len(old_profiles or [])} old profiles")
        for p in old_profiles or []:
            uid = (
                p.get("user_id") if isinstance(p, dict) else getattr(p, "user_id", None)
            )
            p_dict = p if isinstance(p, dict) else p.to_dict()
            has_explicit = "explicit_info" in p_dict
            logger.info(
                f"[LifeProfile] Old profile: user_id={uid}, has_explicit_info={has_explicit}, keys={list(p_dict.keys())[:5]}"
            )
            if uid and has_explicit:
                old_profiles_dict[uid] = ProfileMemoryLife.from_dict(p_dict)
                logger.info(
                    f"[LifeProfile] Loaded profile for {uid}: {old_profiles_dict[uid].total_items()} items"
                )

        results: List[ProfileMemoryLife] = []
        logger.info(
            f"[LifeProfile] user_id_list={user_id_list}, old_profiles_dict keys={list(old_profiles_dict.keys())}"
        )

        # Extract for each user
        for user_id in user_id_list:
            old_profile = old_profiles_dict.get(user_id)
            logger.info(
                f"[LifeProfile] Looking for user_id={user_id}, found={old_profile is not None}"
            )

            # Build request
            request = ProfileMemoryLifeExtractRequest(
                new_episode=new_context,
                cluster_episodes=cluster_contexts,
                old_profile=old_profile,
                user_id=user_id,
                group_id=group_id or self.group_id,
                max_items=max_items,
            )

            # Extract with retry
            for attempt in range(self.config.max_retries):
                try:
                    logger.info(
                        f"Extracting Life profile for user {user_id} (attempt {attempt + 1})..."
                    )

                    result = await self._profile_extractor_life.extract_memory(request)

                    if result:
                        self._stats["successful_extractions"] += 1
                        logger.info(
                            f"Life profile extracted for {user_id}: {result.total_items()} items "
                            f"(explicit: {len(result.explicit_info)}, implicit: {len(result.implicit_traits)})"
                        )
                        results.append(result)
                    else:
                        logger.warning(
                            f"Life profile extraction returned None for {user_id}"
                        )
                        if old_profile:
                            results.append(old_profile)
                    break

                except Exception as e:
                    logger.warning(
                        f"Life profile extraction attempt {attempt + 1} for {user_id} failed: {e}"
                    )
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        logger.error(
                            f"All Life profile extraction attempts failed for {user_id}"
                        )
                        if old_profile:
                            results.append(old_profile)

        return results
