"""Profile Manager - Pure computation component for profile extraction.

This module provides ProfileManager, a pure computation component that extracts
user profiles from memcells using LLM.

IMPORTANT: This is a pure computation component. The caller is responsible
for loading old profiles and saving new profiles.

Usage:
    from memory_layer.profile_manager import ProfileManager, ProfileManagerConfig
    
    # Initialize
    config = ProfileManagerConfig(
        scenario="group_chat",  # or "assistant"
        min_confidence=0.6,
    )
    profile_mgr = ProfileManager(llm_provider, config)
    
    # Caller loads old profiles
    old_profiles = list((await storage.get_all_profiles()).values())
    
    # Pure computation - extract profiles
    new_profiles = await profile_mgr.extract_profiles(
        memcells=memcell_list,
        old_profiles=old_profiles,
        user_id_list=["user1", "user2"],
    )
    
    # Caller saves new profiles
    for profile in new_profiles:
        await storage.save_profile(profile.user_id, profile)
"""

from memory_layer.profile_manager.config import ProfileManagerConfig, ScenarioType
from memory_layer.profile_manager.manager import ProfileManager

__all__ = [
    "ProfileManager",
    "ProfileManagerConfig",
    "ScenarioType",
]
