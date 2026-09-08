"""Data processing utilities for group profile extraction."""

from typing import Any, Dict, List, Optional, Set
import re

from core.observation.logger import get_logger

logger = get_logger(__name__)


class GroupProfileDataProcessor:
    """Data processor - encapsulates data validation, transformation, and mapping logic"""

    def __init__(self, conversation_source: str = "original"):
        """
        Initialize the data processor

        Args:
            conversation_source: conversation source, "original" or "episode"
        """
        self.conversation_source = conversation_source

    def validate_and_filter_memcell_ids(
        self,
        memcell_ids: List[str],
        valid_ids: Set[str],
        user_id: Optional[str] = None,
        memcell_list: Optional[List] = None,
    ) -> List[str]:
        """
        Validate and filter memcell_ids (used to validate newly output evidences from LLM)

        Validation rules:
        1. memcell_id must be in valid_ids (existence check)
        2. If user_id is specified, also check whether the user is in memcell.participants

        Args:
            memcell_ids: memcell_ids to be validated (newly output by LLM)
            valid_ids: set of valid memcell_ids (constructed from current memcell_list)
            user_id: optional, if provided, verify whether the user is in participants (used for roles)
            memcell_list: optional, must be provided when user_id is provided

        Returns:
            Filtered valid memcell_ids
        """
        if not memcell_ids:
            return []

        # Step 1: Validate existence
        valid_memcell_ids = [mid for mid in memcell_ids if mid in valid_ids]
        invalid_memcell_ids = [mid for mid in memcell_ids if mid not in valid_ids]

        if invalid_memcell_ids:
            # Show first 5 invalid IDs as examples
            sample_size = min(5, len(invalid_memcell_ids))
            sample_ids = invalid_memcell_ids[:sample_size]
            if len(invalid_memcell_ids) > sample_size:
                logger.warning(
                    f"[validate_and_filter_memcell_ids] Filtered {len(invalid_memcell_ids)} non-existent memcell_ids. "
                    f"Examples: {sample_ids} (and {len(invalid_memcell_ids) - sample_size} more...)"
                )
            else:
                logger.warning(
                    f"[validate_and_filter_memcell_ids] Filtered {len(invalid_memcell_ids)} non-existent memcell_ids: {invalid_memcell_ids}"
                )

        # Step 2: If needed, validate participants
        if user_id is not None:
            if memcell_list is None:
                logger.error(
                    "[validate_and_filter_memcell_ids] user_id provided but memcell_list is None"
                )
                return valid_memcell_ids

            # Build memcell participants mapping
            memcell_participants = {}
            for memcell in memcell_list:
                if hasattr(memcell, 'event_id'):
                    memcell_id = str(memcell.event_id)
                    participants = (
                        set(memcell.participants)
                        if hasattr(memcell, 'participants') and memcell.participants
                        else set()
                    )
                    memcell_participants[memcell_id] = participants

            # Filter: keep only memcells where the user participated
            participant_valid = []
            participant_invalid = []

            for memcell_id in valid_memcell_ids:
                # In theory, memcell_id must be in memcell_participants, use get as fallback
                participants = memcell_participants.get(memcell_id, set())
                if user_id in participants:
                    participant_valid.append(memcell_id)
                else:
                    participant_invalid.append(memcell_id)

            if participant_invalid:
                sample_size = min(3, len(participant_invalid))
                sample_ids = participant_invalid[:sample_size]
                logger.warning(
                    f"[validate_and_filter_memcell_ids] User {user_id} not in participants of {len(participant_invalid)} memcells. "
                    f"Examples: {sample_ids}{'...' if len(participant_invalid) > sample_size else ''}"
                )

            return participant_valid

        return valid_memcell_ids

    def merge_memcell_ids(
        self,
        historical: Optional[List[str]],
        new: List[str],
        valid_ids: Set[str],
        memcell_list: List,
        user_id: Optional[str] = None,
        max_count: int = 50,
    ) -> List[str]:
        """
        Merge historical and new memcell_ids. Keep historical order unchanged, sort only new memcell_ids by timestamp.

        Args:
            historical: historical memcell_ids (no validation, keep original order)
            new: new memcell_ids (need validation, will be sorted by timestamp)
            valid_ids: set of currently valid memcell_ids (used only to validate new memcell_ids)
            memcell_list: current memcell list (used to get timestamps for sorting)
            user_id: optional, if provided, verify whether the user is in participants (used for roles)
            max_count: maximum number to retain

        Returns:
            Merged and deduplicated memcell_ids (historical order unchanged, new ones appended in time order, up to max_count)
        """
        from common_utils.datetime_utils import get_now_with_timezone
        from memory_layer.memory_extractor.group_profile_memory_extractor import convert_to_datetime

        historical = historical or []

        # Historical memcell_ids are kept directly without validation (since corresponding memcells are not in current input)
        # Only validate new memcell_ids (including existence and optional participants check)
        valid_new = self.validate_and_filter_memcell_ids(
            new, valid_ids, user_id=user_id, memcell_list=memcell_list
        )

        # Build mapping from memcell_id to timestamp (used to sort new memcell_ids)
        memcell_id_to_timestamp = {}
        for memcell in memcell_list:
            if hasattr(memcell, 'event_id') and hasattr(memcell, 'timestamp'):
                # Convert to string to match LLM output format
                memcell_id = str(memcell.event_id)
                timestamp = convert_to_datetime(memcell.timestamp)
                memcell_id_to_timestamp[memcell_id] = timestamp

        # Sort new memcell_ids by timestamp (older first, newer last)
        valid_new_sorted = sorted(
            valid_new,
            key=lambda mid: memcell_id_to_timestamp.get(
                mid, get_now_with_timezone().replace(year=1900)
            ),
        )

        # Merge: keep historical order, append new ones (deduplicated)
        seen = set(historical)  # IDs already present in history
        merged = list(historical)  # Keep historical order

        for mid in valid_new_sorted:
            if mid not in seen:
                merged.append(mid)
                seen.add(mid)

        # Limit count (keep the latest, i.e., those at the end of the list)
        if len(merged) > max_count:
            logger.debug(
                f"[merge_memcell_ids] Limiting from {len(merged)} to {max_count} memcell_ids "
                f"(historical: {len(historical)}, new: {len(valid_new)})"
            )
            merged = merged[-max_count:]

        return merged

    def get_comprehensive_speaker_mapping(
        self,
        memcell_list: List,
        existing_roles: Optional[Dict[str, List[Dict[str, str]]]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Get comprehensive speaker mapping, combining current memcell and historical roles information

        Args:
            memcell_list: current memcell list
            existing_roles: historical roles information, format: role -> [{"user_id": "xxx", "user_name": "xxx"}]

        Returns:
            mapping from speaker_id -> {"user_id": speaker_id, "user_name": speaker_name}
        """
        # 1. Build mapping from current memcells
        current_mapping = {}
        for memcell in memcell_list:
            if hasattr(memcell, 'original_data') and memcell.original_data:
                for data in memcell.original_data:
                    speaker_id = data.get('speaker_id', '')
                    speaker_name = data.get('speaker_name', '')
                    if speaker_id and speaker_name:
                        current_mapping[speaker_id] = {
                            "user_id": speaker_id,
                            "user_name": speaker_name,
                        }

        # 2. Extract speaker mapping from historical roles
        historical_mapping = {}
        if existing_roles:
            for role, users in existing_roles.items():
                for user_info in users:
                    user_id = user_info.get("user_id", "")
                    user_name = user_info.get("user_name", "")
                    if (
                        user_id
                        and user_name
                        and user_id not in ["not_found", "unknown"]
                    ):
                        historical_mapping[user_id] = {
                            "user_id": user_id,
                            "user_name": user_name,
                        }

        # 3. Merge mappings: current takes precedence, historical supplements
        comprehensive_mapping = current_mapping.copy()
        for speaker_id, info in historical_mapping.items():
            if speaker_id not in comprehensive_mapping:
                comprehensive_mapping[speaker_id] = info

        return comprehensive_mapping

    def get_conversation_text(self, data_list: List[Any]) -> str:
        """Convert raw data to conversation text format."""
        lines = []
        for data in data_list:
            if hasattr(data, 'content'):
                speaker_name = data.content.get('speaker_name', '')
                speaker_id = data.content.get('speaker_id', '')
                speaker = (
                    f"{speaker_name}(user_id:{speaker_id})"
                    if speaker_id
                    else speaker_name
                )
                content = data.content.get('content')
            else:
                speaker_name = data.get('speaker_name', '')
                speaker_id = data.get('speaker_id', '')
                speaker = (
                    f"{speaker_name}(user_id:{speaker_id})"
                    if speaker_id
                    else speaker_name
                )
                content = data.get('content')

            if not content:
                continue
            # No longer include timestamps to avoid confusing LLM
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def get_episode_text(self, memcell) -> str:
        """Extract episode text from memcell."""
        if hasattr(memcell, 'episode') and memcell.episode:
            return memcell.episode
        return ""

    def combine_conversation_text_with_ids(self, memcell_list: List) -> str:
        """Combine conversation text with memcell IDs for evidence extraction."""
        all_conversation_text = []

        for memcell in memcell_list:
            # Ensure memcell_id is a string (handle MongoDB ObjectId)
            raw_id = getattr(memcell, 'event_id', f'unknown_{id(memcell)}')
            memcell_id = str(raw_id)

            if self.conversation_source == "original":
                # Method 1: use only original_data (current method)
                conversation_text = self.get_conversation_text(memcell.original_data)
                # Use more distinct delimiters to avoid confusion with timestamp brackets
                annotated_text = (
                    f"=== MEMCELL_ID: {memcell_id} ===\n{conversation_text}"
                )
                all_conversation_text.append(annotated_text)

            elif self.conversation_source == "episode":
                # Method 2: use only episode field
                episode_text = self.get_episode_text(memcell)
                if episode_text:
                    annotated_text = f"=== MEMCELL_ID: {memcell_id} ===\n{episode_text}"
                    all_conversation_text.append(annotated_text)
                else:
                    # If no episode, fall back to original_data
                    logger.warning(
                        f"No episode found for memcell {memcell_id}, using original_data as fallback"
                    )
                    conversation_text = self.get_conversation_text(
                        memcell.original_data
                    )
                    annotated_text = f"=== MEMCELL_ID: {memcell_id} ===\n[FALLBACK] {conversation_text}"
                    all_conversation_text.append(annotated_text)

            else:
                raise ValueError(
                    f"Unsupported conversation_source: {self.conversation_source}"
                )

        return "\n\n".join(all_conversation_text)

    def extract_existing_group_profile(
        self, old_memory_list: Optional[List]
    ) -> Optional[Dict]:
        """
        Extract existing group profile from old memories.

        Extracts all topics/roles with their evidences and confidence.
        Returns separate fields for easier processing.
        """
        from datetime import datetime
        from api_specs.memory_types import MemoryType

        if not old_memory_list:
            return None

        for memory in old_memory_list:
            if memory.memory_type == MemoryType.GROUP_PROFILE:
                existing_topics = getattr(memory, "topics", [])
                # Ensure not None
                if existing_topics is None:
                    existing_topics = []

                # Convert TopicInfo objects to dict, preserving evidences and confidence
                topics_list = []
                if existing_topics:
                    for topic in existing_topics:
                        if hasattr(topic, '__dict__'):
                            topic_dict = topic.__dict__.copy()
                            # Convert datetime to ISO string
                            if isinstance(topic_dict.get('last_active_at'), datetime):
                                topic_dict['last_active_at'] = topic_dict[
                                    'last_active_at'
                                ].isoformat()
                            topics_list.append(topic_dict)
                        elif isinstance(topic, dict):
                            topics_list.append(topic)

                # Roles already include evidences and confidence in new format
                existing_roles = getattr(memory, "roles", {})
                # Ensure not None
                if existing_roles is None:
                    existing_roles = {}

                return {
                    "topics": topics_list,  # includes evidences and confidence
                    "summary": getattr(memory, "summary", ""),
                    "subject": getattr(memory, "subject", ""),
                    "roles": existing_roles,  # includes evidences and confidence
                }
        return None

    def get_user_name(
        self, user_id: str, speaker_mapping: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Get user name from comprehensive_speaker_mapping, fallback to user_id if not found"""
        if speaker_mapping and user_id in speaker_mapping:
            return speaker_mapping[user_id]["user_name"]
        return user_id
