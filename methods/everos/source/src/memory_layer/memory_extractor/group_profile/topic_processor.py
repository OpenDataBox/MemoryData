"""Topic management utilities for group profile extraction."""

from typing import List, Dict, Set, Optional
from datetime import datetime
import uuid

from common_utils.datetime_utils import get_now_with_timezone, from_iso_format
from core.observation.logger import get_logger

logger = get_logger(__name__)


class TopicProcessor:
    """Topic processor - responsible for incremental updates and management of topics"""

    def __init__(self, data_processor):
        """
        Initialize topic processor

        Args:
            data_processor: GroupProfileDataProcessor instance, used to validate and merge memcell_ids
        """
        self.data_processor = data_processor

    def find_topic_to_replace(
        self, topics: List, reference_time: Optional[datetime] = None
    ) -> object:
        """
        Find the topic to be replaced
        Strategy:
        1. Prioritize replacing topics that were implemented over 30 days ago
        2. Otherwise, replace the oldest topic (regardless of status)

        This protects recently completed important projects while reasonably phasing out long-implemented outdated topics.

        Args:
            topics: List of TopicInfo objects
            reference_time: Reference time point (usually the latest time in the memcell list).
                           If not provided, use current system time.

        Returns:
            The TopicInfo object to be replaced
        """
        from datetime import timedelta

        # Use provided reference time, otherwise use current time
        now = reference_time if reference_time else get_now_with_timezone()
        threshold = now - timedelta(days=30)  # 30-day threshold

        return sorted(
            topics,
            key=lambda t: (
                # First priority: not "old implemented" (newest and non-implemented come later)
                not (
                    t.status == "implemented" and (t.last_active_at or now) < threshold
                ),
                # Second priority: sort from oldest to newest
                t.last_active_at or get_now_with_timezone().replace(year=1900),
            ),
        )[0]

    def get_latest_memcell_timestamp(
        self, memcell_list: List, memcell_ids: Optional[List[str]] = None
    ) -> datetime:
        """
        Get the latest timestamp from memcell list.

        Args:
            memcell_list: List of all memcells
            memcell_ids: Optional list of memcell IDs to filter by.
                        If provided and not empty, only consider memcells with these IDs.
                        If not provided or empty, consider all memcells.

        Returns:
            Latest timestamp from (filtered) memcells, or current time if no valid timestamp found.
        """
        from memory_layer.memory_extractor.group_profile_memory_extractor import convert_to_datetime

        # If memcell_ids provided and not empty, create a set for fast lookup
        filter_ids = set(memcell_ids) if memcell_ids else None

        latest_time = None
        matched_count = 0

        for memcell in memcell_list:
            # If filter_ids provided, only consider memcells in the filter
            # Convert to string to match format in filter_ids
            if filter_ids and hasattr(memcell, 'event_id'):
                memcell_id_str = str(memcell.event_id)
                if memcell_id_str not in filter_ids:
                    continue

            if hasattr(memcell, 'timestamp') and memcell.timestamp:
                matched_count += 1
                memcell_time = convert_to_datetime(memcell.timestamp)
                if latest_time is None or memcell_time > latest_time:
                    latest_time = memcell_time

        if filter_ids and matched_count > 0:
            logger.debug(
                f"[get_latest_memcell_timestamp] Found {matched_count} memcells matching {len(filter_ids)} IDs"
            )

        return latest_time if latest_time else get_now_with_timezone()

    def apply_topic_incremental_updates(
        self,
        llm_topics: List[Dict],
        existing_topics_with_evidences: List,  # Historical topics containing evidences
        memcell_list: List,
        valid_memcell_ids: Set[str],  # Set of valid memcell_ids
        max_topics: int = 5,
    ) -> List:
        """
        Apply incremental topic updates based on LLM output.

        Now handles evidences merging internally.

        Args:
            llm_topics: List of topics output by LLM (containing evidences and confidence)
            existing_topics_with_evidences: List of historical topics (containing evidences)
            memcell_list: Current memcell list
            valid_memcell_ids: Set of valid memcell_ids
            max_topics: Maximum number of topics

        Returns:
            Processed list of TopicInfo objects (sorted by last_active_at)
        """
        from memory_layer.memory_extractor.group_profile_memory_extractor import TopicInfo

        # Calculate the latest time in memcell list as reference time point
        # Used for time judgment in topic replacement strategy (offline batch processing scenario)
        reference_time = self.get_latest_memcell_timestamp(memcell_list)

        # Parse existing topics (preserving evidences)
        existing_topics = []
        existing_topics_map = {}  # id -> topic_data_with_evidences

        for topic_data in existing_topics_with_evidences:
            if isinstance(topic_data, dict):
                last_active_str = topic_data.get("last_active_at", "")
                try:
                    if last_active_str:
                        last_active_at = from_iso_format(last_active_str)
                    else:
                        last_active_at = get_now_with_timezone()
                except Exception as e:
                    logger.warning(
                        f"Failed to parse last_active_at: {last_active_str}, error: {e}"
                    )
                    last_active_at = get_now_with_timezone()

                topic_id = topic_data.get(
                    "id", f"topic_{str(uuid.uuid4()).replace('-', '')[:8]}"
                )
                topic_info = TopicInfo(
                    id=topic_id,
                    name=topic_data.get("name", ""),
                    summary=topic_data.get("summary", ""),
                    status=topic_data.get("status", "exploring"),
                    last_active_at=last_active_at,
                    evidences=topic_data.get("evidences", []),
                    confidence=topic_data.get("confidence", "strong"),
                    update_type=topic_data.get("update_type") or "new",
                )
                existing_topics.append(topic_info)
                existing_topics_map[topic_id] = topic_data

        # Create topic dict for fast lookup
        topic_dict = {topic.id: topic for topic in existing_topics}

        # Process LLM topics
        for llm_topic in llm_topics:
            update_type = llm_topic.get("update_type") or "new"
            old_topic_id = llm_topic.get("old_topic_id")
            topic_name = llm_topic.get("name", "")

            # Get evidences and confidence from LLM output
            llm_evidences = llm_topic.get("evidences", [])
            llm_confidence = llm_topic.get("confidence", "weak")

            if update_type == "update" and old_topic_id and old_topic_id in topic_dict:
                # Update existing topic - merge historical and new evidences
                old_topic = topic_dict[old_topic_id]
                historical_evidences = old_topic.evidences or []

                # Merge evidences (validation done internally)
                merged_evidences = self.data_processor.merge_memcell_ids(
                    historical=historical_evidences,
                    new=llm_evidences,
                    valid_ids=valid_memcell_ids,
                    memcell_list=memcell_list,
                    max_count=10,
                )

                # Calculate last_active_at based on merged evidences
                last_active_at = self.get_latest_memcell_timestamp(
                    memcell_list, merged_evidences
                )
                logger.debug(
                    f"[TopicIncremental] Topic '{topic_name}' has {len(merged_evidences)} valid evidences, "
                    f"confidence: {llm_confidence}, last_active_at: {last_active_at}"
                )

                updated_topic = TopicInfo(
                    id=old_topic.id,
                    name=llm_topic.get("name", old_topic.name),
                    summary=llm_topic.get("summary", old_topic.summary),
                    status=llm_topic.get("status", old_topic.status),
                    last_active_at=last_active_at,
                    evidences=merged_evidences,
                    confidence=llm_confidence,
                    update_type=update_type,
                )
                topic_dict[old_topic.id] = updated_topic
                logger.debug(
                    f"[TopicIncremental] Updated topic: {old_topic.id} -> {updated_topic.name}, "
                    f"evidences: {len(merged_evidences)}"
                )

            elif update_type == "new":
                # Add new topic - validate evidences
                valid_llm_evidences = (
                    self.data_processor.validate_and_filter_memcell_ids(
                        llm_evidences, valid_memcell_ids
                    )
                )

                # Calculate last_active_at based on evidences
                last_active_at = self.get_latest_memcell_timestamp(
                    memcell_list, valid_llm_evidences
                )
                logger.debug(
                    f"[TopicIncremental] New topic '{topic_name}' has {len(valid_llm_evidences)} valid evidences, "
                    f"confidence: {llm_confidence}, last_active_at: {last_active_at}"
                )

                new_id = f"topic_{str(uuid.uuid4()).replace('-', '')[:8]}"
                new_topic = TopicInfo(
                    id=new_id,
                    name=llm_topic.get("name", ""),
                    summary=llm_topic.get("summary", ""),
                    status=llm_topic.get("status", "exploring"),
                    last_active_at=last_active_at,
                    evidences=valid_llm_evidences,
                    confidence=llm_confidence,
                    update_type=update_type,
                )

                # If at max capacity, replace oldest/implemented topic
                if len(topic_dict) >= max_topics:
                    topic_to_replace = self.find_topic_to_replace(
                        list(topic_dict.values()), reference_time=reference_time
                    )
                    logger.debug(
                        f"[TopicIncremental] Replacing topic: {topic_to_replace.id} ({topic_to_replace.name}) "
                        f"with new topic: {new_topic.name}"
                    )
                    del topic_dict[topic_to_replace.id]

                topic_dict[new_id] = new_topic
                logger.debug(
                    f"[TopicIncremental] Added new topic: {new_id} -> {new_topic.name}"
                )

        # Sort by last_active_at (newest first)
        final_topics = sorted(
            topic_dict.values(),
            key=lambda t: t.last_active_at or datetime.min,
            reverse=True,
        )
        return final_topics
