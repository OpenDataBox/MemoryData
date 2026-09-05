"""Life Profile Memory Extractor - Explicit info + Implicit traits extractor.

Core features:
1. Separation of explicit info and implicit traits
2. Incremental updates per episode
3. ID mapping (to save tokens)
4. Flexible categorization
5. Multi-language support (controlled by MEMORY_LANGUAGE env var)
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ProfileAction(str, Enum):
    """Profile update actions from LLM."""

    NONE = "none"
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


class ProfileItemType(str, Enum):
    """Profile item types."""

    EXPLICIT_INFO = "explicit_info"
    IMPLICIT_TRAITS = "implicit_traits"


from common_utils.datetime_utils import get_now_with_timezone
from core.observation.logger import get_logger
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.base_memory_extractor import MemoryExtractor
from memory_layer.memory_extractor.profile_memory_life.types import (
    ProfileMemoryLife,
    ExplicitInfo,
    ImplicitTrait,
    ProfileMemoryLifeExtractRequest,
)
from memory_layer.memory_extractor.profile_memory_life.id_mapper import (
    create_id_mapping,
    replace_sources,
    get_short_id,
)
from memory_layer.prompts import get_prompt_by
from api_specs.memory_types import MemoryType

logger = get_logger(__name__)


class ProfileMemoryLifeExtractor(MemoryExtractor):
    """Life Profile Extractor - Supports explicit info + implicit traits."""

    DEFAULT_MAX_ITEMS = 25

    def __init__(self, llm_provider: LLMProvider):
        super().__init__(MemoryType.PROFILE)
        self.llm_provider = llm_provider

    async def extract_memory(
        self, request: ProfileMemoryLifeExtractRequest
    ) -> Optional[ProfileMemoryLife]:
        """Extract Life Profile from 3 types of information.

        LLM receives:
        1. old_profile - Current user profile (each item has evidence + sources)
        2. cluster_episodes - Conversations in the same topic cluster (context reference)
        3. new_episode - Latest conversation record

        Note: referenced_episodes not needed as profile's evidence already explains "why it holds"
        """
        new_episode = request.new_episode
        cluster_episodes = request.cluster_episodes or []
        old_profile = request.old_profile
        max_items = request.max_items or self.DEFAULT_MAX_ITEMS

        # Backward compatibility with old episode_list mode
        if not new_episode and request.episode_list:
            # Old mode: use the last episode as new_episode
            episodes = request.episode_list
            if episodes:
                new_episode = episodes[-1]
                # Others as cluster_episodes
                cluster_episodes = episodes[:-1] if len(episodes) > 1 else []

        if not new_episode:
            logger.warning("No new episode provided for Life profile extraction")
            return old_profile

        # Initialize profile
        if old_profile is None:
            logger.info(
                f"[LifeExtractor] No old_profile for user={request.user_id}, creating new"
            )
            current_profile = ProfileMemoryLife(
                memory_type=MemoryType.PROFILE,
                user_id=request.user_id or "",
                group_id=request.group_id or "",
                timestamp=get_now_with_timezone(),
                ori_event_id_list=[],
            )
        else:
            logger.info(
                f"[LifeExtractor] Using old_profile for user={request.user_id}: "
                f"explicit={len(old_profile.explicit_info)}, implicit={len(old_profile.implicit_traits)}"
            )
            current_profile = old_profile

        # Check if already processed
        ep_id = new_episode.get("id")
        if ep_id in current_profile.processed_episode_ids:
            logger.info(f"Episode {ep_id} already processed, skipping")
            return current_profile

        # Create ID mapping (stateless)
        all_ids = (
            list(current_profile.processed_episode_ids)
            + [ep.get("id") for ep in cluster_episodes]
            + [new_episode.get("id")]
        )
        id_map = create_id_mapping(all_ids)

        logger.info(f"Processing Life profile: cluster={len(cluster_episodes)}, new=1")

        # Call LLM to update (pass 3 types of info)
        updated_dict = await self._llm_update_profile(
            current_profile=current_profile,
            cluster_episodes=cluster_episodes,
            new_episode=new_episode,
            id_map=id_map,
        )

        if updated_dict:
            # Update profile
            # Filter out empty items
            current_profile.explicit_info = [
                ExplicitInfo.from_dict(d)
                for d in updated_dict.get(ProfileItemType.EXPLICIT_INFO, [])
                if d.get("description", "").strip()  # Must have description
            ]
            current_profile.implicit_traits = [
                ImplicitTrait.from_dict(d)
                for d in updated_dict.get(ProfileItemType.IMPLICIT_TRAITS, [])
                if d.get("description", "").strip()  # Must have description
            ]
            current_profile.last_updated = get_now_with_timezone()

        # Mark as processed
        new_ep_id = new_episode.get("id", "")
        if new_ep_id:
            current_profile.processed_episode_ids.append(new_ep_id)

        # Check capacity (with buffer to avoid frequent compacting)
        # Compact threshold = max_items * 1.5 (e.g., 25 -> 37)
        # Compact target = max_items * 0.7 (e.g., 25 -> 17), leaving room for new items
        compact_threshold = int(max_items * 1.5)
        compact_target = int(max_items * 0.7)

        if current_profile.total_items() > compact_threshold:
            logger.info(
                f"Profile has {current_profile.total_items()} items (threshold={compact_threshold}), compacting to {compact_target}..."
            )
            current_profile = await self._compact_profile(
                current_profile, compact_target, id_map
            )

        return current_profile

    async def _llm_update_profile(
        self,
        current_profile: ProfileMemoryLife,
        cluster_episodes: List[Dict[str, Any]],
        new_episode: Dict[str, Any],
        id_map: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Call LLM for incremental update using operations-based approach.

        LLM outputs operations (add/update/delete/none) instead of full profile.
        This prevents accidental deletion of items.
        """

        # Convert to dict and use short IDs
        profile_dict = current_profile.to_dict()
        profile_short = replace_sources(profile_dict, id_map)

        # Format profile with index numbers
        profile_text = self._format_profile_with_index(profile_short)

        # Combine cluster_episodes + new_episode into one conversations block
        all_episodes = (cluster_episodes or []) + ([new_episode] if new_episode else [])
        conversations_text = self._format_episodes_for_llm(all_episodes, id_map)

        # Get prompt template
        prompt_template = get_prompt_by("PROFILE_LIFE_UPDATE_PROMPT")
        prompt = prompt_template.format(
            current_profile=profile_text if profile_text else "(Empty, no records yet)",
            conversations=(
                conversations_text if conversations_text else "(No conversations)"
            ),
        )

        try:
            response = await self.llm_provider.generate(prompt, temperature=0.3)
            result = self._parse_profile_response(response)
            if not result:
                return None

            # Apply operations to current profile
            operations = result.get("operations", [])

            # Start with current profile as base
            explicit_list = [info.to_dict() for info in current_profile.explicit_info]
            implicit_list = [
                trait.to_dict() for trait in current_profile.implicit_traits
            ]

            # Build timestamp mapping
            id_to_ts = self._build_timestamp_map(
                current_profile, cluster_episodes, new_episode
            )

            for op in operations:
                action = op.get("action", ProfileAction.NONE)

                if action == ProfileAction.NONE:
                    continue

                elif action == ProfileAction.ADD:
                    op_type = op.get("type")
                    data = op.get("data", {})
                    if not data.get("description", "").strip():
                        continue
                    # Attach timestamps to sources
                    data["sources"] = [
                        self._attach_ts(s, id_to_ts) for s in data.get("sources", [])
                    ]
                    if op_type == ProfileItemType.EXPLICIT_INFO:
                        explicit_list.append(data)
                        logger.info(
                            f"[Profile] Added explicit_info: {data.get('description', '')[:30]}..."
                        )
                    elif op_type == ProfileItemType.IMPLICIT_TRAITS:
                        implicit_list.append(data)
                        logger.info(
                            f"[Profile] Added implicit_trait: {data.get('trait', '')}..."
                        )

                elif action == ProfileAction.UPDATE:
                    op_type = op.get("type")
                    index = op.get("index", -1)
                    data = op.get("data", {})
                    target_list = (
                        explicit_list
                        if op_type == ProfileItemType.EXPLICIT_INFO
                        else implicit_list
                    )
                    if 0 <= index < len(target_list):
                        # Merge data into existing item
                        for key, val in data.items():
                            if val:  # Only update non-empty values
                                if key == "sources":
                                    # Merge sources
                                    old_sources = target_list[index].get("sources", [])
                                    new_sources = [
                                        self._attach_ts(s, id_to_ts) for s in val
                                    ]
                                    target_list[index]["sources"] = list(
                                        set(old_sources + new_sources)
                                    )
                                else:
                                    target_list[index][key] = val
                        logger.info(f"[Profile] Updated {op_type}[{index}]")

                elif action == ProfileAction.DELETE:
                    op_type = op.get("type")
                    index = op.get("index", -1)
                    reason = op.get("reason", "")
                    target_list = (
                        explicit_list
                        if op_type == ProfileItemType.EXPLICIT_INFO
                        else implicit_list
                    )
                    if 0 <= index < len(target_list) and reason:
                        deleted = target_list.pop(index)
                        logger.warning(
                            f"[Profile] Deleted {op_type}[{index}]: {reason}"
                        )

            # Convert short IDs back to long IDs
            result_dict = {
                ProfileItemType.EXPLICIT_INFO: explicit_list,
                ProfileItemType.IMPLICIT_TRAITS: implicit_list,
            }
            result_long = replace_sources(result_dict, id_map, reverse=True)

            return result_long

        except Exception as e:
            logger.error(f"LLM update profile failed: {e}")
            return None

    def _build_timestamp_map(
        self,
        profile: ProfileMemoryLife,
        cluster_episodes: List[Dict[str, Any]],
        new_episode: Dict[str, Any],
    ) -> Dict[str, str]:
        """Build episode_id -> timestamp mapping."""
        id_to_ts = {}

        # From old profile sources
        for info in profile.explicit_info:
            for src in info.sources:
                if "|" in src:
                    ts, eid = src.rsplit("|", 1)
                    id_to_ts[eid.strip()] = ts.strip()
        for trait in profile.implicit_traits:
            for src in trait.sources:
                if "|" in src:
                    ts, eid = src.rsplit("|", 1)
                    id_to_ts[eid.strip()] = ts.strip()

        # From current episodes
        for ep in (cluster_episodes or []) + ([new_episode] if new_episode else []):
            eid = ep.get("id")
            ts = self._format_timestamp(ep.get("created_at"))
            if eid and ts:
                id_to_ts[str(eid)] = ts

        return id_to_ts

    def _attach_ts(self, s: Any, id_to_ts: Dict[str, str]) -> str:
        """Attach timestamp to source if missing."""
        if not isinstance(s, str) or not s:
            return s
        if "|" in s:
            return s
        sid = s.strip()
        ts = id_to_ts.get(sid)
        return f"{ts}|{sid}" if ts else sid

    def _format_profile_with_index(self, profile_dict: Dict[str, Any]) -> str:
        """Format Profile with index numbers for LLM."""
        explicit = profile_dict.get(ProfileItemType.EXPLICIT_INFO, [])
        implicit = profile_dict.get(ProfileItemType.IMPLICIT_TRAITS, [])

        if not explicit and not implicit:
            return ""

        lines = []
        if explicit:
            lines.append("【Explicit Info】")
            for i, item in enumerate(explicit):
                cat = item.get("category", "")
                desc = item.get("description", "")
                evidence = item.get("evidence", "")
                lines.append(f"  [{i}] [{cat}] {desc}")
                if evidence:
                    lines.append(f"      evidence: {evidence}")

        if implicit:
            lines.append("\n【Implicit Traits】")
            for i, item in enumerate(implicit):
                name = item.get("trait", "")
                desc = item.get("description", "")
                evidence = item.get("evidence", "")
                lines.append(f"  [{i}] {name}: {desc}")
                if evidence:
                    lines.append(f"      evidence: {evidence}")

        return "\n".join(lines)

    async def _compact_profile(
        self, profile: ProfileMemoryLife, max_items: int, id_map: Dict[str, str]
    ) -> ProfileMemoryLife:
        """Let LLM compact the over-limit Profile."""

        profile_dict = profile.to_dict()
        profile_short = replace_sources(profile_dict, id_map)
        profile_text = self._format_profile_for_llm(profile_short)
        total = profile.total_items()

        # Get prompt template
        prompt_template = get_prompt_by("PROFILE_LIFE_COMPACT_PROMPT")
        prompt = prompt_template.format(
            total_items=total, max_items=max_items, profile_text=profile_text
        )

        try:
            response = await self.llm_provider.generate(prompt, temperature=0.3)
            result = self._parse_profile_response(response)

            if result:
                result_long = replace_sources(result, id_map, reverse=True)
                # Filter out empty items
                profile.explicit_info = [
                    ExplicitInfo.from_dict(d)
                    for d in result_long.get(ProfileItemType.EXPLICIT_INFO, [])
                    if d.get("description", "").strip()
                ]
                profile.implicit_traits = [
                    ImplicitTrait.from_dict(d)
                    for d in result_long.get(ProfileItemType.IMPLICIT_TRAITS, [])
                    if d.get("description", "").strip()
                ]
                profile.last_updated = get_now_with_timezone()

            return profile

        except Exception as e:
            logger.error(f"LLM compact profile failed: {e}")
            return profile

    def _merge_missing_items(
        self, old_profile: ProfileMemoryLife, new_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge back items that LLM accidentally dropped.

        Compare old profile with LLM output, add back any missing items.
        Uses description similarity to detect if an item was kept or dropped.
        """
        # Build set of descriptions from new result
        new_explicit_descs = {
            item.get("description", "").lower().strip()
            for item in (new_dict.get(ProfileItemType.EXPLICIT_INFO) or [])
        }
        new_implicit_descs = {
            item.get("description", "").lower().strip()
            for item in (new_dict.get(ProfileItemType.IMPLICIT_TRAITS) or [])
        }

        # Check for dropped explicit_info
        dropped_explicit = []
        for info in old_profile.explicit_info:
            desc_lower = info.description.lower().strip() if info.description else ""
            if desc_lower and desc_lower not in new_explicit_descs:
                # This item was dropped - add it back
                dropped_explicit.append(info.to_dict())
                logger.warning(
                    f"[Profile] Recovered dropped explicit_info: {info.description[:50]}..."
                )

        # Check for dropped implicit_traits
        dropped_implicit = []
        for trait in old_profile.implicit_traits:
            desc_lower = trait.description.lower().strip() if trait.description else ""
            if desc_lower and desc_lower not in new_implicit_descs:
                # This item was dropped - add it back
                dropped_implicit.append(trait.to_dict())
                logger.warning(
                    f"[Profile] Recovered dropped implicit_trait: {trait.trait_name}..."
                )

        # Merge back dropped items
        if dropped_explicit:
            new_dict[ProfileItemType.EXPLICIT_INFO] = (
                new_dict.get(ProfileItemType.EXPLICIT_INFO) or []
            ) + dropped_explicit
            logger.info(
                f"[Profile] Merged back {len(dropped_explicit)} dropped explicit_info items"
            )

        if dropped_implicit:
            new_dict[ProfileItemType.IMPLICIT_TRAITS] = (
                new_dict.get(ProfileItemType.IMPLICIT_TRAITS) or []
            ) + dropped_implicit
            logger.info(
                f"[Profile] Merged back {len(dropped_implicit)} dropped implicit_traits items"
            )

        return new_dict

    def _format_profile_for_llm(self, profile_dict: Dict[str, Any]) -> str:
        """Format Profile dict into LLM-readable text."""
        explicit = profile_dict.get(ProfileItemType.EXPLICIT_INFO, [])
        implicit = profile_dict.get(ProfileItemType.IMPLICIT_TRAITS, [])

        if not explicit and not implicit:
            return ""

        lines = []
        if explicit:
            lines.append("【Explicit Info】")
            for i, item in enumerate(explicit, 1):
                cat = item.get("category", "")
                desc = item.get("description", "")
                evidence = item.get("evidence", "")
                sources = item.get("sources", [])
                lines.append(f"  {i}. [{cat}] {desc}")
                if evidence:
                    lines.append(f"     evidence: {evidence}")
                lines.append(f"     sources: {', '.join(sources)}")

        if implicit:
            lines.append("\n【Implicit Traits】")
            for i, item in enumerate(implicit, 1):
                name = item.get("trait", "")
                desc = item.get("description", "")
                basis = item.get("basis", "")
                evidence = item.get("evidence", "")
                sources = item.get("sources", [])
                lines.append(f"  {i}. {name}: {desc}")
                if basis:
                    lines.append(f"     basis: {basis}")
                if evidence:
                    lines.append(f"     evidence: {evidence}")
                lines.append(f"     sources: {', '.join(sources)}")

        return "\n".join(lines)

    def _format_episodes_for_llm(
        self, episodes: List[Dict[str, Any]], id_map: Dict[str, str]
    ) -> str:
        """Format Episode list into LLM-readable text (using short IDs)."""
        if not episodes:
            return ""

        lines = []
        for ep in episodes:
            long_id = ep.get("id")
            short_id = get_short_id(long_id, id_map)
            timestamp = self._format_timestamp(ep.get("created_at"))

            lines.append(f"[{short_id}] ({timestamp})")

            # Use original_data (actual messages) instead of summary
            original_data = ep.get("original_data", [])
            if original_data and isinstance(original_data, list):
                for msg in original_data:
                    speaker = msg.get("speaker_name", "Unknown")
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                    if content:
                        lines.append(f"  [{timestamp}]【{speaker}】: {content}\n\n")
            else:
                # Fallback to summary if no original_data
                episode = ep.get("episode")
                if episode:
                    lines.append(f"  [Episode Memory] {episode}")

            lines.append("")

        return "\n".join(lines)

    def _format_single_episode(self, ep: Dict[str, Any], id_map: Dict[str, str]) -> str:
        """Format single episode (using short ID)."""
        long_id = ep.get("id", "unknown")
        short_id = get_short_id(long_id, id_map)
        timestamp = self._format_timestamp(ep.get("created_at"))

        lines = [f"[{short_id}] ({timestamp})"]

        # Use original_data (actual messages) instead of summary
        original_data = ep.get("original_data", [])
        if original_data and isinstance(original_data, list):
            for msg in original_data:
                speaker = msg.get("speaker_name", "Unknown")
                content = msg.get("content", "")
                if content:
                    lines.append(f"  {speaker}: {content}")
        else:
            # Fallback to summary if no original_data
            summary = ep.get("episode") or ep.get("summary", "")
            if summary:
                lines.append(f"  [Summary] {summary}")

        return "\n".join(lines)

    def _format_timestamp(self, ts: Any) -> str:
        """Convert various timestamp formats to string."""
        if not ts:
            return ""
        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M")
        if isinstance(ts, str):
            return ts[:16] if len(ts) >= 16 else ts
        return str(ts)[:16]

    def _parse_profile_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse Profile JSON returned by LLM."""
        if not response:
            return None

        # Try to extract from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        raw_json = json_match.group(1) if json_match else response

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            # Try to find JSON directly
            brace_start = response.find("{")
            brace_end = response.rfind("}") + 1
            if brace_start >= 0 and brace_end > brace_start:
                try:
                    data = json.loads(response[brace_start:brace_end])
                except Exception:
                    logger.warning("Failed to parse profile response JSON")
                    return None
            else:
                logger.warning("No JSON found in profile response")
                return None

        # Log update note
        update_note = data.get("update_note") or data.get("compact_note")
        if update_note:
            logger.info(f"Profile update: {update_note}")

        return data
