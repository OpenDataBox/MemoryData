"""LLM interaction handler for group profile extraction."""

import asyncio
import json
import re
from typing import Dict, List, Optional

from core.observation.logger import get_logger

logger = get_logger(__name__)


class GroupProfileLLMHandler:
    """LLM interaction processor - responsible for prompt construction, invocation, and response parsing"""

    def __init__(self, llm_provider, max_topics: int):
        """
        Initialize LLM processor

        Args:
            llm_provider: LLM provider instance
            max_topics: Maximum number of topics
        """
        self.llm_provider = llm_provider
        self.max_topics = max_topics

    async def execute_parallel_analysis(
        self,
        conversation_text: str,
        group_id: str,
        group_name: str,
        memcell_list: List,
        existing_profile: Optional[Dict],
        user_organization: Optional[List],
        timespan: str,
    ) -> Optional[Dict]:
        """
        Execute parallel LLM analysis and return parsed results

        Encapsulates:
        - Building prompts
        - Parallel LLM invocation
        - Response parsing
        - Result merging

        Args:
            conversation_text: Conversation text
            group_id: Group ID
            group_name: Group name
            memcell_list: List of memcells
            existing_profile: Historical profile data
            user_organization: User organization information
            timespan: Time span

        Returns:
            Parsed result dictionary containing topics, roles, summary, subject
        """
        # Extract historical data
        existing_topics = existing_profile.get("topics", []) if existing_profile else []
        existing_summary = (
            existing_profile.get("summary", "") if existing_profile else ""
        )
        existing_subject = (
            existing_profile.get("subject", "") if existing_profile else ""
        )
        existing_roles = existing_profile.get("roles", {}) if existing_profile else {}

        # Build prompts
        content_prompt = self.build_content_analysis_prompt(
            conversation_text,
            group_id,
            group_name,
            existing_topics,
            existing_summary,
            existing_subject,
            timespan,
        )
        behavior_prompt = self.build_behavior_analysis_prompt(
            conversation_text,
            group_id,
            group_name,
            memcell_list,
            existing_roles,
            user_organization,
        )

        # Parallel execution
        logger.info(f"[LLMHandler] Executing parallel analysis for group: {group_name}")
        content_task = self._execute_with_retry(
            "Content Analysis",
            lambda: self.llm_provider.generate(content_prompt, temperature=0.3),
        )
        behavior_task = self._execute_with_retry(
            "Behavior Analysis",
            lambda: self.llm_provider.generate(behavior_prompt, temperature=0.3),
        )

        content_response, behavior_response = await asyncio.gather(
            content_task, behavior_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(content_response, Exception):
            logger.error(f"[Content Analysis] Failed: {content_response}")
            content_response = None
        if isinstance(behavior_response, Exception):
            logger.error(f"[Behavior Analysis] Failed: {behavior_response}")
            behavior_response = None

        # Merge and parse results
        return self._merge_parallel_analysis_results(
            content_response,
            behavior_response,
            group_id,
            group_name,
            memcell_list,
            existing_profile,
        )

    def build_content_analysis_prompt(
        self,
        conversation_text: str,
        group_id: str,
        group_name: str,
        existing_topics: List,
        existing_summary: str,
        existing_subject: str,
        timespan: str,
    ) -> str:
        """
        Build content analysis prompt for topics, summary, and subject extraction.

        Only passes topics-related info to LLM, excluding evidences (but keeping confidence).
        """
        from memory_layer.prompts import get_prompt_by
        CONTENT_ANALYSIS_PROMPT = get_prompt_by("CONTENT_ANALYSIS_PROMPT")

        # Build existing_profile for LLM, including only required fields, excluding evidences
        existing_profile_for_llm = {
            "topics": (
                [
                    {
                        "id": t.id if hasattr(t, 'id') else t.get("id"),
                        "name": t.name if hasattr(t, 'name') else t.get("name"),
                        "summary": (
                            t.summary if hasattr(t, 'summary') else t.get("summary")
                        ),
                        "status": t.status if hasattr(t, 'status') else t.get("status"),
                        "confidence": (
                            t.confidence
                            if hasattr(t, 'confidence')
                            else t.get("confidence", "strong")
                        ),
                        # Exclude evidences field
                    }
                    for t in existing_topics
                ]
                if existing_topics
                else []
            ),
            "summary": existing_summary,
            "subject": existing_subject,
        }

        existing_profile_json = json.dumps(existing_profile_for_llm, ensure_ascii=False)

        return CONTENT_ANALYSIS_PROMPT.format(
            conversation=conversation_text,
            group_id=group_id or "",
            group_name=group_name or "",
            existing_profile=existing_profile_json,
            timespan=timespan,
            max_topics=self.max_topics,
        )

    def build_behavior_analysis_prompt(
        self,
        conversation_text: str,
        group_id: str,
        group_name: str,
        memcell_list: List,
        existing_roles: Dict,
        user_organizations: Optional[List] = None,
    ) -> str:
        """
        Build behavior analysis prompt for roles extraction.

        Only passes roles info to LLM, excluding evidences (but keeping confidence).
        """
        from memory_layer.prompts import get_prompt_by
        BEHAVIOR_ANALYSIS_PROMPT = get_prompt_by("BEHAVIOR_ANALYSIS_PROMPT")

        # Build existing_profile for LLM, including only roles, excluding evidences
        existing_profile_for_llm = {
            "roles": (
                {
                    role_name: [
                        {
                            "user_id": a.get("user_id"),
                            "user_name": a.get("user_name"),
                            "confidence": a.get("confidence", "strong"),
                            # Exclude evidences field
                        }
                        for a in assignments
                    ]
                    for role_name, assignments in existing_roles.items()
                }
                if existing_roles
                else {}
            )
        }

        existing_profile_json = json.dumps(existing_profile_for_llm, ensure_ascii=False)

        # Build speaker info (requires help from data_processor, extract from conversation_text here)
        # Extract current speakers from conversation
        current_speakers = set()
        for line in conversation_text.split('\n'):
            match = re.search(r'\(user_id:([^)]+)\):', line)
            if match:
                speaker_id = match.group(1).strip()
                current_speakers.add(speaker_id)

        # Extract all known speakers from historical roles
        all_available_speakers = current_speakers.copy()
        for role_assignments in existing_roles.values():
            for assignment in role_assignments:
                speaker_id = assignment.get("user_id")
                if speaker_id:
                    all_available_speakers.add(speaker_id)

        # Build organization mapping
        org_mapping = {}
        if user_organizations:
            for org in user_organizations:
                org_mapping[org.user_id] = {
                    "full_name": org.full_name,
                    "team": org.team or "Unknown Team",
                    "role": org.role or "Unknown Role",
                    "direct_manager": org.direct_manager or "Unknown Manager",
                    "skip_level_manager": org.skip_level_manager
                    or "Unknown Skip Manager",
                }

        # Build speaker info string
        has_org_info = bool(user_organizations and org_mapping)
        speaker_info_title = (
            "**Available Speakers with Organization Context:**"
            if has_org_info
            else "**Available Speakers:**"
        )
        speaker_info = f"\n{speaker_info_title}\n"

        # Try to extract speaker names from memcells
        speaker_names = {}
        for memcell in memcell_list:
            if hasattr(memcell, 'original_data') and memcell.original_data:
                for data in memcell.original_data:
                    speaker_id = data.get('speaker_id', '')
                    speaker_name = data.get('speaker_name', '')
                    if speaker_id and speaker_name:
                        speaker_names[speaker_id] = speaker_name

        for speaker_id in sorted(all_available_speakers):
            speaker_name = speaker_names.get(speaker_id, "Unknown")

            # Add organization info if available
            org_info = ""
            if has_org_info and speaker_id in org_mapping:
                org_data = org_mapping[speaker_id]
                org_info = f" | Role: {org_data['role']} | Team: {org_data['team']} | Manager: {org_data['direct_manager']}"
                if org_data['skip_level_manager'] != "Unknown Skip Manager":
                    org_info += f" | Skip Manager: {org_data['skip_level_manager']}"

            speaker_info += f"- {speaker_id}: {speaker_name}{org_info}\n"

        if not has_org_info:
            speaker_info += (
                "\nNote: Organization context not available for this analysis.\n"
            )

        return BEHAVIOR_ANALYSIS_PROMPT.format(
            conversation=conversation_text,
            group_id=group_id or "",
            group_name=group_name or "",
            existing_profile=existing_profile_json,
            speaker_info=speaker_info,
        )

    async def _execute_with_retry(
        self, task_name: str, task_func, max_retries: int = 1
    ) -> Optional[str]:
        """Execute LLM task with retry mechanism."""
        for attempt in range(max_retries + 1):
            try:
                response = await task_func()
                logger.debug(
                    f"[{task_name}] Success on attempt {attempt + 1}, response length: {len(response)} chars"
                )
                return response
            except Exception as e:
                if attempt < max_retries:
                    logger.exception(
                        f"[{task_name}] Attempt {attempt + 1} failed: {e}, retrying..."
                    )
                else:
                    logger.exception(
                        f"[{task_name}] All {max_retries + 1} attempts failed: {e}"
                    )
                    return None
        return None

    def _parse_content_response(
        self, response: str, existing_profile: Optional[Dict], memcell_list: List
    ) -> Dict:
        """Parse content analysis response to extract topics, summary, and subject.

        Returns raw topic data (list of dicts) without converting to TopicInfo yet,
        so that evidences and confidence can be extracted first.
        """
        try:
            # Extract JSON from response
            json_match = re.search(
                r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL
            )
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response)

            # Return raw data without converting to TopicInfo yet
            # Conversion will happen in extract_memory after evidence extraction
            return {
                "topics": data.get("topics", []),  # Keep as raw dicts
                "summary": data.get("summary", ""),
                "subject": data.get("subject", "not_found"),
            }

        except Exception as e:
            logger.exception(f"[ContentParse] Error parsing content response: {e}")
            return self._get_content_fallback(existing_profile)

    def _parse_behavior_response(
        self, response: str, memcell_list: List, existing_profile: Optional[Dict]
    ) -> Dict:
        """Parse behavior analysis response to extract roles.

        Returns raw role data (dict with speaker/evidences/confidence) without filtering,
        so that evidences can be extracted first.
        """
        try:
            # Extract JSON from response
            json_match = re.search(
                r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL
            )
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response)

            logger.debug(f"[_parse_behavior_response] Parsed data: {data}")

            # Return raw data without filtering
            # Filtering will happen in extract_memory after evidence extraction
            return {"roles": data.get("roles", {})}

        except Exception as e:
            logger.error(
                f"[BehaviorParse] Error parsing behavior response: {e}", exc_info=True
            )
            return self._get_behavior_fallback(existing_profile)

    def _get_content_fallback(self, existing_profile: Optional[Dict]) -> Dict:
        """Get fallback content data from existing profile."""
        if existing_profile:
            return {
                "topics": existing_profile.get("topics", []),
                "summary": existing_profile.get("summary", ""),
                "subject": existing_profile.get("subject", "not_found"),
            }
        return {"topics": [], "summary": "", "subject": "not_found"}

    def _get_behavior_fallback(self, existing_profile: Optional[Dict]) -> Dict:
        """Get fallback behavior data from existing profile."""
        if existing_profile:
            return {"roles": existing_profile.get("roles", {})}
        return {"roles": {}}

    def _merge_parallel_analysis_results(
        self,
        content_response: Optional[str],
        behavior_response: Optional[str],
        group_id: str,
        group_name: str,
        memcell_list: List,
        existing_profile: Optional[Dict],
    ) -> Optional[Dict]:
        """Merge content and behavior analysis results into a single result."""
        try:
            # Parse content analysis results
            content_data = {}
            if content_response:
                content_data = self._parse_content_response(
                    content_response, existing_profile, memcell_list
                )
            else:
                logger.debug(
                    "[Merge] Content analysis failed, using fallback from existing profile"
                )
                content_data = self._get_content_fallback(existing_profile)

            # Parse behavior analysis results
            behavior_data = {}
            if behavior_response:
                behavior_data = self._parse_behavior_response(
                    behavior_response, memcell_list, existing_profile
                )
            else:
                logger.debug(
                    "[Merge] Behavior analysis failed, using fallback from existing profile"
                )
                behavior_data = self._get_behavior_fallback(existing_profile)

            # Merge into final result
            merged_result = {
                "topics": content_data.get("topics", []),
                "summary": content_data.get("summary", ""),
                "subject": content_data.get("subject", "not_found"),
                "roles": behavior_data.get("roles", {}),
            }

            logger.debug(
                f"[Merge] Successfully merged parallel analysis results: "
                f"{len(content_data.get('topics', []))} topics, {len(behavior_data.get('roles', {}))} role types"
            )
            return merged_result

        except Exception as e:
            logger.error(
                f"[Merge] Error merging parallel analysis results: {e}", exc_info=True
            )
            return None
