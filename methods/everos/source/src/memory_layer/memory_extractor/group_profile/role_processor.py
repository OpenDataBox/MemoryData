"""Role management utilities for group profile extraction."""

from typing import Dict, List, Set, Optional
from enum import Enum

from core.observation.logger import get_logger

logger = get_logger(__name__)


class RoleProcessor:
    """Role processor - responsible for role assignment and evidence management"""

    def __init__(self, data_processor):
        """
        Initialize the role processor

        Args:
            data_processor: GroupProfileDataProcessor instance, used to validate and merge memcell_ids
        """
        self.data_processor = data_processor

    def process_roles_with_evidences(
        self,
        role_data: Dict,
        speaker_mapping: Dict[str, Dict[str, str]],
        existing_roles: Dict,
        valid_memcell_ids: Set[str],
        memcell_list: List,
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Process all roles (including weak ones), merge historical evidences, and sort by confidence (strong first)

        Args:
            role_data: Role data output by LLM (includes evidences and confidence)
            speaker_mapping: Mapping from speaker_id to user_name
            existing_roles: Historical role data (includes evidences and confidence)
            valid_memcell_ids: Set of valid memcell_ids
            memcell_list: Current list of memcells (used to get timestamps for sorting)

        Returns:
            Processed roles, formatted as role -> [{"user_id": "xxx", "user_name": "xxx", "confidence": "strong|weak", "evidences": [...]}]
        """
        from memory_layer.memory_extractor.group_profile_memory_extractor import GroupRole

        # Define valid roles list (based on GroupRole enum)
        VALID_ROLES = {role.value for role in GroupRole}

        def validate_and_filter_roles(roles_dict: Dict, source: str) -> Dict:
            """Validate and filter invalid roles"""
            if not roles_dict:
                return {}

            filtered = {}
            invalid_roles = []

            for role_name, assignments in roles_dict.items():
                if role_name in VALID_ROLES:
                    filtered[role_name] = assignments
                else:
                    invalid_roles.append(role_name)

            if invalid_roles:
                logger.warning(
                    f"[process_roles_with_evidences] Filtered out {len(invalid_roles)} invalid roles from {source}: {invalid_roles}"
                )

            return filtered

        # Filter invalid roles in LLM output and historical data
        role_data = validate_and_filter_roles(role_data, "LLM output")
        existing_roles = validate_and_filter_roles(existing_roles, "historical data")

        processed_roles = {}

        # 1. Build mapping for historical roles (role_name, user_id) -> evidences
        historical_role_map = {}
        for role_name, assignments in existing_roles.items():
            for assignment in assignments:
                user_id = assignment.get("user_id", "")
                if user_id:
                    key = (role_name, user_id)
                    historical_role_map[key] = {
                        "evidences": assignment.get("evidences", []),
                        "confidence": assignment.get("confidence", "weak"),
                    }

        # 2. Process roles from LLM output
        for role_name, assignments in role_data.items():
            if not assignments:
                continue

            processed_assignments = []
            for assignment in assignments:
                # Handle both old format (string) and new format (dict)
                if isinstance(assignment, str):
                    # Old format: assignment is just speaker_id, treat as weak
                    speaker_id = assignment
                    confidence = "weak"
                    llm_evidences = []
                elif isinstance(assignment, dict):
                    # New format: assignment has speaker, confidence, evidences
                    speaker_id = assignment.get("speaker", "")
                    confidence = assignment.get("confidence", "weak")
                    llm_evidences = assignment.get("evidences", [])
                else:
                    continue

                if not speaker_id:
                    continue

                user_name = self.data_processor.get_user_name(
                    speaker_id, speaker_mapping
                )

                # 3. Merge historical evidences
                key = (role_name, speaker_id)
                if key in historical_role_map:
                    historical_evidences = historical_role_map[key]["evidences"]
                    historical_confidence = historical_role_map[key]["confidence"]

                    # Merge evidences (will validate if user is in participants)
                    merged_evidences = self.data_processor.merge_memcell_ids(
                        historical=historical_evidences,
                        new=llm_evidences,
                        valid_ids=valid_memcell_ids,
                        memcell_list=memcell_list,
                        user_id=speaker_id,
                        max_count=50,
                    )

                    # Update confidence (if the new one is stronger)
                    if confidence == "strong" or historical_confidence == "strong":
                        final_confidence = "strong"
                    else:
                        final_confidence = confidence
                else:
                    # New role, validate evidences (including participants check)
                    merged_evidences = (
                        self.data_processor.validate_and_filter_memcell_ids(
                            llm_evidences,
                            valid_memcell_ids,
                            user_id=speaker_id,
                            memcell_list=memcell_list,
                        )
                    )
                    final_confidence = confidence

                processed_assignments.append(
                    {
                        "user_id": speaker_id,
                        "user_name": user_name,
                        "confidence": final_confidence,
                        "evidences": merged_evidences,
                    }
                )

            # Sort assignments: strong first, then weak
            processed_assignments.sort(
                key=lambda x: (x["confidence"] != "strong", x["user_name"])
            )

            if processed_assignments:
                processed_roles[role_name] = processed_assignments

        return processed_roles
