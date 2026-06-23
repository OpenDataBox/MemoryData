"""Compatibility facade aggregating profile normalization helpers.

Historically this module contained all helper implementations. The functionality
now lives in dedicated modules, but we re-export the same public API so that
existing imports keep working.
"""

from __future__ import annotations

from memory_layer.memory_extractor.profile_memory.evidence_utils import (
    conversation_id_from_evidence,
    ensure_str_list,
    format_evidence_entry,
    merge_evidences_recursive,
)
from memory_layer.memory_extractor.profile_memory.profile_helpers import (
    accumulate_old_memory_entry,
    merge_profiles,
    merge_single_profile,
    profile_payload_to_memory,
    remove_evidences_from_profile,
)
from memory_layer.memory_extractor.profile_memory.project_helpers import (
    convert_projects_to_dataclass,
    merge_projects_participated,
    project_to_dict,
)
from memory_layer.memory_extractor.profile_memory.skill_helpers import merge_skill_lists, normalize_skills_with_evidence
from memory_layer.memory_extractor.profile_memory.value_helpers import (
    extract_values_with_evidence,
    merge_value_with_evidences_lists,
)

__all__ = [
    # Evidence utilities
    "ensure_str_list",
    "format_evidence_entry",
    "conversation_id_from_evidence",
    "merge_evidences_recursive",
    # Value helpers
    "merge_value_with_evidences_lists",
    "extract_values_with_evidence",
    # Skill helpers
    "normalize_skills_with_evidence",
    "merge_skill_lists",
    # Project helpers
    "project_to_dict",
    "convert_projects_to_dataclass",
    "merge_projects_participated",
    # Profile helpers
    "remove_evidences_from_profile",
    "accumulate_old_memory_entry",
    "profile_payload_to_memory",
    "merge_single_profile",
    "merge_profiles",
]
