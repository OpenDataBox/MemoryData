"""Episode ID Mapper - Stateless functions for Long ID <-> Short ID conversion.

This helps reduce hallucination and token consumption of long IDs by the language model.
"""

import copy
from typing import Any, Dict, List


def create_id_mapping(long_ids: List[str]) -> Dict[str, str]:
    """Create long_id -> short_id mapping.

    Args:
        long_ids: List of long episode IDs

    Returns:
        Dict mapping long_id -> short_id (e.g., "ep1", "ep2", ...)
    """
    return {lid: f"ep{i+1}" for i, lid in enumerate(long_ids) if lid}


def replace_sources(
    profile_dict: Dict[str, Any], id_map: Dict[str, str], reverse: bool = False
) -> Dict[str, Any]:
    """Replace source IDs in profile dict.

    Args:
        profile_dict: Profile dictionary with explicit_info and implicit_traits
        id_map: Mapping from long_id -> short_id
        reverse: If True, convert short_id back to long_id

    Returns:
        New profile dict with replaced IDs
    """
    mapping = {v: k for k, v in id_map.items()} if reverse else id_map
    result = copy.deepcopy(profile_dict)

    def _map_source(source: Any) -> Any:
        if not isinstance(source, str) or not source:
            return source
        if "|" in source:
            prefix, sid = source.rsplit("|", 1)
            sid = sid.strip()
            return f"{prefix}|{mapping.get(sid, sid)}"
        return mapping.get(source, source)

    for item in result.get("explicit_info", []):
        item["sources"] = [_map_source(s) for s in item.get("sources", [])]
    for item in result.get("implicit_traits", []):
        item["sources"] = [_map_source(s) for s in item.get("sources", [])]

    return result


def get_short_id(long_id: str, id_map: Dict[str, str]) -> str:
    """Get short ID from mapping, return original if not found."""
    return id_map.get(long_id, long_id)
