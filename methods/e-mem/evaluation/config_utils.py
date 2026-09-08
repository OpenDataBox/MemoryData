"""Helpers for resolving benchmark-local configuration files."""

from __future__ import annotations

from pathlib import Path


def resolve_eval_config_path(script_file: str, config_path: str) -> str:
    """Resolve a config path relative to the eval script or project root."""
    script_path = Path(script_file).resolve()
    script_dir = script_path.parent
    project_root = script_path.parents[2]
    candidate = Path(config_path)

    if candidate.is_absolute():
        return str(candidate)

    script_relative = script_dir / candidate
    if script_relative.exists():
        return str(script_relative)

    return str(project_root / candidate)
