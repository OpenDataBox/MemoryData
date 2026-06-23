import os
import re

from utils.locomo_utils import get_locomo_category_slug


_MEMORY_AGENT_HINTS = (
    "mem0",
    "cognee",
    "memtree",
    "memochat",
    "memoryos",
    "letta",
    "zep",
    "simplemem",
    "lightmem",
    "a_mem",
    "everos",
    "MemOS",
)


def resolve_artifact_root(command_line_args=None, agent_config=None):
    """Resolve an optional artifact root override from CLI or config."""
    cli_value = getattr(command_line_args, "artifact_root", None) if command_line_args is not None else None
    config_value = agent_config.get("artifact_root") if agent_config is not None else None
    value = str(cli_value or config_value or "").strip()
    if not value:
        return ""
    return value.rstrip("/\\")


def _strip_leading_current_dir(path):
    normalized = str(path or "").replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def rebase_legacy_artifact_path(path, artifact_root):
    """Rebase legacy results/log paths under a new artifact root."""
    root = str(artifact_root or "").strip().rstrip("/\\")
    if not root:
        return path

    normalized = _strip_leading_current_dir(path)
    if normalized.startswith("results/"):
        suffix = normalized[len("results/") :]
        return os.path.join(root, suffix)
    if normalized.startswith("logs/"):
        suffix = normalized[len("logs/") :]
        return os.path.join(root, "logs", suffix)
    return path


def resolve_results_artifact_path(artifact_root, *parts):
    """Build a path under the legacy results root, then optionally rebase it."""
    return rebase_legacy_artifact_path(os.path.join(".", "results", *parts), artifact_root)


def resolve_logs_artifact_path(artifact_root, *parts):
    """Build a path under the legacy logs root, then optionally rebase it."""
    return rebase_legacy_artifact_path(os.path.join("logs", *parts), artifact_root)


def normalize_config_values(raw_value):
    """Normalize a scalar/list config value into a list of non-empty strings."""
    if raw_value in (None, ""):
        return []
    if isinstance(raw_value, (list, tuple, set)):
        values = raw_value
    else:
        values = str(raw_value).split(",")
    return [str(value).strip() for value in values if str(value).strip()]


def get_dataset_variant_components(dataset_config):
    """Return stable dataset-variant suffixes to prevent path collisions."""
    if str(dataset_config.get("dataset", "")).lower() != "locomo":
        return []

    components = []
    include_values = normalize_config_values(dataset_config.get("locomo_categories"))
    exclude_values = normalize_config_values(dataset_config.get("locomo_exclude_categories"))

    if include_values:
        normalized = "-".join(sorted(get_locomo_category_slug(value) for value in include_values))
        prefix = "category" if "-" not in normalized else "categories"
        components.append(f"{prefix}_{normalized}")

    if exclude_values:
        normalized = "-".join(sorted(get_locomo_category_slug(value) for value in exclude_values))
        components.append(f"exclude_{normalized}")

    test_file = str(dataset_config.get("test_files") or "").strip()
    if test_file:
        split_stem = os.path.splitext(os.path.basename(test_file))[0].strip()
        if split_stem and split_stem != "locomo10":
            components.append(f"split_{split_stem}")

    return components


def get_dataset_variant_suffix(dataset_config):
    components = get_dataset_variant_components(dataset_config)
    return f"_{'_'.join(components)}" if components else ""


def resolve_agent_chunk_component(agent_config, dataset_config):
    value = agent_config.get("agent_chunk_size", dataset_config.get("chunk_size", "unknown"))
    return str(value) if value is not None else "unknown"


def resolve_retrieve_component(agent_config, agent_name):
    if "simplemem" in agent_name:
        value = agent_config.get("retrieve_num", agent_config.get("simplemem_retrieve_num", 10))
    else:
        value = agent_config.get("retrieve_num", "unknown")
    return str(value) if value is not None else "unknown"


def get_agent_variant_tag(agent_config):
    raw_value = str(agent_config.get("agent_variant_tag", "") or "").strip()
    if not raw_value:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw_value).strip("-_.").lower()


def is_memory_agent_name(agent_name):
    return any(agent_type in agent_name for agent_type in _MEMORY_AGENT_HINTS)


def generate_memory_agent_base_path(agent_config, dataset_config, artifact_root=""):
    agent_name = agent_config["agent_name"]
    dataset_variant_suffix = get_dataset_variant_suffix(dataset_config)
    chunk_component = resolve_agent_chunk_component(agent_config, dataset_config)
    variant_tag = get_agent_variant_tag(agent_config)
    variant_suffix = f"_variant{variant_tag}" if variant_tag else ""
    base_path = resolve_results_artifact_path(
        artifact_root,
        "agents",
        (
            f"{agent_name}_{dataset_config['sub_dataset']}{dataset_variant_suffix}"
            f"_chunk{chunk_component}_model{agent_config['model']}{variant_suffix}"
        ),
    )
    if "letta" in agent_name:
        return f"{base_path}_mode{agent_config['letta_mode']}"
    return base_path


def generate_rag_agent_path(agent_config, dataset_config, current_context_index, artifact_root=""):
    dataset_variant_suffix = get_dataset_variant_suffix(dataset_config)
    folder_name = (
        f"{agent_config['agent_name']}_{dataset_config['sub_dataset']}{dataset_variant_suffix}"
        f"_k{agent_config['retrieve_num']}_chunk{dataset_config['chunk_size']}"
        f"_model{agent_config['model']}"
    )
    return resolve_results_artifact_path(
        artifact_root,
        "agents",
        folder_name,
        f"exp_{current_context_index}",
    )


def generate_default_agent_path(agent_config, dataset_config, current_context_index, artifact_root=""):
    dataset_variant_suffix = get_dataset_variant_suffix(dataset_config)
    return resolve_results_artifact_path(
        artifact_root,
        "agents",
        f"{agent_config['agent_name']}_{dataset_config['sub_dataset']}{dataset_variant_suffix}",
        f"exp_{current_context_index}",
    )


def generate_agent_save_folder_path(agent_config, dataset_config, current_context_index, artifact_root=""):
    agent_name = agent_config["agent_name"]
    if is_memory_agent_name(agent_name):
        base_path = generate_memory_agent_base_path(agent_config, dataset_config, artifact_root)
        return os.path.join(base_path, f"exp_{current_context_index}")
    if "rag" in agent_name:
        return generate_rag_agent_path(agent_config, dataset_config, current_context_index, artifact_root)
    return generate_default_agent_path(agent_config, dataset_config, current_context_index, artifact_root)


def build_memorag_cache_dir(agent_config, dataset_config, chunk_size, context_id, artifact_root=""):
    max_context_tokens = agent_config.get("memorag_max_context_tokens")
    truncation_strategy = agent_config.get("memorag_truncation_strategy", "head_tail")
    token_tag = f"maxtok_{max_context_tokens if max_context_tokens else 'full'}_{truncation_strategy}"
    return resolve_results_artifact_path(
        artifact_root,
        "outputs",
        "rag_retrieved",
        "MemoRAG",
        dataset_config["sub_dataset"],
        f"chunksize_{chunk_size}",
        token_tag,
        f"context_id_{context_id}",
    )
