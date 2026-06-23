import asyncio
import logging
import os
import json
import re
import time
import yaml
import shutil
from collections import defaultdict
from utils.conversation_creator import ConversationCreator
from utils.agent import AgentWrapper, _build_zep_local_namespace
from tqdm import tqdm
from utils.eval_other_utils import metrics_summarization
from utils.provider_utils import (
    resolve_base_url as _resolve_base_url,
    resolve_env_value as _resolve_env_value,
)
from utils.locomo_utils import get_locomo_category_slug
from utils.artifact_paths import (
    build_memorag_cache_dir,
    generate_default_agent_path as _generate_default_agent_path_with_root,
    generate_memory_agent_base_path as _generate_memory_agent_base_path_with_root,
    generate_rag_agent_path as _generate_rag_agent_path_with_root,
    generate_agent_save_folder_path,
    rebase_legacy_artifact_path,
    resolve_artifact_root,
)


logger = logging.getLogger(__name__)

A_MEM_CHECKPOINT_INTERVAL = 5


# ============================================================================
# MAIN WORKFLOW FUNCTIONS (in typical execution order)
# ============================================================================

def setup_configs_and_directories(command_line_args):
    """
    Setup configurations, handle ablations, cleanup, and create output directories.
    
    Args:
        command_line_args: Parsed command line arguments
        
    Returns:
        tuple: (agent_config, dataset_config, output_file_path)
    """
    # Load configuration files
    agent_config = _load_yaml_config(command_line_args.agent_config)
    dataset_config = _load_yaml_config(command_line_args.dataset_config)

    artifact_root = resolve_artifact_root(command_line_args, agent_config)
    if artifact_root:
        agent_config["artifact_root"] = artifact_root
        agent_config["output_dir"] = rebase_legacy_artifact_path(
            agent_config["output_dir"],
            artifact_root,
        )
    
    # Apply ablation study parameters if specified
    _apply_ablation_parameters(command_line_args, agent_config, dataset_config)
    
    # Clean up previous agent data if necessary
    _cleanup_agent_directories(agent_config)
    
    # Create output directory and file path
    output_file_path = _create_output_path(agent_config, dataset_config)
    
    return agent_config, dataset_config, output_file_path


def create_agent_and_fetch_data(agent_config, dataset_config):
    """
    Create conversation creator and fetch chunks and query_and_answers.
    
    Args:
        agent_config: Configuration dictionary for the agent
        dataset_config: Configuration dictionary for the dataset
        
    Returns:
        tuple: (start_time, all_context_chunks, all_query_answer_pairs)
    """
    start_time = time.time()
    
    # Create conversation creator to handle data loading and processing
    conversation_creator = ConversationCreator(agent_config, dataset_config)
    
    # Fetch processed chunks and query-answer pairs
    return start_time, conversation_creator.get_chunks(), conversation_creator.get_query_and_answers()


def load_existing_results(output_file_path, dataset_config, all_query_answer_pairs, retry_failed_queries=False):
    """
    Load existing results from output file and initialize variables.
    
    Args:
        output_file_path: Path to the output results file
        dataset_config: Configuration dictionary for the dataset
        all_query_answer_pairs: List of query-answer pairs for all contexts
        
    Returns:
        tuple: (metrics, results, completed_context_ids, skipped_query_ids)
    """
    if not os.path.exists(output_file_path):
        return defaultdict(list), [], set(), set()
    
    # Load existing results from file
    with open(output_file_path, "r") as file:
        saved_output = json.load(file)
        
    # Initialize data structures
    metrics, results = defaultdict(list), []
    
    skipped_query_ids = set()

    # Process each saved result entry
    for fallback_query_id, saved_data_entry in enumerate(saved_output['data']):
        existing_query_id = saved_data_entry.get('query_id', fallback_query_id)
        if saved_data_entry.get("status") == "failed":
            if retry_failed_queries:
                continue
            results.append(saved_data_entry)
            skipped_query_ids.add(existing_query_id)
            continue

        query = saved_data_entry['query']
        
        # Handle both list and string answer formats
        answer = (saved_data_entry['answer'][0] 
                 if isinstance(saved_data_entry['answer'], list) 
                 else saved_data_entry['answer'])
        
        # Reconstruct output format expected by metrics_summarization
        reconstructed_output = {
            "output": saved_data_entry['output'],
            "input_len": saved_data_entry['input_len'],
            "output_len": saved_data_entry['output_len'],
            "memory_construction_time": saved_data_entry.get('memory_construction_time', 0),
            "query_time_len": saved_data_entry['query_time_len'],
        }
        if "retrieved_source_id_groups" in saved_data_entry:
            reconstructed_output["retrieved_source_id_groups"] = saved_data_entry["retrieved_source_id_groups"]
        if "requested_recall_k" in saved_data_entry:
            reconstructed_output["requested_recall_k"] = saved_data_entry["requested_recall_k"]
        for key, value in saved_data_entry.items():
            if key.startswith("membench_native_"):
                reconstructed_output[key] = value
        
        # Extract existing identifiers
        existing_query_id = saved_data_entry.get('query_id', fallback_query_id)
        existing_context_id = saved_data_entry.get('context_id')
        existing_qa_pair_id = saved_data_entry.get('qa_pair_id')
        existing_eval_metadata = saved_data_entry.get('eval_metadata')
        
        metrics, results = metrics_summarization(
            reconstructed_output, query, answer, dataset_config, 
            metrics, results, existing_query_id, existing_context_id, existing_qa_pair_id,
            eval_metadata=existing_eval_metadata,
        )
        skipped_query_ids.add(existing_query_id)
    
    completed_context_ids = _calculate_completed_context_ids(
        all_query_answer_pairs, skipped_query_ids
    )
    
    return metrics, results, completed_context_ids, skipped_query_ids


def _calculate_completed_context_ids(all_query_answer_pairs, skipped_query_ids):
    """Return the context indices whose queries are all already accounted for."""
    completed_context_ids = set()
    query_index = 0

    for context_index, query_answer_pairs in enumerate(all_query_answer_pairs):
        context_query_ids = range(query_index, query_index + len(query_answer_pairs))
        if all(query_id in skipped_query_ids for query_id in context_query_ids):
            completed_context_ids.add(context_index)
        query_index += len(query_answer_pairs)

    return completed_context_ids


def _a_mem_checkpoint_path(agent_save_folder):
    return os.path.join(agent_save_folder, "a_mem_checkpoint.json")


def _load_a_mem_checkpoint(agent_save_folder):
    checkpoint_path = _a_mem_checkpoint_path(agent_save_folder)
    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load A-MEM checkpoint from %s", checkpoint_path)
        return None

    next_chunk_index = payload.get("next_chunk_index", 0)
    try:
        payload["next_chunk_index"] = max(0, int(next_chunk_index))
    except (TypeError, ValueError):
        logger.warning("Invalid A-MEM checkpoint index in %s", checkpoint_path)
        return None

    return payload


def _save_a_mem_checkpoint(agent_save_folder, next_chunk_index, total_chunks):
    checkpoint_path = _a_mem_checkpoint_path(agent_save_folder)
    os.makedirs(agent_save_folder, exist_ok=True)
    payload = {
        "next_chunk_index": int(next_chunk_index),
        "total_chunks": int(total_chunks),
        "updated_at": time.time(),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _clear_a_mem_checkpoint(agent_save_folder):
    checkpoint_path = _a_mem_checkpoint_path(agent_save_folder)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def generate_agent_save_folder(agent_config, dataset_config, current_context_index):
    """
    Generate the agent save folder path based on agent type and configuration.
    
    Args:
        agent_config: Configuration dictionary for the agent
        dataset_config: Configuration dictionary for the dataset
        current_context_index: Index of the current context being processed
        
    Returns:
        str: Path to the agent save folder
    """
    return generate_agent_save_folder_path(
        agent_config,
        dataset_config,
        current_context_index,
        agent_config.get("artifact_root"),
    )


def initialize_and_memorize_agent(agent_config, dataset_config, agent_save_folder,
                                 context_chunks, current_context_index, total_contexts_count,
                                 force_rebuild=False, load_existing_only=False,
                                 backfill_longmemeval_recall_debug=False):
    """
    Initialize agent and handle memorization if needed.
    
    Args:
        agent_config: Configuration dictionary for the agent
        dataset_config: Configuration dictionary for the dataset
        agent_save_folder: Path to folder where agent state is saved
        context_chunks: List of text chunks for the current context
        current_context_index: Index of the current context
        total_contexts_count: Total number of contexts to process
        force_rebuild: Whether to discard local agent state and rebuild from scratch
        load_existing_only: Whether to require an already-saved agent state
        backfill_longmemeval_recall_debug: Whether to enable query-only recall debug backfill
        
    Returns:
        tuple: (AgentWrapper, optional MemBench native timing sidecar)
    """
    if force_rebuild and load_existing_only:
        raise ValueError("load_existing_only cannot be combined with force_rebuild.")

    if force_rebuild:
        # --force should rebuild from scratch instead of silently reusing local
        # agent state from a previous run. Reset external persistence before
        # deleting the local folder, because some remote resets rely on ids
        # cached in the marker directory.
        _reset_external_persistent_state(
            agent_config,
            dataset_config,
            agent_save_folder,
            current_context_index,
        )
        if os.path.exists(agent_save_folder):
            shutil.rmtree(agent_save_folder)

    # Initialize the agent wrapper
    effective_agent_config = dict(agent_config)
    if backfill_longmemeval_recall_debug:
        effective_agent_config["_backfill_longmemeval_recall_debug"] = True
    agent = AgentWrapper(effective_agent_config, dataset_config, load_agent_from=agent_save_folder)
    native_timing_sidecar = None
    
    # Handle memorization or loading based on whether saved state exists
    should_load_existing_agent = (not force_rebuild) and os.path.exists(agent_save_folder)
    agent_name = agent_config.get("agent_name", "")
    if "letta" in agent_name:
        letta_agent_id_path = os.path.join(agent_save_folder, "agent_id.txt")
        if "api" in agent_name:
            should_load_existing_agent = os.path.exists(letta_agent_id_path)
        else:
            letta_sqlite_path = os.path.join(agent_save_folder, "letta_runtime", "sqlite.db")
            should_load_existing_agent = (
                os.path.exists(letta_agent_id_path) and os.path.exists(letta_sqlite_path)
            )
    if "zep_local" in agent_name:
        # zep_local persists its graph in Neo4j, not in the marker folder.
        # Reusing only the local folder is therefore insufficient to prove the
        # graph still exists, so we rebuild conservatively for now.
        should_load_existing_agent = False
    elif "zep" in agent_name:
        # Zep state lives in the remote service rather than the local folder.
        # The current local save/load flow does not recreate remote users,
        # threads, or graphs, so reusing a local folder leads to 404 errors.
        should_load_existing_agent = False
    if "simplemem" in agent_name:
        simplemem_marker_path = os.path.join(agent_save_folder, "simplemem_ready.txt")
        should_load_existing_agent = os.path.exists(simplemem_marker_path)
    if "mem0" in agent_name:
        mem0_source_map_path = os.path.join(agent_save_folder, "mem0_source_map.json")
        should_load_existing_agent = os.path.exists(mem0_source_map_path)
    if "lightmem" in agent_name:
        lightmem_marker_path = os.path.join(agent_save_folder, "lightmem_ready.txt")
        should_load_existing_agent = os.path.exists(lightmem_marker_path)
    if "a_mem" in agent_name:
        a_mem_marker_path = os.path.join(agent_save_folder, "a_mem_ready.txt")
        should_load_existing_agent = os.path.exists(a_mem_marker_path)
    if "cognee" in agent_name:
        cognee_marker_path = os.path.join(agent_save_folder, "cognee_ready.txt")
        should_load_existing_agent = os.path.exists(cognee_marker_path)
    if "memtree" in agent_name:
        memtree_marker_path = os.path.join(agent_save_folder, "memtree_ready.txt")
        should_load_existing_agent = os.path.exists(memtree_marker_path)
    if "everos" in agent_name:
        # EverOS state lives in an external service and the adapter intentionally
        # creates an isolated remote group per benchmark run to avoid stale data.
        should_load_existing_agent = False
    if "memochat" in agent_name:
        memochat_marker_path = os.path.join(agent_save_folder, "memochat_ready.txt")
        should_load_existing_agent = os.path.exists(memochat_marker_path)
    if "memoryos" in agent_name:
        memoryos_marker_path = os.path.join(agent_save_folder, "memoryos_ready.txt")
        should_load_existing_agent = os.path.exists(memoryos_marker_path)
    if "MemOS" in agent_name:
        memos_marker_path = os.path.join(agent_save_folder, "memos_ready.txt")
        should_load_existing_agent = os.path.exists(memos_marker_path)
    a_mem_checkpoint = None
    if "a_mem" in agent_name:
        a_mem_checkpoint = _load_a_mem_checkpoint(agent_save_folder)

    if load_existing_only and not should_load_existing_agent:
        raise FileNotFoundError(
            f"Existing state required for {agent_name} context {current_context_index}, "
            f"but no reusable saved agent was found at {agent_save_folder}."
        )

    if should_load_existing_agent:
        agent.load_agent()
        print("\n\n Agent loaded...\n\n")
        if "a_mem" in agent_name:
            _clear_a_mem_checkpoint(agent_save_folder)
        native_timing_sidecar = _load_membench_native_timing_sidecar(agent_save_folder, dataset_config)
    else:
        start_chunk_index = 0
        checkpoint_callback = None

        if "a_mem" in agent_name:
            a_mem_state_path = getattr(agent, "a_mem_state_path", None)
            if (
                a_mem_state_path
                and os.path.exists(a_mem_state_path)
                and a_mem_checkpoint is not None
            ):
                start_chunk_index = min(
                    a_mem_checkpoint.get("next_chunk_index", 0),
                    len(context_chunks),
                )
                if start_chunk_index > 0:
                    agent.a_mem.load()
                    logger.info(
                        "Resuming partial A-MEM memorization from chunk %s/%s for context %s",
                        start_chunk_index,
                        len(context_chunks),
                        current_context_index,
                    )

            def checkpoint_callback(next_chunk_index):
                os.makedirs(agent_save_folder, exist_ok=True)
                agent.a_mem.save()
                _save_a_mem_checkpoint(
                    agent_save_folder,
                    next_chunk_index=next_chunk_index,
                    total_chunks=len(context_chunks),
                )

        write_times = _memorize_context_chunks(
            agent,
            context_chunks,
            current_context_index,
            total_contexts_count,
            start_chunk_index=start_chunk_index,
            checkpoint_interval=A_MEM_CHECKPOINT_INTERVAL if "a_mem" in agent_name else None,
            checkpoint_callback=checkpoint_callback,
        )
        agent.save_agent()
        if "a_mem" in agent_name:
            _clear_a_mem_checkpoint(agent_save_folder)
        native_timing_sidecar = _build_membench_native_timing_sidecar(
            dataset_config=dataset_config,
            agent_save_folder=agent_save_folder,
            current_context_index=current_context_index,
            context_chunks=context_chunks,
            write_times=write_times,
        )
        _save_membench_native_timing_sidecar(agent_save_folder, native_timing_sidecar)
        
    return agent, native_timing_sidecar


def _read_saved_agent_id(agent_save_folder):
    agent_id_path = os.path.join(agent_save_folder, "agent_id.txt")
    if not os.path.exists(agent_id_path):
        return None
    with open(agent_id_path, "r") as file:
        agent_id = file.read().strip()
    return agent_id or None


def _is_missing_resource_error(exc):
    message = str(exc).lower()
    return (
        "404" in message
        or "not found" in message
        or "does not exist" in message
        or "no data found" in message
    )


def _reset_external_persistent_state(agent_config, dataset_config, agent_save_folder, current_context_index):
    """Best-effort reset for state that lives outside the local agent folder."""
    agent_name = agent_config.get("agent_name", "")

    if "letta" in agent_name and "api" in agent_name:
        _reset_letta_api_agent(agent_config, agent_save_folder)

    if "memo_rag" in agent_name:
        _reset_memorag_cache(agent_config, dataset_config, current_context_index)

    if "zep_local" in agent_name:
        _reset_zep_local_graph(agent_config, dataset_config, current_context_index)
    elif "zep" in agent_name:
        _reset_zep_cloud_state(dataset_config, current_context_index)


def _reset_letta_api_agent(agent_config, agent_save_folder):
    agent_id = _read_saved_agent_id(agent_save_folder)
    if not agent_id:
        return

    from letta_client import Letta

    base_url = _resolve_base_url(
        agent_config.get("base_url"),
        agent_config.get("base_url_env"),
    )
    token = _resolve_env_value(agent_config.get("api_key_env"), ["Letta_API_KEY"])
    client_kwargs = {"api_key": token}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = Letta(**client_kwargs)
    try:
        client.agents.delete(agent_id)
        logger.info(f"--force reset deleted remote Letta agent {agent_id}")
    except Exception as exc:
        if _is_missing_resource_error(exc):
            logger.info(f"--force reset skipped missing Letta agent {agent_id}")
            return
        raise RuntimeError(f"Failed to reset remote Letta agent {agent_id}: {exc}") from exc
    finally:
        client.close()


def _reset_zep_cloud_state(dataset_config, current_context_index):
    from zep_cloud.client import Zep

    user_id = f"user_{current_context_index}_{dataset_config['sub_dataset']}"
    thread_id = f"thread_{current_context_index}_{dataset_config['sub_dataset']}"

    zep_api_key = os.getenv("ZEP_API_KEY")
    zep_base_url = os.getenv("ZEP_API_URL")
    client_kwargs = {}
    if zep_api_key:
        client_kwargs["api_key"] = zep_api_key
    if zep_base_url:
        client_kwargs["base_url"] = (
            zep_base_url if zep_base_url.endswith("/api/v2") else f"{zep_base_url.rstrip('/')}/api/v2"
        )

    client = Zep(**client_kwargs)
    for step_name, func, identifier in [
        ("memory.delete", client.memory.delete, thread_id),
        ("user.delete", client.user.delete, user_id),
    ]:
        try:
            func(identifier)
            logger.info(f"--force reset deleted Zep resource via {step_name}: {identifier}")
        except Exception as exc:
            if _is_missing_resource_error(exc):
                logger.info(f"--force reset skipped missing Zep resource via {step_name}: {identifier}")
                continue
            raise RuntimeError(f"Failed to reset Zep resource via {step_name} ({identifier}): {exc}") from exc


def _reset_zep_local_graph(agent_config, dataset_config, current_context_index):
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.nodes import Node

    namespace = _build_zep_local_namespace(
        current_context_index,
        dataset_config["sub_dataset"],
        (agent_config.get("zep_local_namespace_prefix") or "").strip(),
    )

    async def _reset():
        driver = Neo4jDriver(
            uri=agent_config.get("neo4j_uri", "bolt://localhost:7687"),
            user=agent_config.get("neo4j_user", "neo4j"),
            password=agent_config.get("neo4j_password", "neo4jneo4j"),
        )
        try:
            await Node.delete_by_group_id(driver, namespace)
            logger.info(f"--force reset deleted zep_local namespace {namespace} from Neo4j")
        finally:
            await driver.close()

    asyncio.run(_reset())


def _reset_memorag_cache(agent_config, dataset_config, current_context_index):
    cache_dir = build_memorag_cache_dir(
        agent_config,
        dataset_config,
        dataset_config["chunk_size"],
        current_context_index,
        agent_config.get("artifact_root"),
    )
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info(f"--force reset deleted MemoRAG cache {cache_dir}")


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def _load_yaml_config(config_file_path):
    """Load and return YAML configuration from file."""
    with open(config_file_path, 'r') as file:
        return yaml.safe_load(file)


def _apply_ablation_parameters(command_line_args, agent_config, dataset_config):
    """Apply ablation study parameters to override default configurations."""
    # Handle chunk size ablation
    if command_line_args.chunk_size_ablation > 0:
        _apply_chunk_size_ablation(command_line_args, agent_config, dataset_config)
    
    # Handle max test queries ablation
    if command_line_args.max_test_queries_ablation > 0:
        dataset_config['max_test_queries'] = command_line_args.max_test_queries_ablation
        print(f"\n\nUsing max_test_queries: {dataset_config['max_test_queries']}\n\n")


def _apply_chunk_size_ablation(command_line_args, agent_config, dataset_config):
    """Apply chunk size ablation based on agent type."""
    new_chunk_size = command_line_args.chunk_size_ablation
    
    # Check if this is a memory agent that uses agent_chunk_size
    if any(agent_name in agent_config['agent_name'] for agent_name in ['mem0', 'letta', 'cognee', 'memtree', 'memochat', 'memoryos', 'zep', 'simplemem', 'lightmem', 'a_mem', 'everos', 'MemOS']):
        agent_config['agent_chunk_size'] = new_chunk_size
        dataset_config['chunk_size'] = new_chunk_size
        print(f"\n\nUsing agent chunk_size: {agent_config['agent_chunk_size']}\n\n")
    else:
        dataset_config['chunk_size'] = new_chunk_size
        print(f"\n\nUsing new chunk_size: {dataset_config['chunk_size']}\n\n")


def _cleanup_agent_directories(agent_config):
    """Clean up previous agent data directories if necessary."""
    if "cognee" in agent_config['agent_name']:
        for directory_path in [
            './methods/cognee/source/cognee/.data_storage/data',
            './methods/cognee/source/cognee/.cognee_system/databases',
            './methods/cognee/source/cognee/.cognee_cache',
        ]:
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)


# ============================================================================
# OUTPUT PATH GENERATION HELPERS
# ============================================================================

def _create_output_path(agent_config, dataset_config):
    """
    Create output directory and return the output file path.
    
    Args:
        agent_config: Configuration dictionary for the agent
        dataset_config: Configuration dictionary for the dataset
        
    Returns:
        str: Path to the output results file
    """
    # Generate name tag based on agent type and configuration
    name_tag = _generate_output_name_tag(agent_config, dataset_config)
    
    # Create output directory for this dataset
    output_directory = os.path.join(agent_config['output_dir'], dataset_config['dataset'])
    os.makedirs(output_directory, exist_ok=True)
    
    # Create complete output file path
    return os.path.join(output_directory, f"{name_tag}_results.json")


def _normalize_config_values(raw_value):
    """Normalize a scalar/list config value into a list of non-empty strings."""
    if raw_value in (None, ""):
        return []
    if isinstance(raw_value, (list, tuple, set)):
        values = raw_value
    else:
        values = str(raw_value).split(",")
    return [str(value).strip() for value in values if str(value).strip()]


def _get_dataset_variant_components(dataset_config):
    """Return stable dataset-variant suffixes to prevent path collisions."""
    if str(dataset_config.get("dataset", "")).lower() != "locomo":
        return []

    components = []
    include_values = _normalize_config_values(dataset_config.get("locomo_categories"))
    exclude_values = _normalize_config_values(dataset_config.get("locomo_exclude_categories"))

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


def _get_dataset_variant_suffix(dataset_config):
    """Return a filesystem-friendly dataset variant suffix for state paths."""
    components = _get_dataset_variant_components(dataset_config)
    return f"_{'_'.join(components)}" if components else ""


def _resolve_agent_chunk_component(agent_config, dataset_config):
    """Resolve the chunk size that determines memorized state layout."""
    value = agent_config.get("agent_chunk_size", dataset_config.get("chunk_size", "unknown"))
    return str(value) if value is not None else "unknown"


def _resolve_retrieve_component(agent_config, agent_name):
    """Resolve retrieval-k for naming, including adapters with implicit defaults."""
    if "simplemem" in agent_name:
        value = agent_config.get("retrieve_num", agent_config.get("simplemem_retrieve_num", 10))
    else:
        value = agent_config.get("retrieve_num", "unknown")
    return str(value) if value is not None else "unknown"


def _get_agent_variant_tag(agent_config):
    """Return a filesystem-safe variant tag when the config defines one."""
    raw_value = str(agent_config.get("agent_variant_tag", "") or "").strip()
    if not raw_value:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_value).strip("-_.").lower()
    return normalized


def _generate_output_name_tag(agent_config, dataset_config):
    """Generate a descriptive name tag for output files based on configuration."""
    def safe_get(config_dict, key, default="unknown"):
        """Helper function to safely get config values and convert to string."""
        value = config_dict.get(key, default)
        return str(value) if value is not None else default
    
    # Base components for all agents
    base_components = [
        safe_get(dataset_config, 'sub_dataset'),
        safe_get(dataset_config, 'tag'),
        f"in{safe_get(dataset_config, 'context_max_length')}",
        f"size{safe_get(dataset_config, 'generation_max_length')}",
        f"shots{safe_get(dataset_config, 'shots')}",
        f"max_samples{safe_get(dataset_config, 'max_test_samples')}"
    ]
    base_components.extend(_get_dataset_variant_components(dataset_config))
    
    # Agent-specific components
    agent_name = safe_get(agent_config, 'agent_name')
    agent_components = []
    
    if "letta" in agent_name:
        agent_components = [
            f"chunk{safe_get(agent_config, 'agent_chunk_size')}",
            f"mode{safe_get(agent_config, 'letta_mode')}"
        ]
    elif any(agent_type in agent_name for agent_type in ["mem0", "cognee", "memtree", "memochat", "memoryos", "zep", "simplemem", "lightmem", "a_mem", "everos", "MemOS"]):
        agent_components = [
            f"k{_resolve_retrieve_component(agent_config, agent_name)}",
            f"chunk{_resolve_agent_chunk_component(agent_config, dataset_config)}"
        ]
    elif "rag" in agent_name:
        agent_components = [
            f"k{safe_get(agent_config, 'retrieve_num')}",
            f"chunk{safe_get(dataset_config, 'chunk_size')}"
        ]

    variant_tag = _get_agent_variant_tag(agent_config)
    if variant_tag:
        agent_components.append(f"variant{variant_tag}")
    
    return "_".join(base_components + agent_components)


# ============================================================================
# RESULTS LOADING HELPERS
# ============================================================================

def _calculate_last_completed_context_id(all_query_answer_pairs, total_queries_processed):
    """
    Calculate how many complete contexts have been processed based on total queries.
    
    Args:
        all_query_answer_pairs: List of query-answer pairs for all contexts
        total_queries_processed: Total number of queries that have been processed
        
    Returns:
        int: Number of completely processed contexts
    """
    queries_counted = 0
    
    for context_id, query_answer_pairs in enumerate(all_query_answer_pairs):
        if queries_counted + len(query_answer_pairs) <= total_queries_processed:
            queries_counted += len(query_answer_pairs)
        else:
            return context_id
            
    return len(all_query_answer_pairs)


# ============================================================================
# AGENT FOLDER GENERATION HELPERS
# ============================================================================

def _generate_memory_agent_base_path(agent_config, dataset_config):
    """Generate base path for memory agents (letta, mem0, cognee, memtree, memochat, memoryos, zep)."""
    return _generate_memory_agent_base_path_with_root(
        agent_config,
        dataset_config,
        agent_config.get("artifact_root"),
    )


def _generate_rag_agent_path(agent_config, dataset_config, current_context_index):
    """Generate path for RAG agents."""
    return _generate_rag_agent_path_with_root(
        agent_config,
        dataset_config,
        current_context_index,
        agent_config.get("artifact_root"),
    )


def _generate_default_agent_path(agent_config, dataset_config, current_context_index):
    """Generate path for default agents."""
    return _generate_default_agent_path_with_root(
        agent_config,
        dataset_config,
        current_context_index,
        agent_config.get("artifact_root"),
    )


# ============================================================================
# AGENT INITIALIZATION HELPERS
# ============================================================================

def _memorize_context_chunks(
    agent,
    context_chunks,
    current_context_index,
    total_contexts_count,
    start_chunk_index=0,
    checkpoint_interval=None,
    checkpoint_callback=None,
):
    """Handle the memorization process for context chunks."""
    print("\n\n Agent Memorizing...\n\n")
    progress_description = f"Processing experiments {current_context_index + 1}/{total_contexts_count}"
    write_times = []

    remaining_chunks = context_chunks[start_chunk_index:]
    total_remaining_chunks = len(remaining_chunks)

    for processed_offset, chunk in enumerate(
        tqdm(remaining_chunks, total=total_remaining_chunks, desc=progress_description),
        start=1,
    ):
        write_start_time = time.perf_counter()
        agent.send_message(chunk, memorizing=True, context_id=current_context_index)
        write_times.append(time.perf_counter() - write_start_time)
        next_chunk_index = start_chunk_index + processed_offset
        should_checkpoint = (
            checkpoint_callback is not None
            and checkpoint_interval
            and (
                next_chunk_index % checkpoint_interval == 0
                or next_chunk_index == len(context_chunks)
            )
        )
        if should_checkpoint:
            checkpoint_callback(next_chunk_index)

    return write_times


def _is_membench_dataset(dataset_config):
    return str(dataset_config.get("dataset", "")).lower() == "membench"


def _membench_native_timing_sidecar_path(agent_save_folder):
    return os.path.join(agent_save_folder, "membench_native_rtwt_sidecar.json")


def _load_membench_native_timing_sidecar(agent_save_folder, dataset_config):
    """Load a persisted MemBench native timing sidecar when available."""
    if not _is_membench_dataset(dataset_config):
        return None

    sidecar_path = _membench_native_timing_sidecar_path(agent_save_folder)
    if not os.path.exists(sidecar_path):
        return None

    try:
        with open(sidecar_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load MemBench native timing sidecar from %s", sidecar_path)
        return None


def _build_membench_native_timing_sidecar(
    dataset_config,
    agent_save_folder,
    current_context_index,
    context_chunks,
    write_times,
):
    """Build a persisted sidecar for runner-boundary MemBench write timings."""
    if not _is_membench_dataset(dataset_config):
        return None

    normalized_write_times = [float(value) for value in (write_times or [])]
    total_write_ops = len(normalized_write_times)
    total_write_time = sum(normalized_write_times)

    return {
        "sidecar_type": "membench_native_rtwt",
        "timing_source": "measured_during_memorization",
        "dataset": str(dataset_config.get("dataset", "")),
        "sub_dataset": str(dataset_config.get("sub_dataset", "")),
        "context_id": current_context_index,
        "agent_save_folder": agent_save_folder,
        "context_chunk_count": len(context_chunks or []),
        "total_write_ops": total_write_ops,
        "total_write_time_s": round(total_write_time, 6),
        "mean_write_time_s": round(total_write_time / total_write_ops, 6) if total_write_ops else None,
        "write_times_s": [round(value, 6) for value in normalized_write_times],
    }


def _save_membench_native_timing_sidecar(agent_save_folder, native_timing_sidecar):
    """Persist the MemBench native timing sidecar next to the saved agent state."""
    if not native_timing_sidecar:
        return

    os.makedirs(agent_save_folder, exist_ok=True)
    sidecar_path = _membench_native_timing_sidecar_path(agent_save_folder)
    with open(sidecar_path, "w", encoding="utf-8") as file:
        json.dump(native_timing_sidecar, file, indent=2, ensure_ascii=False)

    
