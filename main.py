import os
import yaml
import dotenv
import time
import json
import traceback
from argparse import ArgumentParser
from pathlib import Path
from utils.initialization import (
    load_existing_results, 
    create_agent_and_fetch_data, 
    setup_configs_and_directories, 
    generate_agent_save_folder, 
    initialize_and_memorize_agent
)
from tqdm import tqdm
from collections import defaultdict
import logging
import numpy as np
from utils.eval_other_utils import metrics_summarization
from utils.locomo_utils import get_locomo_category_label

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables
dotenv.load_dotenv()

BACKFILL_LONGMEMEVAL_RECALL_METHODS = (
    ("simplemem", "SimpleMem"),
    ("lightmem", "LightMem"),
    ("mem0", "Mem0"),
    ("cognee", "Cognee"),
    ("memoryos", "MemoryOS"),
    ("memochat", "MemoChat"),
)


def parse_command_line_arguments():
    """Parse and return command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--agent_config', type=str, default='config/reference_long_context_agent.yaml',
                       help='Path to agent configuration file')
    parser.add_argument('--dataset_config', type=str, default='benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml',
                       help='Path to dataset configuration file')
    parser.add_argument('--chunk_size_ablation', type=int, default=0,
                       help='Override chunk size for ablation studies (0 = use config default)')
    parser.add_argument('--max_test_queries_ablation', type=int, default=0,
                       help='Limit maximum test queries for ablation studies (0 = no limit)')
    parser.add_argument(
        '--artifact_root',
        type=str,
        default='',
        help='Optional top-level artifact root override. Keeps inner results/agents/logs layout unchanged.',
    )
    parser.add_argument('--force', action='store_true', default=False,
                       help='Full-reset re-run: delete saved results, rebuild local agent state, and reset supported external persistence')
    parser.add_argument('--retry_failed_queries', action='store_true', default=False,
                       help='When resuming, retry previously failed queries instead of skipping them')
    parser.add_argument('--backfill_longmemeval_recall_debug', action='store_true', default=False,
                       help='Reuse saved state to rerun queries and backfill LongMemEval recall debug files without touching the main results JSON')
    return parser.parse_args()


def should_skip_context(force_rerun, context_index, completed_context_ids):
    """Determine if we should skip a context that has already been processed."""
    return not force_rerun and context_index in completed_context_ids


def should_skip_query(query_index, skipped_query_ids):
    """Determine if we should skip a query that has already been processed."""
    return query_index in skipped_query_ids


def has_reached_query_limit(max_queries, current_query_index):
    """Determine if we should stop processing due to reaching the query limit."""
    return max_queries > 0 and current_query_index >= max_queries


def _resolve_backfill_longmemeval_method(agent_name):
    normalized_agent_name = str(agent_name or "").strip().lower()
    for method_key, display_name in BACKFILL_LONGMEMEVAL_RECALL_METHODS:
        if method_key in normalized_agent_name:
            return method_key, display_name
    return None, None


def save_results_to_file(output_path, agent_config, dataset_config, results, metrics, time_cost_list, start_time):
    """Save current results to the output file."""
    # Calculate averaged metrics for logging
    averaged_metrics = {
        key: np.mean(values) * (1 if ("_len" in key) or ("_time" in key) else 100) 
        for key, values in metrics.items()
    }
    
    # Log current metrics
    for key, value in averaged_metrics.items():
        logger.info(f"{key}: {value:.02f}")
    
    # Prepare output data structure
    time_cost_list.append(time.time() - start_time)
    output_data = {
        "agent_config": agent_config,
        "dataset_config": dataset_config,
        "data": results,
        "metrics": metrics,
        "time_cost": time_cost_list,
        "averaged_metrics": averaged_metrics,
    }

    locomo_breakdown = _build_locomo_category_breakdown(results)
    if locomo_breakdown is not None:
        output_data.update(locomo_breakdown)

    membench_breakdown = _build_membench_breakdown(results)
    if membench_breakdown is not None:
        output_data.update(membench_breakdown)

    membench_native_rtwt_sidecar = _build_membench_native_rtwt_sidecar(results)
    if membench_native_rtwt_sidecar is not None:
        output_data.update(membench_native_rtwt_sidecar)
    
    # Write to file
    with open(output_path, "w") as file:
        json.dump(output_data, file, indent=4)
    logger.info(f"Results saved at {output_path}")


def _build_locomo_category_breakdown(results):
    """Aggregate LoCoMo QA and recall metrics by category for convenient reporting."""
    f1_values = defaultdict(list)
    recall_values = defaultdict(lambda: defaultdict(list))
    category_counts = defaultdict(int)

    for result in results:
        if result.get("status") == "failed":
            continue
        eval_metadata = result.get("eval_metadata") or {}
        if eval_metadata.get("dataset") != "locomo_qa":
            continue

        category = str(eval_metadata.get("category") or "unknown")
        category_counts[category] += 1

        if "f1" in result:
            f1_values[category].append(float(result["f1"]))

        for metric_name, metric_value in result.items():
            if metric_name.startswith("locomo_recall@"):
                recall_values[metric_name][category].append(float(metric_value))

    if not category_counts:
        return None

    def _average_with_count(bucket):
        return {
            category: {
                "value": round(float(np.mean(values)) * 100, 2),
                "count": len(values),
            }
            for category, values in sorted(bucket.items())
            if values
        }

    return {
        "locomo_category_labels": {
            category: get_locomo_category_label(category)
            for category in sorted(category_counts.keys())
        },
        "locomo_category_counts": {category: count for category, count in sorted(category_counts.items())},
        "locomo_f1_by_category": _average_with_count(f1_values),
        "locomo_recall_by_category": {
            metric_name: _average_with_count(category_map)
            for metric_name, category_map in sorted(recall_values.items())
        },
    }


def _build_membench_breakdown(results):
    """Aggregate MemBench accuracy, recall, and overhead into stable summary blocks."""
    slice_counts = defaultdict(int)
    slice_branch_counts = defaultdict(int)
    accuracy_by_slice = defaultdict(list)
    accuracy_by_slice_branch = defaultdict(list)
    recall_by_slice = defaultdict(lambda: defaultdict(list))
    recall_by_slice_branch = defaultdict(lambda: defaultdict(list))
    overhead_by_slice = defaultdict(lambda: defaultdict(list))
    overhead_by_slice_branch = defaultdict(lambda: defaultdict(list))
    total_write_ops_by_slice = defaultdict(int)
    total_write_ops_by_slice_branch = defaultdict(int)
    total_read_ops_by_slice = defaultdict(int)
    total_read_ops_by_slice_branch = defaultdict(int)

    overall_accuracy = []
    overall_recall = defaultdict(list)
    overall_overhead = defaultdict(list)
    overall_total_write_ops = 0
    overall_total_read_ops = 0

    overhead_field_map = {
        "avg_memory_construction_time_s": "memory_construction_time",
        "avg_query_time_s": "query_time_len",
        "avg_write_time_per_op_s": "membench_write_time_per_op_s",
        "avg_read_time_per_op_s": "membench_read_time_per_op_s",
        "avg_total_overhead_time_s": "membench_total_overhead_time_s",
        "avg_input_tokens": "input_len",
        "avg_output_tokens": "output_len",
    }
    eval_overhead_field_map = {
        "avg_context_chunk_count": "context_chunk_count",
        "avg_context_length": "context_length",
        "avg_session_count": "session_count",
    }

    for result in results:
        if result.get("status") == "failed":
            continue
        eval_metadata = result.get("eval_metadata") or {}
        if eval_metadata.get("dataset") != "membench_qa":
            continue

        slice_name = str(eval_metadata.get("slice") or "unknown")
        branch_name = str(eval_metadata.get("branch") or "unknown")
        slice_branch_key = f"{slice_name}/{branch_name}"

        slice_counts[slice_name] += 1
        slice_branch_counts[slice_branch_key] += 1

        if "exact_match" in result:
            exact_match = float(result["exact_match"])
            accuracy_by_slice[slice_name].append(exact_match)
            accuracy_by_slice_branch[slice_branch_key].append(exact_match)
            overall_accuracy.append(exact_match)

        for metric_name, metric_value in result.items():
            if not metric_name.startswith("membench_recall@"):
                continue
            recall_value = float(metric_value)
            recall_by_slice[metric_name][slice_name].append(recall_value)
            recall_by_slice_branch[metric_name][slice_branch_key].append(recall_value)
            overall_recall[metric_name].append(recall_value)

        for summary_name, result_field in overhead_field_map.items():
            _append_numeric_value(overhead_by_slice[summary_name][slice_name], result.get(result_field))
            _append_numeric_value(overhead_by_slice_branch[summary_name][slice_branch_key], result.get(result_field))
            _append_numeric_value(overall_overhead[summary_name], result.get(result_field))

        for summary_name, eval_field in eval_overhead_field_map.items():
            eval_value = eval_metadata.get(eval_field)
            _append_numeric_value(overhead_by_slice[summary_name][slice_name], eval_value)
            _append_numeric_value(overhead_by_slice_branch[summary_name][slice_branch_key], eval_value)
            _append_numeric_value(overall_overhead[summary_name], eval_value)

        context_chunk_count = _coerce_int(eval_metadata.get("context_chunk_count"))
        if context_chunk_count is not None:
            total_write_ops_by_slice[slice_name] += context_chunk_count
            total_write_ops_by_slice_branch[slice_branch_key] += context_chunk_count
            overall_total_write_ops += context_chunk_count

        total_read_ops_by_slice[slice_name] += 1
        total_read_ops_by_slice_branch[slice_branch_key] += 1
        overall_total_read_ops += 1

    if not slice_counts:
        return None

    overall_summary = {
        "count": int(sum(slice_counts.values())),
        "accuracy": _safe_mean(overall_accuracy, scale=100),
        "total_write_ops": overall_total_write_ops,
        "total_read_ops": overall_total_read_ops,
    }
    for metric_name, values in sorted(overall_recall.items()):
        overall_summary[metric_name] = _safe_mean(values, scale=100)
    for summary_name, values in sorted(overall_overhead.items()):
        overall_summary[summary_name] = _safe_mean(values, scale=1)

    return {
        "membench_slice_counts": dict(sorted(slice_counts.items())),
        "membench_slice_branch_counts": dict(sorted(slice_branch_counts.items())),
        "membench_accuracy_by_slice": _average_with_count(accuracy_by_slice, scale=100),
        "membench_accuracy_by_slice_branch": _average_with_count(accuracy_by_slice_branch, scale=100),
        "membench_recall_by_slice": {
            metric_name: _average_with_count(grouped_values, scale=100)
            for metric_name, grouped_values in sorted(recall_by_slice.items())
        },
        "membench_recall_by_slice_branch": {
            metric_name: _average_with_count(grouped_values, scale=100)
            for metric_name, grouped_values in sorted(recall_by_slice_branch.items())
        },
        "membench_overhead_by_slice": {
            summary_name: _average_with_count(grouped_values, scale=1)
            for summary_name, grouped_values in sorted(overhead_by_slice.items())
        },
        "membench_overhead_by_slice_branch": {
            summary_name: _average_with_count(grouped_values, scale=1)
            for summary_name, grouped_values in sorted(overhead_by_slice_branch.items())
        },
        "membench_total_write_ops_by_slice": dict(sorted(total_write_ops_by_slice.items())),
        "membench_total_write_ops_by_slice_branch": dict(sorted(total_write_ops_by_slice_branch.items())),
        "membench_total_read_ops_by_slice": dict(sorted(total_read_ops_by_slice.items())),
        "membench_total_read_ops_by_slice_branch": dict(sorted(total_read_ops_by_slice_branch.items())),
        "membench_overall_summary": overall_summary,
    }


def _build_membench_native_rtwt_sidecar(results):
    """Aggregate runner-boundary MemBench RT/WT sidecar measurements."""
    overall_bucket = _create_native_rtwt_bucket()
    slice_buckets = defaultdict(_create_native_rtwt_bucket)
    slice_branch_buckets = defaultdict(_create_native_rtwt_bucket)
    seen_write_context_keys = set()
    has_any_values = False

    for result in results:
        if result.get("status") == "failed":
            continue
        eval_metadata = result.get("eval_metadata") or {}
        if eval_metadata.get("dataset") != "membench_qa":
            continue

        slice_name = str(eval_metadata.get("slice") or "unknown")
        branch_name = str(eval_metadata.get("branch") or "unknown")
        slice_branch_key = f"{slice_name}/{branch_name}"

        read_time_s = _coerce_float(result.get("membench_native_read_time_s"))
        if read_time_s is not None:
            _accumulate_native_read_time(overall_bucket, read_time_s)
            _accumulate_native_read_time(slice_buckets[slice_name], read_time_s)
            _accumulate_native_read_time(slice_branch_buckets[slice_branch_key], read_time_s)
            has_any_values = True

        context_key = _resolve_membench_native_context_key(result, eval_metadata)
        if context_key in seen_write_context_keys:
            continue

        write_time_total_s = _coerce_float(result.get("membench_native_write_time_total_s"))
        write_ops = _coerce_int(result.get("membench_native_write_ops"))
        if write_time_total_s is None or write_ops is None or write_ops <= 0:
            continue

        seen_write_context_keys.add(context_key)
        _accumulate_native_write_time(overall_bucket, write_time_total_s, write_ops)
        _accumulate_native_write_time(slice_buckets[slice_name], write_time_total_s, write_ops)
        _accumulate_native_write_time(slice_branch_buckets[slice_branch_key], write_time_total_s, write_ops)
        has_any_values = True

    if not has_any_values:
        return None

    return {
        "membench_native_rtwt_sidecar_definition": {
            "write_time": "Wall-clock time around each memorization call: agent.send_message(chunk, memorizing=True).",
            "read_time": "Wall-clock time around each query call: agent.send_message(query, memorizing=False). Depending on agent internals this may include retrieval plus answer generation.",
            "note": "This sidecar complements the unified overhead metrics and is stored separately to preserve benchmark-style RT/WT reporting.",
        },
        "membench_native_rtwt_sidecar_overall": _finalize_native_rtwt_bucket(overall_bucket),
        "membench_native_rtwt_sidecar_by_slice": {
            key: _finalize_native_rtwt_bucket(bucket)
            for key, bucket in sorted(slice_buckets.items())
            if _bucket_has_native_rtwt_values(bucket)
        },
        "membench_native_rtwt_sidecar_by_slice_branch": {
            key: _finalize_native_rtwt_bucket(bucket)
            for key, bucket in sorted(slice_branch_buckets.items())
            if _bucket_has_native_rtwt_values(bucket)
        },
    }


def _create_native_rtwt_bucket():
    return {
        "total_write_time_s": 0.0,
        "total_read_time_s": 0.0,
        "total_write_ops": 0,
        "total_read_ops": 0,
        "contexts_with_write_sidecar": 0,
        "queries_with_read_sidecar": 0,
    }


def _accumulate_native_write_time(bucket, write_time_total_s, write_ops):
    bucket["total_write_time_s"] += float(write_time_total_s)
    bucket["total_write_ops"] += int(write_ops)
    bucket["contexts_with_write_sidecar"] += 1


def _accumulate_native_read_time(bucket, read_time_s):
    bucket["total_read_time_s"] += float(read_time_s)
    bucket["total_read_ops"] += 1
    bucket["queries_with_read_sidecar"] += 1


def _bucket_has_native_rtwt_values(bucket):
    return bool(bucket["total_write_ops"] or bucket["total_read_ops"])


def _finalize_native_rtwt_bucket(bucket):
    total_write_ops = int(bucket["total_write_ops"])
    total_read_ops = int(bucket["total_read_ops"])
    total_write_time_s = round(float(bucket["total_write_time_s"]), 6)
    total_read_time_s = round(float(bucket["total_read_time_s"]), 6)
    return {
        "mean_write_time_s": round(total_write_time_s / total_write_ops, 6) if total_write_ops else None,
        "mean_read_time_s": round(total_read_time_s / total_read_ops, 6) if total_read_ops else None,
        "total_write_time_s": total_write_time_s,
        "total_read_time_s": total_read_time_s,
        "total_write_ops": total_write_ops,
        "total_read_ops": total_read_ops,
        "contexts_with_write_sidecar": int(bucket["contexts_with_write_sidecar"]),
        "queries_with_read_sidecar": int(bucket["queries_with_read_sidecar"]),
    }


def _resolve_membench_native_context_key(result, eval_metadata):
    return (
        result.get("membench_native_context_key")
        or result.get("context_id")
        or eval_metadata.get("sample_id")
        or eval_metadata.get("qa_pair_id")
        or id(result)
    )


def _average_with_count(bucket, scale):
    """Return grouped averages with sample counts."""
    return {
        group_key: {
            "value": _safe_mean(values, scale=scale),
            "count": len(values),
        }
        for group_key, values in sorted(bucket.items())
        if values
    }


def _safe_mean(values, scale=1):
    if not values:
        return None
    return round(float(np.mean(values)) * scale, 4)


def _append_numeric_value(bucket, value):
    try:
        bucket.append(float(value))
    except (TypeError, ValueError):
        return


def _coerce_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def initialize_progress_tracking(output_path, dataset_config, all_query_answer_pairs, force_rerun, retry_failed_queries):
    """Return progress state, optionally discarding stale results for --force."""
    if force_rerun:
        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"--force enabled: removed existing results file {output_path}")
        return defaultdict(list), [], set(), set()

    return load_existing_results(
        output_path,
        dataset_config,
        all_query_answer_pairs,
        retry_failed_queries=retry_failed_queries,
    )


def process_single_query(agent, query, answer, dataset_config, metrics, results, 
                        query_index, context_index, qa_pair_id=None, eval_metadata=None,
                        context_native_timing=None):
    """Process a single query and update metrics and results."""
    # Send query to agent and get response
    query_start_time = time.perf_counter()
    agent_output = agent.send_message(
        query,
        memorizing=False,
        query_id=query_index,
        context_id=context_index,
        eval_metadata=eval_metadata,
    )
    query_wall_time_s = time.perf_counter() - query_start_time

    _attach_membench_native_rtwt_to_output(
        agent_output=agent_output,
        dataset_config=dataset_config,
        eval_metadata=eval_metadata,
        context_index=context_index,
        context_native_timing=context_native_timing,
        query_wall_time_s=query_wall_time_s,
    )
    
    # Calculate metrics and update results
    return metrics_summarization(
        agent_output, query, answer, dataset_config, metrics, results,
        query_index, context_index, qa_pair_id, eval_metadata=eval_metadata,
    )


def unpack_query_data(query_data):
    """Unpack query data handling both old and new formats."""
    if len(query_data) == 4:
        return query_data
    if len(query_data) == 3:
        return (*query_data, None)
    raise ValueError(f"Unexpected query tuple length: {len(query_data)}")


def build_failed_query_record(query, answer, query_index, context_index, exc, qa_pair_id=None, eval_metadata=None):
    """Create a result entry for a query that failed mid-run."""
    result_record = {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "output": "",
        "input_len": 0,
        "output_len": 0,
        "memory_construction_time": 0,
        "query_time_len": 0,
        "answer": answer,
        "query": query,
        "query_id": query_index,
        "context_id": context_index,
        "pred": None,
        "judge": False,
    }
    if qa_pair_id is not None:
        result_record["qa_pair_id"] = qa_pair_id
    if eval_metadata is not None:
        result_record["eval_metadata"] = eval_metadata
        for key in ("_id", "domain", "sub_domain", "difficulty", "length", "slice", "branch", "sample_id"):
            if key in eval_metadata:
                result_record[key] = eval_metadata[key]
    return result_record


def process_queries_for_context(agent, query_answer_pairs, dataset_config, metrics, results,
                               query_index, context_index, skipped_query_ids, max_queries,
                               agent_config, output_path, time_cost_list, start_time,
                               context_native_timing=None):
    """Process all queries for a given context."""
    print(f"\n!!!!!Processing {len(query_answer_pairs)} queries for context {context_index}!!!!!\n")
    
    for query_data in tqdm(query_answer_pairs, total=len(query_answer_pairs)):
        query, answer, qa_pair_id, eval_metadata = unpack_query_data(query_data)
        
        # Skip queries that have already been processed
        if should_skip_query(query_index, skipped_query_ids):
            logger.info(f"!!!!!Query {query_index} already processed, skipping...\n")
            query_index += 1
            continue
        
        # Check if we've reached the query limit for ablation studies
        if has_reached_query_limit(max_queries, query_index):
            break
        
        # Process the current query
        try:
            metrics, results = process_single_query(
                agent, query, answer, dataset_config, metrics, results,
                query_index, context_index, qa_pair_id, eval_metadata,
                context_native_timing=context_native_timing,
            )
        except Exception as exc:
            logger.exception(
                "Query %s failed for context %s; recording failure and continuing.",
                query_index,
                context_index,
            )
            results.append(
                build_failed_query_record(
                    query=query,
                    answer=answer,
                    query_index=query_index,
                    context_index=context_index,
                    qa_pair_id=qa_pair_id,
                    eval_metadata=eval_metadata,
                    exc=exc,
                )
            )
        query_index += 1
        
        # Save results after each query (freq = 1)
        save_results_to_file(output_path, agent_config, dataset_config, results, 
                           metrics, time_cost_list, start_time)
        
    return metrics, results, query_index


def process_context(context_index, context_chunks, query_answer_pairs, agent_config, dataset_config,
                   metrics, results, query_index, completed_context_ids, skipped_query_ids,
                   max_queries, output_path, time_cost_list, start_time, force_rerun, total_contexts):
    """Process a single context and its queries."""
    # Skip contexts that have already been fully processed
    if should_skip_context(force_rerun, context_index, completed_context_ids):
        logger.info(f"\n\n!!!!!Experiment {context_index} already finished, skipping...\n")
        return metrics, results, query_index + len(query_answer_pairs), False
    
    # Break early if we've reached the query limit
    if has_reached_query_limit(max_queries, query_index):
        return metrics, results, query_index, True
    
    # Initialize agent for the current context
    agent_save_folder = generate_agent_save_folder(agent_config, dataset_config, context_index)
    agent, context_native_timing = initialize_and_memorize_agent(
        agent_config,
        dataset_config,
        agent_save_folder,
        context_chunks,
        context_index,
        total_contexts,
        force_rebuild=force_rerun,
    )
    
    # Process all queries for this context
    metrics, results, query_index = process_queries_for_context(
        agent, query_answer_pairs, dataset_config, metrics, results,
        query_index, context_index, skipped_query_ids, max_queries,
        agent_config, output_path, time_cost_list, start_time,
        context_native_timing=context_native_timing,
    )
    
    return metrics, results, query_index, False


def _attach_membench_native_rtwt_to_output(
    agent_output,
    dataset_config,
    eval_metadata,
    context_index,
    context_native_timing,
    query_wall_time_s,
):
    """Attach runner-boundary MemBench native RT/WT sidecar fields to query outputs."""
    if not isinstance(agent_output, dict):
        return
    if str(dataset_config.get("dataset", "")).lower() != "membench":
        return
    if (eval_metadata or {}).get("dataset") != "membench_qa":
        return

    agent_output["membench_native_read_time_s"] = round(float(query_wall_time_s), 6)
    agent_output["membench_native_context_key"] = (
        (eval_metadata or {}).get("sample_id")
        or context_index
    )

    if not context_native_timing:
        return

    total_write_time_s = _coerce_float(context_native_timing.get("total_write_time_s"))
    mean_write_time_s = _coerce_float(context_native_timing.get("mean_write_time_s"))
    total_write_ops = _coerce_int(context_native_timing.get("total_write_ops"))
    timing_source = context_native_timing.get("timing_source")

    if total_write_time_s is not None:
        agent_output["membench_native_write_time_total_s"] = round(total_write_time_s, 6)
    if mean_write_time_s is not None:
        agent_output["membench_native_write_time_mean_s"] = round(mean_write_time_s, 6)
    if total_write_ops is not None:
        agent_output["membench_native_write_ops"] = total_write_ops
    if timing_source is not None:
        agent_output["membench_native_timing_source"] = timing_source


def _validate_backfill_longmemeval_request(args, agent_config, dataset_config):
    """Validate the query-only LongMemEval recall backfill mode."""
    from evaluation.longmemeval.memoryagentbench_longmemeval_recall import (
        supports_memoryagentbench_longmemeval_recall,
    )

    if args.force:
        raise ValueError(
            "--force cannot be combined with --backfill_longmemeval_recall_debug; "
            "this mode only reuses existing saved state."
        )

    if not supports_memoryagentbench_longmemeval_recall(dataset_config):
        raise ValueError(
            "--backfill_longmemeval_recall_debug only supports MemoryAgentBench / LongMemEval datasets."
        )

    _, method_label = _resolve_backfill_longmemeval_method(agent_config.get("agent_name"))
    if method_label is None:
        supported_methods = ", ".join(label for _, label in BACKFILL_LONGMEMEVAL_RECALL_METHODS)
        raise ValueError(
            "--backfill_longmemeval_recall_debug currently supports only: "
            f"{supported_methods}."
        )

    if args.retry_failed_queries:
        logger.info(
            "--retry_failed_queries is ignored in backfill mode because no main results JSON is loaded."
        )

    return method_label


def _run_backfill_longmemeval_recall_debug(
    args,
    agent_config,
    dataset_config,
    all_context_chunks,
    all_query_answer_pairs,
):
    """Reuse saved agent state, rerun queries, and backfill LongMemEval recall debug files."""
    from evaluation.longmemeval.memoryagentbench_longmemeval_recall import (
        build_score_debug_report,
        default_score_outfile,
        write_report,
    )

    method_label = _validate_backfill_longmemeval_request(args, agent_config, dataset_config)
    logger.info("Starting LongMemEval recall backfill for %s", method_label)

    max_queries = args.max_test_queries_ablation
    total_contexts = len(all_context_chunks)
    query_index = 0
    successful_queries = 0
    failed_queries = []
    debug_dir = None

    for context_index, (context_chunks, query_answer_pairs) in enumerate(
        tqdm(zip(all_context_chunks, all_query_answer_pairs), total=total_contexts)
    ):
        if has_reached_query_limit(max_queries, query_index):
            break

        agent_save_folder = generate_agent_save_folder(agent_config, dataset_config, context_index)
        agent, _ = initialize_and_memorize_agent(
            agent_config,
            dataset_config,
            agent_save_folder,
            context_chunks,
            context_index,
            total_contexts,
            force_rebuild=False,
            load_existing_only=True,
            backfill_longmemeval_recall_debug=True,
        )
        if debug_dir is None:
            debug_dir = Path(agent._get_retrieval_debug_root()).resolve()
            debug_dir.mkdir(parents=True, exist_ok=True)

        for query_data in tqdm(query_answer_pairs, total=len(query_answer_pairs), leave=False):
            query, _answer, _qa_pair_id, eval_metadata = unpack_query_data(query_data)
            if has_reached_query_limit(max_queries, query_index):
                break

            try:
                agent.send_message(
                    query,
                    memorizing=False,
                    query_id=query_index,
                    context_id=context_index,
                    eval_metadata=eval_metadata,
                )
                successful_queries += 1
            except Exception as exc:
                logger.exception(
                    "LongMemEval recall backfill query %s failed for context %s.",
                    query_index,
                    context_index,
                )
                failed_queries.append(
                    {
                        "query_id": query_index,
                        "context_id": context_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            query_index += 1

    if debug_dir is None:
        raise RuntimeError("Backfill mode did not initialize any agent, so no debug directory was created.")

    report = build_score_debug_report(
        debug_dir=debug_dir,
        max_queries=max_queries or None,
        allow_missing_debug=False,
    )
    report["backfill_method"] = method_label
    report["backfill_successful_queries"] = successful_queries
    report["backfill_failed_queries"] = failed_queries

    report_path = write_report(report, default_score_outfile(None, debug_dir))
    logger.info("LongMemEval recall backfill report saved at %s", report_path)
    if failed_queries:
        logger.warning(
            "LongMemEval recall backfill finished with %s failed queries; see %s",
            len(failed_queries),
            report_path,
        )
    else:
        logger.info(
            "LongMemEval recall backfill completed successfully for %s queries.",
            successful_queries,
        )

    return report, report_path


def main():
    """Main function to run the memory agent benchmark evaluation."""
    # Parse command line arguments and setup configurations
    args = parse_command_line_arguments()
    agent_config, dataset_config, output_path = setup_configs_and_directories(args)
    
    # Create agent and fetch evaluation data
    start_time, all_context_chunks, all_query_answer_pairs = create_agent_and_fetch_data(
        agent_config, dataset_config
    )

    if args.backfill_longmemeval_recall_debug:
        _run_backfill_longmemeval_recall_debug(
            args,
            agent_config,
            dataset_config,
            all_context_chunks,
            all_query_answer_pairs,
        )
        logger.info(f"Total time taken: {time.time() - start_time}")
        return
    
    # Load existing results and initialize tracking variables
    time_cost_list = []
    metrics, results, completed_context_ids, skipped_query_ids = initialize_progress_tracking(
        output_path, dataset_config, all_query_answer_pairs, args.force, args.retry_failed_queries
    )
    
    # Start evaluation loop - process each context and its associated queries
    query_index = 0  # Tracks total queries processed across all contexts
    total_contexts = len(all_context_chunks)
    
    for context_index, (context_chunks, query_answer_pairs) in enumerate(
        tqdm(zip(all_context_chunks, all_query_answer_pairs), total=total_contexts)
    ):
        metrics, results, query_index, should_break = process_context(
            context_index, context_chunks, query_answer_pairs, agent_config, dataset_config,
            metrics, results, query_index, completed_context_ids, skipped_query_ids,
            args.max_test_queries_ablation, output_path, time_cost_list, start_time,
            args.force, total_contexts
        )
        
        if should_break:
            break

    _maybe_generate_post_run_reports(output_path, dataset_config)
    
    # Log completion
    logger.info(f"Total time taken: {time.time() - start_time}")


def _maybe_generate_post_run_reports(output_path, dataset_config):
    """Generate supplementary evaluation reports for datasets that support them."""
    try:
        from evaluation.longmemeval.memoryagentbench_longmemeval_recall import (
            generate_memoryagentbench_longmemeval_recall_report,
            supports_memoryagentbench_longmemeval_recall,
        )
    except ImportError:
        logger.exception("Failed to import supplementary LongMemEval recall report generator.")
        raise

    if not supports_memoryagentbench_longmemeval_recall(dataset_config):
        return

    report, report_path = generate_memoryagentbench_longmemeval_recall_report(
        output_path,
        raise_on_error=False,
    )
    status = report.get("status", "unknown")
    if status == "error":
        logger.error(
            "LongMemEval recall report generation failed; wrote diagnostic file at %s (%s)",
            report_path,
            report.get("details", "no details"),
        )
    elif status == "skipped":
        logger.info(
            "LongMemEval recall report skipped (%s); wrote status file at %s",
            report.get("reason", "unknown"),
            report_path,
        )
    else:
        logger.info("LongMemEval recall report saved at %s", report_path)


if __name__ == '__main__':
    main()
