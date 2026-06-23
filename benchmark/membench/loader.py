"""MemBench dataset loader for repository evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path

from utils.locomo_utils import build_locomo_storage_text


def load_membench_eval_data(dataset_config):
    """Load MemBench rows and convert them into the shared evaluation schema."""
    dataset_path = _resolve_membench_dataset_path(dataset_config)
    max_test_samples = dataset_config.get("max_test_samples")
    branch_filter = _normalize_filter_values(dataset_config.get("membench_branches"))
    scenario_filter = _normalize_int_filter(dataset_config.get("membench_scenario_ids"))
    variant_filter = _normalize_int_filter(dataset_config.get("membench_variant_indices"))
    max_scenarios_per_branch = _safe_positive_int(dataset_config.get("membench_max_scenarios_per_branch"))
    scenario_stride = _safe_positive_int(dataset_config.get("membench_scenario_stride"))
    trajectories_per_scenario = _safe_positive_int(dataset_config.get("membench_trajectory_group_size")) or 3

    with open(dataset_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    processed_rows = []
    source_name = str(dataset_config.get("sub_dataset") or "membench").strip()
    slice_name = _resolve_slice_name(dataset_config)
    agent_view = str(dataset_config.get("membench_agent_view") or "FirstAgent").strip() or "FirstAgent"

    for branch_name, trajectories in raw_data.items():
        normalized_branch_name = str(branch_name).strip()
        if branch_filter and normalized_branch_name not in branch_filter:
            continue

        trajectory_groups = _group_trajectories(trajectories, trajectories_per_scenario)
        selected_scenarios = _select_scenario_indices(
            total_scenarios=len(trajectory_groups),
            scenario_filter=scenario_filter,
            max_scenarios=max_scenarios_per_branch,
            scenario_stride=scenario_stride,
        )

        for scenario_index in selected_scenarios:
            if scenario_index >= len(trajectory_groups):
                continue
            for variant_index, trajectory in enumerate(trajectory_groups[scenario_index]):
                if variant_filter and variant_index not in variant_filter:
                    continue
                processed_rows.append(
                    _convert_membench_trajectory(
                        trajectory=trajectory,
                        source_name=source_name,
                        slice_name=slice_name,
                        branch_name=normalized_branch_name,
                        scenario_index=scenario_index,
                        variant_index=variant_index,
                        agent_view=agent_view,
                    )
                )

    if max_test_samples:
        processed_rows = processed_rows[: int(max_test_samples)]

    return {"data": processed_rows}


def _resolve_membench_dataset_path(dataset_config):
    test_file = str(dataset_config.get("test_files") or "").strip()
    if test_file:
        return Path(test_file)

    slice_name = _resolve_slice_name(dataset_config)
    agent_view = str(dataset_config.get("membench_agent_view") or "FirstAgent").strip() or "FirstAgent"

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "datasets" / "MemBench" / "MemData" / agent_view / f"{slice_name}.json"


def _resolve_slice_name(dataset_config):
    configured_slice = str(dataset_config.get("membench_slice") or "").strip()
    if configured_slice:
        return configured_slice

    sub_dataset = str(dataset_config.get("sub_dataset") or "").strip()
    if sub_dataset.lower().startswith("membench_"):
        return sub_dataset[len("membench_") :]

    return sub_dataset or "simple"


def _group_trajectories(trajectories, trajectories_per_scenario):
    if (
        trajectories_per_scenario > 1
        and len(trajectories) >= trajectories_per_scenario
        and len(trajectories) % trajectories_per_scenario == 0
    ):
        return [
            trajectories[start_index : start_index + trajectories_per_scenario]
            for start_index in range(0, len(trajectories), trajectories_per_scenario)
        ]
    return [[trajectory] for trajectory in trajectories]


def _select_scenario_indices(total_scenarios, scenario_filter, max_scenarios, scenario_stride):
    if total_scenarios <= 0:
        return []

    if scenario_filter:
        return [index for index in sorted(scenario_filter) if 0 <= index < total_scenarios]

    candidate_indices = list(range(total_scenarios))

    if scenario_stride and scenario_stride > 1:
        candidate_indices = candidate_indices[::scenario_stride]

    if max_scenarios and max_scenarios < len(candidate_indices):
        return _select_evenly_spaced_indices(candidate_indices, max_scenarios)

    return candidate_indices


def _select_evenly_spaced_indices(candidate_indices, target_count):
    if target_count <= 0 or target_count >= len(candidate_indices):
        return list(candidate_indices)

    step = len(candidate_indices) / target_count
    selected_indices = []
    for sample_index in range(target_count):
        selected_indices.append(candidate_indices[int(sample_index * step)])

    deduped_indices = []
    seen = set()
    for scenario_index in selected_indices:
        if scenario_index in seen:
            continue
        deduped_indices.append(scenario_index)
        seen.add(scenario_index)

    return deduped_indices


def _convert_membench_trajectory(
    trajectory,
    source_name,
    slice_name,
    branch_name,
    scenario_index,
    variant_index,
    agent_view,
):
    qa = trajectory.get("QA", {}) or {}
    question_id = str(qa.get("qid", 0))
    sample_id = f"{slice_name}_{branch_name}_scenario{scenario_index}_variant{variant_index}"
    qa_pair_id = f"{sample_id}_q{question_id}"
    target_step_ids = _normalize_target_step_ids(qa.get("target_step_id") or [])
    target_source_ids = _target_step_ids_to_source_ids(target_step_ids)
    context_chunks, session_count = _build_context_chunks(
        sample_id,
        trajectory.get("message_list", []),
    )
    context_chunk_count = len(context_chunks)
    context_length = sum(len(str(chunk.get("text", ""))) for chunk in context_chunks)
    choices = qa.get("choices", {}) or {}

    answer_text = _format_option_value(qa.get("answer"))

    return {
        "source": source_name,
        "sample_id": sample_id,
        "context_chunks": context_chunks,
        "context_length": context_length,
        "context_chunk_count": context_chunk_count,
        "session_count": session_count,
        "questions": [str(qa.get("question", "") or "")],
        "answers": [str(qa.get("ground_truth", "") or "").strip().upper()],
        "question_ids": [qa_pair_id],
        "qa_pair_ids": [qa_pair_id],
        "question_dates": [str(qa.get("time", "") or "")],
        "question_types": [branch_name],
        "choice_A": _format_option_value(choices.get("A")),
        "choice_B": _format_option_value(choices.get("B")),
        "choice_C": _format_option_value(choices.get("C")),
        "choice_D": _format_option_value(choices.get("D")),
        "eval_metadata": [
            {
                "dataset": "membench_qa",
                "source": source_name,
                "slice": slice_name,
                "branch": branch_name,
                "agent_view": agent_view,
                "sample_id": sample_id,
                "question_id": qa_pair_id,
                "qa_pair_id": qa_pair_id,
                "trajectory_tid": trajectory.get("tid"),
                "scenario_index": scenario_index,
                "variant_index": variant_index,
                "target_step_id": target_step_ids,
                "target_source_ids": target_source_ids,
                "question_time": str(qa.get("time", "") or ""),
                "ground_truth": str(qa.get("ground_truth", "") or "").strip().upper(),
                "answer_text": answer_text,
                "context_chunk_count": context_chunk_count,
                "context_length": context_length,
                "session_count": session_count,
            }
        ],
    }


def _build_context_chunks(sample_id, raw_message_list):
    sessions = _normalize_sessions(raw_message_list)
    chunks = []
    global_turn_index = 0

    for session_index, session_turns in enumerate(sessions):
        for turn_index, turn in enumerate(session_turns):
            source_ids = _build_turn_source_ids(global_turn_index, session_index)
            formatted_turn = _format_turn(turn, session_index, turn_index)
            if formatted_turn:
                chunk_id = f"{sample_id}_s{session_index}_t{turn_index}"
                chunks.append(
                    {
                        "text": formatted_turn,
                        "storage_text": build_locomo_storage_text(formatted_turn, chunk_id, source_ids),
                        "chunk_id": chunk_id,
                        "source_ids": source_ids,
                        "session_index": session_index,
                        "turn_index": turn_index,
                        "global_turn_index": global_turn_index,
                    }
                )
            global_turn_index += 1

    return chunks, len(sessions)


def _normalize_sessions(raw_message_list):
    if not isinstance(raw_message_list, list):
        return [[raw_message_list]]

    if not raw_message_list:
        return []

    if all(isinstance(item, dict) for item in raw_message_list):
        return [raw_message_list]

    normalized_sessions = []
    for item in raw_message_list:
        if isinstance(item, list):
            normalized_sessions.append(item)
        else:
            normalized_sessions.append([item])
    return normalized_sessions


def _format_turn(turn, session_index, turn_index):
    turn_header = f"Session {session_index + 1} - Turn {turn_index + 1}"
    if isinstance(turn, str):
        body = turn.strip()
        return f"{turn_header}\n{body}".strip() if body else turn_header

    if not isinstance(turn, dict):
        body = str(turn).strip()
        return f"{turn_header}\n{body}".strip() if body else turn_header

    timestamp = str(turn.get("time", "") or "").strip()
    place = str(turn.get("place", "") or "").strip()
    metadata = []
    if timestamp:
        metadata.append(f"time: {timestamp}")
    if place:
        metadata.append(f"place: {place}")

    user_text = str(turn.get("user_message") or turn.get("user") or "").strip()
    assistant_text = str(turn.get("assistant_message") or turn.get("assistant") or turn.get("agent") or "").strip()
    message_text = str(turn.get("message", "") or "").strip()

    formatted_lines = []
    turn_label = turn_header
    if metadata:
        turn_label += f" [{'; '.join(metadata)}]"
    formatted_lines.append(turn_label)

    if user_text or assistant_text:
        if user_text:
            formatted_lines.append(f"User: {user_text}")
        if assistant_text:
            formatted_lines.append(f"Assistant: {assistant_text}")
    elif message_text:
        formatted_lines.append(message_text)

    return "\n".join(line for line in formatted_lines if line).strip()


def _normalize_target_step_ids(target_step_ids):
    normalized_ids = []
    for step_id in target_step_ids or []:
        if isinstance(step_id, (list, tuple)):
            normalized_step = tuple(step_id)
            if normalized_step:
                normalized_ids.append(normalized_step)
        elif step_id is not None:
            normalized_ids.append(step_id)
    return normalized_ids


def _target_step_ids_to_source_ids(target_step_ids):
    source_ids = []
    for step_id in target_step_ids:
        source_ids.extend(_target_step_id_to_source_ids(step_id))
    return _dedupe_preserve_order(source_ids)


def _target_step_id_to_source_ids(step_id):
    if isinstance(step_id, (list, tuple)):
        if len(step_id) >= 2:
            return [_format_membench_source_id(step_id[0], step_id[1])]
        if len(step_id) == 1:
            return [_format_membench_scalar_source_id(step_id[0])]
        return []
    return [_format_membench_scalar_source_id(step_id)]


def _build_turn_source_ids(global_turn_index, session_index):
    return _dedupe_preserve_order(
        [
            _format_membench_scalar_source_id(global_turn_index),
            _format_membench_source_id(global_turn_index, session_index),
        ]
    )


def _format_membench_source_id(global_turn_index, session_index):
    return f"{_safe_index_value(global_turn_index)}|{_safe_index_value(session_index)}"


def _format_membench_scalar_source_id(global_turn_index):
    return str(_safe_index_value(global_turn_index))


def _safe_index_value(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value).strip()


def _dedupe_preserve_order(values):
    deduped_values = []
    seen = set()
    for value in values or []:
        normalized_value = str(value or "").strip()
        if not normalized_value or normalized_value in seen:
            continue
        deduped_values.append(normalized_value)
        seen.add(normalized_value)
    return deduped_values


def _format_option_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _normalize_filter_values(raw_value):
    if raw_value in (None, ""):
        return set()

    if isinstance(raw_value, (list, tuple, set)):
        values = raw_value
    else:
        values = str(raw_value).split(",")

    return {
        str(value).strip()
        for value in values
        if str(value).strip()
    }


def _normalize_int_filter(raw_value):
    normalized_values = _normalize_filter_values(raw_value)
    integer_values = set()
    for value in normalized_values:
        try:
            integer_values.add(int(value))
        except (TypeError, ValueError):
            continue
    return integer_values


def _safe_positive_int(raw_value):
    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return parsed_value if parsed_value > 0 else None
