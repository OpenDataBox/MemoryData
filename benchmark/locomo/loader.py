"""LoCoMo dataset loader for repository evaluation runs."""

import json
from pathlib import Path

from utils.locomo_utils import build_locomo_storage_text, dedupe_preserve_order


def load_locomo_eval_data(dataset_config):
    """Load LoCoMo QA data as pre-chunked conversation contexts."""
    dataset_path = _resolve_locomo_dataset_path(dataset_config)
    chunk_size = int(dataset_config.get("chunk_size", 4096))
    max_test_samples = dataset_config.get("max_test_samples")
    include_categories = _normalize_category_filter(dataset_config.get("locomo_categories"))
    exclude_categories = _normalize_category_filter(dataset_config.get("locomo_exclude_categories"))

    with open(dataset_path, "r", encoding="utf-8") as file:
        raw_samples = json.load(file)

    processed_samples = [
        _convert_locomo_sample(
            sample,
            sample_index,
            chunk_size,
            include_categories=include_categories,
            exclude_categories=exclude_categories,
        )
        for sample_index, sample in enumerate(raw_samples)
    ]
    processed_samples = [
        sample for sample in processed_samples
        if sample.get("questions")
    ]
    if max_test_samples:
        processed_samples = processed_samples[: int(max_test_samples)]
    return {"data": processed_samples}


def _resolve_locomo_dataset_path(dataset_config):
    test_file = str(dataset_config.get("test_files") or "").strip()
    if test_file:
        return Path(test_file)

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "datasets" / "LoCoMo" / "locomo10.json"


def _convert_locomo_sample(sample, sample_index, chunk_size, include_categories=None, exclude_categories=None):
    sample_id = str(sample.get("sample_id", f"locomo_{sample_index}"))
    conversation = sample.get("conversation", {})
    context_chunks = _build_locomo_context_chunks(sample_id, conversation, chunk_size)

    questions = []
    answers = []
    qa_pair_ids = []
    question_ids = []
    question_types = []
    eval_metadata = []

    for qa_index, qa_item in enumerate(sample.get("qa", [])):
        category = qa_item.get("category")
        category_int = _safe_int(category)
        category_str = "" if category is None else str(category)
        if include_categories is not None and category_str not in include_categories:
            continue
        if exclude_categories is not None and category_str in exclude_categories:
            continue

        answer = (
            qa_item.get("adversarial_answer")
            if category_int == 5 and qa_item.get("adversarial_answer") is not None
            else qa_item.get("answer")
        )
        qa_pair_id = str(qa_item.get("question_id") or f"{sample_id}_qa{qa_index}")

        questions.append(qa_item.get("question", ""))
        answers.append(_normalize_locomo_answer(answer))
        qa_pair_ids.append(qa_pair_id)
        question_ids.append(qa_pair_id)
        question_types.append(category_str)
        eval_metadata.append(
            {
                "dataset": "locomo_qa",
                "sample_id": sample_id,
                "question_id": qa_pair_id,
                "qa_pair_id": qa_pair_id,
                "category": None if category is None else category_str,
                "evidence": [str(item) for item in qa_item.get("evidence", []) if str(item).strip()],
            }
        )

    return {
        "source": "locomo_qa",
        "sample_id": sample_id,
        "context_chunks": context_chunks,
        "context_length": sum(len(chunk["text"]) for chunk in context_chunks),
        "questions": questions,
        "answers": answers,
        "qa_pair_ids": qa_pair_ids,
        "question_ids": question_ids,
        "question_types": question_types,
        "eval_metadata": eval_metadata,
    }


def _build_locomo_context_chunks(sample_id, conversation, chunk_size):
    chunks = []
    current_units = []
    current_source_ids = []
    current_length = 0

    for session_key in _iter_session_keys(conversation):
        session_turns = conversation.get(session_key, []) or []
        session_time = str(conversation.get(f"{session_key}_date_time", "") or "").strip()
        session_label = session_key.replace("_", " ").title()

        for turn_index, turn in enumerate(session_turns):
            turn_text = _format_turn_text(turn, session_label, session_time, include_session_header=(turn_index == 0))
            if not turn_text.strip():
                continue

            candidate_length = len(turn_text) if not current_units else current_length + 2 + len(turn_text)
            if current_units and candidate_length > chunk_size:
                chunks.append(_create_chunk(sample_id, len(chunks), current_units, current_source_ids))
                current_units = []
                current_source_ids = []
                current_length = 0

            current_units.append(turn_text)
            current_source_ids.extend([str(turn.get("dia_id", "")).strip()])
            current_length = len(turn_text) if current_length == 0 else current_length + 2 + len(turn_text)

    if current_units:
        chunks.append(_create_chunk(sample_id, len(chunks), current_units, current_source_ids))

    return chunks


def _create_chunk(sample_id, chunk_index, units, source_ids):
    chunk_id = f"{sample_id}_chunk_{chunk_index}"
    text = "\n\n".join(unit.strip() for unit in units if unit and unit.strip()).strip()
    normalized_source_ids = dedupe_preserve_order(source_ids)
    return {
        "text": text,
        "storage_text": build_locomo_storage_text(text, chunk_id, normalized_source_ids),
        "chunk_id": chunk_id,
        "source_ids": normalized_source_ids,
    }


def _iter_session_keys(conversation):
    session_keys = [
        key for key, value in conversation.items()
        if key.startswith("session_")
        and not key.endswith("_date_time")
        and isinstance(value, list)
    ]
    return sorted(session_keys, key=lambda key: int(key.split("_")[1]))


def _format_turn_text(turn, session_label, session_time, include_session_header=False):
    text = str(turn.get("text", "") or "").strip()
    caption = str(turn.get("blip_caption", "") or "").strip()
    if caption:
        text = f"[Image: {caption}] {text}".strip()

    speaker = str(turn.get("speaker", "Unknown") or "Unknown").strip()
    turn_text = f"{speaker}: {text}".strip()
    if include_session_header:
        header = f"{session_label}"
        if session_time:
            header += f" ({session_time})"
        turn_text = f"{header}\n{turn_text}".strip()
    return turn_text


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_locomo_answer(answer):
    """Normalize LoCoMo answers to the string/list shapes expected by eval."""
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, (list, tuple)):
        return [_normalize_locomo_answer(item) for item in answer]
    return str(answer)


def _normalize_category_filter(raw_value) -> set[str] | None:
    """Normalize category filters from config into a set of strings."""
    if raw_value is None or raw_value == "":
        return None

    if isinstance(raw_value, (list, tuple, set)):
        normalized_values = raw_value
    else:
        normalized_values = str(raw_value).split(",")

    categories = {
        str(value).strip()
        for value in normalized_values
        if str(value).strip()
    }
    return categories or None
