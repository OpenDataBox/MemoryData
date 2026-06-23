"""Helpers for LongMemEval LLM-as-Judge post-processing.

This module is intentionally side-effect light:
- it never mutates existing benchmark result JSON files
- it writes only dedicated sidecar judge outputs
- it prefers canonical eval_metadata.question_id lookups
- it can fall back to legacy query-text matching only when question_id is absent
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.provider_utils import resolve_base_url, resolve_env_value


LONGMEMEVAL_SINGLE_SESSION_PROMPT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {golden_answer}\n\n"
    "Model Response: {generated_answer}\n\n"
    "Is the model response correct? Answer yes or no only."
)

LONGMEMEVAL_TEMPORAL_PROMPT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for "
    "the number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {golden_answer}\n\n"
    "Model Response: {generated_answer}\n\n"
    "Is the model response correct? Answer yes or no only."
)

LONGMEMEVAL_KNOWLEDGE_UPDATE_PROMPT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {golden_answer}\n\n"
    "Model Response: {generated_answer}\n\n"
    "Is the model response correct? Answer yes or no only."
)

LONGMEMEVAL_PREFERENCE_PROMPT = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\n"
    "Question: {question}\n\n"
    "Rubric: {golden_answer}\n\n"
    "Model Response: {generated_answer}\n\n"
    "Is the model response correct? Answer yes or no only."
)

LONGMEMEVAL_ABSTENTION_PROMPT = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information is "
    "not.\n\n"
    "Question: {question}\n\n"
    "Explanation: {golden_answer}\n\n"
    "Model Response: {generated_answer}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes "
    "or no only."
)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for LongMemEval. "
    "Return JSON only with a single key: label. "
    'Valid labels are "CORRECT" or "WRONG".'
)

QUESTION_TYPE_TO_PROMPT_KEY = {
    "single-session-user": "single-session-user",
    "single-session-assistant": "single-session-user",
    "multi-session": "single-session-user",
    "temporal-reasoning": "temporal-reasoning",
    "knowledge-update": "knowledge-update",
    "single-session-preference": "single-session-preference",
    "abstention": "abstention",
}

PROMPT_KEY_TO_TEMPLATE = {
    "single-session-user": LONGMEMEVAL_SINGLE_SESSION_PROMPT,
    "temporal-reasoning": LONGMEMEVAL_TEMPORAL_PROMPT,
    "knowledge-update": LONGMEMEVAL_KNOWLEDGE_UPDATE_PROMPT,
    "single-session-preference": LONGMEMEVAL_PREFERENCE_PROMPT,
    "abstention": LONGMEMEVAL_ABSTENTION_PROMPT,
}


def utc_now_iso() -> str:
    """Return an RFC3339-ish UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load JSON payload from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def dump_json_atomic(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid corrupting sidecars on interruption."""
    path = str(path)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)


def expand_result_paths(
    result_files: list[str] | tuple[str, ...],
    result_globs: list[str] | tuple[str, ...],
) -> list[str]:
    """Expand explicit paths and glob patterns into a stable deduped path list."""
    import glob

    paths = [os.path.normpath(path) for path in result_files]
    for pattern in result_globs:
        paths.extend(os.path.normpath(path) for path in glob.glob(pattern))

    deduped = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return sorted(deduped)


def looks_like_longmemeval_result(result_payload: dict[str, Any]) -> bool:
    """Return True when the result payload points at a LongMemEval dataset."""
    dataset_config = result_payload.get("dataset_config") or {}
    sub_dataset = str(dataset_config.get("sub_dataset") or "").strip().lower()
    return "longmemeval" in sub_dataset


def resolve_longmemeval_dataset_json(
    sub_dataset: str,
    explicit_dataset_json: str | None = None,
) -> Path:
    """Map the result payload sub-dataset name to a canonical LongMemEval JSON path."""
    if explicit_dataset_json:
        return Path(explicit_dataset_json)

    normalized = str(sub_dataset or "").strip().lower()
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "datasets" / "LongMemEval" / "eval_dataset_collection"

    if "longmemeval_oracle" in normalized:
        return dataset_root / "longmemeval_oracle" / "longmemeval_oracle.json"
    if "longmemeval_m" in normalized:
        return dataset_root / "longmemeval_m" / "longmemeval_m_cleaned.json"
    if "longmemeval_s" in normalized:
        return dataset_root / "longmemeval_s" / "longmemeval_s_cleaned.json"

    raise ValueError(f"Unsupported LongMemEval sub_dataset: {sub_dataset}")


def load_longmemeval_question_index(
    dataset_json: str | os.PathLike[str],
) -> dict[str, dict[str, Any]]:
    """Build a question_id -> canonical question record index."""
    records = load_json(dataset_json)
    if not isinstance(records, list):
        raise ValueError(f"Expected LongMemEval JSON list in {dataset_json}")

    index = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        question_id = str(record.get("question_id") or "").strip()
        if question_id:
            index[question_id] = record
    return index


def normalize_longmemeval_question_text(question: str | None) -> str:
    """Normalize question text for exact legacy fallback matching."""
    return re.sub(r"\s+", " ", str(question or "")).strip()


def build_longmemeval_question_text_index(
    question_index: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a normalized question-text -> canonical record index."""
    index = {}
    for record in question_index.values():
        normalized_question = normalize_longmemeval_question_text(record.get("question"))
        if not normalized_question:
            continue

        existing = index.get(normalized_question)
        if existing and existing.get("question_id") != record.get("question_id"):
            raise ValueError(
                "Ambiguous LongMemEval question text encountered while building "
                f"legacy fallback index: {normalized_question!r}"
            )
        index[normalized_question] = record
    return index


def extract_longmemeval_question_from_query(query: str | None) -> str | None:
    """Extract the benchmark question text from a legacy result query."""
    content = str(query or "").strip()
    if not content:
        return None

    patterns = [
        r"Now Answer the Question:\s*(.*?)\s*\n+\s*Answer:\s*$",
        r"\bQuestion:\s*(.*?)\s*\n+\s*Answer:\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        normalized_question = normalize_longmemeval_question_text(match.group(1))
        if normalized_question:
            return normalized_question
    return None


def normalize_longmemeval_question_type(question_id: str, question_type: str | None) -> str:
    """Normalize LongMemEval question types, treating *_abs as abstention."""
    normalized_question_id = str(question_id or "").strip()
    if normalized_question_id.endswith("_abs"):
        return "abstention"
    return str(question_type or "").strip()


def prompt_key_for_question(question_id: str, question_type: str | None) -> str:
    """Map a canonical LongMemEval question type onto a judge prompt key."""
    normalized_type = normalize_longmemeval_question_type(question_id, question_type)
    prompt_key = QUESTION_TYPE_TO_PROMPT_KEY.get(normalized_type)
    if not prompt_key:
        raise ValueError(f"Unsupported LongMemEval question type: {normalized_type}")
    return prompt_key


def build_judge_messages(
    question: str,
    golden_answer: Any,
    generated_answer: str,
    question_id: str,
    question_type: str | None,
) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages for one LongMemEval judge call."""
    prompt_key = prompt_key_for_question(question_id, question_type)
    prompt_template = PROMPT_KEY_TO_TEMPLATE[prompt_key]
    user_prompt = prompt_template.format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_object(text: str) -> str | None:
    """Extract the first likely JSON object from a judge response."""
    content = str(text or "").strip()
    if not content:
        return None

    if content.startswith("{") and content.endswith("}"):
        return content

    code_block_start = content.find("```")
    if code_block_start != -1:
        code_block_end = content.find("```", code_block_start + 3)
        if code_block_end != -1:
            fenced = content[code_block_start + 3:code_block_end].strip()
            if fenced.startswith("json"):
                fenced = fenced[4:].strip()
            if fenced.startswith("{") and fenced.endswith("}"):
                return fenced

    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        return content[first_brace:last_brace + 1]
    return None


def parse_judge_label(raw_response: str) -> str | None:
    """Parse CORRECT/WRONG or YES/NO from a judge response."""
    content = str(raw_response or "").strip()
    if not content:
        return None

    json_blob = extract_json_object(content)
    if json_blob:
        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            label = str(parsed.get("label") or "").strip().upper()
            if label in {"CORRECT", "WRONG", "YES", "NO"}:
                return label

    normalized = content.upper()
    for label in ("CORRECT", "WRONG", "YES", "NO"):
        if label in normalized:
            return label
    return None


def judge_label_to_bool(label: str | None) -> bool | None:
    """Convert parsed judge labels into booleans."""
    normalized = str(label or "").strip().upper()
    if normalized in {"CORRECT", "YES"}:
        return True
    if normalized in {"WRONG", "NO"}:
        return False
    return None


def normalize_sidecar_suffix(sidecar_suffix: str | None) -> str | None:
    """Normalize a user-provided sidecar suffix into a filesystem-safe label."""
    suffix = str(sidecar_suffix or "").strip()
    if not suffix:
        return None

    suffix = re.sub(r"\s+", "_", suffix)
    if os.sep:
        suffix = suffix.replace(os.sep, "_")
    if os.altsep:
        suffix = suffix.replace(os.altsep, "_")
    suffix = re.sub(r"[^A-Za-z0-9._-]", "_", suffix)
    suffix = re.sub(r"_+", "_", suffix).strip("._-")
    if not suffix:
        raise ValueError(f"Invalid sidecar suffix: {sidecar_suffix!r}")
    return suffix


def sidecar_path_for_result(
    result_path: str | os.PathLike[str],
    sidecar_suffix: str | None = None,
    sidecar_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the dedicated sidecar path for a result file."""
    if sidecar_path is not None:
        return Path(sidecar_path)

    result_path = Path(result_path)
    normalized_suffix = normalize_sidecar_suffix(sidecar_suffix)
    filename = f"{result_path.stem}.longmemeval_judge"
    if normalized_suffix:
        filename += f".{normalized_suffix}"
    filename += ".json"
    return result_path.with_name(filename)


def sidecar_row_key(row_payload: dict[str, Any]) -> tuple[int, str]:
    """Return the stable resume key for a judged row."""
    return (
        int(row_payload.get("source_row_index", -1)),
        str(row_payload.get("question_id") or ""),
    )


def is_complete_sidecar_row(row_payload: dict[str, Any], num_runs: int) -> bool:
    """Return True when a sidecar row already contains a full judge result."""
    labels = row_payload.get("judge_labels")
    judge_mean = row_payload.get("judge_mean")
    return (
        isinstance(labels, list)
        and len(labels) == int(num_runs)
        and all(isinstance(item, bool) for item in labels)
        and isinstance(judge_mean, (int, float))
    )


def load_existing_sidecar(
    sidecar_path: str | os.PathLike[str],
    source_result_file: str,
    dataset_json: str,
    judge_model: str,
    num_runs: int,
    overwrite: bool = False,
) -> tuple[dict[tuple[int, str], dict[str, Any]], str | None]:
    """Load and validate an existing sidecar for resume support."""
    if overwrite or not Path(sidecar_path).exists():
        return {}, None

    payload = load_json(sidecar_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected sidecar JSON object in {sidecar_path}")

    expected_pairs = {
        "source_result_file": str(source_result_file),
        "dataset_json": str(dataset_json),
        "judge_model": str(judge_model),
        "num_runs": int(num_runs),
    }
    for key, expected in expected_pairs.items():
        actual = payload.get(key)
        if actual != expected:
            raise ValueError(
                f"Existing sidecar {sidecar_path} is incompatible for resume: "
                f"{key}={actual!r} != {expected!r}. Use --overwrite."
            )

    rows = {}
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        rows[sidecar_row_key(row)] = row
    return rows, payload.get("created_at")


def extract_supported_longmemeval_rows(
    result_payload: dict[str, Any],
    question_index: dict[str, dict[str, Any]],
    question_text_index: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Extract non-failed judgeable rows from a LongMemEval result payload."""
    rows = result_payload.get("data") or []
    if not isinstance(rows, list):
        raise ValueError("Expected result payload to contain a list under data")

    summary = {
        "total_rows": len(rows),
        "supported_rows": 0,
        "failed_rows_skipped": 0,
        "unsupported_rows": 0,
        "legacy_query_fallback_rows": 0,
    }
    supported_specs = []
    question_text_index = question_text_index or {}

    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            summary["unsupported_rows"] += 1
            continue

        if row.get("status") == "failed":
            summary["failed_rows_skipped"] += 1
            continue

        eval_metadata = row.get("eval_metadata")
        if not isinstance(eval_metadata, dict):
            eval_metadata = {}

        question_id = str(eval_metadata.get("question_id") or "").strip()
        canonical = None
        matching_strategy = "eval_metadata_question_id"
        if question_id:
            canonical = question_index.get(question_id)
            if canonical is None:
                summary["unsupported_rows"] += 1
                continue
        else:
            parsed_question = extract_longmemeval_question_from_query(row.get("query"))
            if not parsed_question:
                summary["unsupported_rows"] += 1
                continue

            canonical = question_text_index.get(parsed_question)
            if canonical is None:
                summary["unsupported_rows"] += 1
                continue

            question_id = str(canonical.get("question_id") or "").strip()
            if not question_id:
                summary["unsupported_rows"] += 1
                continue

            matching_strategy = "legacy_query_question"
            summary["legacy_query_fallback_rows"] += 1

        question_type = normalize_longmemeval_question_type(
            question_id,
            canonical.get("question_type") or eval_metadata.get("question_type"),
        )
        supported_specs.append(
            {
                "question_id": question_id,
                "question_type": question_type,
                "query_id": row.get("query_id"),
                "qa_pair_id": row.get("qa_pair_id") or eval_metadata.get("qa_pair_id"),
                "source_row_index": row_index,
                "question": canonical.get("question", ""),
                "gold_answer": canonical.get("answer", ""),
                "generated_answer": str(row.get("output") or ""),
                "matching_strategy": matching_strategy,
            }
        )

    summary["supported_rows"] = len(supported_specs)
    return summary, supported_specs


def build_sidecar_summary(
    base_counts: dict[str, int],
    supported_specs: list[dict[str, Any]],
    judged_rows: list[dict[str, Any]],
    num_runs: int,
) -> dict[str, Any]:
    """Build aggregate judge metrics for a sidecar payload."""
    summary = dict(base_counts)
    complete_rows = [
        row for row in judged_rows
        if is_complete_sidecar_row(row, num_runs)
    ]
    summary["judged_rows"] = len(complete_rows)

    if not complete_rows:
        summary["judge_accuracy"] = None
        summary["judge_accuracy_std"] = None
        summary["judge_accuracy_by_type"] = {}
        return summary

    row_scores = [float(row["judge_mean"]) for row in complete_rows]
    summary["judge_accuracy"] = sum(row_scores) / len(row_scores)

    run_accuracies = []
    for run_index in range(int(num_runs)):
        run_values = []
        for row in complete_rows:
            labels = row.get("judge_labels") or []
            if run_index < len(labels):
                run_values.append(1.0 if labels[run_index] else 0.0)
        if run_values:
            run_accuracies.append(sum(run_values) / len(run_values))

    if len(run_accuracies) <= 1:
        summary["judge_accuracy_std"] = 0.0
    else:
        mean_value = sum(run_accuracies) / len(run_accuracies)
        variance = sum((value - mean_value) ** 2 for value in run_accuracies) / len(run_accuracies)
        summary["judge_accuracy_std"] = variance ** 0.5

    supported_type_counts = defaultdict(int)
    for spec in supported_specs:
        supported_type_counts[spec["question_type"]] += 1

    judged_by_type = defaultdict(list)
    std_rows_by_type = defaultdict(list)
    for row in complete_rows:
        question_type = str(row.get("question_type") or "unknown")
        judged_by_type[question_type].append(float(row["judge_mean"]))
        std_rows_by_type[question_type].append(row)

    by_type = {}
    for question_type in sorted(supported_type_counts.keys()):
        row_values = judged_by_type.get(question_type, [])
        entry = {
            "supported_count": supported_type_counts[question_type],
            "judged_count": len(row_values),
            "accuracy": (sum(row_values) / len(row_values)) if row_values else None,
            "std": None,
        }
        complete_type_rows = std_rows_by_type.get(question_type, [])
        if row_values:
            type_run_accuracies = []
            for run_index in range(int(num_runs)):
                run_values = []
                for row in complete_type_rows:
                    labels = row.get("judge_labels") or []
                    if run_index < len(labels):
                        run_values.append(1.0 if labels[run_index] else 0.0)
                if run_values:
                    type_run_accuracies.append(sum(run_values) / len(run_values))
            if len(type_run_accuracies) <= 1:
                entry["std"] = 0.0
            else:
                mean_value = sum(type_run_accuracies) / len(type_run_accuracies)
                variance = sum((value - mean_value) ** 2 for value in type_run_accuracies) / len(type_run_accuracies)
                entry["std"] = variance ** 0.5
        by_type[question_type] = entry

    summary["judge_accuracy_by_type"] = by_type
    return summary


def build_sidecar_payload(
    source_result_file: str,
    dataset_json: str,
    judge_model: str,
    num_runs: int,
    sidecar_suffix: str | None,
    created_at: str,
    base_counts: dict[str, int],
    supported_specs: list[dict[str, Any]],
    rows_map: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    """Build the full sidecar JSON payload."""
    ordered_rows = [
        rows_map[key]
        for key in sorted(rows_map.keys())
    ]
    return {
        "source_result_file": str(source_result_file),
        "dataset_json": str(dataset_json),
        "judge_model": str(judge_model),
        "num_runs": int(num_runs),
        "sidecar_suffix": normalize_sidecar_suffix(sidecar_suffix),
        "created_at": created_at,
        "updated_at": utc_now_iso(),
        "summary": build_sidecar_summary(
            base_counts=base_counts,
            supported_specs=supported_specs,
            judged_rows=ordered_rows,
            num_runs=num_runs,
        ),
        "rows": ordered_rows,
    }


class LongMemEvalJudgeClient:
    """OpenAI-compatible client wrapper for LongMemEval judge calls."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        base_url: str | None = None,
        max_concurrency: int = 8,
    ) -> None:
        from openai import AsyncOpenAI

        api_key = resolve_env_value(api_key_env, ["OPENAI_API_KEY"])
        api_key = str(api_key or "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key for LongMemEval judge. Set {api_key_env} or OPENAI_API_KEY."
            )

        client_kwargs = {"api_key": api_key}
        resolved_base_url = resolve_base_url(base_url, base_url_env)
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def close(self) -> None:
        """Close the underlying async client when supported."""
        close_method = getattr(self.client, "close", None)
        if close_method is None:
            return
        maybe_coro = close_method()
        if asyncio.iscoroutine(maybe_coro):
            await maybe_coro

    async def judge_once(
        self,
        question: str,
        golden_answer: Any,
        generated_answer: str,
        question_id: str,
        question_type: str,
    ) -> str:
        """Run one judge call and return the raw text response."""
        messages = build_judge_messages(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer,
            question_id=question_id,
            question_type=question_type,
        )
        async with self._semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
        return response.choices[0].message.content or ""

    async def judge_row(self, row_spec: dict[str, Any], num_runs: int) -> dict[str, Any]:
        """Run judge evaluation for one supported result row."""
        raw_responses = []
        parsed_labels = []
        judge_labels = []

        for _ in range(int(num_runs)):
            raw_response = await self.judge_once(
                question=row_spec["question"],
                golden_answer=row_spec["gold_answer"],
                generated_answer=row_spec["generated_answer"],
                question_id=row_spec["question_id"],
                question_type=row_spec["question_type"],
            )
            parsed_label = parse_judge_label(raw_response)
            judged_bool = judge_label_to_bool(parsed_label)
            if judged_bool is None:
                raise ValueError(
                    f"Unable to parse judge label for question_id={row_spec['question_id']}: "
                    f"{raw_response!r}"
                )
            raw_responses.append(raw_response)
            parsed_labels.append(parsed_label)
            judge_labels.append(judged_bool)

        judge_mean = sum(1.0 if value else 0.0 for value in judge_labels) / len(judge_labels)
        return {
            "question_id": row_spec["question_id"],
            "question_type": row_spec["question_type"],
            "query_id": row_spec["query_id"],
            "qa_pair_id": row_spec["qa_pair_id"],
            "source_row_index": row_spec["source_row_index"],
            "gold_answer": row_spec["gold_answer"],
            "generated_answer": row_spec["generated_answer"],
            "matching_strategy": row_spec.get("matching_strategy"),
            "judge_labels": judge_labels,
            "judge_mean": judge_mean,
            "parsed_label": parsed_labels,
            "raw_judge_response": raw_responses,
        }


async def process_longmemeval_result_file(
    result_path: str | os.PathLike[str],
    judge_model: str = "gpt-4o",
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    base_url: str | None = None,
    max_concurrency: int = 8,
    num_runs: int = 1,
    dataset_json: str | None = None,
    sidecar_suffix: str | None = None,
    sidecar_path: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    judge_client: LongMemEvalJudgeClient | None = None,
) -> dict[str, Any]:
    """Process one result file and write a sidecar if supported rows are present."""
    result_path = str(result_path)
    result_payload = load_json(result_path)
    if not isinstance(result_payload, dict):
        raise ValueError(f"Expected JSON object in {result_path}")

    if not looks_like_longmemeval_result(result_payload):
        return {
            "path": result_path,
            "skipped": True,
            "written": False,
            "reason": "not_longmemeval_result",
        }

    dataset_config = result_payload.get("dataset_config") or {}
    resolved_dataset_json = str(
        resolve_longmemeval_dataset_json(
            sub_dataset=str(dataset_config.get("sub_dataset") or ""),
            explicit_dataset_json=dataset_json,
        )
    )
    question_index = load_longmemeval_question_index(resolved_dataset_json)
    question_text_index = build_longmemeval_question_text_index(question_index)
    base_counts, supported_specs = extract_supported_longmemeval_rows(
        result_payload=result_payload,
        question_index=question_index,
        question_text_index=question_text_index,
    )

    normalized_sidecar_suffix = normalize_sidecar_suffix(sidecar_suffix)
    sidecar_path = sidecar_path_for_result(
        result_path,
        sidecar_suffix=normalized_sidecar_suffix,
        sidecar_path=sidecar_path,
    )
    if not supported_specs:
        return {
            "path": result_path,
            "skipped": True,
            "written": False,
            "reason": "no_supported_rows",
            **base_counts,
            "sidecar_path": str(sidecar_path),
        }

    created_at = utc_now_iso()
    rows_map, existing_created_at = load_existing_sidecar(
        sidecar_path=sidecar_path,
        source_result_file=result_path,
        dataset_json=resolved_dataset_json,
        judge_model=judge_model,
        num_runs=num_runs,
        overwrite=overwrite,
    )
    if existing_created_at:
        created_at = existing_created_at

    supported_keys = {
        (int(spec["source_row_index"]), str(spec["question_id"]))
        for spec in supported_specs
    }
    rows_map = {
        key: row
        for key, row in rows_map.items()
        if key in supported_keys
    }

    pending_specs = [
        spec for spec in supported_specs
        if not is_complete_sidecar_row(
            rows_map.get((int(spec["source_row_index"]), str(spec["question_id"])), {}),
            num_runs=num_runs,
        )
    ]

    created_client = False
    if judge_client is None:
        judge_client = LongMemEvalJudgeClient(
            model=judge_model,
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            base_url=base_url,
            max_concurrency=max_concurrency,
        )
        created_client = True

    try:
        async def _run_one(spec: dict[str, Any]) -> tuple[tuple[int, str], dict[str, Any]]:
            key = (int(spec["source_row_index"]), str(spec["question_id"]))
            try:
                row = await judge_client.judge_row(spec, num_runs=num_runs)
            except Exception as exc:
                row = {
                    "question_id": spec["question_id"],
                    "question_type": spec["question_type"],
                    "query_id": spec["query_id"],
                    "qa_pair_id": spec["qa_pair_id"],
                    "source_row_index": spec["source_row_index"],
                    "gold_answer": spec["gold_answer"],
                    "generated_answer": spec["generated_answer"],
                    "matching_strategy": spec.get("matching_strategy"),
                    "judge_error": f"{type(exc).__name__}: {exc}",
                }
            return key, row

        tasks = [asyncio.create_task(_run_one(spec)) for spec in pending_specs]
        for completed in asyncio.as_completed(tasks):
            key, row = await completed
            rows_map[key] = row
            payload = build_sidecar_payload(
                source_result_file=result_path,
                dataset_json=resolved_dataset_json,
                judge_model=judge_model,
                num_runs=num_runs,
                sidecar_suffix=normalized_sidecar_suffix,
                created_at=created_at,
                base_counts=base_counts,
                supported_specs=supported_specs,
                rows_map=rows_map,
            )
            dump_json_atomic(sidecar_path, payload)

        final_payload = build_sidecar_payload(
            source_result_file=result_path,
            dataset_json=resolved_dataset_json,
            judge_model=judge_model,
            num_runs=num_runs,
            sidecar_suffix=normalized_sidecar_suffix,
            created_at=created_at,
            base_counts=base_counts,
            supported_specs=supported_specs,
            rows_map=rows_map,
        )
        dump_json_atomic(sidecar_path, final_payload)
        return {
            "path": result_path,
            "skipped": False,
            "written": True,
            "sidecar_path": str(sidecar_path),
            **final_payload["summary"],
        }
    finally:
        if created_client:
            await judge_client.close()
