#!/usr/bin/env python3
"""Bridge MemoryAgentBench longmemeval_s* with LongMemEval-style recall.

This helper supports two workflows:

1. `convert`
   Flatten MemoryAgentBench `longmemeval_s*` questions into the same query order
   used by the benchmark runner, then export the corresponding original
   LongMemEval entries as a JSON list that can be consumed by
   `run_retrieval.py`.

2. `score-debug`
   Read retrieval debug files under `results/outputs/rag_retrieved/...` and
   compute LongMemEval-style recall metrics by aligning retrieved paragraphs
   against the original LongMemEval gold sessions / user turns.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pyarrow as pa
import pyarrow.ipc as ipc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAB_ARROW = (
    REPO_ROOT
    / "datasets/MemoryAgentBench/eval_dataset_collection/Accurate_Retrieval/data-00000-of-00001.arrow"
)
DEFAULT_LONGMEMEVAL_FILE = (
    REPO_ROOT
    / "datasets/LongMemEval/eval_dataset_collection/longmemeval_s/longmemeval_s_cleaned.json"
)
DEFAULT_SOURCE = "longmemeval_s*"
DEFAULT_KS = (1, 3, 5, 10, 30, 50)
DEBUG_FILE_RE = re.compile(r"^query_(?P<query_id>\d+)_context_(?P<context_id>\d+)\.json$")
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert and score MemoryAgentBench longmemeval_s* with LongMemEval recall gold."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Export the MemoryAgentBench longmemeval_s* subset as original LongMemEval entries.",
    )
    add_shared_dataset_args(convert_parser)
    convert_parser.add_argument("--out-file", type=Path, required=True)

    score_parser = subparsers.add_parser(
        "score-debug",
        help="Score retrieval debug files with LongMemEval-style recall metrics.",
    )
    add_shared_dataset_args(score_parser)
    score_parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Optional benchmark results JSON used to infer the rag_retrieved debug directory.",
    )
    score_parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory containing query_{id}_context_{id}.json retrieval debug files.",
    )
    score_parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Where to save the recall report. Defaults next to the debug directory or results file.",
    )
    score_parser.add_argument(
        "--ks",
        type=str,
        default=",".join(str(k) for k in DEFAULT_KS),
        help="Comma-separated K values for Recall@K. Default: 1,3,5,10,30,50",
    )
    score_parser.add_argument(
        "--min-anchor-tokens",
        type=int,
        default=8,
        help="Minimum token window used when matching a retrieved paragraph back to a gold user turn.",
    )
    score_parser.add_argument(
        "--max-anchors-per-turn",
        type=int,
        default=5,
        help="Maximum number of text anchors generated per user turn.",
    )
    score_parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def add_shared_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memoryagentbench-arrow",
        type=Path,
        default=DEFAULT_MAB_ARROW,
        help="Path to the Accurate_Retrieval Arrow file or directory.",
    )
    parser.add_argument(
        "--longmemeval-file",
        type=Path,
        default=DEFAULT_LONGMEMEVAL_FILE,
        help="Path to the original LongMemEval JSON file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        help="MemoryAgentBench metadata.source value to filter.",
    )


def resolve_arrow_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir():
        candidate = path / "data-00000-of-00001.arrow"
        if candidate.exists():
            return candidate
    return path


def load_arrow_rows(path: Path) -> List[dict]:
    path = resolve_arrow_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arrow file not found: {path}")

    with pa.memory_map(str(path), "r") as source:
        try:
            table = ipc.RecordBatchStreamReader(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            table = ipc.RecordBatchFileReader(source).read_all()
    return table.to_pylist()


def load_longmemeval_entries(path: Path) -> Tuple[List[dict], Dict[str, dict]]:
    path = path.resolve()
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    by_qid = {}
    for entry in data:
        question_id = entry["question_id"]
        if question_id in by_qid:
            raise ValueError(f"Duplicate LongMemEval question_id: {question_id}")
        by_qid[question_id] = entry
    return data, by_qid


def load_memoryagentbench_longmemeval_subset(
    arrow_path: Path,
    source_name: str,
) -> List[dict]:
    rows = load_arrow_rows(arrow_path)
    filtered_rows = [row for row in rows if row["metadata"].get("source") == source_name]
    if not filtered_rows:
        raise ValueError(f"No rows found in MemoryAgentBench for metadata.source={source_name!r}")

    flattened: List[dict] = []
    next_query_id = 0
    for context_id, row in enumerate(filtered_rows):
        metadata = row["metadata"]
        questions = row["questions"]
        answers = row["answers"]
        question_ids = metadata["question_ids"]
        question_types = metadata["question_types"]
        question_dates = metadata["question_dates"]
        qa_pair_ids = metadata["qa_pair_ids"]
        haystack_sessions = metadata["haystack_sessions"]

        lengths = {
            "questions": len(questions),
            "answers": len(answers),
            "question_ids": len(question_ids),
            "question_types": len(question_types),
            "question_dates": len(question_dates),
            "qa_pair_ids": len(qa_pair_ids),
            "haystack_sessions": len(haystack_sessions),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Mismatched longmemeval_s* field lengths in context {context_id}: {lengths}"
            )

        for question_index in range(len(questions)):
            flattened.append(
                {
                    "query_id": next_query_id,
                    "context_id": context_id,
                    "question_index": question_index,
                    "question_id": question_ids[question_index],
                    "question": questions[question_index],
                    "answer": answers[question_index],
                    "question_type": question_types[question_index],
                    "question_date": question_dates[question_index],
                    "qa_pair_id": qa_pair_ids[question_index],
                    "source": source_name,
                    "memoryagentbench_haystack_sessions": haystack_sessions[question_index],
                }
            )
            next_query_id += 1

    return flattened


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)

    translation = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
        }
    )
    text = text.translate(translation)
    text = text.replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text.lower()).strip()
    text = PUNCT_SPACE_RE.sub(r"\1", text)
    return text


def build_text_anchors(
    normalized_text: str,
    min_anchor_tokens: int,
    max_anchors_per_turn: int,
) -> List[str]:
    tokens = normalized_text.split()
    if not tokens:
        return []
    if len(tokens) <= max(2 * min_anchor_tokens, 16):
        return [normalized_text]

    window = min(16, max(min_anchor_tokens, len(tokens) // 4))
    candidate_starts = [
        0,
        max(0, len(tokens) // 4 - window // 2),
        max(0, len(tokens) // 2 - window // 2),
        max(0, (3 * len(tokens)) // 4 - window // 2),
        max(0, len(tokens) - window),
    ]
    anchors: List[str] = []
    seen = set()
    for start in candidate_starts:
        anchor = " ".join(tokens[start : start + window]).strip()
        if len(anchor) < 24 or anchor in seen:
            continue
        anchors.append(anchor)
        seen.add(anchor)
        if len(anchors) >= max_anchors_per_turn:
            break

    if not anchors:
        anchors.append(normalized_text)
    return anchors


def processed_session_corpus_id(raw_session_id: str, session_turns: Sequence[dict]) -> str:
    user_turns = [turn for turn in session_turns if turn.get("role") == "user"]
    if "answer" in raw_session_id and all(not bool(turn.get("has_answer")) for turn in user_turns):
        return raw_session_id.replace("answer", "noans")
    return raw_session_id


def processed_turn_corpus_id(raw_session_id: str, turn_index: int, turn: dict) -> str:
    base_id = f"{raw_session_id}_{turn_index}"
    if "answer" not in raw_session_id:
        return base_id
    if bool(turn.get("has_answer")):
        return base_id
    return base_id.replace("answer", "noans")


def build_longmemeval_gold_index(
    entry: dict,
    min_anchor_tokens: int,
    max_anchors_per_turn: int,
) -> dict:
    turn_records = []
    gold_turn_ids: Set[str] = set()
    gold_session_ids: Set[str] = set()

    haystack_session_ids = entry["haystack_session_ids"]
    haystack_sessions = entry["haystack_sessions"]
    if len(haystack_session_ids) != len(haystack_sessions):
        raise ValueError(
            f"LongMemEval entry {entry['question_id']} has mismatched haystack lengths: "
            f"{len(haystack_session_ids)} session ids vs {len(haystack_sessions)} sessions"
        )

    for raw_session_id, session_turns in zip(haystack_session_ids, haystack_sessions):
        session_corpus_id = processed_session_corpus_id(raw_session_id, session_turns)
        if "answer" in session_corpus_id:
            gold_session_ids.add(session_corpus_id)

        for turn_index, turn in enumerate(session_turns, start=1):
            if turn.get("role") != "user":
                continue
            turn_corpus_id = processed_turn_corpus_id(raw_session_id, turn_index, turn)
            if "answer" in turn_corpus_id:
                gold_turn_ids.add(turn_corpus_id)

            normalized_turn = normalize_text(turn.get("content", ""))
            turn_records.append(
                {
                    "turn_corpus_id": turn_corpus_id,
                    "session_corpus_id": session_corpus_id,
                    "normalized_text": normalized_turn,
                    "anchors": build_text_anchors(
                        normalized_turn,
                        min_anchor_tokens=min_anchor_tokens,
                        max_anchors_per_turn=max_anchors_per_turn,
                    ),
                }
            )

    return {
        "gold_turn_ids": gold_turn_ids,
        "gold_session_ids": gold_session_ids,
        "turn_records": turn_records,
    }


def match_paragraph_to_gold(paragraph: object, turn_records: Sequence[dict]) -> Tuple[Set[str], Set[str]]:
    normalized_paragraph = normalize_text(paragraph)
    matched_turn_ids: Set[str] = set()
    matched_session_ids: Set[str] = set()
    if not normalized_paragraph:
        return matched_turn_ids, matched_session_ids

    for record in turn_records:
        normalized_turn = record["normalized_text"]
        if not normalized_turn:
            continue

        matched = False
        if len(normalized_turn) <= 128 and normalized_turn in normalized_paragraph:
            matched = True
        elif any(anchor in normalized_paragraph for anchor in record["anchors"]):
            matched = True

        if matched:
            matched_turn_ids.add(record["turn_corpus_id"])
            matched_session_ids.add(record["session_corpus_id"])

    return matched_turn_ids, matched_session_ids


def compute_prefix_metrics(
    matched_groups: Sequence[Set[str]],
    gold_ids: Set[str],
    ks: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not gold_ids:
        for k in ks:
            metrics[f"recall_any@{k}"] = 0.0
            metrics[f"recall_all@{k}"] = 0.0
            metrics[f"coverage@{k}"] = 0.0
        return metrics

    for k in ks:
        covered_ids: Set[str] = set()
        for group in matched_groups[:k]:
            covered_ids.update(group)
        covered_gold = covered_ids & gold_ids
        metrics[f"recall_any@{k}"] = float(bool(covered_gold))
        metrics[f"recall_all@{k}"] = float(gold_ids.issubset(covered_ids))
        metrics[f"coverage@{k}"] = len(covered_gold) / len(gold_ids)
    return metrics


def flatten_prefixed_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def aggregate_metric_dicts(metric_dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    metric_names = sorted({name for metric_dict in metric_dicts for name in metric_dict})
    return {
        name: sum(metric_dict.get(name, 0.0) for metric_dict in metric_dicts) / len(metric_dicts)
        for name in metric_names
    }


def parse_ks(raw_ks: str) -> List[int]:
    values = []
    for part in raw_ks.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"K must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one K value is required")
    return sorted(set(values))


def load_results_file(path: Path | str) -> dict:
    with open(Path(path).resolve(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _count_query_debug_files(debug_dir: Path) -> int:
    if not debug_dir.exists():
        return 0
    return sum(
        1
        for path in debug_dir.glob("query_*_context_*.json")
        if DEBUG_FILE_RE.match(path.name)
    )


def infer_debug_dir_from_results(results_file: Path, report: dict) -> Path:
    agent_config = report.get("agent_config", {})
    dataset_config = report.get("dataset_config", {})
    agent_name = agent_config.get("agent_name")
    retrieve_num = agent_config.get("retrieve_num")
    sub_dataset = dataset_config.get("sub_dataset")
    chunk_size = dataset_config.get("chunk_size")
    output_dir = agent_config.get("output_dir")
    context_max_length = dataset_config.get("context_max_length", "unknown")
    max_test_samples = dataset_config.get("max_test_samples", "unknown")

    missing_fields = [
        field_name
        for field_name, value in (
            ("agent_config.agent_name", agent_name),
            ("agent_config.retrieve_num", retrieve_num),
            ("dataset_config.sub_dataset", sub_dataset),
            ("dataset_config.chunk_size", chunk_size),
        )
        if value is None
    ]
    if missing_fields:
        raise ValueError(
            f"Cannot infer debug dir from {results_file}: missing {', '.join(missing_fields)}"
        )

    legacy_debug_dir = (
        REPO_ROOT
        / "results/outputs/rag_retrieved"
        / str(agent_name)
        / f"k_{retrieve_num}"
        / str(sub_dataset)
        / f"chunksize_{chunk_size}"
    )
    if not output_dir:
        return legacy_debug_dir

    output_label = Path(str(output_dir)).name
    scoped_debug_dir = (
        REPO_ROOT
        / "results/outputs/rag_retrieved"
        / output_label
        / str(agent_name)
        / f"k_{retrieve_num}"
        / str(sub_dataset)
        / f"in{context_max_length}_max_samples{max_test_samples}"
        / f"chunksize_{chunk_size}"
    )
    if scoped_debug_dir.exists() and legacy_debug_dir.exists():
        expected_query_count = len(report.get("data", []) or [])
        scoped_count = _count_query_debug_files(scoped_debug_dir)
        legacy_count = _count_query_debug_files(legacy_debug_dir)
        if expected_query_count and legacy_count >= expected_query_count and scoped_count < expected_query_count:
            return legacy_debug_dir
        if legacy_count > scoped_count:
            return legacy_debug_dir
    if scoped_debug_dir.exists() or not legacy_debug_dir.exists():
        return scoped_debug_dir
    return legacy_debug_dir


def find_debug_files(debug_dir: Path) -> Dict[Tuple[int, int], Path]:
    debug_dir = debug_dir.resolve()
    if not debug_dir.exists():
        raise FileNotFoundError(f"Debug directory not found: {debug_dir}")

    files: Dict[Tuple[int, int], Path] = {}
    for path in sorted(debug_dir.glob("query_*_context_*.json")):
        match = DEBUG_FILE_RE.match(path.name)
        if match is None:
            continue
        query_id = int(match.group("query_id"))
        context_id = int(match.group("context_id"))
        files[(query_id, context_id)] = path
    return files


def read_debug_payload(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_retrieved_paragraphs(payload: object) -> List[object]:
    if isinstance(payload, dict):
        paragraphs = payload.get("retrieved_context_paragraphs")
        if paragraphs is None:
            paragraphs = payload.get("retrieval_context")
    elif isinstance(payload, list):
        paragraphs = payload
    elif payload is None:
        paragraphs = []
    else:
        paragraphs = [payload]

    if paragraphs is None:
        return []
    if isinstance(paragraphs, list):
        return paragraphs
    return [paragraphs]


def default_score_outfile(results_file: Optional[Path], debug_dir: Optional[Path] = None) -> Path:
    if results_file is not None:
        stem = results_file.name
        if stem.endswith(".json"):
            stem = stem[:-5]
        return results_file.with_name(f"{stem}.longmemeval_recall.json")
    if debug_dir is None:
        raise ValueError("debug_dir is required when results_file is not provided")
    return debug_dir / "memoryagentbench_longmemeval_recall.json"


def write_report(report: dict, out_file: Path) -> Path:
    out_file = out_file.resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return out_file


def supports_memoryagentbench_longmemeval_recall(dataset_config: Optional[dict]) -> bool:
    if not isinstance(dataset_config, dict):
        return False
    dataset_name = str(dataset_config.get("dataset", "")).strip().lower()
    sub_dataset = str(dataset_config.get("sub_dataset", "")).strip().lower()
    return dataset_name == "accurate_retrieval" and "longmemeval" in sub_dataset


def build_non_scored_report(
    *,
    status: str,
    reason: str,
    source: str,
    results_file: Optional[Path] = None,
    debug_dir: Optional[Path] = None,
    memoryagentbench_arrow: Optional[Path] = None,
    longmemeval_file: Optional[Path] = None,
    ks: Optional[Sequence[int]] = None,
    details: Optional[str] = None,
) -> dict:
    report = {
        "mode": "score-debug",
        "status": status,
        "reason": reason,
        "source": source,
        "memoryagentbench_arrow": (
            str(resolve_arrow_path(memoryagentbench_arrow))
            if memoryagentbench_arrow is not None
            else None
        ),
        "longmemeval_file": str(longmemeval_file.resolve()) if longmemeval_file is not None else None,
        "results_file": str(results_file.resolve()) if results_file is not None else None,
        "debug_dir": str(debug_dir.resolve()) if debug_dir is not None else None,
        "ks": list(ks or DEFAULT_KS),
        "num_candidate_queries": 0,
        "num_evaluated_queries": 0,
        "num_missing_queries": 0,
        "queries_with_any_text_match": 0,
        "paragraph_match_rate": 0.0,
        "context_id_mismatch_count": 0,
        "aggregate_metrics": {},
        "missing_queries": [],
        "context_id_mismatches": [],
        "per_query": [],
    }
    if details:
        report["details"] = details
    return report


def _get_result_eval_metadata(result_row: dict) -> dict:
    eval_metadata = result_row.get("eval_metadata")
    return eval_metadata if isinstance(eval_metadata, dict) else {}


def build_candidate_entries(
    flattened_subset: Sequence[dict],
    results_report: Optional[dict] = None,
    max_queries: Optional[int] = None,
) -> List[dict]:
    flat_lookup_by_query = {item["query_id"]: item for item in flattened_subset}
    flat_lookup_by_question = {item["question_id"]: item for item in flattened_subset}

    if results_report is None:
        candidates = list(flattened_subset)
    else:
        candidates = []
        result_rows = sorted(
            results_report.get("data", []),
            key=lambda row: int(row.get("query_id", 0)),
        )
        for result_row in result_rows:
            if "query_id" not in result_row:
                continue
            query_id = int(result_row["query_id"])
            eval_metadata = _get_result_eval_metadata(result_row)
            fallback = flat_lookup_by_query.get(query_id)

            question_id = (
                result_row.get("question_id")
                or eval_metadata.get("question_id")
                or (fallback or {}).get("question_id")
            )
            second_fallback = flat_lookup_by_question.get(question_id) if question_id else None
            base_item = fallback or second_fallback or {}

            candidate = dict(base_item)
            candidate.update(
                {
                    "query_id": query_id,
                    "context_id": result_row.get("context_id", base_item.get("context_id")),
                    "qa_pair_id": (
                        result_row.get("qa_pair_id")
                        or eval_metadata.get("qa_pair_id")
                        or base_item.get("qa_pair_id")
                    ),
                    "question_id": question_id,
                    "question_type": (
                        result_row.get("question_type")
                        or eval_metadata.get("question_type")
                        or base_item.get("question_type")
                    ),
                    "question_date": (
                        result_row.get("question_date")
                        or eval_metadata.get("question_date")
                        or base_item.get("question_date")
                    ),
                    "status": result_row.get("status"),
                }
            )
            candidates.append(candidate)

    if max_queries is not None:
        candidates = candidates[:max_queries]
    return candidates


def locate_debug_path(
    debug_files: Dict[Tuple[int, int], Path],
    query_id: int,
    context_id: Optional[int],
) -> Optional[Path]:
    if context_id is not None:
        exact_match = debug_files.get((int(query_id), int(context_id)))
        if exact_match is not None:
            return exact_match

    matching_paths = [
        path
        for (file_query_id, _), path in debug_files.items()
        if int(file_query_id) == int(query_id)
    ]
    if len(matching_paths) == 1:
        return matching_paths[0]
    return None


def build_score_debug_report(
    *,
    results_file: Optional[Path] = None,
    debug_dir: Optional[Path] = None,
    memoryagentbench_arrow: Path = DEFAULT_MAB_ARROW,
    longmemeval_file: Path = DEFAULT_LONGMEMEVAL_FILE,
    source: str = DEFAULT_SOURCE,
    ks: Sequence[int] = DEFAULT_KS,
    min_anchor_tokens: int = 8,
    max_anchors_per_turn: int = 5,
    max_queries: Optional[int] = None,
    allow_missing_debug: bool = False,
) -> dict:
    flattened_subset = load_memoryagentbench_longmemeval_subset(
        memoryagentbench_arrow,
        source,
    )
    _, longmemeval_by_qid = load_longmemeval_entries(longmemeval_file)

    results_report = None
    if results_file is not None:
        results_file = results_file.resolve()
        results_report = load_results_file(results_file)
        if debug_dir is None:
            try:
                debug_dir = infer_debug_dir_from_results(results_file, results_report)
            except Exception as exc:
                if allow_missing_debug:
                    return build_non_scored_report(
                        status="skipped",
                        reason="debug_dir_not_inferable",
                        source=source,
                        results_file=results_file,
                        memoryagentbench_arrow=memoryagentbench_arrow,
                        longmemeval_file=longmemeval_file,
                        ks=ks,
                        details=str(exc),
                    )
                raise

    if debug_dir is None:
        raise ValueError("Either results_file or debug_dir must be provided")
    debug_dir = debug_dir.resolve()

    if allow_missing_debug and not debug_dir.exists():
        return build_non_scored_report(
            status="skipped",
            reason="debug_dir_not_found",
            source=source,
            results_file=results_file,
            debug_dir=debug_dir,
            memoryagentbench_arrow=memoryagentbench_arrow,
            longmemeval_file=longmemeval_file,
            ks=ks,
        )

    debug_files = find_debug_files(debug_dir)
    if allow_missing_debug and not debug_files:
        return build_non_scored_report(
            status="skipped",
            reason="no_debug_files_found",
            source=source,
            results_file=results_file,
            debug_dir=debug_dir,
            memoryagentbench_arrow=memoryagentbench_arrow,
            longmemeval_file=longmemeval_file,
            ks=ks,
        )

    gold_index_cache: Dict[str, dict] = {}
    evaluated_queries = []
    missing_queries = []
    context_id_mismatches = []

    candidate_entries = build_candidate_entries(
        flattened_subset,
        results_report=results_report,
        max_queries=max_queries,
    )

    for subset_item in candidate_entries:
        query_id = int(subset_item["query_id"])
        context_id = subset_item.get("context_id")
        question_id = subset_item.get("question_id")
        debug_path = locate_debug_path(debug_files, query_id, context_id)
        if debug_path is None:
            missing_queries.append(
                {
                    "query_id": query_id,
                    "context_id": context_id,
                    "question_id": question_id,
                    "status": subset_item.get("status"),
                }
            )
            continue

        if question_id is None:
            raise ValueError(f"Missing question_id for query_id={query_id} in longmemeval recall scoring")

        payload = read_debug_payload(debug_path)
        retrieved_paragraphs = extract_retrieved_paragraphs(payload)

        longmemeval_entry = longmemeval_by_qid.get(question_id)
        if longmemeval_entry is None:
            raise ValueError(
                f"Question id {question_id} exists in MemoryAgentBench/results but not in LongMemEval"
            )

        if question_id not in gold_index_cache:
            gold_index_cache[question_id] = build_longmemeval_gold_index(
                longmemeval_entry,
                min_anchor_tokens=min_anchor_tokens,
                max_anchors_per_turn=max_anchors_per_turn,
            )
        gold_index = gold_index_cache[question_id]

        turn_groups = []
        session_groups = []
        matched_paragraph_count = 0
        paragraph_match_counts = []

        for paragraph in retrieved_paragraphs:
            matched_turn_ids, matched_session_ids = match_paragraph_to_gold(
                paragraph,
                gold_index["turn_records"],
            )
            if matched_turn_ids or matched_session_ids:
                matched_paragraph_count += 1
            paragraph_match_counts.append(
                {
                    "matched_turns": len(matched_turn_ids),
                    "matched_sessions": len(matched_session_ids),
                }
            )
            turn_groups.append(matched_turn_ids)
            session_groups.append(matched_session_ids)

        turn_metrics = compute_prefix_metrics(turn_groups, gold_index["gold_turn_ids"], ks)
        session_metrics = compute_prefix_metrics(session_groups, gold_index["gold_session_ids"], ks)
        combined_metrics = {}
        combined_metrics.update(flatten_prefixed_metrics("turn", turn_metrics))
        combined_metrics.update(flatten_prefixed_metrics("session", session_metrics))

        file_match = DEBUG_FILE_RE.match(debug_path.name)
        observed_context_id = int(file_match.group("context_id")) if file_match is not None else None
        if context_id is not None and observed_context_id is not None and observed_context_id != int(context_id):
            context_id_mismatches.append(
                {
                    "query_id": query_id,
                    "expected_context_id": context_id,
                    "observed_context_id": observed_context_id,
                    "path": str(debug_path),
                }
            )

        evaluated_queries.append(
            {
                "query_id": query_id,
                "context_id": context_id if context_id is not None else observed_context_id,
                "qa_pair_id": subset_item.get("qa_pair_id"),
                "question_id": question_id,
                "question_type": subset_item.get("question_type"),
                "question_date": subset_item.get("question_date"),
                "result_status": subset_item.get("status"),
                "retrieved_paragraph_count": len(retrieved_paragraphs),
                "matched_paragraph_count": matched_paragraph_count,
                "gold_answer_session_ids": sorted(gold_index["gold_session_ids"]),
                "gold_answer_turn_ids": sorted(gold_index["gold_turn_ids"]),
                "paragraph_match_counts": paragraph_match_counts,
                "metrics": combined_metrics,
            }
        )

    aggregate_metrics = aggregate_metric_dicts([entry["metrics"] for entry in evaluated_queries])
    total_retrieved_paragraphs = sum(entry["retrieved_paragraph_count"] for entry in evaluated_queries)
    total_matched_paragraphs = sum(entry["matched_paragraph_count"] for entry in evaluated_queries)
    queries_with_any_match = sum(1 for entry in evaluated_queries if entry["matched_paragraph_count"] > 0)

    return {
        "mode": "score-debug",
        "status": "ok",
        "source": source,
        "memoryagentbench_arrow": str(resolve_arrow_path(memoryagentbench_arrow)),
        "longmemeval_file": str(longmemeval_file.resolve()),
        "results_file": str(results_file.resolve()) if results_file is not None else None,
        "debug_dir": str(debug_dir),
        "ks": list(ks),
        "num_candidate_queries": len(candidate_entries),
        "num_evaluated_queries": len(evaluated_queries),
        "num_missing_queries": len(missing_queries),
        "queries_with_any_text_match": queries_with_any_match,
        "paragraph_match_rate": (
            total_matched_paragraphs / total_retrieved_paragraphs if total_retrieved_paragraphs else 0.0
        ),
        "context_id_mismatch_count": len(context_id_mismatches),
        "aggregate_metrics": aggregate_metrics,
        "missing_queries": missing_queries[:50],
        "context_id_mismatches": context_id_mismatches[:20],
        "per_query": evaluated_queries,
    }


def generate_memoryagentbench_longmemeval_recall_report(
    results_file: Path | str,
    *,
    out_file: Optional[Path | str] = None,
    memoryagentbench_arrow: Path = DEFAULT_MAB_ARROW,
    longmemeval_file: Path = DEFAULT_LONGMEMEVAL_FILE,
    source: str = DEFAULT_SOURCE,
    ks: Sequence[int] = DEFAULT_KS,
    min_anchor_tokens: int = 8,
    max_anchors_per_turn: int = 5,
    max_queries: Optional[int] = None,
    raise_on_error: bool = False,
) -> Tuple[dict, Path]:
    results_file = Path(results_file).resolve()
    out_path = Path(out_file).resolve() if out_file is not None else default_score_outfile(results_file)

    try:
        results_report = load_results_file(results_file)
        dataset_config = results_report.get("dataset_config", {})
        if not supports_memoryagentbench_longmemeval_recall(dataset_config):
            report = build_non_scored_report(
                status="skipped",
                reason="unsupported_dataset",
                source=source,
                results_file=results_file,
                memoryagentbench_arrow=memoryagentbench_arrow,
                longmemeval_file=longmemeval_file,
                ks=ks,
            )
            return report, write_report(report, out_path)

        report = build_score_debug_report(
            results_file=results_file,
            memoryagentbench_arrow=memoryagentbench_arrow,
            longmemeval_file=longmemeval_file,
            source=source,
            ks=ks,
            min_anchor_tokens=min_anchor_tokens,
            max_anchors_per_turn=max_anchors_per_turn,
            max_queries=max_queries,
            allow_missing_debug=True,
        )
        return report, write_report(report, out_path)
    except Exception as exc:
        report = build_non_scored_report(
            status="error",
            reason="report_generation_failed",
            source=source,
            results_file=results_file,
            memoryagentbench_arrow=memoryagentbench_arrow,
            longmemeval_file=longmemeval_file,
            ks=ks,
            details=str(exc),
        )
        write_report(report, out_path)
        if raise_on_error:
            raise
        return report, out_path


def run_convert(args: argparse.Namespace) -> None:
    flattened_subset = load_memoryagentbench_longmemeval_subset(
        args.memoryagentbench_arrow,
        args.source,
    )
    _, longmemeval_by_qid = load_longmemeval_entries(args.longmemeval_file)

    converted_entries = []
    missing_question_ids = []
    for item in flattened_subset:
        question_id = item["question_id"]
        longmemeval_entry = longmemeval_by_qid.get(question_id)
        if longmemeval_entry is None:
            missing_question_ids.append(question_id)
            continue

        exported_entry = copy.deepcopy(longmemeval_entry)
        exported_entry["query_id"] = item["query_id"]
        exported_entry["context_id"] = item["context_id"]
        exported_entry["qa_pair_id"] = item["qa_pair_id"]
        exported_entry["memoryagentbench_metadata"] = {
            "source": item["source"],
            "context_id": item["context_id"],
            "question_index": item["question_index"],
            "query_id": item["query_id"],
            "qa_pair_id": item["qa_pair_id"],
            "question_date": item["question_date"],
        }
        converted_entries.append(exported_entry)

    if missing_question_ids:
        raise ValueError(
            "Missing question ids in the original LongMemEval file: "
            + ", ".join(sorted(missing_question_ids)[:10])
        )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as handle:
        json.dump(converted_entries, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "mode": "convert",
                "source": args.source,
                "num_exported_entries": len(converted_entries),
                "out_file": str(args.out_file.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_score_debug(args: argparse.Namespace) -> None:
    ks = parse_ks(args.ks)
    report = build_score_debug_report(
        results_file=args.results_file.resolve() if args.results_file is not None else None,
        debug_dir=args.debug_dir.resolve() if args.debug_dir is not None else None,
        memoryagentbench_arrow=args.memoryagentbench_arrow,
        longmemeval_file=args.longmemeval_file,
        source=args.source,
        ks=ks,
        min_anchor_tokens=args.min_anchor_tokens,
        max_anchors_per_turn=args.max_anchors_per_turn,
        max_queries=args.max_queries,
        allow_missing_debug=False,
    )
    out_file = (
        args.out_file.resolve()
        if args.out_file is not None
        else default_score_outfile(
            args.results_file.resolve() if args.results_file is not None else None,
            args.debug_dir.resolve() if args.debug_dir is not None else None,
        )
    )
    write_report(report, out_file)

    print(
        json.dumps(
            {
                "mode": "score-debug",
                "status": report.get("status"),
                "num_evaluated_queries": report["num_evaluated_queries"],
                "num_missing_queries": report["num_missing_queries"],
                "paragraph_match_rate": report["paragraph_match_rate"],
                "aggregate_metrics": report["aggregate_metrics"],
                "out_file": str(out_file),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.command == "convert":
        run_convert(args)
        return
    if args.command == "score-debug":
        run_score_debug(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
