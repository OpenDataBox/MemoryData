"""Validate LoCoMo predictions and report official F1 and observed API usage."""

import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def read_jsonl(path):
    """Never repair or modify a journal, including a partially written last line."""
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    return rows


def expected_questions(dataset_path, expected_count):
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8-sig"))
    expected, samples = {}, set()
    for sample in dataset:
        sample_id = sample["sample_id"]
        if not isinstance(sample_id, str) or sample_id in samples:
            raise ValueError("Dataset has an invalid or duplicate sample_id")
        samples.add(sample_id)
        for index, qa in enumerate(sample["qa"]):
            if type(qa["category"]) is not int:
                raise ValueError("Dataset category must be an integer")
            if qa["category"] in (1, 2, 3, 4):
                for key in ("question", "answer", "category", "evidence"):
                    if key not in qa:
                        raise ValueError(f"Dataset QA is missing {key}")
                expected[(sample_id, index)] = qa
    if len(expected) != expected_count:
        raise ValueError(f"Expected {expected_count} QA, found {len(expected)}")
    return expected


def official_scores(rows, scorer_path):
    spec = importlib.util.spec_from_file_location("locomo_official_comparison", scorer_path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the supplied official scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):
        scores, _, _ = module.eval_question_answering(rows, eval_key="prediction", metric="f1")
    values = [float(value) for value in scores]
    if len(values) != len(rows) or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Official scorer returned invalid F1 values")
    return values


def validate_predictions(rows, expected):
    valid, seen, issues = [], set(), []
    for number, row in enumerate(rows, 1):
        sample, index = row.get("sample"), row.get("index")
        if not isinstance(sample, str) or type(index) is not int:
            issues.append({"row": number, "error": "invalid_id"})
            continue
        key = (sample, index)
        if key in seen:
            issues.append({"sample": sample, "index": index, "error": "duplicate_id"})
            continue
        seen.add(key)
        if key not in expected:
            issues.append({"sample": sample, "index": index, "error": "unexpected_id"})
            continue
        mismatches = [field for field in ("question", "answer", "category", "evidence")
                      if field not in row or json.dumps(row[field], sort_keys=True) !=
                      json.dumps(expected[key][field], sort_keys=True)]
        if mismatches or not isinstance(row.get("prediction"), str):
            issues.append({"sample": sample, "index": index, "error": "invalid_prediction",
                           "metadata_mismatches": mismatches,
                           "prediction_is_string": isinstance(row.get("prediction"), str)})
            continue
        # Context recall from the official function is deliberately not reported.
        valid.append({field: row[field] for field in
                      ("sample", "index", "question", "answer", "category", "evidence", "prediction")})
    missing = sorted(set(expected) - {(row["sample"], row["index"]) for row in valid})
    return valid, issues, [{"sample": sample, "index": index} for sample, index in missing]


def _token_totals():
    return {"calls": 0, **dict.fromkeys(TOKEN_FIELDS, 0)}


def summarize_usage(rows, supplied):
    totals = _token_totals()
    by_phase, by_kind = defaultdict(_token_totals), defaultdict(_token_totals)
    request_ids, response_ids = Counter(), Counter()
    unmetered = invalid = failed = truncated = unknown_phase = missing_ids = incomplete_streams = 0
    for row in rows:
        kind = row.get("request_kind", "chat_completion")
        phase = row.get("phase") or "unknown"
        unknown_phase += phase == "unknown"
        request_id, response_id = row.get("request_id"), row.get("response_id")
        if request_id:
            request_ids[str(request_id)] += 1
        if response_id:
            response_ids[str(response_id)] += 1
        missing_ids += bool(row.get("schema_version") == 1 and not request_id)
        failed += bool(row.get("success") is False or row.get("error") or row.get("error_type") or
                       (row.get("http_status") is not None and not 200 <= row["http_status"] < 300))
        finish = row.get("finish_reasons", [row.get("finish_reason")])
        truncated += any(reason in ("length", "content_filter") for reason in finish)
        incomplete_streams += bool(row.get("stream") and row.get("stream_complete") is not True)
        usage = row.get("usage")
        if usage is None:
            unmetered += 1
            continue
        required = ("prompt_tokens", "total_tokens") if kind == "embedding" else TOKEN_FIELDS
        if (not isinstance(usage, dict) or row.get("usage_status") == "invalid" or
                any(type(usage.get(field)) is not int or usage[field] < 0 for field in required)):
            invalid += 1
            continue
        completion = usage.get("completion_tokens", 0) if kind == "embedding" else usage["completion_tokens"]
        if type(completion) is not int or completion < 0 or usage["total_tokens"] != usage["prompt_tokens"] + completion:
            invalid += 1
            continue
        if kind != "embedding" and not response_id:
            missing_ids += 1
        # HTTP failures with reported tokens still consumed work and remain in these totals.
        values = {**usage, "completion_tokens": completion}
        for aggregate in (totals, by_phase[phase], by_kind[kind]):
            aggregate["calls"] += 1
            for field in TOKEN_FIELDS:
                aggregate[field] += values[field]
    duplicate_requests = sum(count - 1 for count in request_ids.values() if count > 1)
    duplicate_responses = sum(count - 1 for count in response_ids.values() if count > 1)
    complete = bool(supplied and rows and not any((unmetered, invalid, unknown_phase, missing_ids,
                                                  duplicate_requests, duplicate_responses, incomplete_streams)))
    return {"complete": complete, "supplied": supplied, "requests": len(rows),
            "metered_requests": totals["calls"], "unmetered_requests": unmetered,
            "invalid_usage_requests": invalid, "failed_requests": failed,
            "truncated_requests": truncated, "incomplete_streams": incomplete_streams,
            "unknown_phase_requests": unknown_phase, "missing_id_requests": missing_ids,
            "duplicate_request_ids": duplicate_requests, "duplicate_response_ids": duplicate_responses,
            "reported_tokens": {key: value for key, value in totals.items() if key != "calls"},
            "by_phase": dict(by_phase), "by_request_kind": dict(by_kind),
            "note": "Reported tokens include metered failed attempts. Missing or invalid usage is unknown, "
                    "not zero. Completeness covers the supplied journal; every model call must be routed "
                    "through the meter. Embedding tokens are separate from generation by_request_kind."}


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def runtime_metrics(rows, telemetry_path, hourly_rate_usd):
    telemetry = json.loads(Path(telemetry_path).read_text(encoding="utf-8-sig")) if telemetry_path else {}
    metrics = {key: telemetry.get(key) for key in ("wall_seconds", "construction_seconds", "qa_seconds",
               "peak_vram_mib", "gpu_energy_wh", "mean_gpu_utilization_percent", "storage_bytes")}
    for key, value in metrics.items():
        if value is not None and not _finite_nonnegative(value):
            raise ValueError(f"Invalid runtime metric: {key}")
    if metrics["mean_gpu_utilization_percent"] is not None and metrics["mean_gpu_utilization_percent"] > 100:
        raise ValueError("GPU utilization exceeds 100 percent")
    if hourly_rate_usd is not None and not _finite_nonnegative(hourly_rate_usd):
        raise ValueError("Hourly rate must be finite and nonnegative")
    seconds = sorted(row["seconds"] for row in rows if _finite_nonnegative(row.get("seconds")))
    metrics["question_latency_seconds"] = {
        "count": len(seconds), "mean": statistics.mean(seconds) if seconds else None,
        "p50": statistics.median(seconds) if seconds else None,
        "p95": seconds[math.ceil(0.95 * len(seconds)) - 1] if seconds else None,
        "p95_method": "nearest_rank", "missing_or_invalid": len(rows) - len(seconds)}
    metrics["hourly_rate_usd"] = hourly_rate_usd
    metrics["estimated_rental_cost_usd"] = (metrics["wall_seconds"] * hourly_rate_usd / 3600
        if metrics["wall_seconds"] is not None and hourly_rate_usd is not None else None)
    return metrics


def build_report(dataset_path, predictions_path, scorer_path, usage_path=None, telemetry_path=None,
                 hourly_rate_usd=None, expected_count=1540):
    expected = expected_questions(dataset_path, expected_count)
    rows = read_jsonl(predictions_path)
    valid, issues, missing = validate_predictions(rows, expected)
    scores = official_scores(valid, scorer_path) if valid else []
    categories = defaultdict(list)
    for row, score in zip(valid, scores):
        categories[row["category"]].append(score)
    usage = summarize_usage(read_jsonl(usage_path) if usage_path else [], usage_path is not None)
    predictions_complete = not issues and not missing
    return {"schema_version": 1, "complete": predictions_complete and usage["complete"],
            "predictions_complete": predictions_complete, "usage_complete": usage["complete"],
            "expected": len(expected), "evaluated": len(valid), "prediction_rows": len(rows),
            "missing": missing, "prediction_issues": issues,
            "empty_predictions": sum(not row["prediction"].strip() for row in valid),
            "official_f1": statistics.mean(scores) if scores else None,
            "score_scope": "all_expected_questions" if predictions_complete else "validated_rows_only",
            "by_category": {str(category): {"count": len(categories[category]),
                "expected": sum(qa["category"] == category for qa in expected.values()),
                "f1": statistics.mean(categories[category]) if categories[category] else None}
                for category in (1, 2, 3, 4)},
            "by_sample": dict(Counter(row["sample"] for row in valid)),
            "usage": usage, "runtime": runtime_metrics(rows, telemetry_path, hourly_rate_usd),
            "sources_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in (("dataset", dataset_path), ("official_scorer", scorer_path))}}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "predictions", "scorer"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("usage", "telemetry", "output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--hourly-rate-usd", type=float)
    parser.add_argument("--expected-count", type=int, default=1540)
    args = parser.parse_args()
    inputs = (args.dataset, args.predictions, args.scorer, args.usage, args.telemetry)
    if args.output and any(args.output.resolve() == path.resolve() for path in inputs if path):
        raise ValueError("Report output must not overwrite an input file")
    report = build_report(args.dataset, args.predictions, args.scorer, args.usage, args.telemetry,
                          args.hourly_rate_usd, args.expected_count)
    if args.output:
        write_report(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
