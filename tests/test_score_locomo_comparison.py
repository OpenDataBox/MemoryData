"""Completion gates and official LoCoMo F1 parity, without model calls."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from scripts.score_locomo_comparison import build_report, main, read_jsonl, summarize_usage, write_report


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def usage_record(number=1, **overrides):
    return {"schema_version": 1, "request_id": f"req-{number}", "response_id": f"resp-{number}",
            "phase": "construction", "request_kind": "chat_completion", "http_status": 200,
            "success": True, "finish_reasons": ["stop"], "stream": False,
            "usage_status": "reported",
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            **overrides}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    # The upstream module imports BERTScore even for F1; verify that it is never invoked.
    unused_bert = ModuleType("bert_score")
    def forbidden_bert(*args, **kwargs):
        raise AssertionError("F1 must not invoke BERTScore")
    unused_bert.score = forbidden_bert
    monkeypatch.setitem(sys.modules, "bert_score", unused_bert)
    scorer = Path(__file__).resolve().parents[2] / "HiGMem" / "official_locomo_evaluation.py"
    if not scorer.is_file():
        pytest.skip("The supplied upstream official scorer is not checked out")
    qas = [{"question": f"Question {index}?", "answer": answer, "category": category,
            "evidence": ["D1:1"]} for index, (answer, category) in enumerate([
                ("red, blue", 1), ("not mentioned", 5), ("red red blue", 2),
                ("2024", 2), ("Seoul; alternate gold", 3), (5, 4)])]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([{"sample_id": "conv-test", "qa": qas}]), encoding="utf-8")
    rows = [{"sample": "conv-test", "index": index, **qas[index], "prediction": prediction,
             "seconds": 2 * order} for order, (index, prediction) in enumerate([
                 (0, "red"), (2, "red blue"), (3, "wrong"), (4, "Seoul"), (5, "5")], 1)]
    predictions = jsonl(tmp_path / "predictions.jsonl", rows)
    usage = jsonl(tmp_path / "usage.jsonl", [usage_record()])
    return {"dataset_path": dataset, "predictions_path": predictions,
            "scorer_path": scorer, "usage_path": usage, "expected_count": 5}, rows


def test_official_f1_category_rules_and_question_weighting(experiment):
    paths, _ = experiment
    report = build_report(**paths)
    assert report["complete"] and report["predictions_complete"] and report["usage_complete"]
    assert report["official_f1"] == pytest.approx(0.66)
    assert report["by_category"]["1"] == {"count": 1, "expected": 1, "f1": 0.5}
    assert report["by_category"]["2"]["f1"] == pytest.approx(0.4)
    assert report["by_category"]["3"]["f1"] == report["by_category"]["4"]["f1"] == 1
    assert "recall" not in report and "retrieval_accuracy" not in report
    assert report["runtime"]["question_latency_seconds"]["mean"] == 6
    assert report["runtime"]["question_latency_seconds"]["p50"] == 6
    assert report["runtime"]["question_latency_seconds"]["p95"] == 10
    assert report["runtime"]["estimated_rental_cost_usd"] is None


def test_missing_prediction_keeps_usage_and_prediction_gates_separate(experiment):
    paths, rows = experiment
    jsonl(paths["predictions_path"], rows[:-1])
    report = build_report(**paths)
    assert not report["complete"] and not report["predictions_complete"] and report["usage_complete"]
    assert report["missing"] == [{"sample": "conv-test", "index": 5}]
    assert report["score_scope"] == "validated_rows_only"


@pytest.mark.parametrize("mode", ["duplicate", "unexpected", "answer", "question", "category", "evidence"])
def test_rejects_duplicate_unexpected_and_altered_metadata(experiment, mode):
    paths, rows = experiment
    rows = deepcopy(rows)
    if mode == "duplicate":
        rows.append(rows[0])
    elif mode == "unexpected":
        rows.append({**rows[0], "index": 1})  # Excluded category 5 original index.
    else:
        rows[0][mode] = {"answer": "red", "question": "Different?", "category": True,
                         "evidence": ["D2:9"]}[mode]
    jsonl(paths["predictions_path"], rows)
    report = build_report(**paths)
    assert not report["complete"] and not report["predictions_complete"]
    assert report["prediction_issues"]


def test_no_usage_does_not_claim_complete_or_free_run(experiment):
    paths, _ = experiment
    report = build_report(**{**paths, "usage_path": None})
    assert report["predictions_complete"] and not report["usage_complete"] and not report["complete"]
    assert report["usage"]["supplied"] is False
    assert report["runtime"]["estimated_rental_cost_usd"] is None


def test_metered_http_failure_tokens_are_included_with_tools_and_embeddings():
    rows = [usage_record(), usage_record(2, phase="reflection", success=False, http_status=500,
                                       error_type="http_error"),
            usage_record(3, phase="retrieval", finish_reasons=["tool_calls"]),
            usage_record(4, request_kind="embedding", response_id=None,
                         usage={"prompt_tokens": 7, "total_tokens": 7})]
    report = summarize_usage(rows, True)
    assert report["complete"]
    assert report["failed_requests"] == 1
    assert report["truncated_requests"] == 0
    assert report["reported_tokens"] == {"prompt_tokens": 37, "completion_tokens": 9, "total_tokens": 46}
    assert report["by_phase"]["reflection"]["total_tokens"] == 13
    assert report["by_request_kind"]["embedding"]["total_tokens"] == 7


@pytest.mark.parametrize("change, counter", [
    ({"usage": None, "usage_status": "missing", "success": False}, "unmetered_requests"),
    ({"usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 14}}, "invalid_usage_requests"),
    ({"usage": {"prompt_tokens": True, "completion_tokens": 3, "total_tokens": 4}}, "invalid_usage_requests"),
    ({"usage_status": "invalid"}, "invalid_usage_requests"),
    ({"request_id": None}, "missing_id_requests"),
    ({"response_id": None}, "missing_id_requests"),
    ({"phase": None}, "unknown_phase_requests"),
    ({"stream": True, "stream_complete": False}, "incomplete_streams"),
])
def test_usage_problems_gate_completeness(change, counter):
    report = summarize_usage([usage_record(**change)], True)
    assert not report["complete"] and report[counter] == 1


@pytest.mark.parametrize("field,counter", [("response_id", "duplicate_response_ids"),
                                            ("request_id", "duplicate_request_ids")])
def test_duplicate_usage_identifiers_cannot_pass(field, counter):
    first, second = usage_record(), usage_record(2)
    second[field] = first[field]
    report = summarize_usage([first, second], True)
    assert not report["complete"] and report[counter] == 1


def test_truncated_output_is_visible_but_its_usage_remains_metered():
    report = summarize_usage([usage_record(finish_reasons=["length"])], True)
    assert report["truncated_requests"] == 1
    assert report["reported_tokens"]["total_tokens"] == 13


def test_runtime_cost_requires_observed_duration_and_explicit_price(experiment, tmp_path):
    paths, _ = experiment
    telemetry = tmp_path / "telemetry.json"
    telemetry.write_text(json.dumps({"wall_seconds": 7200, "gpu_energy_wh": 300}), encoding="utf-8")
    report = build_report(**paths, telemetry_path=telemetry, hourly_rate_usd=0.45)
    assert report["runtime"]["estimated_rental_cost_usd"] == pytest.approx(0.9)
    assert report["runtime"]["peak_vram_mib"] is None
    report_without_time = build_report(**paths, hourly_rate_usd=0.45)
    assert report_without_time["runtime"]["estimated_rental_cost_usd"] is None


def test_partial_jsonl_is_not_silently_repaired(tmp_path):
    journal = tmp_path / "usage.jsonl"
    journal.write_text('{"ok": 1}\n{"incomplete":', encoding="utf-8")
    before = journal.read_bytes()
    with pytest.raises(ValueError, match="invalid JSON"):
        read_jsonl(journal)
    assert journal.read_bytes() == before


def test_atomic_report_and_expected_dataset_count(experiment, tmp_path):
    paths, _ = experiment
    with pytest.raises(ValueError, match="Expected 1540 QA, found 5"):
        build_report(**{**paths, "expected_count": 1540})
    report = build_report(**paths)
    target = tmp_path / "nested" / "report.json"
    write_report(target, report)
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert list(target.parent.iterdir()) == [target]


def test_cli_cannot_overwrite_a_scoring_input(experiment, monkeypatch):
    paths, _ = experiment
    original = paths["dataset_path"].read_bytes()
    monkeypatch.setattr(sys, "argv", ["score", "--dataset", str(paths["dataset_path"]),
                        "--predictions", str(paths["predictions_path"]),
                        "--scorer", str(paths["scorer_path"]),
                        "--output", str(paths["dataset_path"])])
    with pytest.raises(ValueError, match="must not overwrite"):
        main()
    assert paths["dataset_path"].read_bytes() == original
