"""Fail-closed queue gates without a GPU, remote host, or model calls."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from scripts import locomo_server_queue as queue


@pytest.fixture
def expected():
    return {
        ("conv-1", 0): dict(question="When?", answer=2023, category=2, evidence=["D1:1"]),
        ("conv-1", 2): dict(question="Who?", answer="Lee", category=1, evidence=["D1:2"]),
    }


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def completed_higmem(tmp_path, expected):
    write_lines(tmp_path / "predictions.jsonl", [
        dict(sample=sample, index=index, prediction=str(qa["answer"]), **qa)
        for (sample, index), qa in expected.items()
    ])
    write_lines(tmp_path / "usage.jsonl", [
        dict(sample=sample, question=index, phase="qa", response_id=f"response-{index}",
             usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12), finish_reason="stop")
        for sample, index in expected
    ])
    report = dict(complete=True, expected=2, evaluated=2, failed_requests=0, missing_usage=0,
                  truncated_requests=0, server_tokens={"qa": dict(calls=2, prompt_tokens=20,
                                                               completion_tokens=4, total_tokens=24)})
    (tmp_path / "report.json").write_text(json.dumps(report))
    return tmp_path


def test_live_writer_wins_over_existing_complete_report(completed_higmem, expected):
    assert queue.higmem_ready("RUNNING", completed_higmem, expected) is False
    assert queue.higmem_ready("EXITED", completed_higmem, expected) is True


@pytest.mark.parametrize("state", ["STOPPED", "FATAL", "BACKOFF", "UNKNOWN"])
def test_abnormal_terminal_state_never_starts_next_job(state, completed_higmem, expected):
    with pytest.raises(RuntimeError):
        queue.higmem_ready(state, completed_higmem, expected)


def test_higmem_missing_qa_usage_is_not_complete(completed_higmem, expected):
    path = completed_higmem / "usage.jsonl"
    write_lines(path, queue.read_jsonl(path)[:1])
    with pytest.raises(RuntimeError, match="missing metered generation"):
        queue.higmem_ready("EXITED", completed_higmem, expected)


def test_higmem_duplicate_response_is_rejected(completed_higmem, expected):
    path = completed_higmem / "usage.jsonl"
    rows = queue.read_jsonl(path)
    rows[1]["response_id"] = rows[0]["response_id"]
    write_lines(path, rows)
    with pytest.raises(RuntimeError, match="duplicate response"):
        queue.higmem_ready("EXITED", completed_higmem, expected)


def test_higmem_stale_report_totals_rejected(completed_higmem, expected):
    path = completed_higmem / "report.json"
    report = json.loads(path.read_text())
    report["server_tokens"]["qa"]["calls"] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="totals differ"):
        queue.higmem_ready("EXITED", completed_higmem, expected)


def test_recovered_connection_failure_allows_queue_but_not_exact_token_claim(completed_higmem, expected):
    path = completed_higmem / "usage.jsonl"
    write_lines(path, queue.read_jsonl(path) + [dict(sample="conv-1", question=0, phase="qa", error="Connection error.")])
    report_path = completed_higmem / "report.json"
    report = json.loads(report_path.read_text())
    report["failed_requests"] = 1
    report_path.write_text(json.dumps(report))
    original_report = report_path.read_bytes()
    accounting = completed_higmem / "separate_accounting.json"
    assert queue.higmem_ready("EXITED", completed_higmem, expected, accounting) is True
    result = json.loads(accounting.read_text())
    assert result["predictions_complete"] is True
    assert result["usage_complete"] is False
    assert result["unmetered_attempts"] == 1
    assert result["exact_total_tokens"] is None
    assert report_path.read_bytes() == original_report


def test_canonical_conversion_restores_integer_only_after_check(tmp_path, expected):
    original = expected[("conv-1", 0)]
    result = {"data": [dict(query=original["question"], answer=["2023"], output="2023", query_time_len=1.2,
               eval_metadata=dict(dataset="locomo_qa", sample_id="conv-1", question_id="conv-1_qa0",
                                  category="2", evidence=["D1:1"]))]}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result))
    row = queue.canonical_predictions([path], expected)[0]
    assert row["answer"] == 2023 and type(row["answer"]) is int
    assert row["index"] == 0 and row["seconds"] == 1.2
    result["data"][0]["answer"] = ["2024"]
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="metadata differs"):
        queue.canonical_predictions([path], expected)


def test_usage_coverage_rejects_wrong_run_or_missing_question(expected):
    rows = [dict(run_id="run", method="a_mem", sample_id=sample, question_id=str(index),
                 phase="qa", request_kind="chat_completion", success=True) for sample, index in expected]
    queue.audit_coverage(rows, expected, "run", "a_mem")
    with pytest.raises(ValueError, match="Missing metered generation"):
        queue.audit_coverage(rows[:1], expected, "run", "a_mem")
    rows[0]["run_id"] = "old-run"
    with pytest.raises(ValueError, match="unexpected run"):
        queue.audit_coverage(rows, expected, "run", "a_mem")


@pytest.mark.parametrize("error", [OSError("missing"), subprocess.TimeoutExpired("nvidia-smi", 10)])
def test_gpu_observation_failure_does_not_stop_benchmark(monkeypatch, error):
    def failing(*args, **kwargs):
        raise error
    monkeypatch.setattr(queue.subprocess, "run", failing)
    assert queue.gpu_sample()["unavailable"] is True


def test_drain_rejects_fsync_error_even_when_no_requests(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url):
            return httpx.Response(503, json={"status": "journal_error", "drained": True})
    monkeypatch.setattr(queue.httpx, "Client", FakeClient)
    with pytest.raises(RuntimeError, match="journal is unhealthy"):
        queue.drain_proxies([SimpleNamespace(poll=lambda: None)])


def test_telemetry_does_not_double_count_nested_memory_operations(tmp_path):
    timing = [dict(event="end", status="success", phase=phase, duration_s=duration)
              for phase, duration in [("initialize", 10), ("memory_add", 8), ("memory_finalize", 1), ("qa", 3)]]
    samples = [dict(monotonic_s=t, vram_mib=100, power_w=100, utilization=50) for t in (0, 36)]
    result = queue.telemetry(samples, timing, 36, tmp_path)
    assert result["construction_seconds"] == 10
    assert result["qa_seconds"] == 3
    assert result["gpu_energy_wh"] == 1
    samples.append(dict(monotonic_s=40, unavailable=True))
    assert queue.telemetry(samples, timing, 40, tmp_path)["gpu_energy_wh"] is None


def vllm_args():
    return ["vllm", "serve", queue.MODEL, "--dtype", "float16", "--revision", "revision",
            "--tool-call-parser", "qwen3_coder", "--generation-config", "vllm", "--max-model-len", "32768",
            "--language-model-only", "--enable-auto-tool-choice"]


@pytest.mark.parametrize("extra", [["--quantization=fp8"], ["--kv-cache-dtype=fp8"],
                                  ["--dtype", "bfloat16"], ["--dtype=bfloat16"]])
def test_vllm_gate_rejects_equals_quantization_and_duplicate_overrides(extra):
    with pytest.raises(RuntimeError):
        queue.validate_vllm_argv(vllm_args() + extra, "revision")


def test_vllm_gate_accepts_unambiguous_fp16_and_requires_no_server_cap():
    args = vllm_args()
    queue.validate_vllm_argv(args, "revision")
    args[args.index("--generation-config") + 1] = "auto"
    with pytest.raises(RuntimeError, match="generation-config"):
        queue.validate_vllm_argv(args, "revision")


def test_absent_runtime_phase_is_unknown_not_zero(tmp_path):
    result = queue.telemetry([], [], 1, tmp_path)
    assert result["construction_seconds"] is None
    assert result["qa_seconds"] is None
    assert result["gpu_energy_wh"] is None


def test_auxiliary_metrics_require_every_original_turn(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([dict(sample_id="sample", conversation={"session_1": [{}, {}]})]))
    plan = {"dataset": str(dataset), "run_id": "run"}
    row = dict(sample_id="sample", run_id="run", method="lightmem", status="complete",
               revision="5f0c82792b7ea14c6484e015b6a072009496b7f2", device="cpu", dtype="torch.float32",
               conversation_turns_processed=2, actual_forward_calls=3, attention_mask_tokens=12,
               padded_token_slots=24, forward_wall_seconds=0.1)
    path = tmp_path / "auxiliary.jsonl"
    write_lines(path, [row])
    assert queue.auxiliary_metrics(plan, "lightmem", tmp_path)["complete"] is True
    row["conversation_turns_processed"] = 1
    write_lines(path, [row])
    with pytest.raises(ValueError, match="coverage mismatch"):
        queue.auxiliary_metrics(plan, "lightmem", tmp_path)
