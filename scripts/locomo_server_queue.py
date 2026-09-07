"""Run the six controlled LoCoMo comparisons after the live HiGMem job exits.

Run this foreground process under Supervisor, with stopasgroup/killasgroup set.
It never shuts down the Vast instance or copies results off the server.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_locomo_comparison import (
    build_report, expected_questions, read_jsonl, validate_predictions, write_report,
)

METHODS = ("a_mem", "mem0", "langmem", "simplemem", "lightmem", "e_mem")
MODEL = "Qwen/Qwen3.5-9B"
LLM_ORIGIN = "http://127.0.0.1:18082"
EMBED_ORIGIN = "http://127.0.0.1:18083"


def validate_plan(plan):
    if plan.get("schema_version") != 1 or plan.get("model") != MODEL:
        raise ValueError("Unsupported comparison plan")
    if [item["method"] for item in plan["methods"]] != list(METHODS):
        raise ValueError("Plan must include all six methods in the agreed order")
    for key in ("dataset", "scorer", "client_python", "server_python"):
        if not Path(plan[key]).is_file():
            raise ValueError(f"Missing {key}")
    expected_questions(plan["dataset"], 1540)
    hashes = plan.get("file_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("A pinned source/config manifest is required")
    required = {str(Path(plan[key]).resolve()) for key in ("dataset", "scorer")}
    for item in plan["methods"]:
        required.update(str(Path(item[key]).resolve()) for key in ("agent_config", "dataset_config"))
    if not required.issubset(hashes):
        raise ValueError("Dataset, scorer and all configs must be pinned")
    for filename, digest in hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned file changed: {filename}")
    if not Path(plan["output"]).is_absolute():
        raise ValueError("Output must be an absolute dedicated run directory")


def higmem_ready(state, directory, expected, accounting_output=None):
    """A live writer always takes precedence over an old report file."""
    if state in {"RUNNING", "STARTING", "STOPPING"}:
        return False
    if state != "EXITED":
        raise RuntimeError(f"HiGMem is not a normally exited job: {state}")
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if not (report.get("complete") is True and report.get("expected") == len(expected)
            and report.get("evaluated") == len(expected)):
        raise RuntimeError("HiGMem did not complete the exact selected QA set")
    if report.get("truncated_requests") != 0:
        raise RuntimeError("HiGMem integrity gate failed: truncated_requests")
    rows = read_jsonl(directory / "predictions.jsonl")
    valid, issues, missing = validate_predictions(rows, expected)
    if issues or missing or any(not row["prediction"].strip() for row in valid):
        raise RuntimeError("HiGMem predictions failed the original-data audit")
    usage = read_jsonl(directory / "usage.jsonl")
    if not usage:
        raise RuntimeError("HiGMem usage journal is empty")
    samples = {sample for sample, _ in expected}
    seen_qa, response_ids, totals = set(), set(), {}
    failed, missing_usage = 0, 0
    for row in usage:
        counts = row.get("usage")
        if row.get("sample") not in samples or row.get("phase") not in {"construction", "qa"}:
            raise RuntimeError("HiGMem usage is outside the selected samples/phases")
        if row["phase"] == "qa" and (type(row.get("question")) is not int
                or (row["sample"], row["question"]) not in expected):
            raise RuntimeError("HiGMem request has an invalid original QA index")
        if "error" in row:
            failed += 1
            continue
        if counts is None:
            missing_usage += 1
            continue
        if row.get("finish_reason") != "stop" or not isinstance(counts, dict):
            raise RuntimeError("HiGMem has an incomplete model request")
        if any(type(counts.get(key)) is not int or counts[key] < 0 for key in
               ("prompt_tokens", "completion_tokens", "total_tokens")):
            raise RuntimeError("HiGMem has invalid token usage")
        if counts["total_tokens"] != counts["prompt_tokens"] + counts["completion_tokens"]:
            raise RuntimeError("HiGMem token totals do not reconcile")
        response_id = row.get("response_id")
        if not isinstance(response_id, str) or not response_id or response_id in response_ids:
            raise RuntimeError("HiGMem has a missing or duplicate response ID")
        response_ids.add(response_id)
        if row["phase"] == "qa":
            seen_qa.add((row["sample"], row["question"]))
        total = totals.setdefault(row["phase"], dict(calls=0, prompt_tokens=0, completion_tokens=0, total_tokens=0))
        total["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] += counts[key]
    if seen_qa != set(expected):
        raise RuntimeError("HiGMem is missing metered generation for some original questions")
    if report.get("server_tokens") != totals:
        raise RuntimeError("HiGMem report token totals differ from its journal")
    if report.get("failed_requests") != failed or report.get("missing_usage") != missing_usage:
        raise RuntimeError("HiGMem report request counts differ from its journal")
    if accounting_output is not None:
        write_report(accounting_output, {
            "predictions_complete": True, "evaluated": len(expected), "official_f1": report.get("official_f1"),
            "usage_complete": failed + missing_usage == 0,
            "failed_requests": failed, "missing_usage": missing_usage,
            "unmetered_attempts": failed + missing_usage, "reported_tokens_by_phase": totals,
            "exact_total_tokens": sum(row["total_tokens"] for row in totals.values()) if not (failed + missing_usage) else None,
            "note": "Recovered requests do not invalidate completed answers. Unmetered attempts remain unknown, "
                    "never zero; reported token totals are not claimed as exact when usage_complete is false. "
                    "HiGMem embedding tokens and full-run GPU energy were not instrumented from its start.",
        })
    return True


def canonical_predictions(result_files, expected):
    """Validate stored metadata before restoring only the loader's scalar types."""
    if len(result_files) != 1:
        raise ValueError("Expected exactly one benchmark result file")
    data = json.loads(Path(result_files[0]).read_text(encoding="utf-8"))["data"]
    id_map = {}
    for (sample, index), qa in expected.items():
        question_id = str(qa.get("question_id") or f"{sample}_qa{index}")
        id_map[(sample, question_id)] = (index, qa)
    rows = []
    for record in data:
        if record.get("status") == "failed":
            raise ValueError("A benchmark query failed")
        metadata = record.get("eval_metadata") or {}
        key = (metadata.get("sample_id"), metadata.get("question_id"))
        if key not in id_map or metadata.get("dataset") != "locomo_qa":
            raise ValueError("Unknown original QA identity")
        index, qa = id_map[key]
        answer = record.get("answer")
        if isinstance(answer, list) and len(answer) == 1 and not isinstance(qa["answer"], list):
            answer = answer[0]
        if (record.get("query") != qa["question"] or str(answer) != str(qa["answer"])
                or str(metadata.get("category")) != str(qa["category"])
                or metadata.get("evidence") != qa["evidence"]
                or not isinstance(record.get("output"), str)):
            raise ValueError("Result metadata differs from the original QA")
        rows.append({"sample": key[0], "index": index,
                     **{name: qa[name] for name in ("question", "answer", "category", "evidence")},
                     "prediction": record["output"], "seconds": record.get("query_time_len")})
    return rows


def audit_coverage(rows, expected, run_id, method):
    seen = set()
    for row in rows:
        if row.get("run_id") != run_id or row.get("method") != method:
            raise ValueError("Usage journal contains an unexpected run or method")
        if row.get("sample_id") not in {sample for sample, _ in expected}:
            raise ValueError("Model request has no valid sample attribution")
        if row.get("phase") == "qa" and row.get("request_kind") == "chat_completion":
            try:
                key = (row["sample_id"], int(row["question_id"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("QA request has no original question index") from exc
            if key not in expected:
                raise ValueError("QA request is outside the evaluation set")
            if row.get("success") is True:
                seen.add(key)
    if seen != set(expected):
        raise ValueError(f"Missing metered generation for {len(set(expected) - seen)} questions")


def supervisor_state(name):
    result = subprocess.run(["supervisorctl", "status", name], capture_output=True, text=True, check=False)
    fields = result.stdout.split()
    if len(fields) < 2 or fields[0] != name:
        raise RuntimeError(f"Cannot observe Supervisor service {name}")
    return fields[1]


def wait_health(origin, process, timeout=180):
    deadline = time.monotonic() + timeout
    with httpx.Client(trust_env=False, timeout=3) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Background service exited before it was ready")
            try:
                response = client.get(origin + "/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
    raise TimeoutError(f"Service readiness timeout: {origin}")


@contextmanager
def service(command, log_path, env=None):
    with Path(log_path).open("ab") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def drain_proxies(processes):
    deadline = time.monotonic() + 120
    with httpx.Client(trust_env=False, timeout=5) as client:
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("Meter exited before its journal was drained")
            health = [client.get(origin + "/health") for origin in (LLM_ORIGIN, EMBED_ORIGIN)]
            if any(response.status_code != 200 or response.json().get("status") != "ok" for response in health):
                raise RuntimeError("A usage journal is unhealthy; completeness cannot be established")
            if all(response.json().get("drained") is True for response in health):
                return
            time.sleep(0.5)
    raise TimeoutError("Outstanding metered requests did not drain")


def gpu_sample():
    observed = time.monotonic()
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,power.draw,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=10, check=False)
        lines = result.stdout.strip().splitlines()
        if result.returncode != 0 or len(lines) != 1:
            raise ValueError("Expected one GPU")
        values = [float(value.strip()) for value in lines[0].split(",")]
        if len(values) != 3 or any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Invalid GPU metrics")
        return {"monotonic_s": observed, "vram_mib": values[0], "power_w": values[1], "utilization": values[2]}
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return {"monotonic_s": observed, "unavailable": True}


def wait_vllm(plan):
    deadline = time.monotonic() + 600
    with httpx.Client(trust_env=False, timeout=5) as client:
        while time.monotonic() < deadline:
            state = supervisor_state("comparison-vllm")
            if state not in {"STARTING", "RUNNING"}:
                raise RuntimeError(f"Comparison vLLM failed to start: {state}")
            try:
                response = client.get("http://127.0.0.1:18080/v1/models")
                if response.status_code == 200 and MODEL in {row["id"] for row in response.json()["data"]}:
                    result = subprocess.run(["supervisorctl", "pid", "comparison-vllm"],
                                            capture_output=True, text=True, check=True)
                    argv = Path(f"/proc/{int(result.stdout.strip())}/cmdline").read_bytes().decode().split("\0")
                    validate_vllm_argv(argv, plan["model_revision"])
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
    raise TimeoutError("Comparison vLLM did not become ready")


def validate_vllm_argv(argv, revision):
    """Normalize both CLI option spellings and reject ambiguous overrides."""
    options = {}
    for index, value in enumerate(argv):
        if not value.startswith("--"):
            continue
        key, separator, assigned = value.partition("=")
        if key in options:
            raise RuntimeError(f"Duplicate vLLM option: {key}")
        options[key] = assigned if separator else (
            argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith("--") else None)
    for key, expected in (("--dtype", "float16"), ("--revision", revision),
                          ("--tool-call-parser", "qwen3_coder"), ("--generation-config", "vllm"),
                          ("--max-model-len", "32768")):
        if options.get(key) != expected:
            raise RuntimeError(f"vLLM precision/revision/tool configuration mismatch: {key}")
    if "--quantization" in options or options.get("--kv-cache-dtype", "auto") != "auto":
        raise RuntimeError("Comparison weights and KV cache must not be quantized")
    if "--enable-auto-tool-choice" not in options or "--language-model-only" not in options:
        raise RuntimeError("Required comparison vLLM flags are missing")
    if "serve" not in argv or argv[argv.index("serve") + 1] != MODEL:
        raise RuntimeError("Comparison must serve the fixed Qwen model")


def telemetry(samples, timing, wall_seconds, output):
    good = [sample for sample in samples if not sample.get("unavailable")]
    energy = None
    if len(good) >= 2 and len(good) == len(samples):
        energy = sum((b["monotonic_s"] - a["monotonic_s"]) * (a["power_w"] + b["power_w"]) / 7200
                     for a, b in zip(good, good[1:]))
    ends = [row for row in timing if row.get("event") == "end" and row.get("status") == "success"]
    construction = [row["duration_s"] for row in ends if row["phase"] == "initialize"]
    qa_times = [row["duration_s"] for row in ends if row["phase"] == "qa"]
    return {"wall_seconds": wall_seconds,
            "construction_seconds": sum(construction) if construction else None,
            "qa_seconds": sum(qa_times) if qa_times else None,
            "peak_vram_mib": max((row["vram_mib"] for row in good), default=None),
            "gpu_energy_wh": energy,
            "mean_gpu_utilization_percent": sum(row["utilization"] for row in good) / len(good) if good else None,
            "storage_bytes": sum(path.stat().st_size for path in Path(output).rglob("*") if path.is_file()),
            "gpu_measurement_note": "5-second device samples; energy is trapezoidal integration, not a hardware energy counter.",
            "gpu_samples": len(samples), "missing_gpu_samples": len(samples) - len(good)}


def auxiliary_metrics(plan, method, output):
    """Local compressor tokens are not API/LLM tokens and remain separate."""
    if method != "lightmem":
        return {"applicable": False, "complete": True}
    rows = read_jsonl(Path(output) / "auxiliary.jsonl")
    dataset = json.loads(Path(plan["dataset"]).read_text())
    expected_turns = {sample["sample_id"]: sum(len(value) for key, value in sample["conversation"].items()
                     if key.startswith("session_") and isinstance(value, list)) for sample in dataset}
    seen = set()
    for row in rows:
        sample = row.get("sample_id")
        if sample not in expected_turns or sample in seen:
            raise ValueError("LightMem auxiliary journal has a missing/duplicate/unknown sample")
        seen.add(sample)
        if (row.get("run_id") != plan["run_id"] or row.get("method") != method or row.get("status") != "complete"
                or row.get("revision") != "5f0c82792b7ea14c6484e015b6a072009496b7f2"
                or row.get("device") != "cpu" or row.get("dtype") != "torch.float32"
                or row.get("conversation_turns_processed") != expected_turns[sample]):
            raise ValueError("LightMem auxiliary runtime or input-coverage mismatch")
        for key in ("actual_forward_calls", "attention_mask_tokens", "padded_token_slots"):
            if type(row.get(key)) is not int or row[key] <= 0:
                raise ValueError("LightMem auxiliary forward/token measurement is missing")
        if (type(row.get("forward_wall_seconds")) not in (float, int)
                or not math.isfinite(row["forward_wall_seconds"]) or row["forward_wall_seconds"] < 0):
            raise ValueError("LightMem auxiliary forward time is invalid")
    if seen != set(expected_turns):
        raise ValueError("LightMem auxiliary metrics are missing some conversations")
    return {"applicable": True, "complete": True, "samples": len(rows),
            "model_revision": "5f0c82792b7ea14c6484e015b6a072009496b7f2",
            "device": "cpu", "dtype": "torch.float32",
            **{key: sum(row[key] for row in rows) for key in
               ("actual_forward_calls", "attention_mask_tokens", "padded_token_slots", "forward_wall_seconds")},
            "note": "Auxiliary LLMLingua compression/topic tokens, not Qwen generation or MiniLM embedding tokens."}


def run_method(plan, item, expected):
    method = item["method"]
    output = Path(plan["output"]) / method
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite or ambiguously resume an existing method: {output}")
    output.mkdir()
    usage_path = output / "usage.jsonl"
    timing_path = output / "timing.jsonl"
    env = os.environ.copy()
    env.update(OPENAI_API_KEY="local-comparison", OPENAI_BASE_URL=LLM_ORIGIN + "/v1",
               EMBEDDING_BASE_URL=EMBED_ORIGIN + "/v1", METER_RUN_ID=plan["run_id"], METER_METHOD=method,
               METER_LLM_PROXY_ORIGIN=LLM_ORIGIN, METER_EMBEDDING_PROXY_ORIGIN=EMBED_ORIGIN,
               METER_TIMING_JOURNAL=str(timing_path), LIGHTMEM_MODEL=MODEL,
               LIGHTMEM_BASE_URL=LLM_ORIGIN + "/v1", LIGHTMEM_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
               LIGHTMEM_EMBEDDING_DIMENSION="384", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false", HF_HOME=plan["hf_home"], BASELINE_STRICT_COMPARISON="1",
               METER_AUXILIARY_JOURNAL=str(output / "auxiliary.jsonl"))
    proxy = [plan["server_python"], str(ROOT / "scripts/metered_openai_proxy.py"),
             "--journal", str(usage_path), "--run-id", plan["run_id"], "--method", method]
    command = [plan["client_python"], str(ROOT / "main.py"), "--agent_config", item["agent_config"],
               "--dataset_config", item["dataset_config"], "--artifact_root", str(output / "artifacts"),
               "--max_test_queries_ablation", "0", "--fail-on-query-error"]
    # Separate proxy journals avoid cross-process append interleaving.
    llm_journal, embed_journal = output / "llm_usage.jsonl", output / "embedding_usage.jsonl"
    llm_command = proxy.copy()
    llm_command[llm_command.index("--journal") + 1] = str(llm_journal)
    embed_command = proxy.copy()
    embed_command[embed_command.index("--journal") + 1] = str(embed_journal)
    with service(llm_command + ["--upstream-base-url", "http://127.0.0.1:18080/v1", "--port", "18082", "--comparison-policy"], output / "llm_proxy.log") as llm, \
         service(embed_command + ["--upstream-base-url", "http://127.0.0.1:18081/v1", "--port", "18083"], output / "embedding_proxy.log") as embed:
        wait_health(LLM_ORIGIN, llm)
        wait_health(EMBED_ORIGIN, embed)
        started = time.monotonic()
        samples = [gpu_sample()]
        with service(command, output / "benchmark.log", env) as process:
            while process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                samples.append(gpu_sample())
            exit_code = process.returncode
        drain_proxies([llm, embed])
        wall = time.monotonic() - started
    usage = read_jsonl(llm_journal) + read_jsonl(embed_journal)
    with usage_path.open("x", encoding="utf-8") as stream:
        for row in usage:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(output / "gpu_samples.json", samples)
    write_report(output / "telemetry.json", telemetry(samples, read_jsonl(timing_path), wall, output))
    if exit_code != 0:
        raise RuntimeError(f"{method} benchmark exited with code {exit_code}")
    rows = canonical_predictions(list((output / "artifacts").rglob("*_results.json")), expected)
    predictions = output / "predictions.jsonl"
    with predictions.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = build_report(plan["dataset"], predictions, plan["scorer"], usage_path,
                          output / "telemetry.json", plan.get("hourly_rate_usd"))
    report["method"] = method
    report["run_id"] = plan["run_id"]
    report["quality_complete"] = report["usage"]["truncated_requests"] == 0 and report["empty_predictions"] == 0
    try:
        report["auxiliary"] = auxiliary_metrics(plan, method, output)
    except (ValueError, OSError) as exc:
        report["auxiliary"] = {"applicable": True, "complete": False, "error": str(exc)}
    try:
        audit_coverage(usage, expected, plan["run_id"], method)
        report["coverage_complete"] = True
    except ValueError as exc:
        report["coverage_complete"] = False
        report["coverage_error"] = str(exc)
    report["complete"] = (report["complete"] and report["quality_complete"] and report["coverage_complete"]
                          and report["auxiliary"]["complete"])
    write_report(output / "report.json", report)
    if not report["complete"]:
        raise RuntimeError(f"{method} failed final completeness or quality gates")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.check_only:
        print(json.dumps({"plan_valid": True, "methods": list(METHODS), "expected_per_method": 1540}))
        return 0
    import fcntl
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "queue_status.json").exists():
            raise RuntimeError("An existing queue run requires explicit review before resuming")
        expected = expected_questions(plan["dataset"], 1540)
        write_report(output / "queue_status.json", {"state": "waiting_for_higmem", "complete": False})
        while not higmem_ready(supervisor_state("higmem-eval"), plan["higmem_run"], expected,
                               output / "higmem_accounting.json"):
            time.sleep(30)
        validate_plan(plan)
        # HiGMem has exited and passed data/token gates before its server is changed.
        subprocess.run(["supervisorctl", "stop", "higmem-vllm"], check=True)
        subprocess.run(["supervisorctl", "start", "comparison-vllm"], check=True)
        wait_vllm(plan)
        reports, failures = [], []
        with service([plan["server_python"], str(ROOT / "scripts/serve_minilm_embeddings.py"),
                      "--model-path", plan["embedding_path"]], output / "embedding_server.log") as encoder:
            wait_health("http://127.0.0.1:18081", encoder)
            for item in plan["methods"]:
                write_report(output / "queue_status.json", {"state": "running", "method": item["method"], "complete": False})
                try:
                    validate_plan(plan)
                    reports.append(run_method(plan, item, expected))
                except Exception as exc:
                    failures.append({"method": item["method"], "error_type": type(exc).__name__, "error": str(exc)})
                    write_report(output / "failures.json", failures)
        complete = not failures and len(reports) == 6
        write_report(output / "queue_status.json", {"state": "complete" if complete else "failed", "complete": complete,
                     "higmem_verified": True, "methods_complete": [item["method"] for item in reports], "failures": failures})
        return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
