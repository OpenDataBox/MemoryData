"""Matched-budget QA evaluation and clustered statistics. No invented results."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np

from .adapter import RetentionMemAdapter
from .config import endpoint
from .datasets import load_cases
from .engine import cost_delta
from .training import append_jsonl, read_jsonl
from .metrics import locomo_score


def normalize(text):
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", str(text).casefold())).split())


def lexical_scores(answer, reference):
    a, b = normalize(answer), normalize(reference)
    left, right = Counter(a.split()), Counter(b.split())
    overlap = sum((left & right).values())
    f1 = 2*overlap/(sum(left.values())+sum(right.values())) if left or right else 1.0
    return {"exact_match": float(a == b), "token_f1": f1}


def judge_answer(judge, question, answer, benchmark, ledger):
    if benchmark == "longmemeval":
        # Reuse the repository's canonical, question-type-specific LongMemEval rubric.
        from evaluation.longmemeval.longmemeval_judge import (
            LONGMEMEVAL_SINGLE_SESSION_PROMPT, LONGMEMEVAL_TEMPORAL_PROMPT,
            LONGMEMEVAL_KNOWLEDGE_UPDATE_PROMPT, LONGMEMEVAL_PREFERENCE_PROMPT,
            LONGMEMEVAL_ABSTENTION_PROMPT,
        )
        prompts = {"temporal-reasoning": LONGMEMEVAL_TEMPORAL_PROMPT,
                   "knowledge-update": LONGMEMEVAL_KNOWLEDGE_UPDATE_PROMPT,
                   "single-session-preference": LONGMEMEVAL_PREFERENCE_PROMPT,
                   "abstention": LONGMEMEVAL_ABSTENTION_PROMPT}
        prompt = prompts.get(question.category, LONGMEMEVAL_SINGLE_SESSION_PROMPT).format(
            question=question.text, golden_answer=question.answer, generated_answer=answer)
        output = judge.chat([{"role": "user", "content": prompt}], ledger=ledger, stage="qa_judge", max_tokens=32)
        cleaned = output.strip().casefold().rstrip(".! ")
        if cleaned not in {"yes", "no"}:
            raise ValueError("Invalid LongMemEval judge response; do not score a parser failure as wrong")
        return cleaned == "yes"
    prompt = ("Judge semantic answer correctness. All required parts must be present. Respect current versus historical facts. "
              "For an unanswerable question require the model to abstain. Return JSON {correct: true|false}.\n" +
              json.dumps({"question": question.text, "reference": question.answer, "prediction": answer,
                          "category": question.category}, ensure_ascii=False))
    result = judge.chat([{"role": "user", "content": prompt}], json_mode=True, ledger=ledger, stage="qa_judge", max_tokens=256)
    if not isinstance(result.get("correct"), bool):
        raise ValueError("Invalid QA judge response")
    return result["correct"]


def run_qa(config, dataset_path, benchmark, output_dir, *, max_cases=None, max_questions=None,
           include_adversarial=False, judge_enabled=True, resume=False, repeated_rounds=1):
    if repeated_rounds not in {1, 2, 4, 8}:
        raise ValueError("Repeated-round protocol supports 1,2,4,8")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir/"predictions.jsonl"
    manifest_path = output_dir/"manifest.json"
    manifest = {"config": config, "dataset": benchmark, "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
                "include_adversarial": include_adversarial, "judge_enabled": judge_enabled,
                "max_cases": max_cases, "max_questions": max_questions, "repeated_rounds": repeated_rounds,
                "token_budget_semantics": "retrieved memory only; prompt and generation also reported",
                "scoring": "locomo_category_f1_and_custom_semantic_judge" if benchmark == "locomo" else "LongMemEval_official_rubric_configured_judge_model",
                "protocol": "official_input_no_history_truncation", "test_artifact": config.get("test_artifact", False)}
    checkpoint = config.get("policy_checkpoint")
    if checkpoint and Path(checkpoint).is_file():
        manifest["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if max_cases is not None and max_cases < 1 or max_questions is not None and max_questions < 1:
        raise ValueError("Smoke-test limits must be positive")
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Run directory already exists; use --resume or a new output path")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume manifest mismatch")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    done = {r["question_id"] for r in read_jsonl(results_path)} if results_path.exists() else set()
    judge = endpoint(config, "judge") if judge_enabled else None
    count = 0
    for case_index, case in enumerate(load_cases(dataset_path, benchmark, include_adversarial)):
        if max_cases is not None and case_index >= max_cases:
            break
        if max_questions is not None and count >= max_questions:
            break
        case_path = output_dir/"memories"/hashlib.sha256(case.id.encode()).hexdigest()[:20]
        adapter = RetentionMemAdapter({"retentionmem": config}, case_path)
        store = adapter.memory.store
        try:
            if not (case_path/"retentionmem_ready.json").exists():
                previous_session = None
                for event in case.events:
                    if previous_session is not None and event.session != previous_session:
                        adapter.memory.finalize()
                    adapter.memory.observe(event)
                    previous_session = event.session
                adapter.memory.finalize()
                # Controlled stress: same source prefix, repeated local rewrites, no new observations.
                for _ in range(1, repeated_rounds):
                    ids = [u.id for u in store.units()]
                    for begin in range(0, len(ids), adapter.memory.region_size):
                        adapter.memory.maintain(ids[begin:begin+adapter.memory.region_size])
                adapter.save()
            for question in case.questions:
                if max_questions is not None and count >= max_questions:
                    break
                count += 1
                if question.id in done:
                    continue
                before = store.costs()
                started = time.perf_counter()
                answer, items, context = adapter.answer(question.text, question.date, config.get("answer_max_tokens", 128))
                latency = time.perf_counter()-started
                correct = judge_answer(judge, question, answer, benchmark, store) if judge else None
                retrieved_ids = {s for item in items for s in item.source_ids}
                gold = set(question.source_ids)
                row = {"question_id": question.id, "group_id": case.id, "benchmark": benchmark,
                       "question": question.text, "reference": question.answer, "prediction": answer,
                       "category": question.category, "judge_correct": correct, **lexical_scores(answer, question.answer),
                       "locomo_f1": locomo_score(answer, question.answer, question.category) if benchmark == "locomo" else None,
                       "retrieved_tokens": adapter.memory.tokenizer.count(context), "query_budget": adapter.query_budget,
                       "retrieved": [asdict(item) for item in items], "latency_seconds": latency,
                       "query_cost": cost_delta(store.costs(), before),
                       "source_reference_recall": len(gold & retrieved_ids)/len(gold) if gold else None,
                       "reference_recall_warning": "linked source references do not establish answer-span retention",
                       "stats": adapter.memory.stats(), "repeated_rounds": repeated_rounds}
                append_jsonl(results_path, row)
                done.add(question.id)
            (case_path/"run_stats.json").write_text(json.dumps(adapter.memory.stats(), indent=2), encoding="utf-8")
        finally:
            store.close()
    report = summarize(list(read_jsonl(results_path))) if results_path.exists() else {"questions": 0}
    (output_dir/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def clustered_interval(rows, metric, seed=42, samples=2000):
    groups = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            groups[row["group_id"]].append(float(row[metric]))
    if not groups:
        return None
    values = list(groups.values())
    mean = float(np.mean([v for group in values for v in group]))
    if len(values) < 2:
        return {"mean": mean, "ci95": None, "clusters": len(values)}
    rng = np.random.default_rng(seed)
    sums = np.asarray([sum(g) for g in values])
    counts = np.asarray([len(g) for g in values])
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrap = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return {"mean": mean, "ci95": np.quantile(bootstrap, [.025, .975]).tolist(), "clusters": len(values)}


def paired_comparison(left, right, metric="judge_correct"):
    a, b = {r["question_id"]: r for r in left}, {r["question_id"]: r for r in right}
    if a.keys() != b.keys():
        raise ValueError("Paired comparison requires exactly the same question IDs")
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError("Duplicate question IDs would bias a paired comparison")
    differences = []
    for key in sorted(a):
        if a[key]["group_id"] != b[key]["group_id"]:
            raise ValueError("Paired clustering mismatch")
        if a[key].get(metric) is None or b[key].get(metric) is None:
            raise ValueError("Cannot compare missing judge labels")
        differences.append({"group_id": a[key]["group_id"], "delta": float(b[key][metric])-float(a[key][metric])})
    return clustered_interval(differences, "delta")


def summarize(rows):
    categories = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
    return {"questions": len(rows), "groups": len({r["group_id"] for r in rows}),
            "cost_accounting": cost_report(rows),
            "judge_accuracy": clustered_interval(rows, "judge_correct"),
            "token_f1": clustered_interval(rows, "token_f1"),
            "locomo_f1": clustered_interval(rows, "locomo_f1"),
            "by_category": {k: {"questions": len(v), "judge_accuracy": clustered_interval(v, "judge_correct"),
                                "token_f1": clustered_interval(v, "token_f1"), "locomo_f1": clustered_interval(v, "locomo_f1")} for k, v in categories.items()},
            "mean_retrieved_tokens": float(np.mean([r["retrieved_tokens"] for r in rows])) if rows else None,
            "latency_p50": float(np.median([r["latency_seconds"] for r in rows])) if rows else None,
            "latency_p95": float(np.quantile([r["latency_seconds"] for r in rows], .95)) if rows else None}


def cost_report(rows):
    """One cumulative snapshot per memory, avoiding double-counting earlier questions."""
    latest = {r["group_id"]: r.get("stats", {}) for r in rows}
    totals = {"llm_input_tokens": 0, "llm_output_tokens": 0, "embedding_input_tokens": 0,
              "judge_input_tokens": 0, "judge_output_tokens": 0, "missing_usage_calls": 0,
              "raw_bytes": 0, "active_bytes": 0, "database_bytes": 0}
    for snapshot in latest.values():
        for key in ("raw_bytes", "active_bytes", "database_bytes"):
            totals[key] += snapshot.get("storage", {}).get(key, 0)
        for stage, cost in snapshot.get("costs", {}).get("by_stage", {}).items():
            totals["missing_usage_calls"] += cost.get("missing_usage_calls", 0)
            if stage == "qa_judge":
                totals["judge_input_tokens"] += cost["input_tokens"]
                totals["judge_output_tokens"] += cost["output_tokens"]
            elif "embedding" in stage:
                totals["embedding_input_tokens"] += cost["input_tokens"]
            else:
                totals["llm_input_tokens"] += cost["input_tokens"]
                totals["llm_output_tokens"] += cost["output_tokens"]
    totals["amortized_llm_tokens_per_question"] = ((totals["llm_input_tokens"]+totals["llm_output_tokens"])/len(rows)
                                                       if rows and not totals["missing_usage_calls"] else None)
    totals.update(usd=None, policy_training_cost_included=False,
                  note="Write/maintenance/read included; judge separate. Add collection ledger and training report; token volume is not monetary cost.")
    return totals


def plot_runs(run_directories, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = defaultdict(list)
    for directory in run_directories:
        directory = Path(directory)
        manifest = json.loads((directory/"manifest.json").read_text())
        if manifest.get("test_artifact"):
            raise ValueError("Demo artifacts cannot be plotted as scientific runs")
        rows = list(read_jsonl(directory/"predictions.jsonl"))
        report = summarize(rows)
        if report["judge_accuracy"] is None:
            raise ValueError("Accuracy-cost figures require actual judge labels")
        cfg = manifest["config"]
        name = f"{cfg['mode']} / {cfg.get('query_budget',4096)} / {cfg.get('compressor','structured')} / {cfg.get('reader',cfg['llm'])['model']} / {cfg.get('policy_checkpoint','control')}"
        cost = report["cost_accounting"]["amortized_llm_tokens_per_question"]
        if cost is None: raise ValueError("Cost curve requires complete API usage records")
        series[name].append((manifest["repeated_rounds"], report["judge_accuracy"]["mean"], cost))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, points in series.items():
        rounds, accuracy, cost = zip(*sorted(points))
        axes[0].plot(rounds, accuracy, marker="o", label=name)
        axes[1].plot(cost, accuracy, marker="o", label=name)
    axes[0].set(xlabel="Consolidation rounds", ylabel="Judge accuracy", xticks=[1, 2, 4, 8])
    axes[1].set(xlabel="LLM tokens/query, including maintenance", ylabel="Judge accuracy")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
