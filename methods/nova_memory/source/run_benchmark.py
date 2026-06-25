"""Nova Memory — Standalone MemoryAgentBench runner.

Bypasses MemoryData's main.py (which requires torch) and runs the benchmark
directly using HF datasets + NovaMemoryAgent. Outputs a JSON in
MemoryData-compatible format so it can be compared with other agents.

Two modes:
  - mock   : use _mock_bench.py (no network) — for CI / local sanity
  - hf     : download ai-hyz/MemoryAgentBench from HuggingFace

Metrics (more informative for extractive memory agents):
  - recall_at_k    : % of QA where any gold answer appears in any of top-k retrieved chunks
  - substring_em   : % of QA where any gold answer appears as substring in prediction
  - llm_judge_f1   : if LLM available, do squad-style F1 over the LLM's answer

Usage:
    python run_benchmark.py --mock
    python run_benchmark.py --benchmark eventqa --max-samples 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


from nova_agent import NovaMemoryAgent  # noqa: E402
from _mock_bench import get_mock_samples  # noqa: E402


# ============================================================================
# Dataset loaders
# ============================================================================

BENCHMARKS = {
    "eventqa": {
        "sub_dataset": "eventqa_full",
        "split": "Accurate_Retrieval",
    },
    "longmemeval": {
        "sub_dataset": "longmemeval_s*",
        "split": "Accurate_Retrieval",
    },
}


def load_memoryagentbench(benchmark: str, max_samples: int = 5):
    from datasets import load_dataset
    cfg = BENCHMARKS[benchmark]
    print(f"[load] ai-hyz/MemoryAgentBench  split={cfg['split']}  sub={cfg['sub_dataset']}")
    ds = load_dataset("ai-hyz/MemoryAgentBench", split=cfg["split"], revision="main")
    print(f"[load] total in split: {len(ds)}")
    filtered = ds.filter(lambda s: s.get("metadata", {}).get("source", "") == cfg["sub_dataset"])
    print(f"[load] filtered to {len(filtered)} samples")
    if max_samples and len(filtered) > max_samples:
        filtered = filtered.select(range(max_samples))
    return list(filtered)


# ============================================================================
# Metrics
# ============================================================================

def _normalize(s: str) -> str:
    return " ".join(str(s).lower().split())


def contains_any_gold(text: str, gold_answers: List[str]) -> bool:
    """True if any gold answer appears as substring in text (case-insensitive)."""
    t = _normalize(text)
    for g in gold_answers:
        if _normalize(g) and _normalize(g) in t:
            return True
    return False


def compute_substring_em(pred: str, gold_answers: List[str]) -> float:
    return float(contains_any_gold(pred, gold_answers))


def compute_token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = {}
    for t in set(pred_toks):
        if t in gold_toks:
            common[t] = min(pred_toks.count(t), gold_toks.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ============================================================================
# Main runner
# ============================================================================

def run_benchmark(samples: List[Dict], source: str, output_path: str, llm_model: str,
                  max_qa_per_sample: int = 0, agent: NovaMemoryAgent = None,
                  reinit_agent_per_sample: bool = False):
    print(f"=== Nova Memory benchmark: {source} ===")
    print(f"  samples: {len(samples)}")
    print(f"  llm_model: {llm_model}")
    print()

    if agent is None:
        api_key = os.environ.get("NOVA_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("NOVA_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
        print(f"  base_url: {base_url or '(default)'}")
        print(f"  api_key: {'set' if api_key else 'NONE (top-chunk fallback)'}")
        print()
        agent = NovaMemoryAgent(
            model=llm_model,
            retrieve_num=5,
            api_key=api_key if api_key else None,
            base_url=base_url if base_url else None,
            agent_save_to_folder=str(HERE / "_bench_state"),
        )

    has_llm = bool(agent.api_key and agent.base_url and agent._get_client() is not None)

    results = []
    metrics = {
        "recall_at_k": [],        # 召回:gold 在 top-k chunks 里
        "substring_em": [],       # 答案里包含 gold
        "token_f1": [],           # token F1 (LLM)
        "first_chunk_hit": [],    # 第一个召回 chunk 是否包含 gold
    }
    t_start = time.time()

    for s_idx, sample in enumerate(samples):
        ctx_id = sample.get("metadata", {}).get("id", s_idx)
        chunks = sample.get("context_chunks", [])
        qa_list = sample.get("qa_list", [])
        if not chunks:
            print(f"[skip] sample {s_idx} no context_chunks")
            continue
        if max_qa_per_sample and len(qa_list) > max_qa_per_sample:
            qa_list = qa_list[:max_qa_per_sample]

        if reinit_agent_per_sample:
            agent = NovaMemoryAgent(
                model=llm_model,
                retrieve_num=5,
                api_key=os.environ.get("NOVA_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "") or None,
                base_url=os.environ.get("NOVA_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "") or None,
                agent_save_to_folder=str(HERE / f"_bench_state_{s_idx}"),
            )

        print(f"\n--- sample {s_idx} (id={ctx_id}, chunks={len(chunks)}, qa={len(qa_list)}) ---")

        # 1) Memorize
        t_memo = time.time()
        for c in chunks:
            if isinstance(c, str):
                agent.memorize_chunk(c)
            elif isinstance(c, dict):
                agent.memorize_chunk(c.get("text", "") or c.get("content", ""))
            else:
                agent.memorize_chunk(str(c))
        memo_time = time.time() - t_memo
        print(f"  memorize: {len(chunks)} chunks in {memo_time:.1f}s")

        # 2) Answer each QA
        for q_idx, qa in enumerate(qa_list):
            question = qa.get("question", "")
            gold_answers = qa.get("answers", [])
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
            if not question or not gold_answers:
                continue
            gold = gold_answers[0]

            # 2a) Pure recall
            retrieved = agent.recall_chunks(question)
            recall_at_k = float(any(contains_any_gold(c, gold_answers) for c in retrieved))
            first_chunk_hit = float(
                bool(retrieved) and contains_any_gold(retrieved[0], gold_answers)
            )

            # 2b) LLM answer
            t_q = time.time()
            pred = agent.send_message(question, memorizing=False)
            q_time = time.time() - t_q

            sub_em = compute_substring_em(pred, gold_answers)
            tok_f1 = max(compute_token_f1(pred, g) for g in gold_answers)

            metrics["recall_at_k"].append(recall_at_k)
            metrics["first_chunk_hit"].append(first_chunk_hit)
            metrics["substring_em"].append(sub_em)
            metrics["token_f1"].append(tok_f1)

            results.append({
                "sample_id": ctx_id,
                "qa_index": q_idx,
                "question": question,
                "gold_answers": gold_answers,
                "prediction": pred,
                "retrieved_top_k": retrieved,
                "recall_at_k": recall_at_k,
                "first_chunk_hit": first_chunk_hit,
                "substring_em": sub_em,
                "token_f1": tok_f1,
                "query_time_s": round(q_time, 3),
            })

            label = "✓" if recall_at_k else "✗"
            print(f"  {label} q{q_idx}: recall@k={recall_at_k:.0f} first={first_chunk_hit:.0f} "
                  f"sub_em={sub_em:.0f} f1={tok_f1:.2f}  Q={question[:40]}")
            print(f"      gold: {gold[:50]}")
            print(f"      pred: {str(pred)[:50]}")

    total_time = time.time() - t_start

    n = max(1, len(metrics["recall_at_k"]))
    summary = {
        "benchmark": source,
        "model": llm_model,
        "n_samples": len(samples),
        "n_qa": n,
        "metrics": {
            "recall_at_k": round(100 * sum(metrics["recall_at_k"]) / n, 2),
            "first_chunk_hit": round(100 * sum(metrics["first_chunk_hit"]) / n, 2),
            "substring_em": round(100 * sum(metrics["substring_em"]) / n, 2),
            "token_f1": round(100 * sum(metrics["token_f1"]) / n, 2),
        },
        "total_time_s": round(total_time, 1),
        "note": "recall@k=1 → 评估纯检索质量;substring_em/token_f1 → LLM 答案质量(没 LLM 时=recall)",
    }
    print(f"\n=== {source} summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    output = {"summary": summary, "results": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="use built-in mock dataset (no network)")
    parser.add_argument("--benchmark", default="eventqa", choices=list(BENCHMARKS))
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-qa-per-sample", type=int, default=0)
    parser.add_argument("--output", default=str(HERE / "_bench_results" / "eventqa_nova.json"))
    parser.add_argument("--llm-model", default=os.environ.get("NOVA_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--reinit-per-sample", action="store_true")
    args = parser.parse_args()

    if args.mock:
        samples = get_mock_samples()
        source = "mock"
    else:
        try:
            samples = load_memoryagentbench(args.benchmark, max_samples=args.max_samples)
            source = f"hf:{args.benchmark}"
        except Exception as e:
            print(f"ERROR loading HF: {e}")
            print("Use --mock for offline testing")
            sys.exit(1)

    try:
        run_benchmark(
            samples=samples,
            source=source,
            output_path=args.output,
            llm_model=args.llm_model,
            max_qa_per_sample=args.max_qa_per_sample,
            reinit_agent_per_sample=args.reinit_per_sample,
        )
    except KeyboardInterrupt:
        print("\n[abort]")
        sys.exit(1)


if __name__ == "__main__":
    main()