"""Local-only diagnostic: future-aware memory selection, without weight training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def grade(answer: str, branch: dict[str, Any]) -> dict[str, Any]:
    """Exact field aliases; no substring matching or model judge."""
    text = answer.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    valid = isinstance(parsed, dict)
    correct = {
        key: bool(valid and isinstance(parsed.get(key), str) and
                  normalize(parsed[key]) in {normalize(a) for a in aliases})
        for key, aliases in branch["aliases"].items()
    }
    return {"valid_json": valid, "field_correct": correct,
            "score": sum(correct.values()) / len(correct)}


def select_candidates(scores: list[list[float]], source: list[float]) -> dict[str, int]:
    """Never receives heldout results. Lowest candidate index breaks all ties."""
    if not scores or not source or any(len(row) != len(source) for row in scores):
        raise ValueError("Every candidate needs the same nonempty selection probes")
    losses = [[max(0.0, full - value) for full, value in zip(source, row)] for row in scores]
    return {"plain": 0,
            "mean": min(range(len(losses)), key=lambda i: sum(losses[i]) / len(source)),
            "worst": min(range(len(losses)), key=lambda i: max(losses[i]))}


def writer_messages(prefix: str, budget: int) -> list[dict[str, str]]:
    # Only the history is accepted, not branches, questions, or gold answers.
    return [{"role": "system", "content":
             "Compress the conversation into a factual memory for later use. "
             f"Use at most {budget} tokens. Output only the memory, without preamble. "
             "Do not invent facts."}, {"role": "user", "content": prefix}]


def reader_messages(memory: str, branch: dict[str, Any]) -> list[dict[str, str]]:
    # Gold values and aliases are deliberately not included in the request.
    fields = ", ".join(branch["gold"])
    return [{"role": "system", "content":
             "Answer using only the memory and new message. "
             f"Return one JSON object with string fields: {fields}. "
             "Use short exact names/values. Use an empty string for unknown fields. Do not explain."},
            {"role": "user", "content":
             f"Memory:\n{memory or '(none)'}\n\nNew message:\n{branch['future']}"
             f"\n\nQuestion:\n{branch['question']}"}]


def validate_local_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Only local HTTP inference is permitted; no paid API")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


class LocalClient:
    def __init__(self, base_url: str, model: str, trace_path: Path) -> None:
        validate_local_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.root_url = self.base_url.removesuffix("/v1")
        self.model = model
        self.trace_path = trace_path
        self.calls = 0
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        validate_local_url(url)
        request = Request(url, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with self.opener.open(request, timeout=180) as response:
            return json.load(response)

    def tokenize(self, text: str) -> list[int]:
        return self.post(f"{self.root_url}/tokenize", {"content": text, "add_special": False})["tokens"]

    def chat(self, messages: list[dict[str, str]], tag: str, *, max_tokens: int,
             seed: int, temperature: float = 0.0) -> dict[str, Any]:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "seed": seed,
                   "chat_template_kwargs": {"enable_thinking": False}}
        start = time.perf_counter()
        response = self.post(f"{self.base_url}/chat/completions", payload)
        choice = response["choices"][0]
        record = {"tag": tag, "request": payload,
                  "answer": choice["message"].get("content") or "",
                  "finish_reason": choice["finish_reason"],
                  "usage": response.get("usage", {}), "seconds": time.perf_counter() - start}
        with self.trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.calls += 1
        return record


def evaluate(client: LocalClient, memory: str, branch: dict[str, Any],
             tag: str, seed: int) -> dict[str, Any]:
    record = client.chat(reader_messages(memory, branch), tag, max_tokens=128, seed=seed)
    return {"answer": record["answer"], **grade(record["answer"], branch)}


def validate_cases(cases: list[dict[str, Any]]) -> None:
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty with unique IDs")
    for case in cases:
        splits = [branch["split"] for branch in case["branches"]]
        if splits.count("selection") != 2 or splits.count("heldout") != 1 or len(splits) != 3:
            raise ValueError("Each case needs two selection branches and one heldout branch")
        for branch in case["branches"]:
            if set(branch["gold"]) != set(branch["aliases"]) or not branch["gold"]:
                raise ValueError("Gold and alias fields must match")
            if any(value not in branch["aliases"][key] for key, value in branch["gold"].items()):
                raise ValueError("Canonical gold must occur in its aliases")


def run_case(client: LocalClient, case: dict[str, Any], budget: int,
             count: int, seed: int) -> dict[str, Any]:
    case_id = case["id"]
    candidates = []
    for index in range(count):
        record = client.chat(writer_messages(case["prefix"], budget), f"{case_id}/write/{index}",
                             max_tokens=budget, temperature=0.7, seed=seed + index)
        memory = record["answer"].strip()
        tokens = client.tokenize(memory)
        clipped = len(tokens) > budget
        if clipped:
            memory = client.post(f"{client.root_url}/detokenize", {"tokens": tokens[:budget]})["content"]
        final_count = len(client.tokenize(memory))
        if final_count > budget:
            raise ValueError(f"Memory exceeds token budget after clipping: {final_count} > {budget}")
        candidates.append({"memory": memory, "tokens": final_count,
                           "explicitly_clipped": clipped, "finish_reason": record["finish_reason"]})
    selection = [b for b in case["branches"] if b["split"] == "selection"]
    heldout = next(b for b in case["branches"] if b["split"] == "heldout")
    controls = {}
    for branch in case["branches"]:
        controls[branch["id"]] = {
            name: evaluate(client, context, branch, f"{case_id}/{name}/{branch['id']}", seed)
            for name, context in (("source", case["prefix"]), ("future_only", ""))}
    for index, candidate in enumerate(candidates):
        candidate["selection"] = [evaluate(client, candidate["memory"], branch,
                                          f"{case_id}/candidate/{index}/{branch['id']}", seed)
                                  for branch in selection]
    source_scores = [controls[b["id"]]["source"]["score"] for b in selection]
    indices = select_candidates([[r["score"] for r in c["selection"]] for c in candidates], source_scores)
    heldout_answers = {index: evaluate(client, candidates[index]["memory"], heldout,
                                      f"{case_id}/heldout/{index}", seed)
                       for index in sorted(set(indices.values()))}
    return {"id": case_id, "source_tokens": len(client.tokenize(case["prefix"])),
            "heldout_id": heldout["id"], "candidates": candidates, "controls": controls,
            "selected_indices": indices, "source_selection_scores": source_scores,
            "heldout": {name: heldout_answers[index] for name, index in indices.items()}}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, list[float]] = {name: [] for name in ("source", "future_only", "plain", "mean", "worst")}
    strict_disagreements = 0
    for case in results:
        for name in ("source", "future_only"):
            scores[name].append(case["controls"][case["heldout_id"]][name]["score"])
        for name in ("plain", "mean", "worst"):
            scores[name].append(case["heldout"][name]["score"])
        losses = [[max(0.0, source - probe["score"])
                   for source, probe in zip(case["source_selection_scores"], candidate["selection"])]
                  for candidate in case["candidates"]]
        mean, worst = (case["selected_indices"][name] for name in ("mean", "worst"))
        strict_disagreements += (sum(losses[mean]) < sum(losses[worst]) and
                                 max(losses[worst]) < max(losses[mean]))
    solved = [i for i, score in enumerate(scores["source"]) if score == 1.0]
    return {"heldout_field_accuracy": {name: sum(values) / len(values) for name, values in scores.items()},
            "source_solved_cases": len(solved),
            "heldout_accuracy_on_source_solved": {
                name: sum(values[i] for i in solved) / len(solved) if solved else None
                for name, values in scores.items()},
            "selector_disagreements": sum(c["selected_indices"]["mean"] != c["selected_indices"]["worst"] for c in results),
            "strict_selector_disagreements": strict_disagreements,
            "distinct_candidate_counts": [len({m["memory"] for m in c["candidates"]}) for c in results]}


def write_report(output: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = ["# Future-memory smoke pilot (untrained)", "",
             "Authored conditional plans; not LoCoMo/LongMemEval scoring or a trained method.",
             "Writer sees only the prefix. Both selectors share candidates and two visible futures; the third condition is held out.",
             "Lowest index breaks ties. Metric: exact-alias field accuracy. No LLM judge or API key.", "",
             "| Heldout reader input | Field accuracy |", "|---|---:|"]
    lines += [f"| {name} | {score:.1%} |" for name, score in summary["heldout_field_accuracy"].items()]
    lines += ["", f"Source-solved heldout cases: {summary['source_solved_cases']}/{len(payload['results'])}.",
              f"Mean/worst selector disagreements: {summary['selector_disagreements']}.",
              f"Strict objective reversals (excluding tie-driven picks): {summary['strict_selector_disagreements']}.",
              f"Local chat calls: {payload['chat_calls']}; elapsed: {payload['seconds']:.1f}s.", "",
              "| Case | Source | No memory | Plain | Mean | Worst | Selected plain/mean/worst |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for case in payload["results"]:
        control = case["controls"][case["heldout_id"]]
        values = [control[n]["score"] for n in ("source", "future_only")]
        values += [case["heldout"][n]["score"] for n in ("plain", "mean", "worst")]
        row = " | ".join(f"{value:.0%}" for value in values)
        picks = "/".join(str(case["selected_indices"][n]) for n in ("plain", "mean", "worst"))
        lines.append(f"| {case['id']} | {row} | {picks} |")
    lines += ["", "## Limits", "",
              "- Six authored prefixes are diagnostics, not statistical evidence or a benchmark win.",
              "- Candidate selection has privileged training-time probes; writer weights are NOT updated.",
              "- The third continuation is held out from selector probes; its mapping is present in the prefix. This is not generalization to arbitrary futures.",
              "- Short-prefix compression is isolated: embeddings are health-checked but not used by the reader.",
              "- Audit memories, JSON errors, writer truncation and source-reader errors in results.json/calls.jsonl.",
              "- Training claims require a trained writer, disjoint histories and matched-compute hypothetical-QA controls."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9911/v1")
    parser.add_argument("--model", default="Qwen3.5-0.8B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:9010/v1")
    parser.add_argument("--embedding-model", default="Qwen3-Embedding-0.6B")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if min(args.budget, args.candidates, args.limit) < 1:
        parser.error("budget, candidates and limit must be positive")
    raw = args.cases.read_bytes()
    cases = json.loads(raw)["cases"][:args.limit]
    validate_cases(cases)
    args.output.mkdir(parents=True, exist_ok=False)
    client = LocalClient(args.base_url, args.model, args.output / "calls.jsonl")
    embedding = LocalClient(args.embedding_base_url, args.embedding_model, args.output / "unused.jsonl")
    health = embedding.post(f"{embedding.base_url}/embeddings", {
        "model": args.embedding_model, "input": ["A short local embedding health check."]})
    dimensions = len(health["data"][0]["embedding"])
    if dimensions != 1024:
        raise ValueError(f"Expected 1024 embedding dimensions, got {dimensions}")
    start = time.perf_counter()
    payload: dict[str, Any] = {"trained": False, "official_benchmark": False,
                              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                              "cases_sha256": hashlib.sha256(raw).hexdigest(),
                              "embedding_dim": dimensions, "results": []}
    for case in cases:
        print(f"Running {case['id']}...", flush=True)
        result = run_case(client, case, args.budget, args.candidates, args.seed)
        payload["results"].append(result)
        payload.update(chat_calls=client.calls, seconds=time.perf_counter() - start)
        (args.output / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  heldout={ {k: r['score'] for k, r in result['heldout'].items()} }", flush=True)
    payload["summary"] = summarize(payload["results"])
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.output, payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
