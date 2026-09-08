"""Explicit, resumable research commands. Defaults never launch a paid sweep."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path
import socket
import sys
from urllib.parse import urlparse

from .config import load_config, endpoint
from .datasets import download_dataset, synthetic_streams
from .training import append_jsonl, read_jsonl


def doctor(config):
    from .features import FeatureExtractor
    from .backends import RemoteEmbedder, Tokenizer
    from .policy import RetentionPolicy
    checks = {}
    for kind in ("llm", "reader", "embedding", "judge"):
        section = config.get(kind, config["llm"] if kind == "reader" else {})
        if not section:
            checks[kind] = {"ready": False, "reason": "missing configuration"}
            continue
        parsed = urlparse(section["base_url"])
        try:
            with socket.create_connection((parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)), timeout=2):
                pass
            checks[kind] = {"tcp_reachable": True, "model_verified": False, "model": section["model"]}
        except OSError:
            checks[kind] = {"tcp_reachable": False, "model": section["model"]}
    try:
        features = FeatureExtractor(RemoteEmbedder(endpoint(config, "embedding"), None), Tokenizer(config.get("tokenizer_encoding", "cl100k_base")))
        RetentionPolicy.load(config.get("policy_checkpoint", ""), features.fingerprint)
        checks["policy"] = {"ready": True}
    except (OSError, ValueError, KeyError) as exc:
        checks["policy"] = {"ready": False, "reason": str(exc)}
    return {"python": sys.executable, "checks": checks,
            "alfworld_installed": importlib.util.find_spec("alfworld") is not None,
            "note": "TCP reachability does not verify served model identity. No generation or credentials are read by doctor."}


def create_matrix(config, output, datasets, *, full=False):
    import yaml
    output = Path(output)
    if (output/"jobs.jsonl").exists():
        raise FileExistsError("Matrix already exists; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    modes = ["raw", "always_merge", "keep", "heuristic", "criteria_verifier", "learned", "learned_residual"]
    budgets = [2048, 4096, 8192]
    rounds = [1, 2, 4, 8] if full else [1]
    styles = ["structured", "topical"] if full else ["structured"]
    readers = [config.get("reader", config["llm"])]
    if full:
        readers.append(config["transfer_reader"])
    jobs = []
    for dataset, mode, budget, repeat, style, reader in itertools.product(datasets, modes, budgets, rounds, styles, readers):
        # Raw/keep have no compressor or repeats; avoid claiming duplicate runs as replications.
        if mode in {"raw", "keep"} and (repeat != 1 or style != "structured"):
            continue
        for seed in ([0, 1, 2] if mode.startswith("learned") else [None]):
            run_config = deepcopy(config)
            run_config.update(mode=mode, query_budget=budget, compressor=style, reader=reader)
            if seed is not None:
                run_config["policy_checkpoint"] = str(Path(config["policy_checkpoint"]).with_name(f"policy_seed{seed}.npz"))
            else:
                run_config.pop("policy_checkpoint", None)
            reader_name = reader["model"].replace("/", "_")
            name = f"{dataset}-{mode}-{budget}-r{repeat}-{style}-{reader_name}-s{seed}"
            path = (output/"configs"/(name+".yaml")).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
            arguments = [sys.executable, "-m", "methods.retentionmem", "run", "--config", str(path),
                         "--benchmark", dataset, "--data", str(Path(config["datasets"][dataset]).resolve()),
                         "--output", str((output/"runs"/name).resolve()), "--rounds", str(repeat)]
            job = {"id": name, "arguments": arguments, "cwd": str(Path.cwd()), "status": "not_run"}
            append_jsonl(output/"jobs.jsonl", job)
            jobs.append(job)
    return {"jobs": len(jobs), "manifest": str(output/"jobs.jsonl"), "executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description="RetentionMem research implementation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--config", required=True)
    p = sub.add_parser("demo"); p.add_argument("--output", required=True)
    p = sub.add_parser("download"); p.add_argument("--benchmark", choices=["locomo", "longmemeval"], required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("synthetic")
    p.add_argument("--output", required=True); p.add_argument("--train", type=int, default=3000); p.add_argument("--validation", type=int, default=750); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("collect")
    p.add_argument("--config", required=True); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--max-candidates", type=int); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--max-regression-rate", type=float, default=.05); p.add_argument("--min-accepted", type=int, default=20)
    p = sub.add_parser("run")
    p.add_argument("--config", required=True); p.add_argument("--benchmark", choices=["locomo", "longmemeval"], required=True)
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--max-cases", type=int); p.add_argument("--max-questions", type=int); p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--include-adversarial", action="store_true"); p.add_argument("--no-judge", action="store_true"); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("matrix"); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--benchmarks", nargs="+", choices=["locomo", "longmemeval"], default=["locomo", "longmemeval"]); p.add_argument("--full", action="store_true")
    p = sub.add_parser("compare"); p.add_argument("--left", required=True); p.add_argument("--right", required=True); p.add_argument("--metric", default="judge_correct")
    p = sub.add_parser("plot"); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("alfworld-streams"); p.add_argument("--trajectories", required=True); p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("adamem-prepare"); p.add_argument("--config", required=True); p.add_argument("--repo", required=True)
    p.add_argument("--memory-model", required=True); p.add_argument("--output", required=True)
    p.add_argument("--subset", choices=["all", "correct_only", "failure_only"], default="correct_only")
    p = sub.add_parser("alfworld-score"); p.add_argument("--trajectories", required=True); p.add_argument("--episode-ids", required=True)
    p.add_argument("--output", required=True); p.add_argument("--mode", default="adamem-low")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        result = doctor(load_config(args.config))
    elif args.command == "demo":
        from .demo import run_demo
        result = run_demo(args.output)
    elif args.command == "download":
        result = download_dataset(args.benchmark, args.output)
    elif args.command == "synthetic":
        if min(args.train, args.validation) < 1: raise ValueError("Both splits need source groups")
        if Path(args.output).exists(): raise FileExistsError(args.output)
        for row in synthetic_streams(args.train, args.validation, args.seed): append_jsonl(args.output, row)
        result = {"source_groups": args.train+args.validation, "output": args.output}
    elif args.command == "collect":
        from .backends import RemoteEmbedder, Tokenizer
        from .generation import CandidateGenerator
        from .storage import MemoryStore
        from .training import collect_transitions
        config = load_config(args.config)
        store = MemoryStore(str(args.output)+".sqlite")
        try:
            writer = endpoint(config)
            generator = CandidateGenerator(writer, store, config.get("compressor", "structured"), config.get("max_probes", 8))
            result = collect_transitions(read_jsonl(args.input), generator, endpoint(config, "reader" if "reader" in config else "llm"),
                                         endpoint(config, "judge"), RemoteEmbedder(endpoint(config, "embedding"), store),
                                         Tokenizer(config.get("tokenizer_encoding", "cl100k_base")), store, args.output,
                                         args.max_candidates, args.resume, config.get("test_artifact", False))
            Path(str(args.output)+".costs.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        finally: store.close()
    elif args.command == "train":
        from .training import train_from_file
        result = train_from_file(args.input, args.output, args.seeds, args.epochs, args.max_regression_rate, args.min_accepted)
    elif args.command == "run":
        from .evaluation import run_qa
        result = run_qa(load_config(args.config), args.data, args.benchmark, args.output, max_cases=args.max_cases,
                        max_questions=args.max_questions, include_adversarial=args.include_adversarial,
                        judge_enabled=not args.no_judge, resume=args.resume, repeated_rounds=args.rounds)
    elif args.command == "matrix":
        result = create_matrix(load_config(args.config), args.output, args.benchmarks, full=args.full)
    elif args.command == "compare":
        from .evaluation import paired_comparison
        result = paired_comparison(list(read_jsonl(args.left)), list(read_jsonl(args.right)), args.metric)
    elif args.command == "plot":
        from .evaluation import plot_runs
        plot_runs(args.runs, args.output); result = {"figure": args.output}
    elif args.command == "alfworld-streams":
        from .alfworld import trajectory_streams
        result = trajectory_streams(args.trajectories, args.output, args.seed)
    elif args.command == "adamem-prepare":
        from .alfworld import prepare_adamem
        result = prepare_adamem(load_config(args.config), args.repo, args.memory_model, args.output, args.subset)
    elif args.command == "alfworld-score":
        from .alfworld import score_trajectories
        result = score_trajectories(args.trajectories, args.episode_ids, args.output, args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
