"""Run the six memory baselines on official LoCoMo and LongMemEval-S."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "langmem": "sequential_langmem.yaml", "mem0": "sequential_mem0.yaml",
    "a_mem": "hybrid_a_mem.yaml", "lightmem": "hybrid_lightmem.yaml",
    "simplemem": "hybrid_simplemem.yaml", "e_mem": "hybrid_e_mem.yaml",
}
DATASETS = {
    "locomo": "benchmark/locomo/config/LoCoMo_official.yaml",
    "longmemeval": "benchmark/longmemeval/config/LongMemEval_s_official.yaml",
}


def build_agent_config(method, args):
    config = yaml.safe_load((ROOT / "config" / METHODS[method]).read_text(encoding="utf-8-sig"))
    config.update(
        model=args.model, provider="openai_compatible",
        api_key_env="OPENAI_API_KEY", base_url=args.base_url,
        base_url_env="OPENAI_BASE_URL", temperature=0.0,
        input_length_limit=32768, buffer_length=1000,
        tokenizer_encoding="cl100k_base", agent_chunk_size=4096,
        embedding_model=args.embedding_model, mem0_embedder_model=args.embedding_model,
        mem0_use_model_name_verbatim=True,
        embedding_base_url=args.embedding_base_url or args.base_url,
        embedding_api_key_env="OPENAI_API_KEY", embedding_dim=args.embedding_dim,
    )
    # Existing adapters use method-specific endpoint options.
    config.update({
        "a_mem_base_url": args.base_url,
        "a_mem_api_key_env": "OPENAI_API_KEY",
        "a_mem_embedding_provider": "openai_compatible",
        "a_mem_embedding_model": args.embedding_model,
        "a_mem_embedding_base_url": args.embedding_base_url or args.base_url,
        "lightmem_base_url": args.base_url, "lightmem_embedding_backend": "openai",
        "lightmem_messages_use": "hybrid",
        "lightmem_embedding_model": args.embedding_model,
        "lightmem_embedding_base_url": args.embedding_base_url or args.base_url,
        "lightmem_embedding_dims": args.embedding_dim,
        "simplemem_base_url": args.base_url,
    })
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-base-url", default=os.environ.get("EMBEDDING_BASE_URL"))
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--locomo-data", type=Path, default=ROOT / "datasets/LoCoMo/locomo10.json")
    parser.add_argument("--longmemeval-data", type=Path,
                        default=ROOT / "datasets/LongMemEval/longmemeval_s_cleaned.json")
    parser.add_argument("--max-samples", type=int, help="Limit histories per dataset for a smoke run")
    parser.add_argument("--max-queries", type=int, default=0, help="Limit total questions per method/dataset run (0 = all)")
    parser.add_argument("--max-context-chunks", type=int,
                        help="Limit context chunks per history for a smoke run")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "results/six_baselines")
    parser.add_argument("--dry-run", action="store_true", help="Write configs and print commands without model calls")
    args = parser.parse_args()
    if not args.base_url:
        parser.error("Set --base-url or OPENAI_BASE_URL.")
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY (a placeholder is sufficient for a keyless local server).")
    if (args.embedding_dim < 1
            or (args.max_samples is not None and args.max_samples < 1)
            or (args.max_context_chunks is not None and args.max_context_chunks < 1)
            or args.max_queries < 0):
        parser.error("Embedding dimension and smoke limits must be positive; query limit must be nonnegative.")
    run_root = args.artifact_root.resolve()
    config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        data_path = getattr(args, dataset + "_data").resolve()
        if not args.dry_run and not data_path.is_file():
            parser.error(f"Missing dataset {data_path}. Run scripts/download_memory_benchmarks.py.")
    failed = []
    for method in args.methods:
        config = build_agent_config(method, args)
        agent_path = config_dir / f"{method}.yaml"
        agent_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        for dataset in args.datasets:
            data_config = yaml.safe_load((ROOT / DATASETS[dataset]).read_text(encoding="utf-8-sig"))
            data_config["test_files"] = str(getattr(args, dataset + "_data").resolve())
            if args.max_samples is not None:
                data_config["max_test_samples"] = args.max_samples
            if args.max_context_chunks is not None:
                data_config["max_context_chunks"] = args.max_context_chunks
            data_path = config_dir / f"{dataset}.yaml"
            data_path.write_text(yaml.safe_dump(data_config), encoding="utf-8")
            command = [
                sys.executable, str(ROOT / "main.py"),
                "--agent_config", str(agent_path), "--dataset_config", str(data_path),
                "--artifact_root", str(run_root / method / dataset),
                "--max_test_queries_ablation", str(args.max_queries),
                "--fail-on-query-error", "--retry_failed_queries",
            ]
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                # Upstream modules and global settings stay isolated by process.
                env = os.environ.copy()
                env.update(LIGHTMEM_MODEL=args.model, LIGHTMEM_BASE_URL=args.base_url,
                           LIGHTMEM_EMBEDDING_MODEL=args.embedding_model,
                           LIGHTMEM_EMBEDDING_DIMENSION=str(args.embedding_dim))
                result = subprocess.run(command, cwd=ROOT, env=env, check=False)
                if result.returncode:
                    failed.append({"method": method, "dataset": dataset, "exit_code": result.returncode})
    if not args.dry_run:
        (run_root / "failures.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

