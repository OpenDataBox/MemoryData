"""Build a source-only comparison bundle or prepare a pinned server run plan."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
METHODS = ("a_mem", "mem0", "langmem", "simplemem", "lightmem", "e_mem")
MODEL = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
LLM_BASE_URL = "http://127.0.0.1:18082/v1"
EMBEDDING_BASE_URL = "http://127.0.0.1:18083/v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_PATH = Path(
    "/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/"
    f"snapshots/{MODEL_REVISION}"
)
COMPRESSOR_PATH = Path(
    "/workspace/.hf_home/hub/"
    "models--microsoft--llmlingua-2-bert-base-multilingual-cased-meetingbank/"
    "snapshots/5f0c82792b7ea14c6484e015b6a072009496b7f2"
)

_EXACT_BUNDLE_PATHS = (
    "main.py",
    "scripts/run_six_baselines.py",
    "scripts/locomo_server_queue.py",
    "scripts/metered_openai_proxy.py",
    "scripts/score_locomo_comparison.py",
    "scripts/serve_minilm_embeddings.py",
    "scripts/prepare_locomo_comparison.py",
    "scripts/preflight_locomo_clients.py",
    "scripts/comparison_shared_runtime.pth",
    "scripts/comparison_runtime_requirements.txt",
    "scripts/comparison_vllm.conf",
    "scripts/comparison_queue.conf",
    "config/sequential_langmem.yaml",
    "config/sequential_mem0.yaml",
    "config/hybrid_a_mem.yaml",
    "config/hybrid_lightmem.yaml",
    "config/hybrid_simplemem.yaml",
    "config/hybrid_e_mem.yaml",
    "methods/langmem/langmem_adapter.py",
    "methods/langmem/pyproject.toml",
    "methods/langmem/README.md",
    "methods/langmem/LICENSE",
    "methods/a_mem/a_mem_adapter.py",
    "methods/simplemem/simplemem_adapter.py",
    "methods/lightmem/lightmem_adapter.py",
    "methods/lightmem/upstream/experiments/locomo/prompts.py",
    "methods/e_mem/e_mem_adapter.py",
)
_TREE_BUNDLE_PATHS = (
    ("utils", frozenset({".py"})),
    ("benchmark", frozenset({".py", ".yaml", ".yml"})),
    ("methods/langmem/src", frozenset({".py"})),
    ("methods/a_mem/source", frozenset({".py"})),
    ("methods/mem0/source", frozenset({".py"})),
    ("methods/simplemem/source", frozenset({".py"})),
    ("methods/lightmem/source", frozenset({".py"})),
    ("methods/e-mem/src", frozenset({".py", ".yaml", ".yml", ".json"})),
)
_FORBIDDEN_COMPONENTS = frozenset(
    {".git", ".env", "results", "__pycache__", "secrets", "private", "privatepayload"}
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _safe_relative_file(root: Path, relative: str | Path) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Bundle path must be repository-relative: {relative}")
    path = root / relative_path
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path resolves outside the repository: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Required bundle source is missing: {relative}")
    return path


def _bundle_relative_paths(root: Path) -> list[Path]:
    paths = {Path(relative) for relative in _EXACT_BUNDLE_PATHS}
    paths.update(path.relative_to(root) for path in root.glob("requirements*.txt") if path.is_file())
    for relative_root, suffixes in _TREE_BUNDLE_PATHS:
        directory = root / relative_root
        if not directory.is_dir():
            raise FileNotFoundError(f"Required bundle source tree is missing: {relative_root}")
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            relative = path.relative_to(root)
            if _FORBIDDEN_COMPONENTS.intersection(part.lower() for part in relative.parts):
                continue
            paths.add(relative)
    return sorted(paths, key=lambda path: path.as_posix())


def _tar_add_bytes(bundle: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    bundle.addfile(info, io.BytesIO(payload))


def bundle_sources(output: Path, *, root: Path = ROOT) -> Path:
    """Create a source-only tarball from the explicit comparison allowlist."""

    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_payloads = {}
    for relative in _bundle_relative_paths(root):
        source = _safe_relative_file(root, relative)
        source_payloads[relative.as_posix()] = source.read_bytes()
    manifest = {
        "schema_version": 1,
        "files": {
            name: _sha256_bytes(payload)
            for name, payload in sorted(source_payloads.items())
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    with tarfile.open(output, mode="x:gz") as bundle:
        for name, payload in sorted(source_payloads.items()):
            _tar_add_bytes(bundle, name, payload)
        _tar_add_bytes(bundle, "source_manifest.json", manifest_bytes)
    return output


def _read_source_manifest(root: Path) -> tuple[Path, dict[str, str]]:
    manifest_path = root / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError("Unsupported source manifest")
    files = payload["files"]
    if not files or any(not isinstance(name, str) or not isinstance(digest, str)
                        for name, digest in files.items()):
        raise ValueError("Source manifest has invalid file entries")
    return manifest_path, files


def _verify_source_manifest(root: Path, files: dict[str, str]) -> dict[str, str]:
    pinned = {}
    for relative, expected_digest in files.items():
        path = _safe_relative_file(root, relative).resolve()
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(f"Bundled source changed: {relative}")
        pinned[str(path)] = actual_digest
    return pinned


def _load_build_agent_config(root: Path):
    source = root / "scripts" / "run_six_baselines.py"
    spec = importlib.util.spec_from_file_location("_comparison_run_six_baselines", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load comparison config builder: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_agent_config


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _absolute_interpreter_path(path: Path) -> Path:
    """Make an interpreter path absolute without resolving its venv symlink."""

    return Path(os.path.abspath(Path(path).expanduser()))


def prepare_comparison(
    *,
    config_dir: Path,
    output: Path,
    run_id: str,
    dataset: Path = Path("/workspace/HiGMem/data/locomo10.json"),
    scorer: Path = Path("/workspace/HiGMem/official_locomo_evaluation.py"),
    client_python: Path = Path("/workspace/comparison-venv/bin/python"),
    server_python: Path = Path("/venv/main/bin/python"),
    higmem_run: Path = Path("/workspace/HiGMem/vast_run"),
    hf_home: Path = Path("/workspace/.hf_home"),
    embedding_path: Path = Path(
        "/workspace/.hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
        "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    ),
    tokenizer_path: Path = TOKENIZER_PATH,
    compressor_path: Path = COMPRESSOR_PATH,
    model_revision: str = MODEL_REVISION,
    hourly_rate_usd: float | None = None,
    root: Path = ROOT,
) -> Path:
    """Create new pinned configs and a queue plan without invoking any model."""

    root = Path(root).resolve()
    config_dir = Path(config_dir).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {output}")
    if config_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing config directory: {config_dir}")
    if _paths_overlap(config_dir, output):
        raise ValueError("Config directory must be outside the output run directory")
    if not str(run_id).strip():
        raise ValueError("run_id must be non-empty")
    if hourly_rate_usd is not None and (
        not math.isfinite(hourly_rate_usd) or hourly_rate_usd < 0
    ):
        raise ValueError("hourly_rate_usd must be a finite nonnegative value")
    if str(Path(tokenizer_path)) != str(TOKENIZER_PATH):
        raise ValueError(f"Comparison tokenizer snapshot is fixed at {TOKENIZER_PATH}")
    if str(Path(compressor_path)) != str(COMPRESSOR_PATH):
        raise ValueError(f"Comparison compressor snapshot is fixed at {COMPRESSOR_PATH}")
    if model_revision != MODEL_REVISION:
        raise ValueError(f"Comparison model revision is fixed at {MODEL_REVISION}")

    dataset = Path(dataset).resolve()
    scorer = Path(scorer).resolve()
    client_python = _absolute_interpreter_path(client_python)
    server_python = _absolute_interpreter_path(server_python)
    for label, path in (
        ("dataset", dataset),
        ("scorer", scorer),
        ("client_python", client_python),
        ("server_python", server_python),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    manifest_path, manifest_files = _read_source_manifest(root)
    source_hashes = _verify_source_manifest(root, manifest_files)
    build_agent_config = _load_build_agent_config(root)

    config_dir.mkdir(parents=True)
    output.mkdir(parents=True)
    builder_args = SimpleNamespace(
        model=MODEL,
        base_url=LLM_BASE_URL,
        embedding_model=EMBEDDING_MODEL,
        embedding_base_url=EMBEDDING_BASE_URL,
        embedding_dim=384,
    )
    common_overrides = {
        "model": MODEL,
        "provider": "openai_compatible",
        "base_url": LLM_BASE_URL,
        "base_url_env": "OPENAI_BASE_URL",
        "temperature": 0.0,
        "input_length_limit": 32768,
        "buffer_length": 0,
        "tokenizer_model": str(Path(tokenizer_path)),
        "tokenizer_encoding": None,
        "agent_chunk_size": 1,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_base_url": EMBEDDING_BASE_URL,
        "embedding_base_url_env": "EMBEDDING_BASE_URL",
        "embedding_dim": 384,
        "mem0_embedder_model": EMBEDDING_MODEL,
        "mem0_use_model_name_verbatim": True,
        "qwen3_disable_thinking": True,
    }
    method_overrides = {
        "a_mem": {"retrieve_num": 10},
        "simplemem": {
            "simplemem_window_size": 40,
            "simplemem_overlap_size": 2,
            "simplemem_enable_thinking": False,
            "simplemem_use_streaming": False,
            "simplemem_enable_planning": True,
            "simplemem_enable_reflection": True,
            "simplemem_max_reflection_rounds": 2,
            "simplemem_enable_parallel_processing": True,
            "simplemem_max_parallel_workers": 4,
            "simplemem_enable_parallel_retrieval": True,
            "simplemem_max_retrieval_workers": 3,
        },
        "lightmem": {
            "retrieve_num": 60,
            "lightmem_comparison_mode": True,
            "lightmem_ingest_mode": "pipeline",
            "lightmem_compressor_model": str(Path(compressor_path)),
            "lightmem_compression_device": "cpu",
        },
        "e_mem": {"e_mem_tokenizer_model": str(Path(tokenizer_path))},
    }

    method_entries = []
    generated_paths = []
    for method in METHODS:
        config = build_agent_config(method, builder_args)
        config.update(common_overrides)
        config.update(method_overrides.get(method, {}))
        config["output_dir"] = f"results/outputs/{method}"
        agent_path = config_dir / f"{method}.yaml"
        agent_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        generated_paths.append(agent_path)
        method_entries.append({"method": method, "agent_config": str(agent_path)})

    dataset_config = yaml.safe_load(
        (root / "benchmark/locomo/config/LoCoMo_official.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    dataset_config.update(
        test_files=str(dataset),
        chunk_size=1,
        generation_max_length=1024,
        max_test_samples=10,
        locomo_categories=[1, 2, 3, 4],
        locomo_repeat_session_header=True,
    )
    dataset_config.pop("max_context_chunks", None)
    dataset_path = config_dir / "locomo.yaml"
    dataset_path.write_text(
        yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8"
    )
    generated_paths.append(dataset_path)
    for entry in method_entries:
        entry["dataset_config"] = str(dataset_path)

    file_hashes = dict(source_hashes)
    file_hashes[str(manifest_path.resolve())] = _sha256_file(manifest_path)
    for path in (dataset, scorer, *generated_paths):
        file_hashes[str(path.resolve())] = _sha256_file(path)

    plan = {
        "schema_version": 1,
        "model": MODEL,
        "dtype": "float16",
        "model_revision": str(model_revision),
        "run_id": str(run_id).strip(),
        "output": str(output),
        "dataset": str(dataset),
        "scorer": str(scorer),
        "client_python": str(client_python),
        "server_python": str(server_python),
        "higmem_run": str(Path(higmem_run).resolve()),
        "hf_home": str(Path(hf_home).resolve()),
        "embedding_path": str(Path(embedding_path).resolve()),
        "methods": method_entries,
        "file_sha256": dict(sorted(file_hashes.items())),
    }
    if hourly_rate_usd is not None:
        plan["hourly_rate_usd"] = hourly_rate_usd

    from scripts.locomo_server_queue import validate_plan

    validate_plan(plan)
    _verify_source_manifest(root, manifest_files)
    plan_path = config_dir / "plan.json"
    with plan_path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return plan_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle", help="Create the allowlisted source tarball")
    bundle.add_argument("--output", type=Path, required=True)

    prepare = commands.add_parser("prepare", help="Create six configs and a pinned queue plan")
    prepare.add_argument("--config-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--dataset", type=Path, default=Path("/workspace/HiGMem/data/locomo10.json"))
    prepare.add_argument("--scorer", type=Path, default=Path("/workspace/HiGMem/official_locomo_evaluation.py"))
    prepare.add_argument("--client-python", type=Path, default=Path("/workspace/comparison-venv/bin/python"))
    prepare.add_argument("--server-python", type=Path, default=Path("/venv/main/bin/python"))
    prepare.add_argument("--higmem-run", type=Path, default=Path("/workspace/HiGMem/vast_run"))
    prepare.add_argument("--hf-home", type=Path, default=Path("/workspace/.hf_home"))
    prepare.add_argument("--embedding-path", type=Path, default=Path(
        "/workspace/.hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
        "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"))
    prepare.add_argument("--hourly-rate-usd", type=float)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bundle":
        archive = bundle_sources(args.output)
        print(json.dumps({"bundle": str(archive)}))
        return 0
    plan = prepare_comparison(
        config_dir=args.config_dir,
        output=args.output,
        run_id=args.run_id,
        dataset=args.dataset,
        scorer=args.scorer,
        client_python=args.client_python,
        server_python=args.server_python,
        higmem_run=args.higmem_run,
        hf_home=args.hf_home,
        embedding_path=args.embedding_path,
        hourly_rate_usd=args.hourly_rate_usd,
    )
    print(json.dumps({"plan": str(plan)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
