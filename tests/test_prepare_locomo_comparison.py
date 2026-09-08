import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import yaml

from scripts.prepare_locomo_comparison import (
    _absolute_interpreter_path,
    bundle_sources,
    prepare_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATASET = ROOT.parent / "HiGMem" / "data" / "locomo10.json"
OFFICIAL_SCORER = ROOT.parent / "HiGMem" / "official_locomo_evaluation.py"
TOKENIZER_SNAPSHOT = Path(
    "/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
COMPRESSOR_SNAPSHOT = Path(
    "/workspace/.hf_home/hub/"
    "models--microsoft--llmlingua-2-bert-base-multilingual-cased-meetingbank/"
    "snapshots/5f0c82792b7ea14c6484e015b6a072009496b7f2"
)


def test_interpreter_absolute_path_preserves_venv_symlink(tmp_path):
    target = tmp_path / "base-python"
    target.write_bytes(b"")
    interpreter = tmp_path / "comparison-venv-python"
    try:
        interpreter.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")

    normalized = _absolute_interpreter_path(interpreter)

    assert normalized == interpreter.absolute()
    assert normalized != target.resolve()
    assert normalized.is_file()


def test_bundle_is_allowlisted_and_manifest_matches_archive_bytes(tmp_path):
    archive = tmp_path / "comparison-source.tar.gz"

    bundle_sources(archive, root=ROOT)

    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
        manifest = json.load(bundle.extractfile("source_manifest.json"))
        assert {
            "main.py",
            "scripts/locomo_server_queue.py",
            "scripts/metered_openai_proxy.py",
            "scripts/prepare_locomo_comparison.py",
            "scripts/preflight_locomo_clients.py",
            "scripts/comparison_shared_runtime.pth",
            "config/sequential_langmem.yaml",
            "utils/agent.py",
            "benchmark/locomo/loader.py",
            "methods/langmem/langmem_adapter.py",
            "methods/a_mem/a_mem_adapter.py",
            "methods/simplemem/simplemem_adapter.py",
            "methods/lightmem/lightmem_adapter.py",
            "methods/e_mem/e_mem_adapter.py",
        }.issubset(names)
        assert set(manifest["files"]) == names - {"source_manifest.json"}
        for name, expected_digest in manifest["files"].items():
            archived_bytes = bundle.extractfile(name).read()
            assert hashlib.sha256(archived_bytes).hexdigest() == expected_digest

    forbidden_parts = {".git", ".env", "results", "__pycache__", "secrets", "privatepayload"}
    assert not any(forbidden_parts.intersection(Path(name).parts) for name in names)
    assert not any(name.endswith((".bin", ".safetensors", ".pt", ".pth.tar")) for name in names)
    assert not any(
        name.startswith("methods/lightmem/upstream/")
        and name != "methods/lightmem/upstream/experiments/locomo/prompts.py"
        for name in names
    )


@pytest.mark.skipif(
    not PUBLIC_DATASET.is_file() or not OFFICIAL_SCORER.is_file(),
    reason="The supplied public HiGMem evaluation checkout is unavailable",
)
def test_prepare_pins_six_configs_and_validates_without_model_calls(tmp_path):
    archive = tmp_path / "source.tar.gz"
    bundle_sources(archive, root=ROOT)
    extracted = tmp_path / "source"
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(extracted, filter="data")

    config_dir = tmp_path / "comparison_config"
    output = tmp_path / "new_run"
    clean_environment = os.environ.copy()
    clean_environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(extracted / "scripts" / "prepare_locomo_comparison.py"),
            "prepare",
            "--config-dir", str(config_dir),
            "--output", str(output),
            "--run-id", "run-fixed",
            "--dataset", str(PUBLIC_DATASET),
            "--scorer", str(OFFICIAL_SCORER),
            "--client-python", sys.executable,
            "--server-python", sys.executable,
            "--higmem-run", str(tmp_path / "higmem_run"),
            "--hf-home", str(tmp_path / "hf_home"),
            "--embedding-path", str(tmp_path / "minilm"),
            "--hourly-rate-usd", "0.3422222222222222",
        ],
        cwd=tmp_path,
        env=clean_environment,
        capture_output=True,
        text=True,
        check=True,
    )

    plan_path = config_dir / "plan.json"
    assert json.loads(result.stdout)["plan"] == str(plan_path)
    assert plan_path == config_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema_version"] == 1
    assert plan["model"] == "Qwen/Qwen3.5-9B"
    assert plan["dtype"] == "float16"
    assert plan["run_id"] == "run-fixed"
    assert plan["output"] == str(output.resolve())
    assert plan["hourly_rate_usd"] == 0.3422222222222222
    assert [item["method"] for item in plan["methods"]] == [
        "a_mem", "mem0", "langmem", "simplemem", "lightmem", "e_mem"
    ]

    dataset_paths = {item["dataset_config"] for item in plan["methods"]}
    assert len(dataset_paths) == 1
    dataset_config = yaml.safe_load(Path(dataset_paths.pop()).read_text(encoding="utf-8"))
    assert dataset_config["chunk_size"] == 1
    assert dataset_config["generation_max_length"] == 1024
    assert dataset_config["locomo_repeat_session_header"] is True
    assert dataset_config["locomo_categories"] == [1, 2, 3, 4]
    assert dataset_config["max_test_samples"] == 10

    configs = {
        item["method"]: yaml.safe_load(
            Path(item["agent_config"]).read_text(encoding="utf-8")
        )
        for item in plan["methods"]
    }
    for method, config in configs.items():
        assert config["model"] == "Qwen/Qwen3.5-9B"
        assert config["base_url"] == "http://127.0.0.1:18082/v1"
        assert config["embedding_base_url"] == "http://127.0.0.1:18083/v1"
        assert config["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert config["embedding_dim"] == 384
        assert config["tokenizer_model"] == str(TOKENIZER_SNAPSHOT)
        assert config["tokenizer_encoding"] is None
        assert config["input_length_limit"] == 32768
        assert config["buffer_length"] == 0
        assert config["agent_chunk_size"] == 1
        assert config["output_dir"] == f"results/outputs/{method}"

    assert configs["a_mem"]["retrieve_num"] == 10
    assert configs["simplemem"]["simplemem_window_size"] == 40
    assert configs["simplemem"]["simplemem_overlap_size"] == 2
    assert configs["simplemem"]["simplemem_enable_planning"] is True
    assert configs["simplemem"]["simplemem_enable_reflection"] is True
    assert configs["simplemem"]["simplemem_max_reflection_rounds"] == 2
    assert configs["simplemem"]["simplemem_max_parallel_workers"] == 4
    assert configs["simplemem"]["simplemem_max_retrieval_workers"] == 3
    assert configs["lightmem"]["retrieve_num"] == 60
    assert configs["lightmem"]["lightmem_comparison_mode"] is True
    assert configs["lightmem"]["lightmem_ingest_mode"] == "pipeline"
    assert configs["lightmem"]["lightmem_compressor_model"] == str(COMPRESSOR_SNAPSHOT)
    assert configs["lightmem"]["lightmem_compression_device"] == "cpu"
    assert configs["e_mem"]["e_mem_tokenizer_model"] == str(TOKENIZER_SNAPSHOT)

    pinned = plan["file_sha256"]
    assert str(PUBLIC_DATASET.resolve()) in pinned
    assert str(OFFICIAL_SCORER.resolve()) in pinned
    for item in plan["methods"]:
        assert item["agent_config"] in pinned
        assert item["dataset_config"] in pinned
    for filename, digest in pinned.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest


def test_prepare_refuses_an_existing_output_without_creating_configs(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    config_dir = tmp_path / "configs"

    with pytest.raises(FileExistsError, match="output"):
        prepare_comparison(
            config_dir=config_dir,
            output=output,
            run_id="run-fixed",
            dataset=tmp_path / "dataset.json",
            scorer=tmp_path / "scorer.py",
            client_python=Path(sys.executable),
            server_python=Path(sys.executable),
            higmem_run=tmp_path / "higmem",
            hf_home=tmp_path / "hf",
            embedding_path=tmp_path / "embed",
            tokenizer_path=TOKENIZER_SNAPSHOT,
            compressor_path=COMPRESSOR_SNAPSHOT,
            model_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            root=ROOT,
        )

    assert not config_dir.exists()


def test_prepare_rejects_source_changed_after_manifest_creation(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    source = root / "main.py"
    original = b"print('pinned')\n"
    source.write_bytes(original)
    (root / "source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {"main.py": hashlib.sha256(original).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    source.write_text("print('changed')\n", encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    scorer = tmp_path / "scorer.py"
    dataset.write_text("[]", encoding="utf-8")
    scorer.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Bundled source changed"):
        prepare_comparison(
            config_dir=tmp_path / "configs",
            output=tmp_path / "output",
            run_id="run-fixed",
            dataset=dataset,
            scorer=scorer,
            client_python=Path(sys.executable),
            server_python=Path(sys.executable),
            root=root,
        )

    assert not (tmp_path / "configs").exists()
    assert not (tmp_path / "output").exists()
