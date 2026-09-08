import hashlib
import os

from utils.artifact_paths import build_hashed_runtime_dir


def test_lightmem_runtime_uses_short_hashed_path(tmp_path):
    """Keep Qdrant below Windows MAX_PATH even when artifact names are descriptive."""
    agent_save_folder = str(
        tmp_path
        / "agents"
        / (
            "Agentic_memory_lightmem_locomo_qa_categories_"
            "multi_hop-open_domain-single_hop-temporal_chunk4096_"
            "modelQwen3.5-0.8B"
        )
        / "exp_0"
    )
    runtime_path = build_hashed_runtime_dir(
        agent_save_folder,
        "_lightmem_runtime",
    )
    expected_key = hashlib.sha256(
        os.path.abspath(agent_save_folder).encode("utf-8")
    ).hexdigest()[:16]
    assert os.path.basename(runtime_path) == expected_key
    assert os.path.basename(os.path.dirname(runtime_path)) == "_lightmem_runtime"
    assert len(os.path.join(runtime_path, "qdrant", "collection", "locomo_qa_lightmem")) < 260
