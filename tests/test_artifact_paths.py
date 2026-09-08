from utils.artifact_paths import generate_memory_agent_base_path, sanitize_path_component


def test_sanitize_path_component_handles_ollama_model_names():
    assert sanitize_path_component("qwen3.5:0.8b") == "qwen3.5-0.8b"


def test_memory_agent_path_is_windows_safe_for_ollama_model_names():
    path = generate_memory_agent_base_path(
        {
            "agent_name": "Structure_rag_mem0_qwen35_0_8b_ollama",
            "agent_chunk_size": 8192,
            "model": "qwen3.5:0.8b",
            "retrieve_num": 5,
        },
        {"dataset": "LoCoMo", "sub_dataset": "locomo_qa"},
    )

    assert "modelqwen3.5-0.8b" in path
    assert ":" not in path
