from utils.agent import AgentWrapper


def test_infers_qwen3_embedding_dimensions_across_endpoint_model_names():
    agent = AgentWrapper.__new__(AgentWrapper)

    assert agent._infer_embedding_dimensions("Qwen/Qwen3-Embedding-0.6B") == 1024
    assert agent._infer_embedding_dimensions("qwen3-embedding:0.6b") == 1024
    assert agent._infer_embedding_dimensions("Qwen3-Embedding-4B") == 2560
    assert agent._infer_embedding_dimensions("Qwen/Qwen3-Embedding-8B") == 4096


def test_explicit_embedding_dimension_wins():
    agent = AgentWrapper.__new__(AgentWrapper)

    assert agent._infer_embedding_dimensions("qwen3-embedding:0.6b", 384) == 384
