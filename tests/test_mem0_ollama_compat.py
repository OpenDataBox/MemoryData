from types import SimpleNamespace

from utils.agent import AgentWrapper
from mem0.llms.openai import OpenAILLM


def test_wrapper_uses_configured_openai_reasoning_effort():
    agent = AgentWrapper.__new__(AgentWrapper)
    agent.agent_config = {
        "qwen3_disable_thinking": False,
        "openai_reasoning_effort": "none",
    }

    assert agent._prepare_openai_request_kwargs({}) == {"reasoning_effort": "none"}


def test_mem0_disables_qwen_thinking_for_local_ollama():
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.config = SimpleNamespace(
        model="qwen3.5:0.8b",
        openai_base_url="http://127.0.0.1:11434/v1",
    )
    params = {}

    llm._inject_ollama_non_thinking_defaults(params)

    assert params == {"reasoning_effort": "none"}
