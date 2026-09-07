import pytest
from langgraph.errors import GraphRecursionError
from types import SimpleNamespace

import utils.agent as agent_module
from methods.langmem.langmem_adapter import LangMemAdapter
from utils.agent import AgentWrapper


class HuggingFaceLikeTokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


def _bare_wrapper(*, strict):
    wrapper = AgentWrapper.__new__(AgentWrapper)
    wrapper.strict_comparison = strict
    wrapper.input_length_limit = 300
    wrapper.model_context_window = None
    wrapper.max_tokens = 16
    wrapper.tokenizer = HuggingFaceLikeTokenizer()
    wrapper.agent_config = {}
    wrapper.agent_name = "langmem"
    wrapper.sub_dataset = "locomo_qa"
    wrapper.record_llm_io = False
    wrapper._last_llm_trace = None
    wrapper._current_eval_metadata = None
    return wrapper


def test_env_enables_strict_comparison_on_agent_wrapper(monkeypatch, tmp_path):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    monkeypatch.setattr(agent_module, "resolve_artifact_root", lambda **kwargs: tmp_path)
    monkeypatch.setattr(AgentWrapper, "_initialize_tokenizer", lambda self: HuggingFaceLikeTokenizer())
    monkeypatch.setattr(AgentWrapper, "_initialize_agent_by_type", lambda *args: None)

    wrapper = AgentWrapper(
        agent_config={
            "agent_name": "langmem",
            "output_dir": str(tmp_path),
            "input_length_limit": 512,
            "buffer_length": 0,
            "model": "Qwen/Qwen3.5-9B",
        },
        dataset_config={
            "sub_dataset": "locomo_qa",
            "context_max_length": 512,
            "dataset": "locomo",
            "generation_max_length": 16,
        },
        load_agent_from=str(tmp_path),
    )

    assert wrapper.strict_comparison is True


def test_strict_retrieval_fitting_rejects_partial_or_dropped_context():
    wrapper = _bare_wrapper(strict=True)

    with pytest.raises(RuntimeError, match="truncate retrieved context"):
        wrapper._fit_retrieved_contexts_to_token_limit(
            contexts=["a" * 200, "b" * 200],
            message="question",
            tokenizer=wrapper.tokenizer,
            reserved_tokens=0,
        )


def test_default_retrieval_fitting_preserves_existing_truncation_behavior():
    wrapper = _bare_wrapper(strict=False)

    fitted = wrapper._fit_retrieved_contexts_to_token_limit(
        contexts=["a" * 200, "b" * 200],
        message="question",
        tokenizer=wrapper.tokenizer,
        reserved_tokens=0,
    )

    assert fitted[0] == "a" * 200
    assert 0 < len(fitted[1]) < 200


def test_strict_context_limit_failure_does_not_retry_with_less_memory(monkeypatch):
    wrapper = _bare_wrapper(strict=True)
    attempts = 0

    def fail_request(messages):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("maximum context length exceeded")

    wrapper._request_openai_completion = fail_request
    monkeypatch.setattr(agent_module, "get_template", lambda *args, **kwargs: "system")

    with pytest.raises(RuntimeError, match="maximum context length"):
        wrapper._generate_answer_from_memories(
            question="question",
            memories_text="memory",
            prompt_override="{memories}\n{question}",
        )

    assert attempts == 1


class RecursingManager:
    def invoke(self, payload):
        raise GraphRecursionError("recursion limit")


def _bare_langmem(*, strict):
    adapter = LangMemAdapter.__new__(LangMemAdapter)
    adapter.manager = RecursingManager()
    adapter.strict_comparison = strict
    return adapter


def test_strict_langmem_propagates_graph_recursion_error():
    adapter = _bare_langmem(strict=True)

    with pytest.raises(GraphRecursionError, match="recursion limit"):
        adapter.add_chunk("dialogue")


def test_default_langmem_still_skips_graph_recursion_error(caplog):
    adapter = _bare_langmem(strict=False)

    assert adapter.add_chunk("dialogue") is None
    assert "skipped chunk after graph recursion limit" in caplog.text


def test_mem0_query_token_accounting_accepts_huggingface_tokenizer(monkeypatch):
    wrapper = _bare_wrapper(strict=False)
    wrapper.model = "Qwen/Qwen3.5-9B"
    wrapper.temperature = 0.0
    wrapper.retrieve_num = 2
    wrapper.mem0_source_map = {}
    wrapper.agent_start_time = 0.0
    wrapper.memory = SimpleNamespace(search=lambda **kwargs: [])
    wrapper.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
                )
            )
        )
    )
    wrapper._maybe_save_longmemeval_backfill_debug = lambda **kwargs: None
    monkeypatch.setattr(
        agent_module,
        "get_template",
        lambda *args, **kwargs: "{memories}\n{question}",
    )

    result = wrapper._handle_mem0_agent("question", False, 0, 0)

    assert result["output"] == "answer"


def test_a_mem_query_token_accounting_accepts_huggingface_tokenizer():
    wrapper = _bare_wrapper(strict=False)
    wrapper.retrieve_num = 2
    wrapper.agent_start_time = 0.0
    wrapper.a_mem = SimpleNamespace(
        retrieve_with_source_groups=lambda question: ("memory", []),
        ask_with_retrieved_context=lambda question, context: "answer",
    )
    wrapper._save_retrieval_debug_payload = lambda *args, **kwargs: None

    result = wrapper._handle_a_mem_agent("question", False, 0, 0)

    assert result["output"] == "answer"
