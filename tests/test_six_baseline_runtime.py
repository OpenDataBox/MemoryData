"""Real SDK construction and common dataset paths, without model calls."""
from types import SimpleNamespace

import pytest
import yaml

from scripts.run_six_baselines import DATASETS, METHODS, ROOT, build_agent_config


def args():
    return SimpleNamespace(model="test-model", base_url="http://127.0.0.1:1/v1",
                           embedding_model="organization/test-embedding", embedding_dim=8,
                           embedding_base_url="http://127.0.0.1:2/v1")


@pytest.mark.parametrize("method", METHODS)
def test_native_sdk_constructor_uses_selected_settings(method, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("MEM0_TELEMETRY", "false")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in ("LIGHTMEM_MODEL", "LIGHTMEM_BASE_URL", "LIGHTMEM_EMBEDDING_MODEL",
                 "LIGHTMEM_EMBEDDING_DIMENSION"):
        monkeypatch.delenv(name, raising=False)
    from utils.agent import AgentWrapper
    config = build_agent_config(method, args())
    config["output_dir"] = str(tmp_path / "outputs")
    dataset = yaml.safe_load((ROOT / DATASETS["longmemeval"]).read_text(encoding="utf-8-sig"))
    agent = AgentWrapper(config, dataset, str(tmp_path / method))
    if method == "mem0":
        assert agent.memory.embedding_model.config.model == args().embedding_model
        assert agent.memory.embedding_model.config.embedding_dims == 8
        assert agent.memory.config.vector_store.config.embedding_model_dims == 8
        agent.memory.vector_store.client.close()
    if method == "e_mem":
        import tiktoken
        from src.agent.base import BaseAgent
        from transformers import AutoTokenizer

        from methods.e_mem.e_mem_adapter import EMemAdapter
        monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                            lambda *a, **kw: tiktoken.get_encoding("cl100k_base"))
        monkeypatch.setattr(BaseAgent, "generate_response", lambda *a, **kw: "Alice lives in Seoul.")
        agent.benchmark_memory.add_chunk("user: Alice lives in Seoul.")
        agent.benchmark_memory.save()
        restored = EMemAdapter(model="test-model", api_key="test",
                               base_url=args().base_url, state_dir=tmp_path / method)
        restored.load()
        monkeypatch.setattr(restored.manager.aggregator_llm.chat.completions, "create",
                            lambda **kw: SimpleNamespace(choices=[
                                SimpleNamespace(message=SimpleNamespace(content="Alice lives in Seoul."))]))
        assert "Seoul" in restored.retrieve("Where does Alice live?")[0]
    if method == "lightmem":
        parsed = agent.lightmem._parse_chunk_messages(
            "Session abc_123 (2023/05/20 (Sat) 12:00)\nuser: Where?\nassistant: Seoul.", None)
        selected = agent.lightmem._select_messages_for_direct_ingest(parsed["messages"])
        assert [entry["role"] for entry in selected] == ["user", "assistant"]
        assert all(entry["time_stamp"] == "2023/05/20 (Sat) 12:00" for entry in selected)
        assert agent.lightmem._parse_timestamp(selected[-1]["time_stamp"]).year == 2023


@pytest.mark.parametrize("dataset", DATASETS)
def test_official_dataset_reaches_common_conversation_creator(dataset):
    from utils.conversation_creator import ConversationCreator
    config = yaml.safe_load((ROOT / DATASETS[dataset]).read_text(encoding="utf-8-sig"))
    path = ROOT / config["test_files"]
    if not path.is_file():
        pytest.skip("Download official datasets to run this integration check")
    config.update(test_files=str(path), max_test_samples=1)
    for method in METHODS:
        creator = ConversationCreator(build_agent_config(method, args()), config)
        assert creator.get_chunks()[0][0]["text"]
        question, answer, qid, metadata = creator.get_query_and_answers()[0][0]
        assert question and answer and qid and metadata
        if dataset == "longmemeval":
            assert metadata["question_date"] in question


def test_answer_uses_benchmark_date_even_after_context_retry(monkeypatch):
    from utils.agent import AgentWrapper
    agent = AgentWrapper.__new__(AgentWrapper)
    agent._current_eval_metadata = {"question_date": "2023/05/20 (Sat) 12:00:00"}
    agent.agent_config = {}
    agent.agent_name = "Agentic_memory_langmem"
    agent.sub_dataset = "longmemeval_s_official"
    prompts = []
    def complete(messages):
        prompts.append(messages[1]["content"])
        if len(prompts) == 1:
            raise RuntimeError("maximum context length")
        return "Seoul", 10, 2
    monkeypatch.setattr(agent, "_request_openai_completion", complete)
    monkeypatch.setattr(agent, "_count_tokens", lambda text: 2000)
    monkeypatch.setattr(agent, "_truncate_text_to_token_limit", lambda text, limit: "Seoul")
    assert agent._generate_answer_from_memories("Where?", "Seoul" * 2000)[0] == "Seoul"
    assert len(prompts) == 2
    assert all("Current Time: 2023/05/20 (Sat) 12:00:00" in prompt for prompt in prompts)


def test_process_context_closes_agent_after_queries(monkeypatch):
    import main as runner

    closed = []
    agent = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(runner, "generate_agent_save_folder", lambda *args: "agent")
    monkeypatch.setattr(
        runner,
        "initialize_and_memorize_agent",
        lambda *args, **kwargs: (agent, {}),
    )
    monkeypatch.setattr(
        runner,
        "process_queries_for_context",
        lambda *args, **kwargs: ({}, [], 1),
    )

    runner.process_context(
        0, ["chunk"], [("question", "answer", "id")], {}, {}, {}, [], 0,
        set(), set(), 1, "output.json", [], 0.0, False, 1,
    )

    assert closed == [True]


def test_batch_failure_flag_reports_recorded_query_failure(monkeypatch):
    import main as runner
    flags = SimpleNamespace(backfill_longmemeval_recall_debug=False, force=False,
                            retry_failed_queries=True, max_test_queries_ablation=1,
                            fail_on_query_error=True)
    monkeypatch.setattr(runner, "parse_command_line_arguments", lambda: flags)
    monkeypatch.setattr(runner, "setup_configs_and_directories", lambda args: ({}, {}, "unused.json"))
    monkeypatch.setattr(runner, "create_agent_and_fetch_data", lambda *a: (0, [["history"]], [["question"]]))
    monkeypatch.setattr(runner, "initialize_progress_tracking", lambda *a: ({}, [], set(), set()))
    monkeypatch.setattr(runner, "process_context", lambda *a: ({}, [{"status": "failed"}], 1, False))
    monkeypatch.setattr(runner, "_maybe_generate_post_run_reports", lambda *a: None)
    with pytest.raises(SystemExit, match="queries failed"):
        runner.main()
