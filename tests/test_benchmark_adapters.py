import json
import os
import types

import pytest

from methods.e_mem.e_mem_adapter import EMemAdapter
from methods.langmem.langmem_adapter import LangMemAdapter


def test_langmem_snapshot_keeps_only_live_memories(tmp_path):
    from langgraph.store.memory import InMemoryStore
    adapter = LangMemAdapter.__new__(LangMemAdapter)
    adapter.state_dir = tmp_path
    adapter.state_path = tmp_path / "langmem_state.json"
    adapter.namespace = ("benchmark", "memories")
    adapter.store = InMemoryStore()
    adapter.store.put(adapter.namespace, "old", {"content": "Busan"})
    adapter.store.put(adapter.namespace, "new", {"content": "Seoul"})
    adapter.store.delete(adapter.namespace, "old")
    adapter.save()
    adapter.store = InMemoryStore()
    adapter.load()
    assert [x.value["content"] for x in adapter.store.search(adapter.namespace)] == ["Seoul"]


def test_emem_histories_are_isolated_and_environment_is_restored(tmp_path, monkeypatch):
    def factory(**kwargs):
        class Handler:
            def __init__(self):
                self.items = []
                path = tmp_path / os.environ["TEXT_DATA_DIR"] / "items.json"
                if not kwargs["clean_cache_first"]:
                    self.items = json.loads(path.read_text())
            def add_memory(self, text):
                self.items.append(text)
            def _save_metadata(self):
                from pathlib import Path
                path = Path(os.environ["TEXT_DATA_DIR"]) / "items.json"
                path.write_text(json.dumps(self.items))
            def query_memory(self, query):
                return "|".join(self.items)
        return types.SimpleNamespace(
            memory_handler=Handler(),
            _aggregate_memory_results=lambda query, result: result,
        )

    fake = types.SimpleNamespace(create_chat_manager=factory)
    monkeypatch.delitem(__import__("sys").modules, "src", raising=False)
    monkeypatch.setattr("methods.e_mem.e_mem_adapter.importlib.import_module", lambda name: fake)
    monkeypatch.setenv("TEXT_DATA_DIR", "original")
    config = {"model": "test", "api_key": "test", "base_url": "http://localhost"}
    first = EMemAdapter(**config, state_dir=tmp_path / "first")
    second = EMemAdapter(**config, state_dir=tmp_path / "second")
    first.add_chunk("Alice in Seoul")
    second.add_chunk("Bob in Busan")
    first.save()
    second.save()
    restored = EMemAdapter(**config, state_dir=tmp_path / "first")
    restored.load()
    assert restored.retrieve("Where?") == ["Alice in Seoul"]
    assert second.retrieve("Where?") == ["Bob in Busan"]
    assert os.environ["TEXT_DATA_DIR"] == "original"
    with pytest.raises(RuntimeError), first._storage_scope():
        raise RuntimeError("upstream failure")
    assert os.environ["TEXT_DATA_DIR"] == "original"


@pytest.mark.parametrize("kind", ["langmem", "e_mem"])
@pytest.mark.parametrize("dataset_name", ["locomo", "longmemeval"])
def test_new_baselines_build_query_and_reload_through_runner(tmp_path, monkeypatch, kind, dataset_name):
    from utils.agent import AgentWrapper
    from utils.initialization import initialize_and_memorize_agent
    from utils.provider_utils import ApproximateTokenizer

    monkeypatch.setattr(
        AgentWrapper,
        '_initialize_tokenizer',
        lambda self: ApproximateTokenizer(),
    )

    class Memory:
        writes = 0
        def __init__(self, state_dir, **kwargs):
            self.path = __import__("pathlib").Path(state_dir) / "state.json"
            self.items = []
        def add_chunk(self, text, **kwargs):
            del kwargs
            self.items.append(text)
            Memory.writes += 1
        def retrieve(self, query):
            return self.items
        def retrieve_with_source_groups(self, query):
            return self.retrieve(query), []
        def save(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.items))
        def load(self):
            self.items = json.loads(self.path.read_text())

    target = ("methods.langmem.langmem_adapter.LangMemAdapter" if kind == "langmem"
              else "methods.e_mem.e_mem_adapter.EMemAdapter")
    monkeypatch.setattr(target, Memory)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    config = {
        "agent_name": f"Agentic_memory_{kind}", "model": "test",
        "provider": "openai_compatible", "base_url_env": "OPENAI_BASE_URL",
        "input_length_limit": 32768, "buffer_length": 1000,
        "output_dir": str(tmp_path), "tokenizer_encoding": "cl100k_base",
        "retrieve_num": 5, "agent_chunk_size": 4096,
    }
    dataset = {"dataset": dataset_name, "sub_dataset": "locomo_qa" if dataset_name == "locomo"
               else "longmemeval_s_official", "context_max_length": 10000000,
               "generation_max_length": 256, "chunk_size": 4096}
    monkeypatch.setattr(AgentWrapper, "_generate_answer_from_memories",
                        lambda self, question, memories_text: (memories_text, 5, 2))
    state = str(tmp_path / kind)
    _agent, _ = initialize_and_memorize_agent(
        config, dataset, state, ["user: Alice lives in Seoul."], 0, 1)
    loaded, _ = initialize_and_memorize_agent(
        config, dataset, state, ["THIS MUST NOT BE MEMORIZED AGAIN"], 0, 1)
    result = loaded.send_message("Where does Alice live?", query_id=0, context_id=0)
    assert "Seoul" in result["output"]
    assert Memory.writes == 1

