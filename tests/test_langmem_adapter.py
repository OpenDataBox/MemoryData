import json
from types import SimpleNamespace

from methods.langmem.langmem_adapter import LangMemAdapter
from utils.locomo_utils import build_locomo_storage_text


class FakeStore:
    def __init__(self):
        self.items = {}

    def put(self, namespace, key, value):
        self.items[(namespace, key)] = value

    def delete(self, namespace, key):
        self.items.pop((namespace, key), None)

    def search(self, namespace, *, query=None, limit=10, offset=0):
        del query
        matches = [
            SimpleNamespace(key=key, value=value)
            for (item_namespace, key), value in self.items.items()
            if item_namespace == namespace
        ]
        return matches[offset : offset + limit]


class FakeManager:
    def __init__(self, store, namespace, outputs):
        self.store = store
        self.namespace = namespace
        self.outputs = iter(outputs)
        self.messages = []

    def invoke(self, payload):
        self.messages.append(payload["messages"][0]["content"])
        output = next(self.outputs)
        for key, value in output.get("puts", []):
            self.store.put(self.namespace, key, value)
        for key in output.get("deletes", []):
            self.store.delete(self.namespace, key)
        return [
            {"namespace": self.namespace, "key": key, "value": value}
            for key, value in output.get("puts", [])
        ]


def make_adapter(tmp_path, manager_outputs=()):
    adapter = LangMemAdapter.__new__(LangMemAdapter)
    adapter.state_dir = tmp_path
    adapter.state_path = tmp_path / "langmem_state.json"
    adapter.retrieve_num = 10
    adapter.namespace = ("benchmark", "memories")
    adapter.store = FakeStore()
    adapter.source_ids_by_key = {}
    adapter.manager = FakeManager(adapter.store, adapter.namespace, manager_outputs)
    return adapter


def test_add_chunk_tracks_update_provenance_and_prunes_deleted_keys(tmp_path):
    adapter = make_adapter(
        tmp_path,
        manager_outputs=[
            {"puts": [("profile", {"kind": "Memory", "content": "Lives in Seoul"})]},
            {"puts": [("profile", {"kind": "Memory", "content": "Lives in Busan"})]},
            {"deletes": ["profile"]},
        ],
    )

    adapter.add_chunk(build_locomo_storage_text("Alice moved to Seoul", "c1", ["D1:1"]))
    adapter.add_chunk(build_locomo_storage_text("Alice moved to Busan", "c2", ["D2:3"]))

    assert adapter.manager.messages == ["Alice moved to Seoul", "Alice moved to Busan"]
    assert adapter.source_ids_by_key == {"profile": ["D1:1", "D2:3"]}

    adapter.add_chunk("Forget the address")
    assert adapter.source_ids_by_key == {}


def test_retrieve_and_snapshot_preserve_ranked_source_groups(tmp_path):
    adapter = make_adapter(tmp_path)
    adapter.store.put(
        adapter.namespace,
        "first",
        {"kind": "Memory", "content": "Alice lives in Seoul"},
    )
    adapter.store.put(
        adapter.namespace,
        "second",
        {"kind": "Memory", "content": "Bob lives in Busan"},
    )
    adapter.source_ids_by_key = {
        "first": ["D1:1"],
        "second": ["D2:2", "D2:3"],
    }

    memories, source_groups = adapter.retrieve_with_source_groups("Where do they live?")
    adapter.save()

    assert json.loads(memories[0])["content"] == "Alice lives in Seoul"
    assert source_groups == [["D1:1"], ["D2:2", "D2:3"]]

    restored = make_adapter(tmp_path)
    restored.load()
    restored_memories, restored_groups = restored.retrieve_with_source_groups("Where?")

    assert restored_memories == memories
    assert restored_groups == source_groups
