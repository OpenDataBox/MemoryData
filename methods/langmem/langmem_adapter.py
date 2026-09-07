"""LangMem memory extraction and retrieval with per-context JSON snapshots."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from langgraph.errors import GraphRecursionError
from utils.locomo_utils import (
    dedupe_preserve_order,
    parse_locomo_source_ids,
    strip_locomo_metadata,
)

logger = logging.getLogger(__name__)

_LANGMEM_SOURCE = Path(__file__).resolve().parent / "src"
if str(_LANGMEM_SOURCE) not in sys.path:
    sys.path.insert(0, str(_LANGMEM_SOURCE))


class LangMemAdapter:
    """Adapt LangMem's memory manager to the benchmark agent lifecycle."""

    _SNAPSHOT_VERSION = 1

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None,
        embedding_model: str,
        embedding_api_key: str,
        embedding_base_url: str | None,
        embedding_dims: int,
        state_dir: str | Path,
        retrieve_num: int = 10,
        query_limit: int = 5,
        strict_comparison: bool = False,
    ) -> None:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langgraph.store.memory import InMemoryStore

        from langmem import create_memory_store_manager

        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / "langmem_state.json"
        self.retrieve_num = int(retrieve_num)
        self.strict_comparison = bool(strict_comparison)
        self.namespace = ("benchmark", "memories")
        self.source_ids_by_key: dict[str, list[str]] = {}

        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
            check_embedding_ctx_length=False,
        )
        self.store = InMemoryStore(
            index={
                "dims": int(embedding_dims),
                "embed": embeddings,
                "fields": ["content"],
            }
        )
        memory_model = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
        self.manager = create_memory_store_manager(
            memory_model,
            namespace=self.namespace,
            store=self.store,
            enable_inserts=True,
            enable_deletes=True,
            query_limit=int(query_limit),
        )

    def _live_items(self) -> list[Any]:
        items = []
        offset = 0
        while True:
            page = self.store.search(self.namespace, limit=100, offset=offset)
            items.extend(page)
            if len(page) < 100:
                break
            offset += len(page)
        return items

    def _prune_source_ids(self) -> None:
        live_keys = {str(item.key) for item in self._live_items()}
        self.source_ids_by_key = {
            key: source_ids
            for key, source_ids in getattr(self, "source_ids_by_key", {}).items()
            if key in live_keys
        }

    def add_chunk(self, text: str, source_ids: list[str] | None = None) -> None:
        """Extract memories from one benchmark chunk and retain its provenance."""

        chunk_source_ids = dedupe_preserve_order(
            list(source_ids or []) + parse_locomo_source_ids(text)
        )
        clean_text = strip_locomo_metadata(text)
        try:
            final_puts = self.manager.invoke(
                {"messages": [{"role": "user", "content": clean_text}]}
            )
        except GraphRecursionError as exc:
            if self.strict_comparison:
                raise
            logger.warning(
                "LangMem skipped chunk after graph recursion limit; source_ids=%s: %s",
                chunk_source_ids,
                exc,
            )
            return

        for put in final_puts:
            key = str(put["key"])
            self.source_ids_by_key[key] = dedupe_preserve_order(
                self.source_ids_by_key.get(key, []) + chunk_source_ids
            )
        self._prune_source_ids()

    def retrieve_with_source_groups(
        self, query: str
    ) -> tuple[list[str], list[list[str]]]:
        """Return ranked memory text and matching LoCoMo source-ID groups."""

        memories = []
        source_groups = []
        for item in self.store.search(
            self.namespace,
            query=query,
            limit=self.retrieve_num,
        ):
            memories.append(json.dumps(item.value, ensure_ascii=False))
            source_ids = self.source_ids_by_key.get(str(item.key), [])
            if source_ids:
                source_groups.append(list(source_ids))
        return memories, source_groups

    def retrieve(self, query: str) -> list[str]:
        memories, _ = self.retrieve_with_source_groups(query)
        return memories

    def save(self) -> None:
        """Atomically persist live memories and their benchmark provenance."""

        live_items = sorted(self._live_items(), key=lambda item: str(item.key))
        live_keys = {str(item.key) for item in live_items}
        payload = {
            "version": self._SNAPSHOT_VERSION,
            "memories": [
                {"key": str(item.key), "value": item.value} for item in live_items
            ],
            "source_ids_by_key": {
                key: source_ids
                for key, source_ids in sorted(
                    getattr(self, "source_ids_by_key", {}).items()
                )
                if key in live_keys
            },
        }

        self.state_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def load(self) -> None:
        """Restore a snapshot, including snapshots from the earlier list format."""

        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            memory_records = payload
            source_ids_by_key = {}
        elif isinstance(payload, dict):
            memory_records = payload.get("memories", [])
            source_ids_by_key = payload.get("source_ids_by_key", {})
        else:
            raise ValueError(f"Invalid LangMem snapshot at {self.state_path}")

        for item in memory_records:
            self.store.put(self.namespace, item["key"], item["value"])

        self.source_ids_by_key = {
            str(key): dedupe_preserve_order(source_ids)
            for key, source_ids in source_ids_by_key.items()
        }
        self._prune_source_ids()
