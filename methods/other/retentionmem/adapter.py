"""MemoryData adapter. It never receives benchmark answers during memorization."""
from __future__ import annotations

import json
from pathlib import Path

from .config import build_runtime, endpoint
from .types import Event


class RetentionMemAdapter:
    def __init__(self, agent_config, directory):
        config = agent_config.get("retentionmem")
        if not isinstance(config, dict):
            raise ValueError("Missing retentionmem runtime configuration")
        self.config = config
        self.directory = Path(directory)
        self.memory = build_runtime(config, self.directory)
        self.reader = endpoint(config, "reader" if "reader" in config else "llm")
        self.query_budget = config.get("query_budget", 4096)
        self._ingest_index = 0

    def add_chunk(self, text, source_ids=()):
        event = Event(f"chunk-{self._ingest_index}", text, actor="conversation", source_ids=tuple(source_ids))
        self.memory.observe(event)
        self._ingest_index += 1

    def answer(self, question, question_date="", max_tokens=128):
        retrieved = self.memory.retrieve(question, self.query_budget)
        context = "\n\n".join(item.text for item in retrieved)
        content = f"Memory:\n{context}\n\nQuestion: {question}"
        if question_date:
            content += "\nQuestion date: " + question_date
        answer = self.reader.chat([
            {"role": "system", "content": "Answer using only the supplied memory. Respect temporal qualifiers and updated facts. If unsupported, say that you do not know. Keep the answer concise."},
            {"role": "user", "content": content}], max_tokens=max_tokens, ledger=self.memory.store, stage="answer")
        return answer, retrieved, context

    def save(self):
        self.memory.finalize()
        self.memory.store.db.commit()
        marker = self.directory/"retentionmem_ready.json"
        marker.write_text(json.dumps({"version": 1, "stats": self.memory.stats()}, indent=2), encoding="utf-8")

    def load(self):
        if not (self.directory/"retentionmem_ready.json").is_file():
            raise FileNotFoundError("Memorization did not complete; rerun ingestion before querying")
        # build_runtime already opens the durable, configuration-checked store.
        self._ingest_index = max((int(e.id.split("-")[1])+1 for e in self.memory.store.events()
                                 if e.id.startswith("chunk-") and e.id.split("-")[1].isdigit()), default=0)
