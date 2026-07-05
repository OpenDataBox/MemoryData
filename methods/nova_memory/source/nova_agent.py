"""NovaMemoryAgent — MemoryData-compatible wrapper over NovaMemoryStore.

Drop-in agent class that mimics the subset of AgentWrapper interface used by
MemoryData/main.py:
  - send_message(text, memorizing: bool, ...) -> str
  - (optionally) chunk_size / context / agent_save_to_folder attributes

Usage from MemoryData:
  1. Add `agent_name: Nova_memory_agent` to your YAML config
  2. Patch utils/agent.py to dispatch 'nova' to NovaMemoryAgent
  OR (cleaner):
  3. Use as standalone — see README.md in this directory
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Allow importing this file even if MemoryData's sys.path layout differs.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from nova_core import NovaMemoryStore, tokenize


class NovaMemoryAgent:
    """Lexical + morphology memory agent with LLM answer generation.

    Stores every observed chunk verbatim; on queries, retrieves top-k by
    substring overlap with query tokens (mimics nova-mvp/memory.py SQLite LIKE
    behavior) and asks the LLM to answer using those chunks as context.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        retrieve_num: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        agent_save_to_folder: str = "./results/outputs/nova",
        chunk_size: int = 4096,
        answer_max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.retrieve_num = retrieve_num
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE")
        self.agent_save_to_folder = agent_save_folder = agent_save_to_folder
        self.chunk_size = chunk_size
        self.answer_max_tokens = answer_max_tokens

        self.store = NovaMemoryStore()
        self._seen_ids = set()

        os.makedirs(agent_save_to_folder, exist_ok=True)

        # Lazy-init OpenAI client
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. `pip install openai` to use NovaMemoryAgent."
            ) from e
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    # ---------------- MemoryData-compatible API ----------------

    def memorize_chunk(self, text: str) -> None:
        """Ingest a chunk of context."""
        if not text or not text.strip():
            return
        if text in self._seen_ids:
            return
        self._seen_ids.add(text)
        self.store.memorize([text], keywords=[text])

    def recall_chunks(self, query: str) -> List[str]:
        """Return top-k chunks relevant to query (lexical overlap)."""
        if not query or not query.strip():
            return []
        hits = self.store.recall(query, k=self.retrieve_num)
        return [c for c, _ in hits]

    def send_message(self, text: str, memorizing: bool = False, **_kwargs) -> str:
        """MemoryData main loop calls this.

        memorizing=True  -> ingest text into memory
        memorizing=False -> answer query using retrieved context
        """
        if memorizing:
            self.memorize_chunk(text)
            return ""
        return self.answer_query(text)

    # ---------------- Answer generation ----------------

    ANSWER_SYSTEM = (
        "You are a helpful assistant. Use ONLY the provided context to answer "
        "the user's question. If the answer is not in the context, say you don't know."
    )

    def answer_query(self, query: str) -> str:
        chunks = self.recall_chunks(query)
        context = "\n".join(f"- {c}" for c in chunks) if chunks else "(no relevant context found)"
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.ANSWER_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=self.answer_max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # LLM 不可用时的兜底:返回检索到的 top chunk 原文
            return chunks[0] if chunks else f"[LLM error: {e}] {user_prompt[:200]}"

    # ---------------- Persistence ----------------

    def save(self) -> None:
        """Dump state to disk so MemoryData can resume."""
        import json
        path = Path(self.agent_save_to_folder) / "nova_state.json"
        path.write_text(
            json.dumps(
                {
                    "chunks": self.store._chunks,
                    "hits": self.store._hits,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> bool:
        """Load state from disk. Returns True if loaded successfully."""
        import json
        path = Path(self.agent_save_to_folder) / "nova_state.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.store._chunks = list(data.get("chunks", []))
            self.store._hits = list(data.get("hits", []))
            self.store._match_texts = [
                c + " " + c for c in self.store._chunks
            ]
            self._seen_ids = set(self.store._chunks)
            return True
        except Exception:
            return False