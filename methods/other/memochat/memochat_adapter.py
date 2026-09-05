"""Repository adapter for the upstream-style MemoChat runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


CURRENT_DIR = Path(__file__).resolve().parent
MEMOCHAT_SOURCE_ROOT = CURRENT_DIR / "source"
if str(MEMOCHAT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(MEMOCHAT_SOURCE_ROOT))

from memochat import MemoChat


class MemoChatAdapter:
    """Thin wrapper that exposes MemoChat through the repository's adapter conventions."""

    def __init__(
        self,
        *,
        state_path: str,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str],
        embedding_model: str,
        embedding_provider: Optional[str],
        embedding_base_url: Optional[str],
        embedding_api_key: Optional[str],
        embedding_dimensions: Optional[int],
        retrieve_num: int,
        summary_trigger_chunks: int,
        keep_recent_chunks: int,
        max_topics_per_window: int,
        summary_chars: int,
        keyword_limit: int,
        topic_top_k: int,
        recent_top_k: int,
        dialogs_per_topic: int,
        llm_max_tokens: int,
        use_llm_topic_segmentation: bool,
    ) -> None:
        self.runtime = MemoChat(
            state_path=state_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_dimensions=embedding_dimensions,
            retrieve_num=retrieve_num,
            summary_trigger_chunks=summary_trigger_chunks,
            keep_recent_chunks=keep_recent_chunks,
            max_topics_per_window=max_topics_per_window,
            summary_chars=summary_chars,
            keyword_limit=keyword_limit,
            topic_top_k=topic_top_k,
            recent_top_k=recent_top_k,
            dialogs_per_topic=dialogs_per_topic,
            llm_max_tokens=llm_max_tokens,
            use_llm_topic_segmentation=use_llm_topic_segmentation,
        )

    def add_chunk(self, content: str, timestamp: Optional[str] = None) -> str:
        return self.runtime.add_chunk(content, timestamp=timestamp)

    def retrieve(self, question: str) -> dict:
        return self.runtime.retrieve(question)

    def save(self) -> None:
        self.runtime.save()

    def load(self) -> None:
        self.runtime.load()

    def memory_count(self) -> int:
        return self.runtime.memory_count()
