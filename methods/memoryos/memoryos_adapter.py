"""Repository adapter for the upstream-style MemoryOS runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


CURRENT_DIR = Path(__file__).resolve().parent
MEMORYOS_SOURCE_ROOT = CURRENT_DIR / "source"
if str(MEMORYOS_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(MEMORYOS_SOURCE_ROOT))

from memoryos import MemoryOS


class MemoryOSAdapter:
    """Thin wrapper that exposes MemoryOS through the repository's adapter conventions."""

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
        short_term_capacity: int,
        mid_term_capacity: int,
        queue_capacity: int,
        topic_similarity_threshold: float,
        heat_threshold: float,
        summary_chars: int,
        keyword_limit: int,
        segment_threshold: float,
        page_threshold: float,
        knowledge_threshold: float,
        llm_max_tokens: int,
    ) -> None:
        self.runtime = MemoryOS(
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
            short_term_capacity=short_term_capacity,
            mid_term_capacity=mid_term_capacity,
            queue_capacity=queue_capacity,
            topic_similarity_threshold=topic_similarity_threshold,
            heat_threshold=heat_threshold,
            summary_chars=summary_chars,
            keyword_limit=keyword_limit,
            segment_threshold=segment_threshold,
            page_threshold=page_threshold,
            knowledge_threshold=knowledge_threshold,
            llm_max_tokens=llm_max_tokens,
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
