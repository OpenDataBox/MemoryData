"""Repository adapter for the vendored SimpleMem package source."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional

from utils.locomo_utils import dedupe_preserve_order, parse_locomo_source_ids, strip_locomo_metadata

SIMPLEMEM_SOURCE_ROOT = Path(__file__).resolve().parent / "source"
if str(SIMPLEMEM_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLEMEM_SOURCE_ROOT))

simplemem_config = importlib.import_module("SimpleMem.config")
SimpleMemSystem = importlib.import_module("SimpleMem.main").SimpleMemSystem


class SimpleMemAdapter:
    """Thin wrapper around SimpleMemSystem for repository evaluations."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: Optional[str],
        embedding_api_key: str,
        embedding_base_url: Optional[str],
        db_path: str,
        table_name: str,
        clear_db: bool,
        enable_thinking: bool,
        use_streaming: bool,
        enable_planning: bool,
        enable_reflection: bool,
        max_reflection_rounds: int,
        enable_parallel_processing: bool,
        max_parallel_workers: int,
        enable_parallel_retrieval: bool,
        max_retrieval_workers: int,
        embedding_model: Optional[str],
        embedding_dimension: Optional[int],
        retrieve_limit: Optional[int],
        semantic_top_k: int,
        keyword_top_k: int,
        structured_top_k: int,
        window_size: int,
        overlap_size: int,
    ) -> None:
        self.db_path = db_path
        self.table_name = table_name
        self.retrieve_limit = retrieve_limit
        self._configure_runtime(
            api_key=api_key,
            model=model,
            base_url=base_url,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            db_path=db_path,
            table_name=table_name,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            max_reflection_rounds=max_reflection_rounds,
            enable_parallel_processing=enable_parallel_processing,
            max_parallel_workers=max_parallel_workers,
            enable_parallel_retrieval=enable_parallel_retrieval,
            max_retrieval_workers=max_retrieval_workers,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            semantic_top_k=semantic_top_k,
            keyword_top_k=keyword_top_k,
            structured_top_k=structured_top_k,
            window_size=window_size,
            overlap_size=overlap_size,
        )
        self.system = SimpleMemSystem(
            api_key=api_key,
            model=model,
            base_url=base_url,
            db_path=db_path,
            table_name=table_name,
            clear_db=clear_db,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            max_reflection_rounds=max_reflection_rounds,
            enable_parallel_processing=enable_parallel_processing,
            max_parallel_workers=max_parallel_workers,
            enable_parallel_retrieval=enable_parallel_retrieval,
            max_retrieval_workers=max_retrieval_workers,
        )
        self.entry_source_map = {}
        self._install_locomo_source_tracking()

    def _configure_runtime(
        self,
        *,
        api_key: str,
        model: str,
        base_url: Optional[str],
        embedding_api_key: str,
        embedding_base_url: Optional[str],
        db_path: str,
        table_name: str,
        enable_thinking: bool,
        use_streaming: bool,
        enable_planning: bool,
        enable_reflection: bool,
        max_reflection_rounds: int,
        enable_parallel_processing: bool,
        max_parallel_workers: int,
        enable_parallel_retrieval: bool,
        max_retrieval_workers: int,
        embedding_model: Optional[str],
        embedding_dimension: Optional[int],
        semantic_top_k: int,
        keyword_top_k: int,
        structured_top_k: int,
        window_size: int,
        overlap_size: int,
    ) -> None:
        simplemem_config.OPENAI_API_KEY = api_key
        simplemem_config.OPENAI_BASE_URL = base_url
        simplemem_config.LLM_MODEL = model
        simplemem_config.EMBEDDING_API_KEY = embedding_api_key
        simplemem_config.EMBEDDING_BASE_URL = embedding_base_url
        if embedding_model:
            simplemem_config.EMBEDDING_MODEL = embedding_model
        if embedding_dimension is not None:
            simplemem_config.EMBEDDING_DIMENSION = embedding_dimension
        simplemem_config.LANCEDB_PATH = db_path
        simplemem_config.MEMORY_TABLE_NAME = table_name
        simplemem_config.ENABLE_THINKING = enable_thinking
        simplemem_config.USE_STREAMING = use_streaming
        simplemem_config.ENABLE_PLANNING = enable_planning
        simplemem_config.ENABLE_REFLECTION = enable_reflection
        simplemem_config.MAX_REFLECTION_ROUNDS = max_reflection_rounds
        simplemem_config.ENABLE_PARALLEL_PROCESSING = enable_parallel_processing
        simplemem_config.MAX_PARALLEL_WORKERS = max_parallel_workers
        simplemem_config.ENABLE_PARALLEL_RETRIEVAL = enable_parallel_retrieval
        simplemem_config.MAX_RETRIEVAL_WORKERS = max_retrieval_workers
        simplemem_config.SEMANTIC_TOP_K = semantic_top_k
        simplemem_config.KEYWORD_TOP_K = keyword_top_k
        simplemem_config.STRUCTURED_TOP_K = structured_top_k
        simplemem_config.WINDOW_SIZE = window_size
        simplemem_config.OVERLAP_SIZE = overlap_size

    def add_chunk(self, content: str, timestamp: Optional[str] = None) -> None:
        self.system.add_dialogue(speaker="Benchmark", content=content, timestamp=timestamp)

    def finalize(self) -> None:
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.system.finalize()

    def ask(self, question: str) -> str:
        return self.system.ask(question)

    def retrieve_entries(self, question: str) -> list[dict]:
        entries = self.system.hybrid_retriever.retrieve(question)
        if self.retrieve_limit is not None and self.retrieve_limit > 0:
            entries = entries[: self.retrieve_limit]
        return [
            {
                "entry_id": entry.entry_id,
                "text": entry.lossless_restatement,
                "source_ids": self.entry_source_map.get(entry.entry_id, []),
            }
            for entry in entries
        ]

    def memory_count(self) -> int:
        return len(self.system.get_all_memories())

    def _install_locomo_source_tracking(self) -> None:
        original_generate_memory_entries = self.system.memory_builder._generate_memory_entries

        def _copy_dialogue(dialogue):
            if hasattr(dialogue, "model_copy"):
                return dialogue.model_copy(deep=True)
            return dialogue.copy(deep=True)

        def wrapped_generate_memory_entries(dialogues, *args, **kwargs):
            normalized_source_ids = []
            sanitized_dialogues = []

            for dialogue in dialogues:
                copied_dialogue = _copy_dialogue(dialogue)
                source_ids = parse_locomo_source_ids(copied_dialogue.content)
                if source_ids:
                    normalized_source_ids.extend(source_ids)
                    copied_dialogue.content = strip_locomo_metadata(copied_dialogue.content)
                sanitized_dialogues.append(copied_dialogue)

            entries = original_generate_memory_entries(
                sanitized_dialogues,
                *args,
                **kwargs,
            )
            tracked_source_ids = dedupe_preserve_order(normalized_source_ids)
            if tracked_source_ids:
                for entry in entries:
                    self.entry_source_map[entry.entry_id] = tracked_source_ids
            return entries

        self.system.memory_builder._generate_memory_entries = wrapped_generate_memory_entries
