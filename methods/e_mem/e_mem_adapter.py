"""Run upstream E-mem text memory with isolated state for each history."""
import importlib
import os
import sys
import threading
import types
from contextlib import contextmanager
from pathlib import Path

_STATE_LOCK = threading.RLock()


class EMemAdapter:
    def __init__(self, *, model, api_key, base_url, state_dir,
                 tokenizer_model="Qwen/Qwen3-4B", memory_model=None,
                 embedding_model="text-embedding-3-small",
                 embedding_api_key=None, embedding_base_url=None,
                 retrieve_num=5, context_window=32768, block_size_ratio=0.125,
                 overlap_ratio=0.1, summary_max_tokens=8192,
                 query_max_tokens=8192, aggregation_max_tokens=2048,
                 parallel_queries=True):
        self.state_dir = Path(state_dir).resolve()
        self.ready_path = self.state_dir / "e_mem_ready.txt"
        self.data_dir = self.state_dir / "e_mem_text"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        upstream = Path(__file__).resolve().parents[1] / "e-mem" / "src"
        with self._storage_scope():
            existing = sys.modules.get("src")
            if existing is not None and str(upstream) not in list(getattr(existing, "__path__", [])):
                raise RuntimeError("E-mem's 'src' package conflicts with another runtime. Run baselines in separate processes.")
            if existing is None:
                package = types.ModuleType("src")
                package.__path__ = [str(upstream)]
                sys.modules["src"] = package
            factory = importlib.import_module("src.conversation_manager.factory")
            api_config = {"model": model, "api_key": api_key}
            if base_url:
                api_config["base_url"] = base_url
            memory_config = {**api_config, "model": memory_model or model}
            self.manager = factory.create_chat_manager(
                storage_mode="text", model_id=tokenizer_model,
                chat_openai_config=api_config, aggregator_openai_config=api_config,
                memory_agent_openai_config=memory_config, router_openai_config=api_config,
                clean_cache_first=not self.ready_path.exists(),
                model_context_window=context_window, block_size_ratio=block_size_ratio,
                overlap_ratio=overlap_ratio, max_blocks=retrieve_num,
                summary_max_tokens=summary_max_tokens,
                query_max_tokens=query_max_tokens,
                enable_router=True, router_type="hybrid",
                hybrid_router_config={
                    "embedding_provider": "openai", "embedding_model": embedding_model,
                    "embedding_config": {
                        "api_key": embedding_api_key or api_key,
                        "base_url": embedding_base_url or base_url,
                    },
                },
            )
            self.manager.aggregator_max_tokens = aggregation_max_tokens
            self.manager.memory_handler.parallel_queries = parallel_queries

    @contextmanager
    def _storage_scope(self):
        # Upstream uses a process-global variable for metadata. Serialize access
        # and restore it, including while its internal query threads are active.
        with _STATE_LOCK:
            previous = os.environ.get("TEXT_DATA_DIR")
            os.environ["TEXT_DATA_DIR"] = str(self.data_dir)
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop("TEXT_DATA_DIR", None)
                else:
                    os.environ["TEXT_DATA_DIR"] = previous

    def add_chunk(self, text):
        with self._storage_scope():
            # Call the handler directly so storage errors propagate to the runner.
            self.manager.memory_handler.add_memory(text)

    def retrieve(self, query):
        with self._storage_scope():
            raw = self.manager.memory_handler.query_memory(query)
            evidence = self.manager._aggregate_memory_results(query, raw)
            return [evidence]

    def save(self):
        with self._storage_scope():
            self.manager.memory_handler._save_metadata()
            self.ready_path.write_text("ready", encoding="utf-8")

    def load(self):
        # Upstream restores blocks, summaries and the active block in __init__.
        if not self.ready_path.is_file():
            raise FileNotFoundError(self.ready_path)

