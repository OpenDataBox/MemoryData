from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

RetrieverMode = Literal[
    "session_level",
    "turn_level",
    "summary_level",
    "keyword_level",
    "hybrid_level",
    "lite",
    "memgas",
]

EmbedderName = Literal["contriever", "mpnet", "minilm", "qaminilm"]
LLMProvider = Literal["openai", "vllm"]


@dataclass
class MemoryConfig:
    storage_dir: str = "./memgas_store"
    embedder: EmbedderName = "contriever"
    device: Optional[str] = None
    batch_size: int = 64

    llm_model: str = "gpt-4o-mini"
    llm_provider: LLMProvider = "openai"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_max_tokens: int = 500
    llm_temperature: float = 0.0
    llm_max_retries: int = 3
    llm_retry_wait_sec: float = 2.0

    default_mode: RetrieverMode = "memgas"
    mem_threshold: int = 30
    n_components: int = 2
    num_seednodes: int = 15
    damping: float = 0.1
    router_temp: float = 0.2

    auto_save: bool = True

    def resolve_api_key(self) -> str:
        if self.llm_provider == "vllm":
            # vLLM OpenAI-compatible server usually accepts any non-empty key.
            return self.llm_api_key or os.getenv("VLLM_API_KEY") or "EMPTY"
        api_key = self.llm_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing API key. Set MemoryConfig.llm_api_key or OPENAI_API_KEY."
            )
        return api_key

    def resolve_base_url(self) -> Optional[str]:
        if self.llm_provider == "vllm":
            return self.llm_base_url or os.getenv("VLLM_BASE_URL") or "http://localhost:8000/v1"
        return self.llm_base_url or os.getenv("OPENAI_BASE_URL")
