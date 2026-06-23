import os
from dataclasses import dataclass
from typing import Any, Optional

from .recurrent import MemAgentRecurrentRunner, RecurrentConfig


@dataclass
class MemAgent7B:
    model: str
    base_url: str
    api_key: str = "123-abc"
    tokenizer_model: Optional[str] = None
    recurrent_chunk_size: int = 5000
    recurrent_max_new: int = 1024
    recurrent_max_context_len: int = 120000
    temperature: float = 0.7
    top_p: float = 0.95

    def __post_init__(self) -> None:
        self.runner = MemAgentRecurrentRunner(
            RecurrentConfig(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                tokenizer_model=self.tokenizer_model,
                recurrent_chunk_size=self.recurrent_chunk_size,
                recurrent_max_new=self.recurrent_max_new,
                recurrent_max_context_len=self.recurrent_max_context_len,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        )

    def answer(self, context: str, query: str, verbose: bool = False) -> str:
        return self.runner.run(context=context, prompt=query, verbose=verbose)


def build_memagent(config: Any) -> MemAgent7B:
    explicit_api_key = config.get("api_key")
    api_key_env = config.get("api_key_env")
    resolved_api_key = explicit_api_key or (os.environ.get(api_key_env) if api_key_env else None) or "123-abc"

    return MemAgent7B(
        model=config["model"],
        base_url=config["base_url"],
        api_key=resolved_api_key,
        tokenizer_model=config.get("tokenizer_model"),
        recurrent_chunk_size=config.get("recurrent_chunk_size", 5000),
        recurrent_max_new=config.get("recurrent_max_new", 1024),
        recurrent_max_context_len=config.get("recurrent_max_context_len", 120000),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.95),
    )
