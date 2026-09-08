import logging
import re
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List

from tokenizers import Tokenizer

from src.agent.base import BaseAgent
from src.memory.kv_block_manager.text_block import TextBlock
from src.utils.prompt import MEMORY_AGENT_SYS_PROMPT, SUMMARY_INSTRUCTION

logger = logging.getLogger(__name__)


class _TokenizerAdapter:
    """Expose the small encode interface used by text-mode E-mem."""

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids


@lru_cache(maxsize=None)
def _load_tokenizer(model_id: str):
    """Load tokenizer.json without importing torch; retain the HF fallback."""
    try:
        model_path = Path(model_id)
        if model_path.is_file():
            tokenizer_path = model_path
        elif model_path.is_dir():
            tokenizer_path = model_path / "tokenizer.json"
        else:
            from huggingface_hub import hf_hub_download

            tokenizer_path = Path(
                hf_hub_download(
                    model_id,
                    "tokenizer.json",
                    local_files_only=True,
                )
            )
        return _TokenizerAdapter(Tokenizer.from_file(str(tokenizer_path)))
    except Exception:
        logger.info(
            "Fast tokenizer.json loading failed for %s; falling back to transformers",
            model_id,
            exc_info=True,
        )
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


class _TextMemoryLLM(BaseAgent):
    """Internal LLM wrapper for TextMemoryAgent."""
    def execute_tool(self, tool_name, arguments):
        # TextMemoryAgent doesn't use tools
        pass


class TextMemoryAgent:
    def __init__(self, model_id: str, openai_config: dict, model_context_window: int = 32768,
                 load_from_block_id: str = None, load_timestamp: str = None,
                 block_size_ratio: float = 0.125, summary_max_tokens: int = 8192,
                 query_max_tokens: int = 8192):
        self.model_id = model_id
        self.model_context_window = model_context_window
        self.block_size_ratio=block_size_ratio
        self.block_size = int(model_context_window * block_size_ratio)
        self.summary_max_tokens = summary_max_tokens
        self.query_max_tokens = query_max_tokens
        self.tokenizer = _load_tokenizer(model_id)
        self.llm = _TextMemoryLLM(openai_config, MEMORY_AGENT_SYS_PROMPT)
        self.is_active = True
        self.summary = None
        
        if load_from_block_id and load_timestamp:
            self.current_block = TextBlock(
                block_id=uuid.UUID(load_from_block_id),
                create_timestamp=load_timestamp,
                block_size=self.block_size
            )
            self.current_block.load()
            logger.info(f"Loaded existing text block: {load_from_block_id}")
        else:
            self.current_block = TextBlock(
                block_id=uuid.uuid4(),
                create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                block_size=self.block_size
            )
            logger.info(f"TextMemoryAgent initialized with block_size={self.block_size}")

    def add(self, text_chunks: List[str]) -> bool:
        if not self.is_active:
            raise RuntimeError("Agent is inactive, cannot add new memories.")
        
        for text in text_chunks:
            token_count = len(self.tokenizer.encode(text))
            block_full = self.current_block.add_chunk(text, token_count)
            
            if block_full:
                logger.info(f"Block {self.current_block.block_id} is full")
                self.is_active = False
                self._create_summaries()
                return

    def _create_summaries(self):
        logger.info("Creating summary for text block")
        all_text = self.current_block.get_all_text()
        prompt = f"{all_text}\n\n{SUMMARY_INSTRUCTION}"
        raw_summary = self.llm.generate_response(
            prompt,
            max_tokens=self.summary_max_tokens,
            repetition_penalty=1.1,
        )
        self.summary = self._remove_thinking_content(raw_summary)
        logger.info(f"Summary created (length: {len(self.summary)} chars)")

    def _remove_thinking_content(self, response: str) -> str:
        """Remove reasoning traces from model output for parity with KV mode."""
        cleaned_response = response
        cleaned_response = re.sub(
            r"<thinking>.*?</thinking>",
            "",
            cleaned_response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned_response = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned_response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned_response = re.sub(r"\n\s*\n", "\n\n", cleaned_response)
        return cleaned_response.strip()

    def preload_cache(self):
        """No-op for text storage mode (no cache to preload)."""
        pass

    def query(self, question: str, max_new_tokens: int | None = None) -> str:
        logger.debug(f"Querying text memory: {question[:50]}...")
        all_text = self.current_block.get_all_text()
        if not all_text:
            return "No knowledge available."
        
        prompt = f"{all_text}\n\nBased on the context information provided above, please extract the original information that is relevant to the question (REMEMBER to give EXACT datetime along with information, and the datetime format is 'YYYY-MM-DD HH:MM:SS'):\n{question}"
        raw_response = self.llm.generate_response(
            prompt,
            max_tokens=max_new_tokens or self.query_max_tokens,
            repetition_penalty=1.1,
        )
        return self._remove_thinking_content(raw_response)

    def get_original_texts(self) -> List[str]:
        """
        Get original text chunks for hybrid routing.
        
        Returns:
            List of original text chunks stored in this block.
        """
        return [chunk['text'] for chunk in self.current_block.chunks]

    @property
    def original_texts(self) -> List[str]:
        """Property alias for get_original_texts for compatibility."""
        return self.get_original_texts()
