"""Chat handlers for KV cache and text storage modes."""

import logging
from typing import Any, Dict, Optional

from src.conversation_manager.base_chat_manager import BaseChatManager
from src.memory.core.text_loop_handler import TextMemoryHandler
from src.utils.prompt import CHAT_SYS_PROMPT

logger = logging.getLogger(__name__)


class ChatManager(BaseChatManager):
    """
    Chat manager using KV cache for memory storage.

    This implementation uses GPU-based KV cache for efficient memory storage
    and retrieval. Uses a shared model architecture to minimize GPU memory usage.

    Args:
        model_id: HuggingFace model ID or local path.
        openai_config: OpenAI API configuration for router and aggregation.
        system_prompt: System prompt for the chat agent.
        clean_cache_first: Whether to clean cache on initialization.
        model_context_window: Model's context window size.
        attn_implementation: Attention implementation ('sdpa', 'flash_attention_2', 'eager').
        device_map: Device mapping strategy.
        router_system_prompt: Custom router system prompt.
        quantization_config: Model quantization configuration.
        max_memory: Max memory per GPU device.
        offload_folder: Folder for offloading weights.
        overlap_mode: Overlap handling strategy ('chunk' or 'token').
        overlap_ratio: Overlap ratio between blocks (0.0-1.0).
        block_size_ratio: Block size relative to context window (0.0-1.0).
        max_memory_segments: Maximum memory segments to return per block query.
        max_blocks: Maximum number of memory blocks to select by router.
        query_batch_size: Queries to batch together for inference.
        max_parallel_cache_loads: Maximum parallel KV cache loads to GPU.
        enable_router: If False, query ALL blocks without LLM routing.
        router_type: Router type ('llm' or 'hybrid'). Default is 'hybrid'.
        hybrid_router_config: Configuration for hybrid router (embedding, BM25 settings).
    """

    def __init__(
        self,
        model_id: str,
        chat_openai_config: Dict[str, Any],
        aggregator_openai_config: Dict[str, Any],
        router_openai_config: Optional[Dict[str, Any]] = None,
        system_prompt: str = CHAT_SYS_PROMPT,
        clean_cache_first: bool = True,
        model_context_window: int = 32768,
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
        router_system_prompt: Optional[str] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        max_memory: Optional[Dict[str, str]] = None,
        offload_folder: Optional[str] = None,
        overlap_mode: str = "chunk",
        overlap_ratio: float = 0.1,
        block_size_ratio: float = 0.125,
        max_memory_segments: Optional[int] = None,
        max_blocks: int = 5,
        query_batch_size: int = 4,
        max_parallel_cache_loads: int = 8,
        enable_router: bool = True,
        router_type: str = "hybrid",
        hybrid_router_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            chat_openai_config,
            aggregator_openai_config=aggregator_openai_config,
            system_prompt=system_prompt,
        )
        self._name = "chat_manager"

        logger.info(f"Initializing ChatManager with model: {model_id}")

        # Validate block_size_ratio
        if not 0 < block_size_ratio <= 1:
            raise ValueError("block_size_ratio must be between 0 and 1")

        # Build memory handler kwargs
        memory_kwargs: Dict[str, Any] = {
            "model_id": model_id,
            "openai_config": router_openai_config,
            "clean_cache_first": clean_cache_first,
            "model_context_window": model_context_window,
            "attn_implementation": attn_implementation,
            "device_map": device_map,
            "overlap_ratio": overlap_ratio,
            "overlap_mode": overlap_mode,
            "block_size_ratio": block_size_ratio,
            "max_memory_segments": max_memory_segments,
            "max_blocks": max_blocks,
            "query_batch_size": query_batch_size,
            "max_parallel_cache_loads": max_parallel_cache_loads,
            "enable_router": enable_router,
            "router_type": router_type,
            "hybrid_router_config": hybrid_router_config,
        }

        if router_system_prompt is not None:
            memory_kwargs["router_system_prompt"] = router_system_prompt
        if quantization_config is not None:
            memory_kwargs["quantization_config"] = quantization_config
        if max_memory is not None:
            memory_kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            memory_kwargs["offload_folder"] = offload_folder

        # KV mode imports transformers/torch. Keep it out of the text-only path,
        # which only needs an API client and a tokenizer for token counting.
        from src.memory.core.loop_handler import MemoryHandler

        self._memory_handler = MemoryHandler(**memory_kwargs)

    @property
    def name(self) -> str:
        """Return the manager name."""
        return self._name

    @property
    def memory_handler(self) -> "MemoryHandler":
        """Return the memory handler instance."""
        return self._memory_handler


class TextStorageChatManager(BaseChatManager):
    """
    Chat manager using text storage for memory.

    This implementation uses text-based storage with LLM API calls for
    memory retrieval. Does not require GPU.

    Args:
        model_id: HuggingFace model ID (used for tokenization).
        openai_config: OpenAI API configuration for all LLM operations.
        system_prompt: System prompt for the chat agent.
        clean_cache_first: Whether to clean cache on initialization.
        model_context_window: Model's context window size.
        router_system_prompt: Custom router system prompt.
        overlap_mode: Overlap handling strategy ('chunk' or 'token').
        overlap_ratio: Overlap ratio between blocks (0.0-1.0).
        block_size_ratio: Block size relative to context window (0.0-1.0).
        max_memory_segments: Maximum memory segments to return per block query.
        max_blocks: Maximum number of memory blocks to select by router.
        enable_router: If False, query ALL blocks without LLM routing.
        router_type: Router type ('llm' or 'hybrid'). Default is 'hybrid'.
        hybrid_router_config: Configuration for hybrid router (embedding, BM25 settings).
    """

    def __init__(
        self,
        model_id: str,
        chat_openai_config: Dict[str, Any],
        aggregator_openai_config: Dict[str, Any],
        memory_agent_openai_config: Dict[str, Any],
        router_openai_config: Optional[Dict[str, Any]] = None,
        system_prompt: str = CHAT_SYS_PROMPT,
        clean_cache_first: bool = True,
        model_context_window: int = 32768,
        router_system_prompt: Optional[str] = None,
        overlap_mode: str = "chunk",
        overlap_ratio: float = 0.1,
        block_size_ratio: float = 0.125,
        max_memory_segments: Optional[int] = None,
        max_blocks: int = 5,
        summary_max_tokens: int = 8192,
        query_max_tokens: int = 8192,
        enable_router: bool = True,
        router_type: str = "hybrid",
        hybrid_router_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            chat_openai_config,
            aggregator_openai_config=aggregator_openai_config,
            system_prompt=system_prompt,
        )
        self._name = "text_chat_manager"

        logger.info(f"Initializing TextStorageChatManager with model: {model_id}")

        # Validate block_size_ratio
        if not 0 < block_size_ratio <= 1:
            raise ValueError("block_size_ratio must be between 0 and 1")

        memory_kwargs: Dict[str, Any] = {
            "model_id": model_id,
            "openai_config": memory_agent_openai_config,
            "router_openai_config": router_openai_config,
            "clean_cache_first": clean_cache_first,
            "model_context_window": model_context_window,
            "overlap_ratio": overlap_ratio,
            "overlap_mode": overlap_mode,
            "block_size_ratio": block_size_ratio,
            "max_memory_segments": max_memory_segments,
            "max_blocks": max_blocks,
            "summary_max_tokens": summary_max_tokens,
            "query_max_tokens": query_max_tokens,
            "enable_router": enable_router,
            "router_type": router_type,
            "hybrid_router_config": hybrid_router_config,
        }

        if router_system_prompt is not None:
            memory_kwargs["router_system_prompt"] = router_system_prompt

        self._memory_handler = TextMemoryHandler(**memory_kwargs)

    @property
    def name(self) -> str:
        """Return the manager name."""
        return self._name

    @property
    def memory_handler(self) -> TextMemoryHandler:
        """Return the memory handler instance."""
        return self._memory_handler
