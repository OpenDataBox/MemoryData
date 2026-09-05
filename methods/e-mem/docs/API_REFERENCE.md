# API Reference

This reference documents the core Python API for E-mem, focusing on initialization and configuration.

If you want a conceptual explanation of which config model field controls which runtime stage, see [Config Model Roles](CONFIG_MODELS.md). If you want the rest of the YAML fields explained, see [Config Reference](CONFIG_REFERENCE.md).

## Initialization

### `create_chat_manager`

The primary entry point for initializing the system. Supports both KV Cache (GPU) and Text (CPU/API) storage modes.

```python
def create_chat_manager(
    storage_mode: Literal["kv_cache", "text"] = "kv_cache",
    model_id: str = "Qwen/Qwen3-4B",
    chat_openai_config: Optional[Dict[str, Any]] = None,
    aggregator_openai_config: Optional[Dict[str, Any]] = None,
    memory_agent_openai_config: Optional[Dict[str, Any]] = None,
    router_openai_config: Optional[Dict[str, Any]] = None,
    clean_cache_first: bool = True,
    model_context_window: int = 32768,
    router_system_prompt: Optional[str] = None,
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
    **kwargs: Any,
) -> Union[ChatManager, TextStorageChatManager]
```

#### Key Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `storage_mode` | `str` | `"kv_cache"` | `"kv_cache"` for GPU tensor storage; `"text"` for JSON/API storage. |
| `model_id` | `str` | `"Qwen/Qwen3-4B"` | HuggingFace model ID or local path. Required for tokenization in both modes. |
| `chat_openai_config` | `dict` | `None` | Top-level manager/tool-calling model config. |
| `aggregator_openai_config` | `dict` | `None` | Memory aggregation model config. |
| `memory_agent_openai_config` | `dict` | `None` | Text-mode memory-agent config. |
| `router_openai_config` | `dict` | `None` | Router or router-fallback model config. |
| `clean_cache_first` | `bool` | `True` | **Warning**: If `True`, wipes all data with the same session id in `kv_data/` on startup. |
| `enable_router` | `bool` | `True` | If `False`, skips retrieval and forces all memory blocks into context (debugging only). |
| `router_type` | `str` | `"hybrid"` | `"hybrid"` or `"llm"` (legacy). |
| `hybrid_router_config` | `dict` | `None` | Overrides for router weights and thresholds (see `HybridRouterConfig`). |
| `router_system_prompt` | `str` | `None` | Custom system prompt for the router. |
| `quantization_config` | `dict` | `None` | HuggingFace quantization config (e.g., bitsandbytes). |
| `max_memory` | `dict` | `None` | Max memory per GPU device (e.g., `{"0": "20GiB"}`). |
| `offload_folder` | `str` | `None` | Folder for offloading model weights if GPU is full. |

These runtime arguments correspond to the role-based YAML fields documented in [Config Model Roles](CONFIG_MODELS.md). In particular:

- `memory_agent_openai_config` maps to `model.memory_agent_model.openai_config`
- `chat_openai_config` maps to `model.general_model.openai_config` by default, or `model.manager_model.openai_config` when overridden
- `aggregator_openai_config` maps to `model.general_model.openai_config` by default, or `model.aggregator_model.openai_config` when overridden
- `router_openai_config` maps to `model.general_model.openai_config` by default, or `model.router_fallback_model.openai_config` when overridden
- `model_id` and `model_context_window` map to the memory-agent model settings

`question_answer_model` is not a `create_chat_manager(...)` argument because it is consumed directly by evaluation scripts.

---

## Configuration Classes

Configuration is validated using Pydantic models.

### `HybridRouterConfig`

Controls the retrieval logic for the Hybrid Router.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embedding_provider` | `str` | `"huggingface"` | `"huggingface"` or `"openai"`. |
| `embedding_model` | `str` | `None` | Embedding model name (e.g., `sentence-transformers/all-MiniLM-L6-v2`). |
| `embedding_config` | `dict` | `None` | Extra config for embeddings (e.g., API key). |
| `summary_weight` | `float` | `0.3` | Weight for **Global Alignment** (summary embedding similarity). |
| `text_weight` | `float` | `0.4` | Weight for **Semantic Association** (raw text embedding similarity). |
| `bm25_weight` | `float` | `0.3` | Weight for **Symbolic Trigger** (keyword matching). |
| `summary_top_k` | `int` | `10` | Top-k summaries to consider. |
| `text_top_k` | `int` | `20` | Top-k text chunks to consider. |
| `bm25_top_k` | `int` | `10` | Top-k BM25 results to consider. |
| `text_chunk_size` | `int` | `512` | Max chunk size for text embeddings (chars). |
| `text_chunk_overlap` | `int` | `50` | Overlap between text chunks (chars). |
| `bm25_boost_threshold` | `float` | `None` | If > 0, bypasses weighted score for strong keyword matches. |
| `use_llm_fallback` | `bool` | `False` | Fallback to LLM routing if embeddings fail. |
| `bm25_use_jieba` | `bool` | `True` | Use jieba tokenizer for Chinese text support. |

### `MemoryConfig`

Controls memory block management and hardware usage.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `storage_mode` | `str` | `"kv_cache"` | Storage backend mode. |
| `max_blocks` | `int` | `5` | Max memory blocks to retrieve per query. |
| `max_parallel_cache_loads` | `int` | `8` | Max concurrent KV cache loads (GPU VRAM dependent). |
| `kv_cache_on_gpu` | `bool` | `False` | Keep inactive caches on GPU (requires massive VRAM, e.g., A100s). |
| `block_size_ratio` | `float` | `0.125` | Target size of each memory block relative to context window. |

---

## Core Interfaces

### `ChatManager` / `TextStorageChatManager`

The high-level interface returned by `create_chat_manager`.

#### Methods

- **`chat(user_input: str, outer_tools: list = None, auto_save: bool = False, save_original_input: bool = False, max_new_tokens: int = 1024) -> str`**
  - Processes a user query, retrieves context, and generates a response.
  - `user_input`: The user's input message.
  - `outer_tools`: Additional tools to pass to the agent.
  - `auto_save`: If `True`, directly saves input to memory without generating a response (fast path).
  - `save_original_input`: If `True`, saves the raw user input instead of LLM-extracted content.
  - `max_new_tokens`: Maximum number of tokens to generate.

- **`add_memory(memory: str) -> str`**
  - Manually injects text into the memory stream without generating a response.
  - Returns a success or failure message.

- **`search_memory(query: str) -> str`**
  - Searches the memory blocks for relevant information using the configured router.
  - Returns aggregated results from retrieved blocks.
