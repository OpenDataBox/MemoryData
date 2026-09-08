# Config Reference

This document explains the main non-model fields in the YAML config files.

If you want the meaning of `memory_agent_model`, `general_model`, or the optional model overrides, read [Config Model Roles](CONFIG_MODELS.md) first.

## Layout

The example configs are organized into these sections:

- `tokenizer`: token counting and block sizing
- `model`: memory model, general model, and optional model overrides
- `memory`: storage mode, chunking, routing, and hardware-related settings
- `locomo_eval` or `hotpotqa_eval`: benchmark-specific settings
- `logging`: log output settings

## `tokenizer`

### `tokenizer.model_id`

The tokenizer reference used for token counting and memory block sizing.

In practice:

- keep it aligned with the memory model you want block sizing to resemble,
- in text mode, this is still useful even though the memory agent may be API-based.

## `memory`

### `memory.storage_mode`

Choose the memory backend:

- `"kv_cache"`: local HuggingFace model, KV cache storage, GPU-oriented
- `"text"`: text storage, API-friendly, easier for debugging and approximate reproduction

### `memory.clean_cache_first`

If `true`, E-mem clears previously stored cache / memory data on startup.

Use it when:

- you want a clean benchmark run,
- you suspect old cached state is stale or incompatible,
- you are iterating on config changes and do not want old memory to leak in.

### `memory.router_system_prompt`

Optional custom system prompt for the router.

Leave it `null` unless you are deliberately experimenting with router prompting.

### `memory.overlap_ratio`

Overlap ratio between adjacent memory blocks.

Practical guidance:

- lower values reduce duplication and save memory,
- higher values improve continuity across block boundaries,
- values above `0.5` are intentionally rejected.

### `memory.overlap_mode`

Controls how overlap is applied:

- `"chunk"`: overlap in chunk units
- `"token"`: overlap in token units

### `memory.block_size_ratio`

Target block size relative to the memory model context window.

This is one of the most important memory-shaping parameters.

Example:

- with `model_context_window = 32768`
- and `block_size_ratio = 0.125`

the target block size is about `4096` tokens before overlap effects.

### `memory.max_concurrent_gpu_operations`

High-level GPU concurrency cap for memory operations.

Usually leave this at the default unless you are tuning throughput on larger hardware.

### `memory.max_memory_segments`

Maximum number of memory segments returned from a block-level query.

Higher values can improve recall, but they also make downstream aggregation noisier and more expensive.

### `memory.max_blocks`

Maximum number of memory blocks the router is allowed to select.

This is one of the main knobs controlling recall vs. latency.

### `memory.query_batch_size`

How many block queries to batch together.

Hardware-oriented rule of thumb:

- single GPU with limited VRAM: `1-2`
- moderate multi-GPU setup: `4-8`
- very large GPU setup: try larger values only after measuring memory headroom

### `memory.max_parallel_cache_loads`

Maximum number of KV caches loaded to GPU in parallel.

This mainly matters in `kv_cache` mode.

Rule of thumb:

- limited VRAM: keep it small
- larger memory systems: increase gradually after profiling

### `memory.enable_router`

If `false`, E-mem skips routing and queries all blocks directly.

This is mostly useful for:

- debugging,
- ablations,
- sanity-checking recall without router effects.

### `memory.router_type`

Router mode:

- `"hybrid"`: recommended, embedding + BM25 based
- `"llm"`: legacy LLM-based routing mode

## `memory.hybrid_router`

These fields only matter when `router_type: "hybrid"`.

### `embedding_provider`

Embedding backend:

- `"huggingface"`: local / sentence-transformers style
- `"openai"`: OpenAI-compatible embedding API

### `embedding_model`

Optional explicit embedding model name.

Leave `null` to use the provider default.

### `embedding_config`

Extra config for the embedding backend, especially when using API-based embeddings.

### `summary_weight`, `text_weight`, `bm25_weight`

The three retrieval scoring components:

- `summary_weight`: summary-level semantic matching
- `text_weight`: chunk-level semantic matching
- `bm25_weight`: keyword / symbolic matching

You usually tune these when changing corpus style or retrieval behavior.

### `summary_top_k`, `text_top_k`, `bm25_top_k`

Top-k candidates considered from each scoring pathway before the router combines them.

### `text_chunk_size`, `text_chunk_overlap`

Chunking settings for text embeddings.

These affect retrieval granularity:

- smaller chunks are more precise,
- larger chunks preserve more context.

### `use_llm_fallback`

If `true`, the hybrid router can fall back to an LLM when needed.

When this happens:

- E-mem uses `router_fallback_model` if you set it,
- otherwise it falls back to `general_model`.

### `bm25_use_jieba`

Whether to use `jieba` tokenization for BM25.

Guidance:

- Chinese or mixed Chinese text: usually keep `true`
- English-only corpus: `false` is often slightly cleaner

### `bm25_boost_threshold`

If a block's normalized BM25 score exceeds this threshold, the block is force-selected regardless of the combined weighted score.

Use it when you want exact keyword matches to dominate.

Practical range:

- around `0.7-0.9` for stricter keyword-triggered inclusion
- `null` to disable the behavior

## Evaluation Sections

### `locomo_eval.dataset_path` / `hotpotqa_eval.dataset_path`

Path to the benchmark dataset file.

### `locomo_eval.output_dir` / `hotpotqa_eval.output_dir`

Directory where benchmark results are written.

### `locomo_eval.ratio` / `hotpotqa_eval.ratio`

Portion of the dataset to evaluate.

Examples:

- `1.0`: full dataset
- `0.1`: 10% sample for a faster smoke test

### `locomo_eval.conversation_auto_save`

Controls whether conversation turns are saved through the auto-save path during LoCoMo evaluation.

### `locomo_eval.categories`

LoCoMo question categories to evaluate.

Valid values are `1-5`.

### `hotpotqa_eval.max_tokens_per_chunk`

Maximum token count per context chunk when HotpotQA contexts are split before memory insertion.

## `logging`

### `logging.log_dir`

Directory where logs are written.

### `logging.log_level`

Standard logging level, such as:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`

## Suggested Workflow

1. Start from `config.kv.yaml` or `config.text.yaml`.
2. Set `memory_agent_model`.
3. Set `general_model`.
4. Tune only a few high-impact memory fields at first:
   - `storage_mode`
   - `block_size_ratio`
   - `max_blocks`
   - `query_batch_size`
   - `max_parallel_cache_loads`
5. Only then add role overrides or router-specific tuning.

## Related Files

- [Config Model Roles](CONFIG_MODELS.md)
- [Operational Guides](GUIDES.md)
- [API Reference](API_REFERENCE.md)
- [KV Example Config](../config.kv.yaml)
- [Text Example Config](../config.text.yaml)
