# Vast.ai: Mem0 with local Qwen3.5-9B

This launcher runs the MemoryData Mem0 benchmark against two local
OpenAI-compatible vLLM services:

- chat and fact extraction: `Qwen/Qwen3.5-9B`
- memory embeddings: `Qwen/Qwen3-Embedding-4B`

The official small dense Qwen3.5 checkpoint is 9B. `Qwen3-8B` is from the
earlier Qwen3 family.

## Instance recommendation

Use a CUDA Vast.ai image with Python 3.11+ and enough disk space for both model
checkpoints. Two GPUs are preferred: a 24 GB or larger GPU for Qwen3.5-9B and a
second 12 GB or larger GPU for Qwen3-Embedding-4B. A single 48 GB or larger GPU
can host both services; the launcher automatically assigns conservative memory
fractions when both services share a GPU.

The default run is a one-query smoke test:

```bash
bash scripts/vastai/run_mem0_qwen35.sh
```

Run the complete configured EventQA benchmark:

```bash
MAX_QUERIES=0 bash scripts/vastai/run_mem0_qwen35.sh
```

Run on explicit GPUs or with another dataset config:

```bash
LLM_GPU=0 \
EMBED_GPU=1 \
MAX_QUERIES=0 \
DATASET_CONFIG=benchmark/membench/config/MemBench_simple.yaml \
bash scripts/vastai/run_mem0_qwen35.sh
```

MemBench data must already exist under `datasets/MemBench/`. MemoryAgentBench
falls back to its Hugging Face dataset when the local copy is absent.

## Common overrides

| Variable | Default | Purpose |
| --- | --- | --- |
| `MAX_QUERIES` | `1` | Smoke-test query count; use `0` for the full benchmark |
| `FORCE` | `0` | Use `1` to discard and rebuild prior result/agent state |
| `LLM_GPU` | `0` | Physical GPU index for the chat server |
| `EMBED_GPU` | `1` when available, otherwise `0` | Physical GPU index for embeddings |
| `MAX_MODEL_LEN` | `32768` | Qwen3.5 vLLM context limit |
| `LLM_MODEL` | `Qwen/Qwen3.5-9B` | Hugging Face model ID or local model directory |
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-4B` | Embedding model ID or local directory |
| `KEEP_SERVERS` | `0` | Use `1` to leave servers running after the benchmark |
| `SKIP_INSTALL` | `0` | Use `1` after the virtual environment is fully prepared |
| `HF_HOME` | `~/.cache/huggingface` | Persistent model cache location |

Server logs and the generated runtime YAML are saved under
`.runtime-cache/vastai-mem0-qwen35/`. Benchmark outputs retain the normal
`results/outputs/` and `results/agents/` layout.
