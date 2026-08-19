# Vast.ai: Mem0 with local Qwen3.5-4B

This launcher runs the MemoryData Mem0 benchmark against two local
OpenAI-compatible vLLM services:

- chat and fact extraction: `Qwen/Qwen3.5-4B`
- memory embeddings: `Qwen/Qwen3-Embedding-4B`

The default benchmark is the MemoryAgentBench `longmemeval_s*` subset using the
bundled 150k-context LongMemEval preset.

## Instance recommendation

Use a CUDA Vast.ai image with Python 3.11+ and enough disk space for both model
checkpoints. Two 16 GB or larger GPUs are preferred, one for Qwen3.5-4B and one
for Qwen3-Embedding-4B. A single 24 GB or larger GPU can host both services; the
launcher automatically assigns conservative memory fractions when they share a
GPU.

The default run is a one-query smoke test:

```bash
bash scripts/vastai/run_mem0_qwen35.sh
```

Run every query in the configured five-sample LongMemEval preset:

```bash
MAX_QUERIES=0 bash scripts/vastai/run_mem0_qwen35.sh
```

Remove the preset's five-sample cap and run every available LongMemEval sample:

```bash
MAX_QUERIES=0 MAX_TEST_SAMPLES=999999 bash scripts/vastai/run_mem0_qwen35.sh
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
| `MAX_QUERIES` | `1` | Smoke-test query count; use `0` for every query in the selected samples |
| `MAX_TEST_SAMPLES` | preset value (`5` for LongMemEval) | Positive integer overriding the dataset sample cap |
| `FORCE` | `0` | Use `1` to discard and rebuild prior result/agent state |
| `LLM_GPU` | `0` | Physical GPU index for the chat server |
| `EMBED_GPU` | `1` when available, otherwise `0` | Physical GPU index for embeddings |
| `MAX_MODEL_LEN` | `32768` | Qwen3.5 vLLM context limit |
| `LLM_MODEL` | `Qwen/Qwen3.5-4B` | Hugging Face model ID or local model directory |
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-4B` | Embedding model ID or local directory |
| `KEEP_SERVERS` | `0` | Use `1` to leave servers running after the benchmark |
| `SKIP_INSTALL` | `0` | Use `1` after the virtual environment is fully prepared |
| `HF_HOME` | `~/.cache/huggingface` | Persistent model cache location |

Server logs and the generated runtime YAML are saved under
`.runtime-cache/vastai-mem0-qwen35/`. Benchmark outputs retain the normal
`results/outputs/` and `results/agents/` layout.
