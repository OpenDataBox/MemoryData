# Baselines (same backbone Qwen3-8B, same LoCoMo 10, our scoring)

Order (after LME / day3 finish; GPU must be free):
1. `bash scripts/baselines/serve_both.sh`          # chat :8000 + embeddings :8001
2. `bash scripts/baselines/setup_memorydata.sh`    # ~15 min, own venv
3. `bash scripts/baselines/run_memorydata.sh`       # mem0 lightmem a_mem simplemem memoryos embedding_rag  (~1-3 h each)
4. `bash scripts/baselines/setup_trimem.sh && bash scripts/baselines/run_trimem.sh`
5. `python scripts/baselines/compare.py --ours runs/v15_locomo/items.csv --external runs/baseline_*/items.csv`
Then `pkill -f "vllm serve"` before running our own scripts again (they load vLLM in-process).

Notes: MemoryData presets were written for Qwen3-8B on local vLLM; we only re-point ports, swap the 4B embedder for the
0.6B we use ourselves, set temperature 0, and use the full locomo10.json (their default is a 600-question subset).
If a method fails, check /workspace/baseline_<m>.log; most failures are a missing pip dep (install into their venv) or an
env var (OPENAI_API_KEY=EMPTY).
