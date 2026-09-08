#!/usr/bin/env bash
# One-shot environment setup. Run on the GPU machine.
set -e
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "vllm>=0.8" "transformers>=4.51" numpy scipy pandas tqdm sentence-transformers rank-bm25 huggingface_hub
# models (Qwen3-8B ~16GB bf16; embedding 0.6B ~1.2GB)
huggingface-cli download Qwen/Qwen3-8B
huggingface-cli download Qwen/Qwen3-Embedding-0.6B
# LoCoMo data
mkdir -p data
curl -L -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python scripts/smoke_test.py data/locomo10.json
echo "setup done"
