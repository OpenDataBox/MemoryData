#!/usr/bin/env bash
set -euo pipefail
cd /workspace/locomo-refinement
export HF_HOME=/workspace/.hf_home
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec .venv/bin/python run_evidence_utility_fit.py
