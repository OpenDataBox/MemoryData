#!/usr/bin/env bash
set -euo pipefail
cd /workspace/locomo-refinement
export HF_HOME=/workspace/.hf_home
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python -m unittest discover -s . -p 'test_transfer_index_alignment.py'
exec .venv/bin/python run_transfer_refinement.py
