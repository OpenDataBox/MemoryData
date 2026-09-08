#!/usr/bin/env bash
set -euo pipefail
cd /workspace/locomo-refinement
export HF_HOME=/workspace/.hf_home
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python -m unittest discover -s . -p 'test_*.py'
.venv/bin/python validate_official_metric.py data/official_evaluation.py
exec .venv/bin/python continuous.py --out runs/continuous_v1 --queue continuous_queue.json --seed-run runs/pilot100_v3
