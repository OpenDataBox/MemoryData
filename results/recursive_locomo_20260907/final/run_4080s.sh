#!/usr/bin/env bash
set -euo pipefail
cd /workspace/locomo-refinement
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_XET=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
for attempt in $(seq 1 180); do
    if test -f environment.json; then break; fi
    if supervisorctl status locomo-setup | grep -Eq 'EXITED|FATAL|STOPPED'; then
        echo 'Environment setup ended without environment.json; inspect setup.log' >&2
        exit 1
    fi
    echo 'WAITING_FOR_ENVIRONMENT'
    sleep 20
done
test -f environment.json
.venv/bin/python -m unittest discover -s . -p test_refine.py
.venv/bin/python validate_official_metric.py data/official_evaluation.py
exec .venv/bin/python refine.py --out runs/pilot100_v3 --parent-run runs/pilot100_v2 --max-rounds 5 --patience 5 --budget 2048 --embed-batch-size 4
