#!/usr/bin/env bash
# Run TriMem on LoCoMo 10 conversations against the local server, then score with our normalized F1 + CI vs our system.
set -e
cd /workspace/baselines/TriMem && source .venv/bin/activate
python test_locomo10.py --dataset test_ref/locomo10.json --num-samples 10 --result-file trimem_qwen3_8b.json --test-workers 4 --skip-categories 5
cd /workspace/certmem && source .venv/bin/activate
python scripts/baselines/score_external.py /workspace/baselines/TriMem/trimem_qwen3_8b.json --system trimem --out runs/baseline_trimem
echo "TRIMEM DONE"
