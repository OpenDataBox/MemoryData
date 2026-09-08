#!/usr/bin/env bash
# Day 2: v10 system on LoCoMo (10 convs), then LongMemEval-S subset.
set -e
export VLLM_USE_V1=1
python scripts/run_system_v10.py --dataset locomo --eps 0.5 --out runs/v10_locomo
bash scripts/get_longmemeval.sh
python scripts/run_system_v10.py --dataset longmemeval --limit_convs 60 --eps 0.5 --configs full,no_adaptive --out runs/v10_lme
tar czf results_day2.tgz runs/v10_*
echo "DAY2 DONE"
