#!/usr/bin/env bash
# No-raw-retention regime on LoCoMo: raw deleted after write. Only memory units (facts + residual + profiles) answer.
# Shows the value of write-time recovery (residual) and adaptive lambda when expansion is impossible.
set -e
export VLLM_USE_V1=1
python scripts/run_system_v15.py --dataset locomo --eps 0.5 --configs full,no_adaptive,no_residual,plain --budgets 400 --no_raw --out runs/v15_noraw
python scripts/analyze_ci.py runs/v15_noraw/items.csv --pairs "full:units_only vs plain:units_only" "full:units_only vs no_residual:units_only" "full:units_only vs no_adaptive:units_only"
echo "NORAW DONE"
