#!/usr/bin/env bash
# Day 3: v12 on LoCoMo (10 convs) + ablations + significance. Then the same with --think (reader reasoning) at 2 budgets.
set -e
export VLLM_USE_V1=1
python scripts/run_system_v15.py --dataset locomo --eps 0.5 --configs full,no_adaptive,no_residual --budgets 400,1600,4000 --out runs/v15_locomo
python scripts/run_system_v15.py --dataset locomo --eps 0.5 --configs full --budgets 400,1600 --no_reform  --out runs/v15_noreform
python scripts/run_system_v15.py --dataset locomo --eps 0.5 --configs full --budgets 400,1600 --no_profile --out runs/v15_noprofile
python scripts/analyze_distance.py runs/v15_locomo/items.csv --config full --budget 400,1600
python scripts/analyze_ci.py runs/v15_locomo/items.csv --by_category --pairs \
  "full:ours_400 vs full:raw_rag_400" "full:ours_1600 vs full:raw_rag_1600" "full:ours_4000 vs full:raw_rag_4000" "full:certified vs full:raw_rag_400" \
  "full:certified vs full:units_only" "full:certified vs no_adaptive:certified" "full:certified vs no_residual:certified"
python scripts/run_system_v15.py --dataset locomo --eps 0.5 --configs full --budgets 400,1600 --think --out runs/v15_think
tar czf results_day3.tgz runs/v15_*
echo "DAY3 DONE"
