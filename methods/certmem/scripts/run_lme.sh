#!/usr/bin/env bash
# LongMemEval-S: 30 histories, v13 full vs RAG at 3 budgets (+ question date), then distance-style analysis by evidence gap.
set -e
export VLLM_USE_V1=1
python scripts/run_system_v15.py --dataset longmemeval --limit_convs 30 --eps 0.5 --configs full --budgets 400,1600,4000 --out runs/v15_lme
python scripts/analyze_ci.py runs/v15_lme/items.csv --pairs "full:ours_400 vs full:raw_rag_400" "full:ours_1600 vs full:raw_rag_1600" "full:ours_4000 vs full:raw_rag_4000" "full:certified vs full:raw_rag_400"
tar czf results_lme.tgz runs/v15_lme
echo "LME DONE"
