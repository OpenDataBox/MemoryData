#!/usr/bin/env bash
# One-shot pipeline (12-18 h on one A100). Each stage logs separately; a failed stage does not stop the rest.
# Stages: LME (30) -> no-raw regime -> LoCoMo final (day3) -> baselines via MemoryData + TriMem -> paired comparison.
cd /workspace/certmem && source .venv/bin/activate && export VLLM_USE_V1=1
run() { echo "===== $(date '+%H:%M') START $1"; bash "$2" > "/workspace/stage_$1.log" 2>&1 && echo "===== $(date '+%H:%M') OK $1" || echo "===== $(date '+%H:%M') FAILED $1 (see /workspace/stage_$1.log)"; }
run lme      scripts/run_lme.sh
run noraw    scripts/run_noraw.sh
run day3     scripts/run_day3.sh
# ---- baselines need the GPU free of in-process vLLM, then two servers
pkill -f VLLM::EngineCore || true; sleep 5
run serve    scripts/baselines/serve_both.sh
run md_setup scripts/baselines/setup_memorydata.sh
run md_run   scripts/baselines/run_memorydata.sh
run tm_setup scripts/baselines/setup_trimem.sh
run tm_run   scripts/baselines/run_trimem.sh
cd /workspace/certmem && source .venv/bin/activate
python scripts/baselines/compare.py --ours runs/v15_locomo/items.csv --external runs/baseline_*/items.csv > /workspace/stage_compare.log 2>&1 && echo "===== OK compare" || echo "===== FAILED compare"
pkill -f "vllm serve" || true
tar czf /workspace/certmem/results_everything.tgz runs/v15_* runs/baseline_* runs/compare_baselines.csv 2>/dev/null
echo "===== EVERYTHING DONE $(date)"
