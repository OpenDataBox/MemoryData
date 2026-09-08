#!/usr/bin/env bash
# Dose-response: 2 conversations x 4 compression levels, fact sensor only (fast).
set -e
export VLLM_USE_V1=1
for cap in 3 6 12 25; do
  python scripts/run_4_2a.py --host mem0 --cap $cap --limit_convs 2 --sensors facts --out runs/sweep_cap$cap
done
python scripts/analyze_sweep.py runs/sweep_cap3 runs/sweep_cap6 runs/sweep_cap12 runs/sweep_cap25
