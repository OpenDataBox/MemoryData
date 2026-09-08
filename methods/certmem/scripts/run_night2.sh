#!/usr/bin/env bash
# Night 2: combined system + ablations on 10 conversations (~2-3 h), then re-run certification table.
set -e
export VLLM_USE_V1=1
python scripts/run_combined.py --out runs/full_combined
python scripts/ltt_gate.py --sweep_glob "runs/full_sweep_cap*" --out runs/full_certified
tar czf results_night2.tgz runs/full_combined runs/full_certified
echo "NIGHT2 DONE"
