#!/usr/bin/env bash
# Full LoCoMo (10 conversations). Run on one A100; ~6-8 h. Split across 2 pods by editing the loops.
set -e
export VLLM_USE_V1=1
# 1. dose-response + fact sensor at 4 compression levels  (Fig 1, Fig 5 input)
for cap in 3 6 12 25; do
  python scripts/run_4_2a.py --host mem0 --cap $cap --sensors facts --out runs/full_sweep_cap$cap
done
# 2. sensors incl. probes at default cap  (Fig 2)
python scripts/run_4_2a.py --host mem0 --sensors facts,probes --out runs/full_sensors
python scripts/analyze_sensors.py runs/full_sensors
python scripts/analyze_fact_match.py runs/full_sensors
python scripts/relabel_and_test.py runs/full_sensors
python scripts/analyze_sweep.py runs/full_sweep_cap3 runs/full_sweep_cap6 runs/full_sweep_cap12 runs/full_sweep_cap25
# 3. certified consolidation from sweep (no GPU)  (Fig 4/5)
python scripts/ltt_gate.py --sweep_glob "runs/full_sweep_cap*" --out runs/full_certified
# 4. write-time residual  (Fig 3)
python scripts/run_recover_v7.py --out runs/full_recover_v7
# 5. read-time expansion vs RAG  (Fig 6)
python scripts/run_read_v9.py --out runs/full_read_v9
tar czf results_full.tgz runs/full_*
echo "ALL DONE"
