#!/usr/bin/env bash
set -e; export VLLM_USE_V1=1
for cap in 3 6 12 25; do python scripts/run_4_2a.py --host mem0 --cap $cap --sensors facts --out runs/full_sweep_cap$cap; done
python scripts/run_4_2a.py --host mem0 --sensors facts,probes --out runs/full_sensors
tar czf results_podA.tgz runs/full_*; echo "POD A DONE"
