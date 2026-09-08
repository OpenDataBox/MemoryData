#!/usr/bin/env bash
set -e; export VLLM_USE_V1=1
python scripts/run_recover_v7.py --out runs/full_recover_v7
python scripts/run_read_v9.py --out runs/full_read_v9
tar czf results_podB.tgz runs/full_*; echo "POD B DONE"
