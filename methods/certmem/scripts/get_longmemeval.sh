#!/usr/bin/env bash
# Download LongMemEval-S (cleaned) into data/. Tries HF dataset repo first, then GitHub release names.
set -e
mkdir -p data
python - << 'PY'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files
repo = "xiaowu0162/longmemeval"
try:
    files = list_repo_files(repo, repo_type="dataset")
    print("files:", files)
    cand = [f for f in files if "longmemeval_s" in f and f.endswith(".json")]
    pick = next((f for f in cand if "clean" in f), cand[0] if cand else None)
    if pick is None: raise SystemExit("no longmemeval_s json found")
    p = hf_hub_download(repo, pick, repo_type="dataset", local_dir="data")
    dst = "data/longmemeval_s_cleaned.json"
    if os.path.abspath(p) != os.path.abspath(dst):
        os.replace(p, dst)
    print("saved", dst)
except Exception as e:
    print("HF download failed:", e); sys.exit(1)
PY
python - << 'PY'
import sys; sys.path.insert(0, ".")
from certmem.longmemeval import load
c = load("data/longmemeval_s_cleaned.json", limit=3)
for k, v in c.items():
    q = v["qa"][0]; print(k, q.qtype, "| sessions:", len(v["sessions"]), "| evidence sessions:", q.sessions(), "| Q:", q.question[:60])
PY
