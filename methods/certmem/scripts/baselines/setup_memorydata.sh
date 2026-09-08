#!/usr/bin/env bash
# MemoryData testbed (Zhou et al. 2026, OpenDataBox/MemoryData): own venv, patched to our local servers, full LoCoMo 10.
set -e
mkdir -p /workspace/baselines && cd /workspace/baselines
[ -d MemoryData ] || git clone --depth 1 https://github.com/OpenDataBox/MemoryData.git
cd MemoryData
python -m venv .venv && source .venv/bin/activate
pip install -q -U pip
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# their manifest minus heavy/optional extras (deepspeed, bitsandbytes, gritlm, timm, spacy, google/groq/cohere clients)
grep -viE "^(deepspeed|bitsandbytes|gritlm|timm|spacy|langchain_google_genai|langchain-groq|langchain-cohere|torchvision|torch)\b" requirements.txt > req_min.txt
pip install -q -r req_min.txt || echo "some optional deps failed; continuing"
pip install -q "openai>=1.40" qdrant-client rank_bm25 dateparser
# point every preset at our two servers, our embedding model, and the full LoCoMo file
python - << 'PY'
import re, glob, os
for f in glob.glob("config/*.yaml"):
    s = open(f).read()
    s = re.sub(r"http://127\.0\.0\.1:99\d\d/v1", "http://127.0.0.1:8000/v1", s)   # chat endpoints 9908-9911
    s = re.sub(r"http://127\.0\.0\.1:90\d\d/v1", "http://127.0.0.1:8001/v1", s)   # embedding endpoints 9009/9010
    s = s.replace("Qwen3-Embedding-4B", "Qwen3-Embedding-0.6B").replace("embedding_dim: 2560", "embedding_dim: 1024")
    s = re.sub(r"(temperature:\s*)[0-9.]+", r"\g<1>0.0", s)
    open(f, "w").write(s)
d = "benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml"; s = open(d).read()
s = re.sub(r"test_files:.*", "test_files: /workspace/certmem/data/locomo10.json", s)
s = re.sub(r"max_test_samples:.*", "max_test_samples: 10", s)
s = re.sub(r"tag:.*", "tag: locomo10_full", s)
open(d, "w").write(s); print("configs patched")
PY
export OPENAI_API_KEY=EMPTY
echo "MemoryData ready"
