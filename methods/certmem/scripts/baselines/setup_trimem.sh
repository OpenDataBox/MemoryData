#!/usr/bin/env bash
# Clone TriMem (Sun et al. 2026), give it its own venv, point it at the local Qwen3-8B server, copy LoCoMo.
set -e
mkdir -p /workspace/baselines && cd /workspace/baselines
[ -d TriMem ] || git clone --depth 1 https://github.com/tmlr-group/TriMem.git
cd TriMem
python -m venv .venv && source .venv/bin/activate
pip install -q -U pip
pip install -q openai lancedb pylance tantivy "sentence-transformers>=3.0" rank-bm25 nltk rouge_score bert_score tqdm pydantic dateparser scikit-learn pandas
mkdir -p test_ref && cp /workspace/certmem/data/locomo10.json test_ref/locomo10.json
# config: local server, no thinking (matches our reader), no streaming
python - << 'PY'
import re
s = open("config.py").read()
s = re.sub(r'OPENAI_API_KEY\s*=.*', 'OPENAI_API_KEY = "EMPTY"', s)
s = re.sub(r'OPENAI_BASE_URL\s*=.*', 'OPENAI_BASE_URL = "http://localhost:8000/v1"', s)
s = re.sub(r'LLM_MODEL\s*=.*', 'LLM_MODEL = "Qwen/Qwen3-8B"', s)
s = re.sub(r'ENABLE_THINKING\s*=.*', 'ENABLE_THINKING = False', s)
s = re.sub(r'USE_STREAMING\s*=.*', 'USE_STREAMING = False', s)
open("config.py", "w").write(s)
print("config patched")
PY
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('punkt_tab', quiet=True)"
echo "TriMem ready"
