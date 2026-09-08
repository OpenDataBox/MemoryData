#!/usr/bin/env bash
set -euo pipefail
cd /workspace/locomo-refinement
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_XET=0
export UV_NO_CACHE=false
export UV_CACHE_DIR=/workspace/locomo-refinement/uv_cache_recovered
if ! test -f .venv/bin/python; then uv venv --python /usr/bin/python3.12 .venv; fi
extra=()
if compgen -G 'wheelhouse/*.whl' > /dev/null; then extra=(wheelhouse/*.whl); fi
uv pip install --python .venv/bin/python "${extra[@]}" 'vllm==0.10.2' 'transformers==4.56.2' 'sentence-transformers==5.1.0' 'rank-bm25==0.2.2' 'nltk==3.9.1' 'numpy<3' pandas
if test -d model_ranges; then
    while ! test -f model_ranges/ready.json; do
        if supervisorctl status locomo-models | grep -Eq 'EXITED|FATAL|STOPPED'; then
            echo 'Parallel model transfer failed; inspect models.log' >&2
            exit 1
        fi
        echo 'WAITING_FOR_VERIFIED_MODEL_RANGES'
        sleep 20
    done
fi
mkdir -p data
curl --fail --location --retry 3 -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
curl --fail --location --retry 3 -o data/official_evaluation.py https://raw.githubusercontent.com/snap-research/locomo/main/task_eval/evaluation.py
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
import importlib.metadata, json, pathlib, hashlib, time
manifest = {'models': {}}
for model in ['Qwen/Qwen3-8B', 'Qwen/Qwen3-Embedding-0.6B']:
    for attempt in range(5):
        try:
            path = snapshot_download(model, ignore_patterns=['*.bin', '*.gguf', '*.onnx', 'onnx/*'])
            break
        except Exception as error:
            if attempt == 4: raise
            print('RETRY_MODEL', model, type(error).__name__, flush=True)
            time.sleep(10)
    manifest['models'][model] = {'path': path, 'revision': pathlib.Path(path).name}
manifest['packages'] = {p: importlib.metadata.version(p) for p in ['torch','vllm','transformers','sentence-transformers','nltk','numpy']}
manifest['data_sha256'] = hashlib.sha256(pathlib.Path('data/locomo10.json').read_bytes()).hexdigest()
pathlib.Path('environment.json').write_text(json.dumps(manifest, indent=2))
print('ENVIRONMENT_READY', flush=True)
PY
