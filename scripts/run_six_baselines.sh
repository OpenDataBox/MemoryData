#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run LangMem, Mem0, A-MEM, LightMem, SimpleMem, and E-Mem on LoCoMo and LongMemEval-S.

Usage:
  bash scripts/run_six_baselines.sh [--full] [run_six_baselines.py options]

The default is a smoke run with one history and one question per method/dataset.
Pass --full to remove those two limits. All remaining arguments are forwarded to
scripts/run_six_baselines.py and can override its defaults.

Environment overrides:
  PYTHON_BIN          Python executable (auto-detected when unset)
  MODEL               Chat model name (default: Qwen3.5-0.8B)
  EMBEDDING_MODEL     Embedding model name (default: Qwen3-Embedding-0.6B)
  EMBEDDING_DIM       Embedding dimension (default: 1024)
  OPENAI_BASE_URL     OpenAI-compatible chat endpoint (default: http://127.0.0.1:9911/v1)
  EMBEDDING_BASE_URL  OpenAI-compatible embedding endpoint (default: http://127.0.0.1:9010/v1)
  OPENAI_API_KEY      API key; use EMPTY for a keyless local server
  ARTIFACT_ROOT       Output root (default depends on smoke/full mode)

Examples:
  bash scripts/run_six_baselines.sh --dry-run
  bash scripts/run_six_baselines.sh --methods e_mem lightmem --datasets locomo
  bash scripts/run_six_baselines.sh --full
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="smoke"
forwarded_args=()

while (($#)); do
  case "$1" in
    --full)
      mode="full"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$repo_root/.venv-six/bin/python" ]]; then
    PYTHON_BIN="$repo_root/.venv-six/bin/python"
  elif [[ -x "$repo_root/.venv-six/Scripts/python.exe" ]]; then
    PYTHON_BIN="$repo_root/.venv-six/Scripts/python.exe"
  else
    PYTHON_BIN="python"
  fi
fi

model="${MODEL:-Qwen3.5-0.8B}"
embedding_model="${EMBEDDING_MODEL:-Qwen3-Embedding-0.6B}"
embedding_dim="${EMBEDDING_DIM:-1024}"
chat_base_url="${OPENAI_BASE_URL:-http://127.0.0.1:9911/v1}"
embedding_base_url="${EMBEDDING_BASE_URL:-http://127.0.0.1:9010/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

limits=()
if [[ "$mode" == "smoke" ]]; then
  artifact_root="${ARTIFACT_ROOT:-$repo_root/results/six_baselines_smoke}"
  limits=(--max-samples "${MAX_SAMPLES:-1}" --max-queries "${MAX_QUERIES:-1}")
else
  artifact_root="${ARTIFACT_ROOT:-$repo_root/results/six_baselines}"
fi

exec "$PYTHON_BIN" "$repo_root/scripts/run_six_baselines.py" \
  --model "$model" \
  --base-url "$chat_base_url" \
  --embedding-model "$embedding_model" \
  --embedding-base-url "$embedding_base_url" \
  --embedding-dim "$embedding_dim" \
  --artifact-root "$artifact_root" \
  "${limits[@]}" \
  "${forwarded_args[@]}"
