#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot Vast.ai launcher for MemoryData + Mem0 + local Qwen3.5-9B.
#
# The script creates/reuses a virtual environment, starts two OpenAI-compatible
# vLLM servers, waits for both APIs, writes a runtime YAML config, and runs the
# selected benchmark. Environment variables below provide all common overrides.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-vast-mem0}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/.runtime-cache/vastai-mem0-qwen35}"
LOG_DIR="${RUNTIME_ROOT}/logs"
RUNTIME_CONFIG="${RUNTIME_ROOT}/sequential_mem0_qwen35_9b.runtime.yaml"

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.5-9B}"
LLM_SERVED_NAME="${LLM_SERVED_NAME:-Qwen3.5-9B}"
LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-9909}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}"
EMBED_SERVED_NAME="${EMBED_SERVED_NAME:-Qwen3-Embedding-4B}"
EMBED_HOST="${EMBED_HOST:-127.0.0.1}"
EMBED_PORT="${EMBED_PORT:-9009}"
EMBED_MAX_MODEL_LEN="${EMBED_MAX_MODEL_LEN:-8192}"

LLM_GPU="${LLM_GPU:-0}"
if [[ ! "${LLM_GPU}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: LLM_GPU must be a non-negative integer." >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  GPU_COUNT=0
fi
if (( GPU_COUNT == 0 )); then
  echo "ERROR: No NVIDIA GPU is visible. Choose a CUDA-enabled Vast.ai instance." >&2
  exit 1
fi
if (( LLM_GPU >= GPU_COUNT )); then
  echo "ERROR: LLM_GPU=${LLM_GPU}, but only ${GPU_COUNT} GPU(s) are visible." >&2
  exit 1
fi

if (( GPU_COUNT > 1 )); then
  EMBED_GPU="${EMBED_GPU:-1}"
else
  EMBED_GPU="${EMBED_GPU:-0}"
fi
if [[ ! "${EMBED_GPU}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EMBED_GPU must be a non-negative integer." >&2
  exit 1
fi
if (( EMBED_GPU >= GPU_COUNT )); then
  echo "ERROR: EMBED_GPU=${EMBED_GPU}, but only ${GPU_COUNT} GPU(s) are visible." >&2
  exit 1
fi

if [[ "${LLM_GPU}" == "${EMBED_GPU}" ]]; then
  LLM_GPU_MEMORY_UTILIZATION="${LLM_GPU_MEMORY_UTILIZATION:-0.65}"
  EMBED_GPU_MEMORY_UTILIZATION="${EMBED_GPU_MEMORY_UTILIZATION:-0.28}"
  echo "WARNING: Chat and embedding servers share GPU ${LLM_GPU}. A 48 GB or larger GPU is recommended." >&2
else
  LLM_GPU_MEMORY_UTILIZATION="${LLM_GPU_MEMORY_UTILIZATION:-0.90}"
  EMBED_GPU_MEMORY_UTILIZATION="${EMBED_GPU_MEMORY_UTILIZATION:-0.85}"
fi

OPENAI_API_KEY="${OPENAI_API_KEY:-local-vastai-key}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-3600}"
MAX_QUERIES="${MAX_QUERIES:-1}"
FORCE="${FORCE:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
VLLM_SPEC="${VLLM_SPEC:-vllm>=0.17.0}"
BASE_AGENT_CONFIG="${BASE_AGENT_CONFIG:-config/sequential_mem0_qwen35_9b.yaml}"
DATASET_CONFIG="${DATASET_CONFIG:-benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/outputs/Qwen3.5-9B-mem0}"

export OPENAI_API_KEY
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
export MEM0_TELEMETRY="${MEM0_TELEMETRY:-false}"

mkdir -p "${LOG_DIR}" "${HF_HOME}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} is not installed in this Vast.ai image." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  python -m pip install --upgrade pip setuptools wheel
  if ! python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
try:
    from packaging.version import Version
    installed = Version(version("vllm"))
except (PackageNotFoundError, ImportError):
    raise SystemExit(1)
raise SystemExit(0 if installed >= Version("0.17.0") else 1)
PY
  then
    python -m pip install "${VLLM_SPEC}"
  fi
  python -m pip install -r requirements-mem0-vast.txt
fi

for required_command in curl vllm; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "ERROR: Required command '${required_command}' is unavailable." >&2
    exit 1
  fi
done

LLM_PID=""
EMBED_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ "${KEEP_SERVERS}" != "1" ]]; then
    for pid in "${EMBED_PID}" "${LLM_PID}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
      fi
    done
  fi
  exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

api_ready() {
  local url=$1
  curl --fail --silent --show-error \
    --header "Authorization: Bearer ${OPENAI_API_KEY}" \
    "${url}/models" >/dev/null 2>&1
}

wait_for_api() {
  local name=$1
  local url=$2
  local pid=$3
  local log_path=$4
  local deadline=$((SECONDS + SERVER_WAIT_SECONDS))

  until api_ready "${url}"; do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "ERROR: ${name} server exited before becoming ready. Last log lines:" >&2
      tail -n 80 "${log_path}" >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: Timed out waiting for ${name} at ${url}. Last log lines:" >&2
      tail -n 80 "${log_path}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "${name} API ready: ${url}"
}

verify_served_model() {
  local name=$1
  local url=$2
  local expected=$3
  curl --fail --silent --show-error \
    --header "Authorization: Bearer ${OPENAI_API_KEY}" \
    "${url}/models" | python -c '
import json, sys
expected = sys.argv[1]
ids = [item.get("id") for item in json.load(sys.stdin).get("data", [])]
if expected not in ids:
    raise SystemExit(f"Expected served model {expected!r}; API returned {ids!r}")
' "${expected}"
  echo "${name} model verified: ${expected}"
}

LLM_BASE_URL="http://${LLM_HOST}:${LLM_PORT}/v1"
EMBED_BASE_URL="http://${EMBED_HOST}:${EMBED_PORT}/v1"

if api_ready "${LLM_BASE_URL}"; then
  echo "Reusing chat API already running at ${LLM_BASE_URL}"
else
  echo "Starting ${LLM_MODEL} on GPU ${LLM_GPU}; log: ${LOG_DIR}/llm.log"
  CUDA_VISIBLE_DEVICES="${LLM_GPU}" vllm serve "${LLM_MODEL}" \
    --served-model-name "${LLM_SERVED_NAME}" \
    --host "${LLM_HOST}" \
    --port "${LLM_PORT}" \
    --api-key "${OPENAI_API_KEY}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${LLM_GPU_MEMORY_UTILIZATION}" \
    --language-model-only \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    >"${LOG_DIR}/llm.log" 2>&1 &
  LLM_PID=$!
fi
wait_for_api "Chat" "${LLM_BASE_URL}" "${LLM_PID}" "${LOG_DIR}/llm.log"
verify_served_model "Chat" "${LLM_BASE_URL}" "${LLM_SERVED_NAME}"

if api_ready "${EMBED_BASE_URL}"; then
  echo "Reusing embedding API already running at ${EMBED_BASE_URL}"
else
  echo "Starting ${EMBED_MODEL} on GPU ${EMBED_GPU}; log: ${LOG_DIR}/embedding.log"
  CUDA_VISIBLE_DEVICES="${EMBED_GPU}" vllm serve "${EMBED_MODEL}" \
    --served-model-name "${EMBED_SERVED_NAME}" \
    --host "${EMBED_HOST}" \
    --port "${EMBED_PORT}" \
    --api-key "${OPENAI_API_KEY}" \
    --task embed \
    --max-model-len "${EMBED_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${EMBED_GPU_MEMORY_UTILIZATION}" \
    >"${LOG_DIR}/embedding.log" 2>&1 &
  EMBED_PID=$!
fi
wait_for_api "Embedding" "${EMBED_BASE_URL}" "${EMBED_PID}" "${LOG_DIR}/embedding.log"
verify_served_model "Embedding" "${EMBED_BASE_URL}" "${EMBED_SERVED_NAME}"

export BASE_AGENT_CONFIG RUNTIME_CONFIG LLM_SERVED_NAME LLM_BASE_URL
export EMBED_SERVED_NAME EMBED_BASE_URL OUTPUT_DIR MAX_MODEL_LEN
python - <<'PY'
import os
from pathlib import Path

import yaml

source_path = Path(os.environ["BASE_AGENT_CONFIG"])
target_path = Path(os.environ["RUNTIME_CONFIG"])
with source_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config.update(
    model=os.environ["LLM_SERVED_NAME"],
    base_url=os.environ["LLM_BASE_URL"],
    model_context_window=int(os.environ["MAX_MODEL_LEN"]),
    mem0_embedder_model=os.environ["EMBED_SERVED_NAME"],
    embedding_base_url=os.environ["EMBED_BASE_URL"],
    output_dir=os.environ["OUTPUT_DIR"],
    qwen3_disable_thinking=True,
)
target_path.parent.mkdir(parents=True, exist_ok=True)
with target_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
print(f"Runtime agent config: {target_path}")
PY

benchmark_args=(
  main.py
  --agent_config "${RUNTIME_CONFIG}"
  --dataset_config "${DATASET_CONFIG}"
)
if [[ "${MAX_QUERIES}" =~ ^[0-9]+$ ]] && (( MAX_QUERIES > 0 )); then
  benchmark_args+=(--max_test_queries_ablation "${MAX_QUERIES}")
elif [[ ! "${MAX_QUERIES}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAX_QUERIES must be a non-negative integer." >&2
  exit 1
fi
if [[ "${FORCE}" == "1" ]]; then
  benchmark_args+=(--force)
fi

echo "Running MemoryData with dataset config: ${DATASET_CONFIG}"
if (( MAX_QUERIES > 0 )); then
  echo "Query limit: ${MAX_QUERIES} (set MAX_QUERIES=0 for the full benchmark)"
else
  echo "Query limit: full benchmark"
fi
python "${benchmark_args[@]}"
