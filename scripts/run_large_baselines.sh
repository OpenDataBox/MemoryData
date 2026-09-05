#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run six memory baselines with larger Qwen3.5 models.

Usage:
  bash scripts/run_large_baselines.sh [run] [--full] [runner options]
  bash scripts/run_large_baselines.sh serve-chat [extra vLLM options]
  bash scripts/run_large_baselines.sh serve-embedding [extra vLLM options]

Start servers in separate terminals; wait for readiness before running.
Run defaults to one history and one question per method/dataset.

Environment:
  CHAT_SIZE            0.8B | 4B | 9B (default) | 27B
  EMBEDDING_SIZE       0.6B (default, 1024 dims) | 4B (2560) | 8B (4096)
  MODEL                Served chat name (default: Qwen3.5-${CHAT_SIZE})
  LLM_MODEL            Checkpoint ID/path (default: Qwen/Qwen3.5-${CHAT_SIZE})
  EMBEDDING_MODEL      Served embedding name (default: Qwen3-Embedding-${EMBEDDING_SIZE})
  EMBED_MODEL          Embedding checkpoint ID/path
  EMBEDDING_DIM        Actual server output dimension (default from size)
  OPENAI_BASE_URL      Chat endpoint (default: http://127.0.0.1:9911/v1)
  EMBEDDING_BASE_URL   Embedding endpoint (default: http://127.0.0.1:9010/v1)
  OPENAI_API_KEY       Key for both APIs (default: EMPTY)
  ARTIFACT_ROOT        Override model/embedding/dimension/mode output directory
  PYTHON_BIN           Benchmark Python executable (existing wrapper auto-detects)

Server environment (Linux/CUDA, compatible vLLM already installed):
  CUDA_VISIBLE_DEVICES   GPU IDs for this server; set explicitly per terminal
  TP                    Tensor parallel size (default: 1)
  GPU_MEMORY_UTILIZATION  Per-server GPU memory fraction (default: 0.9)
  MAX_MODEL_LEN         Chat context (default: 65536)
  EMBED_MAX_MODEL_LEN   Embedding context (default: 8192)
  CHAT_PORT / EMBED_PORT  Local ports (defaults: 9911 / 9010)
  TOOL_CALL_PARSER      Chat tool parser (default: qwen3_coder)

Use environment variables for model, endpoint and artifact settings.
Other runner options (--methods, --datasets, --dry-run) are forwarded.
See scripts/LARGE_BASELINES.md for examples.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
action=run
case "${1:-}" in
  run|serve-chat|serve-embedding) action="$1"; shift ;;
  -h|--help) usage; exit 0 ;;
  ''|--*) ;;
  *) echo "Unknown action: $1" >&2; exit 2 ;;
esac

CHAT_SIZE="${CHAT_SIZE:-9B}"
EMBEDDING_SIZE="${EMBEDDING_SIZE:-0.6B}"
case "$CHAT_SIZE" in
  0.8B|4B|9B|27B) ;;
  *) echo 'CHAT_SIZE must be 0.8B, 4B, 9B or 27B.' >&2; exit 2 ;;
esac
case "$EMBEDDING_SIZE" in
  0.6B) default_dim=1024 ;;
  4B) default_dim=2560 ;;
  8B) default_dim=4096 ;;
  *) echo 'EMBEDDING_SIZE must be 0.6B, 4B or 8B.' >&2; exit 2 ;;
esac

export MODEL="${MODEL:-Qwen3.5-${CHAT_SIZE}}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen3-Embedding-${EMBEDDING_SIZE}}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-$default_dim}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:${CHAT_PORT:-9911}/v1}"
export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:${EMBED_PORT:-9010}/v1}"

case "$action" in
  serve-chat)
    exec vllm serve "${LLM_MODEL:-Qwen/Qwen3.5-${CHAT_SIZE}}" \
      --served-model-name "$MODEL" --host 127.0.0.1 --port "${CHAT_PORT:-9911}" \
      --api-key "$OPENAI_API_KEY" --tensor-parallel-size "${TP:-1}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
      --max-model-len "${MAX_MODEL_LEN:-65536}" --max-num-seqs 1 \
      --language-model-only --enable-auto-tool-choice \
      --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
      --default-chat-template-kwargs '{"enable_thinking": false}' "$@"
    ;;
  serve-embedding)
    exec vllm serve "${EMBED_MODEL:-Qwen/Qwen3-Embedding-${EMBEDDING_SIZE}}" \
      --served-model-name "$EMBEDDING_MODEL" --host 127.0.0.1 --port "${EMBED_PORT:-9010}" \
      --api-key "$OPENAI_API_KEY" --runner pooling --tensor-parallel-size "${TP:-1}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
      --max-model-len "${EMBED_MAX_MODEL_LEN:-8192}" "$@"
    ;;
  run)
    mode=smoke
    for arg in "$@"; do
      case "$arg" in
        --full) mode=full ;;
        --model|--model=*|--embedding-model|--embedding-model=*|--embedding-dim|--embedding-dim=*|--base-url|--base-url=*|--embedding-base-url|--embedding-base-url=*|--artifact-root|--artifact-root=*)
          echo "Use the corresponding environment variable instead of $arg (see --help)." >&2
          exit 2 ;;
      esac
    done
    model_tag="${MODEL//[^a-zA-Z0-9._-]/_}"
    embed_tag="${EMBEDDING_MODEL//[^a-zA-Z0-9._-]/_}"
    export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$repo_root/results/large_baselines/${model_tag}__${embed_tag}__d${EMBEDDING_DIM}/${mode}}"
    printf 'Chat: %s\nEmbedding: %s (%s dimensions)\nArtifacts: %s\n' \
      "$MODEL" "$EMBEDDING_MODEL" "$EMBEDDING_DIM" "$ARTIFACT_ROOT"
    exec bash "$repo_root/scripts/run_six_baselines.sh" "$@"
    ;;
esac