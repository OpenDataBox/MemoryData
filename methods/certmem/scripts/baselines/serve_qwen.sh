#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for baseline systems (run from certmem venv). Leaves GPU room for their embedders.
set -e
export VLLM_USE_V1=1
nohup vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 40960 --gpu-memory-utilization 0.70 \
  --served-model-name Qwen/Qwen3-8B --dtype bfloat16 > /workspace/vllm_server.log 2>&1 &
echo "waiting for server..."
for i in $(seq 1 60); do
  if curl -s http://localhost:8000/v1/models | grep -q Qwen; then echo "server up"; exit 0; fi
  sleep 5
done
echo "server did not start; see /workspace/vllm_server.log"; exit 1
