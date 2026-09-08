#!/usr/bin/env bash
# Two vLLM OpenAI servers on one A100: chat (Qwen3-8B, :8000) and embeddings (Qwen3-Embedding-0.6B, :8001).
set -e
export VLLM_USE_V1=1
pkill -f "vllm serve" || true; sleep 3
nohup vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 40960 --gpu-memory-utilization 0.62 \
  --served-model-name Qwen3-8B Qwen/Qwen3-8B --dtype bfloat16 > /workspace/vllm_chat.log 2>&1 &
sleep 60
nohup vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8001 --gpu-memory-utilization 0.15 \
  --served-model-name Qwen3-Embedding-0.6B Qwen/Qwen3-Embedding-0.6B > /workspace/vllm_embed.log 2>&1 &
for i in $(seq 1 60); do
  a=$(curl -s http://localhost:8000/v1/models | grep -c Qwen || true); b=$(curl -s http://localhost:8001/v1/models | grep -c Qwen || true)
  if [ "$a" -gt 0 ] && [ "$b" -gt 0 ]; then echo "both servers up"; exit 0; fi; sleep 5
done
echo "servers not up; see /workspace/vllm_chat.log /workspace/vllm_embed.log"; exit 1
