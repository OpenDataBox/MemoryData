#!/usr/bin/env bash
# Run 5 memory systems + embedding-RAG reference on LoCoMo 10 via MemoryData, then score each with our metric.
# Usage: bash scripts/baselines/run_memorydata.sh [method ...]   (default: mem0 lightmem a_mem simplemem memoryos embedding_rag)
set -u
METHODS=${@:-"mem0 lightmem a_mem simplemem memoryos embedding_rag"}
declare -A CFG=( [mem0]=sequential_mem0 [lightmem]=hybrid_lightmem [a_mem]=hybrid_a_mem [simplemem]=hybrid_simplemem [memoryos]=hybrid_memoryos [embedding_rag]=reference_embedding_rag )
cd /workspace/baselines/MemoryData && source .venv/bin/activate && export OPENAI_API_KEY=EMPTY
for m in $METHODS; do
  echo "=== $m ==="
  python main.py --agent_config config/${CFG[$m]}.yaml --dataset_config benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml > /workspace/baseline_${m}.log 2>&1 \
    && echo "$m finished" || { echo "$m FAILED (see /workspace/baseline_${m}.log)"; continue; }
  res=$(ls -t results/outputs/*/LoCoMo/*_results.json 2>/dev/null | head -1)
  ( cd /workspace/certmem && source .venv/bin/activate && python scripts/baselines/score_external.py "$res" --system $m --fields query,output,answer,category --out runs/baseline_$m )
done
echo "MEMORYDATA DONE"
