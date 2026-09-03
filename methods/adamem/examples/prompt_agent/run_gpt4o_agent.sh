#!/bin/bash

#SBATCH --account=wangluxy_owned1
#SBATCH --job-name=prompt_alfworld       # Name of the job
#SBATCH --output=logs/gl/prompt_alfworld--%j.log   # File to which the output will be written
#SBATCH --error=logs/gl/prompt_alfworld--%j.log     # File to which the error will be written
#SBATCH --time=07-12:00:00           # Wall time limit of the job (e.g., 1 hour)
#SBATCH --partition=spgpu2           # Partition (or queue) name
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --ntasks=1                # Number of tasks, typically set to 1 for single GPU jobs
#SBATCH --cpus-per-gpu=4         # Number of CPU cores per task
#SBATCH --mem-per-gpu=43GB                 # Amount of memory per node (e.g., 16 GB)
##SBATCH --exclude=gl1506
##SBATCH --dependency=afterany:24617369:24617368

echo "My job ID is $SLURM_JOB_ID"
echo "Running on host $(hostname)"
echo "Starting at $(date)"

source ~/.bashrc
conda activate verl-agent

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 32768 > logs/vllm.log 2>&1 &

echo "Waiting for vLLM to start..."
until curl -fsS http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do
  sleep 2
done
echo "vLLM is ready."

ENV_NAME="alfworld"

if [[ "$ENV_NAME" == "alfworld" ]]; then
  echo "Launching AlfWorld agent..."
  python3 -m examples.prompt_agent.gpt4o_alfworld
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'alfworld'." >&2
  exit 1
fi

echo "Ending at $(date)"
