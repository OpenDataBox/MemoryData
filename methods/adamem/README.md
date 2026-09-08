<h1 align="center">AdaMEM: Test-Time Adaptive Memory for Language Agents</h1>

<p align="center">
  <b>Yunxiang Zhang &nbsp;·&nbsp; Yiheng Li &nbsp;·&nbsp; Ali Payani &nbsp;·&nbsp; Lu Wang</b>
</p>

<p align="center">
  <b>ICML 2026</b>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/pdf/2606.05684">Paper</a>
  &nbsp;|&nbsp;
  <a href="https://github.com/yunx-z/AdaMEM">Code</a>
  &nbsp;|&nbsp;
  <a href="https://yunx-z.github.io/AdaMEM/">Project Page</a>
</p>

---

## Overview

A central challenge for language agents is utilizing past experience to adapt to dynamic test-time conditions. While recent work demonstrates the promise of agentic memory mechanisms, most systems restrict retrieval to episode initiation. Consequently, agents are forced to rely on static guidance that becomes increasingly misaligned as long-horizon tasks unfold. To address this rigidity, we propose the **Adaptive Memory Agent (AdaMEM)**, a novel framework for agent test-time adaptation. Without updating model parameters online, AdaMEM adapts agent behavior via a hybrid memory architecture: it maintains a long-term trajectory memory of raw experiences collected offline while generating dynamic short-term strategy memory on-the-fly to guide decision-making. This mechanism enables the trade-off between token efficiency and adaptability across varying inference-time compute levels. Empirically, AdaMEM significantly outperforms static memory baselines, achieving relative gains of up to 13% on ALFWorld and 11% on WebShop, with consistent leading performance extending to agentic search on HotpotQA. To further enhance this adaptation, we develop **STEP-MFT**, a Step-wise Memory Fine-Tuning technique that trains the policy to synthesize high-quality strategies from retrieved experiences, yielding additional performance gains. Our work establishes a new scaling dimension for agentic memory, supporting continuous reasoning and self-evolution post-deployment in real-world environments.

---

## Memory Types

| Name | Description |
|---|---|
| `no-memory` | Base ReAct agent with no external memory |
| `synapse` | Episode-level trajectory retrieval ([Synapse](https://openreview.net/forum?id=Pc8AU1aF5e) baseline) |
| `reasoningbank` | Episode-level strategy retrieval ([ReasoningBank](https://arxiv.org/abs/2509.25140) baseline) |
| `adamem-high` | Step-wise strategy synthesis — generates a *transient* strategy at each step the agent requests memory (**AdaMEM-HIGH**) |
| `adamem-low` | Step-wise strategy synthesis with persistent strategy and agent-controlled refresh (**AdaMEM-LOW**) |
| `adamem-max` | Generates a fresh strategy at *every* step unconditionally (**AdaMEM-MAX / ablation**) |
| `adamem-max-without-trajectory-memory` | Strategy synthesis without retrieval (no long-term memory; ablation) |
| `adamem-max-without-strategy-memory` | Raw trajectory injection at every step without synthesis (ablation) |

---

## Installation

```bash
git clone https://github.com/yunx-z/AdaMEM.git
cd AdaMEM
pip install -e .
pip install -r requirements.txt
```

Install the ALFWorld and WebShop environments by following the setup instructions in the verl-agent repo:

- **ALFWorld**: https://github.com/langfengQ/verl-agent#1-alfworld
- **WebShop**: https://github.com/langfengQ/verl-agent#2-webshop

---

## Quick Start

### 1. Build the Long-Term Memory Index

Collect successful trajectories on the training split first (see Step 3 below for how to run a training-split rollout), then build the HNSW index:

```bash
# Step-level index (used by all AdaMEM variants)
python build_index.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507 \
  --correct_only

# Episode-level index (for synapse baseline)
python build_index_traj_level.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507

# Strategy index (for reasoningbank baseline)
python build_index_reasoningbank.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507
```

### 2. Run Inference

Start a vLLM server, then launch the agent:

```bash
# Start vLLM
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8001 \
  --dtype bfloat16 --max-model-len 32768

# AdaMEM-LOW on ALFWorld (seen split)
SPLIT=eval_in_distribution \
MEM_TYPE=adamem-low \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld

# AdaMEM-LOW on WebShop
MEM_TYPE=adamem-low \
MODEL_NAME=langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop \
  python -m examples.prompt_agent.gpt4o_webshop
```

Set `MEM_TYPE` to any value from the table above. Omit `MEM_TYPE` (or set it empty) for the `no-memory` baseline.

**Key environment variables**

| Variable | Default | Description |
|---|---|---|
| `SPLIT` | — | ALFWorld split: `train`, `eval_in_distribution`, `eval_out_of_distribution` |
| `MEM_TYPE` | `None` | Memory mechanism (see table above) |
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | Policy model served by vLLM |
| `STRATEGY_MODEL_NAME` | same as `MODEL_NAME` | Strategy synthesis model (can differ from policy for off-policy setup) |
| `RETRIEVAL_TOPK` | `1` | Number of retrieved experiences *k* |
| `OPENAI_BASE_IP_ADDR` | `127.0.0.1` | vLLM server host:port |

### 3. STEP-MFT Training

STEP-MFT trains the model to generate high-utility strategies using process-level supervision.

**Step 1 — Collect training trajectories with `adamem-high`** (generates dense per-step strategy/action pairs):

```bash
SPLIT=train \
MEM_TYPE=adamem-high \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld
```

**Step 2 — Filter SFT data** using `filter_sft_data.py`:

```bash
# STEP-MFT: keep only strategies that changed the agent's action (recommended)
python filter_sft_data.py \
  --traj_file "logs/alfworld/*/traj_train_adamem-high*.json" \
  --mode step-mft \
  --output_file sft_data_step_mft.json

# Outcome-MFT: keep all strategies from successful trajectories (weaker baseline)
python filter_sft_data.py \
  --traj_file "logs/alfworld/*/traj_train_adamem-high*.json" \
  --mode outcome-mft \
  --output_file sft_data_outcome_mft.json
```

**Step 3 — Fine-tune with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**:

```bash
# Register sft_data_step_mft.json as a dataset in LLaMA-Factory, then:
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --finetuning_type lora \
  --lora_rank 32 \
  --dataset sft_data_step_mft \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1
```

We use NVIDIA A40/L40S GPUs (48 GB). Run inference with the fine-tuned adapter by pointing `MODEL_NAME` and `STRATEGY_MODEL_NAME` at the saved adapter path.

---

## Citation

```bibtex
@inproceedings{zhang2026adamem,
  title     = {{AdaMEM}: Test-Time Adaptive Memory for Language Agents},
  author    = {Zhang, Yunxiang and Li, Yiheng and Payani, Ali and Wang, Lu},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
}
```
