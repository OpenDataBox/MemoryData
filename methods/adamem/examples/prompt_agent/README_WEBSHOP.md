# WebShop Prompt Agent

This directory contains the implementation of a prompt-based agent for the WebShop e-commerce environment, with optional memory retrieval capabilities.

## Files

- **`gpt4o_webshop.py`**: Main WebShop agent implementation with memory support
- **`gpt4o_alfworld.py`**: ALFWorld agent implementation (for reference)
- **`run_gpt4o_webshop.sh`**: SLURM script to run WebShop agent with vLLM server
- **`run_gpt4o_agent.sh`**: SLURM script to run ALFWorld agent (for reference)

## Quick Start

### 1. Basic Usage (No Memory)

```bash
export SPLIT="eval"  # 'train' or 'eval'
export MEM_TYPE=""  # Leave empty for no memory
export MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
export OPENAI_BASE_IP_ADDR="127.0.0.1"

python3 -m examples.prompt_agent.gpt4o_webshop
```

### 2. With Memory Retrieval

```bash
export SPLIT="eval"
export MEM_TYPE="step_strategy_optional"  # See memory types below
export MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
export OPENAI_BASE_IP_ADDR="127.0.0.1"

python3 -m examples.prompt_agent.gpt4o_webshop
```

### 3. Using SLURM

```bash
sbatch examples/prompt_agent/run_gpt4o_webshop.sh
```

## Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `SPLIT` | Dataset split | `"eval"` | `"train"`, `"eval"` |
| `MEM_TYPE` | Memory retrieval type | `""` (none) | See below |
| `MODEL_NAME` | LLM model name | `"Qwen/Qwen3-4B-Instruct-2507"` | Any vLLM-compatible model |
| `OPENAI_BASE_IP_ADDR` | vLLM server address | `"127.0.0.1"` | IP address or comma-separated list |
| `OPENAI_API_KEY` | API key for vLLM | `"EMPTY"` | Any string |
| `CORRECT_ONLY` | Filter correct trajectories | `"false"` | `"true"`, `"false"` |
| `MAX_STEPS` | Maximum steps per episode | `15` | Any integer |

## Memory Types

The agent supports 7 memory retrieval strategies:

1. **No Memory** (`MEM_TYPE=""`)
   - Direct action generation without memory

2. **`step_strategy_optional`**
   - Agent decides when to request memory retrieval
   - Two-stage: initial response → (optional) memory retrieval → final action

3. **`step_strategy_random`**
   - Random memory retrieval (50% probability)
   - Single-stage with randomly injected memory

4. **`step_strategy_three_stage`**
   - Agent-requested three-stage process
   - Initial → Strategy from memory → Action from strategy

5. **`step_strategy_three_stage_every_step`**
   - Always retrieve memory at every step
   - Strategy generation → Action from strategy

6. **`step_strategy_three_stage_random_step`**
   - Random three-stage retrieval (50% probability)
   - Either three-stage with memory OR direct action

7. **`step_strategy_three_stage_no_retrieval`**
   - Three-stage without memory (control)
   - Strategy generation → Action from strategy (no memory)

8. **`step_strategy_direct_retrieval_every_step`**
   - Always retrieve and directly use memory
   - No intermediate strategy generation

## Dataset Splits

### Evaluation (`SPLIT="eval"`)
- **500 goals** (indices 0-499)
- **Deterministic**: Each goal evaluated exactly once
- **Requirement**: `env_num` must equal 500
- **Test times**: 1 (single pass through all goals)

### Training (`SPLIT="train"`)
- **~12K goals** (indices 500+)
- **Stochastic**: Random sampling with replacement
- **Flexibility**: `env_num` can be any number
- **Test times**: 20 (default, can be adjusted)

## Output Files

Results are saved to `logs/webshop/{MODEL_NAME}/`:

- **`traj_{split}.json`**: Trajectory data with step-by-step information
- **`traj_{split}_{mem_type}.json`**: Trajectories with memory type suffix
- **`stats_{split}.txt`**: Summary statistics and logs

## Key Parameters

```python
max_steps = 15  # WebShop episodes typically complete within 15 steps
env_num = 500   # Must be 500 for eval (deterministic)
test_times = 1  # Single pass for eval (deterministic)
```

## Evaluation Metrics

The agent tracks:

1. **Success Rate**: Binary task completion (reward = 1.0)
2. **Task Score**: Continuous score (0.0-1.0) measuring match quality
3. **Memory Statistics** (if memory enabled):
   - Retrieval rate
   - Action change rate
   - Steps/retrievals per trajectory

## Implementation Details

### Deterministic Evaluation
- Modified `WebshopMultiProcessEnv` to assign fixed goals to workers
- Each worker gets a specific goal during evaluation
- Ensures reproducible results across runs

### Memory Integration
- Imports prompt templates from `agent_system/environments/prompts/webshop.py`
- Uses `utils.get_top_k_memories()` for retrieval (requires separate memory index)
- Supports all 7 memory strategies from ALFWorld

### Action Format
- **Search**: `search[<query>]`
- **Click**: `click[<text>]`
- Actions must be enclosed in `<action>...</action>` tags
- Reasoning must be in `<think>...</think>` tags

## Differences from ALFWorld

| Feature | ALFWorld | WebShop |
|---------|----------|---------|
| Max Steps | 50 | 15 |
| Eval Tasks | 134-140 | 500 |
| Success Metric | Binary only | Binary + Score (0-1) |
| Action Format | Natural language | `search[...]` / `click[...]` |
| Task Types | 6 categories | Open-ended shopping |

## Troubleshooting

### Common Issues

1. **"num_processes should equal number of eval goals"**
   - For eval, set `env_num=500` exactly
   - Ensure `SPLIT="eval"` is set correctly

2. **Empty responses from vLLM**
   - Check vLLM server is running: `curl http://127.0.0.1:8001/v1/models`
   - Increase `max_tokens` for complex reasoning

3. **Memory retrieval errors**
   - Ensure memory index is built (see `build_index.py`)
   - Check `utils.py` has `get_top_k_memories()` function

## Related Files

- **Environment**: `agent_system/environments/env_package/webshop/`
- **Prompts**: `agent_system/environments/prompts/webshop.py`
- **Manager**: `agent_system/environments/env_manager.py` (WebshopEnvironmentManager)
- **Projection**: `agent_system/environments/env_package/webshop/projection.py`

## Citation

If you use this code, please cite the original WebShop paper and the verl-agent framework.
