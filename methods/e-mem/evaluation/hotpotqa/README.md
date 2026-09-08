# HotpotQA Evaluation

Evaluation script for HotpotQA dataset using E-mem.

## Setup
1. Download the dataset
  - Download the hotpotqa dataset from [`HotpotQA`](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main).
    - `eval_400.json`
    - `eval_800.json`
    - `eval_1600.json`
2. Copy the example config:
```bash
cp evaluation/hotpotqa/config.kv.yaml evaluation/hotpotqa/config.yaml
# Or use text mode:
# cp evaluation/hotpotqa/config.text.yaml evaluation/hotpotqa/config.yaml
```

3. Edit `evaluation/hotpotqa/config.yaml` with your settings:
   - Set your OpenAI API key and base URL
   - Configure model settings
   - Set dataset path
   - Other configurations

The repository-root `config.yaml` is for quickstart/examples. `bash scripts/eval_hotpotqa.sh` defaults to `evaluation/hotpotqa/config.yaml`.

If you are not sure what each `model.*` field means, read [Config Model Roles](../../docs/CONFIG_MODELS.md) first.
If you want the meaning of the memory, router, evaluation, or logging fields, read [Config Reference](../../docs/CONFIG_REFERENCE.md).

## Usage

### Using Shell Script
```bash
bash scripts/eval_hotpotqa.sh

# Override specific settings
bash scripts/eval_hotpotqa.sh --ratio 0.1
```

## How It Works

For each sample:
1. **Create Agent**: Initialize ChatManager with `clean_cache_first=True`
2. **Add Context**: Use `add_memory()` to store the context
3. **Query**: Use `search_memory()` to answer the question
4. **Cleanup**: Delete agent and clear GPU cache to avoid OOM
5. **Calculate Metrics**: Token-level F1 between prediction and ground truth

## Output

Results are saved to `evaluation/hotpotqa/results/` with:
  - Fine-grained results of each question
  - Overall metrics (average F1 score) and the F1 score of each qa

Logs are saved to `evaluation/hotpotqa/logs/hotpotqa_eval_<timestamp>.log`
