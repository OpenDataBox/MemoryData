# Config Model Roles

This document explains what each model-related field in [`config.kv.yaml`](../config.kv.yaml) and [`config.text.yaml`](../config.text.yaml) does.

The most important idea is:

- `memory_agent_model` is always configured explicitly.
- `general_model` is the default OpenAI-compatible model for the rest of the pipeline.
- `manager_model`, `aggregator_model`, `router_fallback_model`, and `question_answer_model` are optional overrides.
- If an override is omitted or set to `null`, E-mem uses `general_model` for that role.

## The Recommended Mental Model

Most users only need to think about two model layers:

- `memory_agent_model`: the model used by the memory blocks themselves
- `general_model`: the default model used everywhere else

That is enough for:

- normal usage,
- reproducing the main experiment setup,
- approximate reproduction in either `kv_cache` mode or `text` mode.

You only need role-specific overrides when you want finer experimental control, such as:

- testing a stronger `aggregator_model`,
- using a different final `question_answer_model` for benchmark scoring,
- giving the router its own fallback model,
- or isolating the interactive `manager_model`.

## Override Rule

The inheritance rule is simple:

```yaml
model:
  memory_agent_model: ...
  general_model:
    openai_config: ...

  manager_model: null
  aggregator_model: null
  question_answer_model: null
  router_fallback_model: null
```

When a role override is `null` or missing:

- `manager_model -> general_model`
- `aggregator_model -> general_model`
- `question_answer_model -> general_model`
- `router_fallback_model -> general_model`

`memory_agent_model` never falls back to `general_model`.

## YAML To Runtime Mapping

If you also use the Python factory interface, the YAML fields map to runtime kwargs like this:

| YAML field | Factory kwarg(s) | Meaning |
| --- | --- | --- |
| `tokenizer.model_id` | `model_id` | Tokenizer reference used for token counting and block sizing |
| `model.memory_agent_model.model_id` | `model_id` | KV-cache memory model |
| `model.memory_agent_model.openai_config` | `memory_agent_openai_config` | Text-mode memory agent model |
| `model.memory_agent_model.model_context_window` | `model_context_window` | Context window used for block sizing |
| `model.general_model.openai_config` | `chat_openai_config`, `aggregator_openai_config`, `router_openai_config` | Default OpenAI config for non-memory roles |
| `model.manager_model.openai_config` | `chat_openai_config` | Optional manager override |
| `model.aggregator_model.openai_config` | `aggregator_openai_config` | Optional aggregator override |
| `model.router_fallback_model.openai_config` | `router_openai_config` | Optional router fallback override |

`question_answer_model` does not map to `create_chat_manager(...)` because it is only consumed directly by evaluation scripts.

## Field-by-Field Meaning

### `tokenizer.model_id`

This is the tokenizer reference used for token counting and memory block sizing.

Use it when:

- you want block sizing to match a specific OSS model,
- you are in text mode and the memory agent itself is API-based,
- you still need a tokenizer locally to estimate chunk size.

Typical value:

```yaml
tokenizer:
  model_id: "Qwen/Qwen3-4B"
```

### `model.memory_agent_model`

This is the model block for memory storage itself.

In `kv_cache` mode:

- `model_id` is required,
- it points to the HuggingFace / local causal LM used to build KV caches,
- `model_context_window`, `attn_implementation`, `device_map`, and `quantization_config` belong here.

In `text` mode:

- `openai_config` is required,
- it is the API model used inside the text memory agent for summarization and block-level querying,
- `model_context_window` is still important because block size is still computed from it.

Typical `kv_cache` example:

```yaml
model:
  memory_agent_model:
    model_id: "Qwen/Qwen3-4B"
    model_context_window: 32768
    attn_implementation: "sdpa"
    device_map: "auto"
```

Typical `text` example:

```yaml
model:
  memory_agent_model:
    openai_config:
      api_key: "..."
      base_url: "..."
      model: "qwen3-4b"
    model_context_window: 32768
```

### `model.general_model`

This is the default OpenAI-compatible model for all non-memory roles.

By default it supplies the model used for:

- top-level manager behavior,
- retrieval aggregation,
- router LLM fallback,
- evaluation final answer generation.

If you do not care about role-by-role ablations, this is the only general LLM field you need to set.

Typical example:

```yaml
model:
  general_model:
    openai_config:
      api_key: "..."
      base_url: "..."
      model: "gpt-4o-mini"
```

### `model.manager_model`

This is an optional override for the top-level manager model used by `ChatManager` / `TextStorageChatManager`.

It is responsible for:

- deciding whether to call tools like `add_memory` or `query_memory`,
- driving the top-level interaction loop,
- acting as the controller model in normal chat usage.

If this field is `null` or omitted, E-mem uses `general_model`.

> [!warning]
> In the current LoCoMo and HotpotQA evaluation scripts, `manager_model` is not on the main answer-generation path.
> Those scripts store memory, retrieve evidence, and then generate the scored answer through `question_answer_model` or, if that override is absent, through `general_model`.

### `model.aggregator_model`

This is an optional override for the aggregation model used after retrieval.

It takes:

- raw block-level results,
- multiple memory snippets,
- possibly redundant or noisy retrieval outputs,

and turns them into a cleaner aggregated memory result.

If this field is `null` or omitted, E-mem uses `general_model`.

### `model.router_fallback_model`

This is an optional override for the router LLM fallback.

That fallback is relevant when:

- `memory.router_type: "llm"`, or
- `memory.router_type: "hybrid"` and `memory.hybrid_router.use_llm_fallback: true`.

If this field is `null` or omitted, E-mem uses `general_model`.

If your router never needs an LLM fallback, the field may remain `null` forever.

### `model.question_answer_model`

This is an optional override for evaluation-only final answer generation.

It is mainly used by scripts such as LoCoMo and HotpotQA.

More concretely:

- `memory_agent_model` and retrieval logic find the relevant evidence,
- `aggregator_model` or `general_model` cleans and merges the retrieved evidence,
- `question_answer_model` or `general_model` writes the final benchmark answer that is scored.

If this field is `null` or omitted, E-mem uses `general_model`.

In real product usage, there is often no separate evaluation answer-writer stage.
Instead, the top-level user-facing response is usually produced through the manager path, so `manager_model` is the closest practical counterpart.

That said, the two roles are still not identical:

- `manager_model` is responsible for orchestration and tool use in interactive chat,
- `question_answer_model` is a pure final-answer role used by evaluation.

## Which Fields Matter In Each Mode

### `kv_cache` mode

Required in practice:

- `tokenizer.model_id`
- `model.memory_agent_model.model_id`
- `model.memory_agent_model.model_context_window`
- `model.general_model.openai_config`

Optional overrides:

- `model.manager_model.openai_config`
- `model.aggregator_model.openai_config`
- `model.router_fallback_model.openai_config`
- `model.question_answer_model.openai_config`

All of these optional overrides fall back to `general_model`.

### `text` mode

Required in practice:

- `tokenizer.model_id`
- `model.memory_agent_model.openai_config`
- `model.memory_agent_model.model_context_window`
- `model.general_model.openai_config`

Optional overrides:

- `model.manager_model.openai_config`
- `model.aggregator_model.openai_config`
- `model.router_fallback_model.openai_config`
- `model.question_answer_model.openai_config`

All of these optional overrides fall back to `general_model`.

## How To Choose Values

### The default setup

- `memory_agent_model`: the memory model from the paper setup
- `general_model`: the shared API model for the rest of the system
- all overrides: leave as `null`

This is the recommended starting point.

### A finer-grained experiment setup

Start from the default setup, then override only the role you want to isolate:

- stronger `aggregator_model` for retrieval consolidation,
- different `question_answer_model` for benchmark scoring,
- separate `manager_model` for interactive quality,
- separate `router_fallback_model` for routing ablations.

## Example Mental Model

For a LoCoMo run in `text` mode, you can think of the fields like this:

- `tokenizer.model_id`: how to estimate block size
- `memory_agent_model`: how each memory block summarizes and answers locally
- `general_model`: the default LLM for the non-memory pipeline
- `manager_model`: optional override for interactive manager behavior
- `aggregator_model`: optional override for retrieval merging
- `router_fallback_model`: optional override when the router needs an LLM
- `question_answer_model`: optional override for the final benchmark answer

## Recommended Practice

- Start with only `memory_agent_model` and `general_model`.
- Add role overrides only when you actually want a role-specific experiment.
- Keep `question_answer_model` explicit only if you want benchmark final-answer ablations.
- Keep `router_fallback_model` explicit only if you want router-specific ablations.
- Do not assume `manager_model` and `aggregator_model` are interchangeable, even if they share the same default config.

## Related Files

- [Config Reference](CONFIG_REFERENCE.md)
- [API Reference](API_REFERENCE.md)
- [Operational Guides](GUIDES.md)
- [KV Example Config](../config.kv.yaml)
- [Text Example Config](../config.text.yaml)
