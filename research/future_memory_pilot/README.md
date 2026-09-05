# Future-continuation memory pilot

This is an untrained, six-case mechanism pilot, not an official LoCoMo or
LongMemEval benchmark result. No weights are updated. The cases are authored
conditional examples, not randomly sampled benchmark questions.

The question is narrow: when several equally small memories are built from the
same history prefix, does selecting one that survives different continuations
help on a continuation held out from selector probes? Its conditional mapping
is already in the prefix, so this is not testing unseen knowledge.

## Protocol

1. Generate a shared pool of three candidate memories from each prefix alone.
   The default memory budget is 96 tokens. Candidate generation cannot see the
   continuation or its questions and answers.
2. Use two selection continuations to estimate each candidate's downstream
   error. Compare selection by mean excess error and worst excess error against
   choosing the first candidate. All selectors use the same candidate pool.
3. Evaluate on one held-out continuation per case. Its questions and answers
   must not influence candidate generation or candidate selection.
4. Also evaluate the original prefix plus continuation (source control), and
   continuation alone (future-only control). These distinguish compression
   damage from reader failure and questions answerable without prefix memory.

The relevant comparison is held-out answer quality under the same memory
budget. A win here is only evidence that this small selection experiment is
worth extending. Six authored cases do not establish generalization, statistical
significance, a trainable method, or superiority over the six existing baselines.
Selection-continuation supervision is an experimental resource, not something
available when an ordinary deployed agent first receives a prefix.

Embedding is health-checked only. The short-prefix experiment isolates memory
compression and continuation handling; it does not test retrieval quality.

## Local reproduction

From the repository root on this Windows workspace:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File research/future_memory_pilot/run_local.ps1
```

Or from Git Bash:

```bash
bash research/future_memory_pilot/run.sh
```

For a smaller smoke check:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File research/future_memory_pilot/run_local.ps1 -Limit 1 -Budget 96
```

The launcher uses the existing `.venv-six/Scripts/python.exe` and the existing
`tools/llama-b10796/bin/llama-server.exe`. It expects these downloaded models in
the parent workspace's `pilot_runtime/models` directory:

- `qwen35/Qwen_Qwen3.5-0.8B-Q8_0.gguf`
- `qwen3-embed/Qwen3-Embedding-0.6B-Q8_0.gguf`

It refuses occupied ports 9911 and 9010, starts hidden localhost-only servers,
waits up to approximately 300 seconds for cold-start health, and runs `run_pilot.py` with
`--output <unique-output-directory>/run --budget 96 --limit 6` by default.
Chat uses context 4096, one slot, six CPU threads, all GPU layers, the model's
Jinja template, and reasoning disabled. Embedding uses context 2048, last-token
pooling, and batch/ubatch 2048. No paid API key is required.

Each launch creates `results/future_memory_pilot_<timestamp>/` for separate
server logs and experiment output. The launcher stops only the processes it
started, including after failure, and returns the runner's exit code. It never
stops a pre-existing server. Review the generated results before interpreting
success; a completed process is not itself evidence that the method works.

The installed Python environment currently has CPU-only PyTorch and no PEFT or
TRL. This reproduction uses llama.cpp inference, not LoRA or RL training.

## First actual run: 2026-09-05

Completed with Qwen3.5-0.8B Q8_0, seed 20260905, three memories per case and a
96-token maximum (actual memories: 49–96 tokens; prefixes: 286–314 tokens).
The metric below is exact-alias accuracy over 24 held-out answer fields, not
six independent benchmark accuracy estimates.

| Reader context / candidate rule | Held-out field accuracy |
|---|---:|
| Original prefix + new message | 83.3% (20/24) |
| New message alone | 0.0% (0/24) |
| First generated memory | 33.3% (8/24) |
| Mean-excess-error selection | 75.0% (18/24) |
| Worst-excess-error selection | 75.0% (18/24) |

Among the five cases the source reader solved, first-memory accuracy was 40%
and both selected-memory accuracies were 90%. The travel source reader chose
the one-day plan even when the new message specified two or four days; this
is a reader failure, not evidence of compression damage.

The selectors picked different candidates only for catering. That difference
was caused by a worst-objective tie, not a strict reversal of preferences:
all candidates had worst excess error 1.0. There were **zero strict objective
reversals** and no held-out accuracy advantage of worst over mean selection.
Thus this run supports investigating candidate quality, not the claimed
novelty or advantage of a robust future objective. Selection has extra probe
supervision compared with the first-memory baseline; its gain is not free.

Nine of eighteen writer outputs hit the generation limit. All stored memories
passed the actual tokenizer budget check. The 100 local chat calls took
450.1 seconds, excluding server startup and the embedding health check.
Cold server startup took about 116 seconds; an earlier attempt with a
120-second health deadline failed and was retained separately. Both servers
started by the successful launcher were stopped after evaluation.

Artifacts: `results/future_memory_pilot_20260905_215512_551/run/` contains
`report.md`, `results.json`, and every prompt/answer/timing in `calls.jsonl`.
The evaluation and Python skills shaped the fixed-budget, no-gold-leakage
protocol, deterministic grading and offline tests. Do not call this a learned
writer, official benchmark result, or a win over Mem0/A-MEM/SimpleMem/LangMem/
E-Mem/LightMem. A next research test needs disjoint natural histories and a
matched-compute hypothetical-QA comparison, followed by actual writer training.
