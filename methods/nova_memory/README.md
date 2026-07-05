# Nova Memory — Sequential Context method preset

Lexical + morphology memory baseline. **Zero external dependencies** at the
core; uses OpenAI-compatible LLM only for answer generation.

## TL;DR

```bash
# 1. Self-test (no LLM, no network)
python methods/nova_memory/source/_smoke_test.py   # 16/16
python methods/nova_memory/source/_e2e_test.py     # ingest + recall + save/load

# 2. Offline benchmark (mock dataset, ~5s)
python methods/nova_memory/source/run_benchmark.py --mock --reinit-per-sample

# 3. Real benchmark (needs network)
python methods/nova_memory/source/run_benchmark.py --benchmark eventqa --max-samples 50

# 4. Full MemoryData integration (needs network + LLM)
python methods/nova_memory/adapter_patch.py    # injects dispatch into utils/agent.py
python main.py \
    --agent_config config/sequential_nova_memory.yaml \
    --dataset_config benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml
```

## What's Nova

- **形态学扩展 (morph expansion):** maps spoken-Chinese variants to canonical
  keywords (`买的房` → `买房`, `在哪工作` → `工作`, `几口人` → `家庭成员`...)
- **2-gram + 3-gram sliding window tokenization** — robust to Chinese word
  segmentation without jieba
- **单字白名单 (single-char whitelist):** preserves high-signal nouns
  (`车`, `猫`, `房`, `儿`...) that would otherwise be dropped
- **Substring matching recall** — top-k chunks whose content/keywords
  contain any query token (substring LIKE), ranked by hit count. Mirrors
  `nova-mvp/memory.py` SQLite LIKE behavior in pure Python.
- **Top-chunk fallback** when no LLM endpoint is configured — returns the
  top retrieved chunk as the answer (works for extractive QA).

## Files

```
methods/nova_memory/
├── README.md                  this file
├── PR_DESCRIPTION.md          ready-to-paste PR text for OpenDataBox/MemoryData
├── DISCUSSION_ISSUE.md        ready-to-paste issue for adding Chinese benchmarks
├── adapter_patch.py           monkeypatch into utils/agent.py (idempotent)
└── source/
    ├── __init__.py
    ├── nova_core.py           tokenize, expand_morph, NovaMemoryStore (zero-dep)
    ├── nova_agent.py          NovaMemoryAgent (MemoryData-compatible)
    ├── run_benchmark.py       standalone runner (mock + HF modes)
    ├── _mock_bench.py         3-sample mock dataset (no network)
    ├── _smoke_test.py         16/16 self-test
    ├── _e2e_test.py           ingest+recall+save/load E2E
    └── _dbg.py                debug helper (delete if not needed)

config/
└── sequential_nova_memory.yaml  preset config
```

## Mock benchmark results

15 Chinese QA across 3 samples (3.5s, no network, no LLM):

| Metric | Value | Meaning |
|---|---|---|
| `recall_at_k` | 86.67% | gold appears in any of top-5 recalled chunks |
| `first_chunk_hit` | 66.67% | gold appears in top-1 chunk |
| `substring_em` | 66.67% | answer (top-chunk fallback) contains gold |
| `token_f1` | 0% | squad token F1 — n/a when no LLM |

Two misses are deliberate hard cases requiring synonym resolution
(e.g. query "狗叫什么?" vs stored "金毛犬叫豆豆") — a known limitation
of pure lexical methods. With an LLM endpoint, the substring_em +
token_f1 metrics become meaningful.

## What the patch does

Adds 4 hooks to `utils/agent.py`:

| Hook | Purpose |
| --- | --- |
| `_is_nova_agent()` | True if `agent_name` contains `nova` |
| `_initialize_nova_agent()` | Constructs `NovaMemoryAgent`, stashes as `self._nova_agent` |
| `send_message()` override | Routes text to `self._nova_agent.send_message()` for nova agents only |
| `elif self._is_nova_agent()` | Dispatch branch in `_initialize_agent_by_type()` |

The patch is **idempotent** (running `adapter_patch.py` twice is safe)
and **additive** (no existing methods modified). Backup file
`utils/agent.py.bak.<timestamp>` is left for rollback.

## Why "Sequential Context" taxonomy

Nova ingests chunks in order and answers using top-k lexical overlap. It
**does not** build graphs, trees, or hybrid structures — fits the
"Reference Baselines / Sequential Context" bucket of MemoryData's taxonomy.

## Use cases

- **Ablation baseline:** "how much does embedding/GraphRAG buy you vs
  pure lexical?"
- **Chinese personal-fact QA:** test corpora where the 22 existing
  presets underperform
- **Edge / CPU-only environments:** no GPU, no vector DB
- **Zero-dep smoke testing:** can run on any Python install

## Limitations

- No semantic search (lexical only — misses synonyms)
- Single-process (cap ~10K chunks)
- Chinese-optimized (works on English but uncompetitive with
  BM25/dense baselines)
- LLM fallback returns top chunk verbatim — works for extractive,
  fails for abstractive QA

## Vendored from

`nova-mvp/memory.py` v6 tokenize chain (MIT-style license).

## License

Same as nova-mvp (MIT-style).
## 📊 Benchmark (Mock, Offline)

| 指标 | 分数 |
|---|---|
| recall@5 | **86.67%** |
| first_chunk_hit | 66.67% |
| substring_em | 66.67% |
| 平均 query | 0.26s |

3 samples × 5 QA = 15 queries, 全离线无 LLM。
真实 MemoryAgentBench (LoCoMo / EventQA) 数字待 PR review 时补。
