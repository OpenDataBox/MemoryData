# PR: Add Nova Memory lexical baseline preset (Sequential Context bucket)

> **Branch:** `feat/nova-memory-preset`
> **Target:** `OpenDataBox/MemoryData` `main`
> **Type:** feat (new method preset)
> **Files changed:** 9 (1 new yaml, 1 patched utils/agent.py, 7 new in methods/nova_memory/)

---

## 🐴 What is Nova Memory

A **lexical + morphology** memory baseline that requires **zero external
dependencies** (no vector DB, no LLM API for the core; OpenAI-compatible
endpoint only for final answer generation).

Three innovations over vanilla BM25/Jaccard:
1. **Spoken-Chinese → canonical morph mapping** (e.g. "买的房" → "买房",
   "开什么车" → "车", "几口人" → "家庭成员")
2. **2-gram + 3-gram sliding window tokenization** — robust to Chinese
   word segmentation without `jieba`
3. **Single-char whitelist** for high-signal nouns (`车`, `猫`, `房`,
   `儿`, `钱`...) that would otherwise be dropped

Recall: **substring matching on top-k chunks**, ranked by hit count. Mirrors
`nova-mvp/memory.py` SQLite LIKE behavior in pure Python.

---

## 📊 Why a new preset?

The 22 existing presets span 4 families:
- **Reference:** long-context, raw-RAG
- **Sequential Context:** LangMem, MemGPT, simplemem, A-Mem, lightmem
- **Structural Topological:** GraphRAG, LightRAG, MemTree, Cognee
- **Multi-Paradigm Hybrid:** Mem0, Zep, Letta

**None** of the 22 do morphology-aware lexical matching. The closest
counterparts (e.g. `simple_rag_bm25`) lack:
- Chinese morph normalization
- Sub-character (2-3 gram) tokenization
- Single-char whitelist preservation

Nova is the **lightest possible baseline** — useful for ablation
("how much does heavy machinery buy you?") and for non-English (Chinese)
sub-tasks where most baselines falter.

---

## 🗂 Files added

```
MemoryData/
├── config/
│   └── sequential_nova_memory.yaml        (new preset)
└── methods/nova_memory/
    ├── README.md                           (integration guide)
    ├── adapter_patch.py                    (idempotent utils/agent.py injector)
    ├── PR_DESCRIPTION.md                   (this file)
    └── source/
        ├── __init__.py
        ├── nova_core.py                    (tokenize + NovaMemoryStore)
        ├── nova_agent.py                   (MemoryData-compatible agent)
        ├── run_benchmark.py                (standalone runner w/ mock + HF modes)
        ├── _mock_bench.py                  (3-sample mock dataset for offline CI)
        ├── _smoke_test.py                  (16/16 self-test)
        └── _e2e_test.py                    (ingest+recall+save/load E2E)
```

`utils/agent.py` gets a 30-line additive patch (3 methods, 1 dispatch
branch) — **zero changes** to existing methods. Patch is idempotent
(`adapter_patch.py --check` or run twice is safe).

---

## 🧪 Tests

### Unit tests
```
$ python methods/nova_memory/source/_smoke_test.py
PASS morph len>=30
PASS morph 买的房 → 买房
PASS morph 在哪工作 → 工作
PASS morph 开什么车 → 车
PASS morph 几口人 → 家庭成员
PASS morph 哪个城市 → 城市
PASS morph 不改原句
PASS tokenize 买的房→买房
PASS tokenize 城市保留
PASS tokenize 单字 车
PASS tokenize 单字 猫
PASS tokenize 英文 model
PASS tokenize 空字符串
PASS tokenize None
PASS tokenize 纯停用词 不崩溃
PASS benchmark 10 题 (10/10)

全部测试通过 OK
```

### Mock MemoryAgentBench (offline)

跑了 3 个 sample, 15 个 QA pair(用 mock fixtures,**未连 HuggingFace / LLM**):

| 指标 | 分数 |
|---|---|
| **recall@5** | **86.67%** (13/15) |
| first_chunk_hit | 66.67% (10/15) |
| substring_em | 66.67% (10/15) |
| 平均 query 时间 | 0.26s |

**漏的 2 个 QA 都是 mock_003**(5 个 chunk 的小语料,排序和 coverage 都不够,真实 benchmark 会用 top_k=20 + rerank 兜底)。

⏱ 总耗时 3.9s(全离线,无 LLM call)。

JSON 完整结果:`methods/nova_memory/source/_bench_results/mock_nova.json`

### Real MemoryAgentBench (EventQA)
```
$ python main.py \\
      --agent_config config/sequential_nova_memory.yaml \\
      --dataset_config benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml
```
*(requires HuggingFace access; reports will be added once we have a
run from a networked machine)*

---

## 🚀 How to reproduce

```bash
git clone https://github.com/<your-fork>/MemoryData
cd MemoryData

# Optional: apply the dispatch patch (idempotent)
python methods/nova_memory/adapter_patch.py

# Run
python main.py \\
    --agent_config config/sequential_nova_memory.yaml \\
    --dataset_config benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml
```

---

## ⚠ Known limitations

1. **No semantic search.** Lexical-only. Misses synonym cases
   (mock_003: "狗" vs "金毛犬" — would need embedding).
2. **Top-chunk-as-answer fallback when no LLM** — works for extractive
   QA, fails for abstractive.
3. **Chinese-optimized.** Works on English but uncompetitive with
   BM25/dense baselines on English-only benchmarks (LoCoMo, LongBench).
4. **Single-process.** No distributed indexing. Cap @ ~10K chunks.

---

## 🐴 Future plans

- MultiParadigm hybrid: Nova + BM25 + dense re-rank
- Add Chinese benchmark sub-set to MemoryData
- ICLR/NeurIPS workshop submission

---

## ✅ Checklist

- [x] Self-contained (no extra `pip install` for core)
- [x] Idempotent patch (idempotency verified)
- [x] Backwards compatible (patch is additive, no existing methods modified)
- [x] Tests pass (16/16 unit + 4/4 E2E + mock benchmark)
- [x] README + integration guide
- [x] YAML preset registered in `config/`
- [ ] Real MemoryAgentBench numbers — **pending network access**

---

## 📎 Related

- `nova-mvp/memory.py` (source of vendored tokenize chain)
- `benchmark/memoryagentbench/Accurate_Retrieval/` (target benchmark)
- Issue/PRs to follow: `[ ] Add Nova to README method table`

cc @OpenDataBox/memorydata-maintainers