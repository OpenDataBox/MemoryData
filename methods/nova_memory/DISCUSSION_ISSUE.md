# Discussion: Should MemoryData add Chinese-specific lexical baselines?

**TL;DR:** Proposing `Nova Memory` as a new preset (Sequential Context
bucket) and asking whether to add Chinese-language sub-benchmarks to
MemoryData in general.

## Background

The 22 existing presets are predominantly English/embedding-centric.
For Chinese personal-fact QA, the dominant failure mode in vanilla
lexical methods is **形态学 (morphology) gap** — spoken-Chinese
variants don't match canonical forms in stored memory.

Example:
- Stored: `用户在杭州买房,花费300万。`
- Query: `我在哪个城市买的房?` (also: `我在哪买的房子啊?`)
- Vanilla BM25/Jaccard: recall drops to ~40% because "买房" doesn't
  tokenize the same way as "买的房"

## What Nova adds

Three techniques, ~200 lines of code, **zero external dependencies**:
1. **Morph map** (40+ entries): `买的房 → 买房`, `开什么车 → 车`,
   `几口人 → 家庭成员`, `之前在哪工作 → 跳槽`...
2. **2-gram + 3-gram sliding window** tokenization
3. **Single-char whitelist** for high-signal nouns

On a 3-sample Chinese mock (15 QA): **86.67% recall@5 in 3.5s** on
CPU. No vector DB, no GPU.

## Proposal

**A) Add Nova as a 23rd preset** (Sequential Context, lexical baseline)

*Pros:*
- Provides the "lightest possible" baseline for ablation
- Works on CPU, < 5s per 100 QA
- First Chinese-aware baseline in the suite

*Cons:*
- May underperform on English-heavy benchmarks (LoCoMo, LongBench)
- Adds maintenance burden for a niche use case

**B) Add Chinese sub-benchmarks** (e.g. `eventqa_zh`, `convqa_zh`)

*Pros:*
- Reflects that 1.5B+ speakers are an underserved market
- Differentiates MemoryData from LoCoMo/LongBench

*Cons:*
- Curating/curating Chinese data is non-trivial (license, quality)
- May dilute the "unified" value proposition

## Questions for maintainers

1. Is the addition of `methods/nova_memory/` welcome?
2. Would a Chinese sub-benchmark fit the 4-family taxonomy?
3. Are there plans for HuggingFace Chinese mirrors of LoCoMo/LongMemEval?

## PR link

Draft PR with full code, tests, and docs:
[`methods/nova_memory/PR_DESCRIPTION.md`](./PR_DESCRIPTION.md)

## Self-test artifacts

- `methods/nova_memory/source/_smoke_test.py` — 16/16
- `methods/nova_memory/source/_e2e_test.py` — ingest + recall + save/load
- `methods/nova_memory/source/run_benchmark.py --mock` — 3-sample mock

Happy to iterate based on feedback. 🐴

---

*cc @OpenDataBox/memorydata-maintainers*