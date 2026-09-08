# certmem — Experiment 4.2-A (probe risk vs. benchmark regression)

Go/no-go experiment for "Certified Consolidation". Question: when a host
memory system compresses session t, does our probe-measured risk R_t predict
the regression on LoCoMo's own questions about session t?

Everything runs on one 24 GB GPU (RTX 3090/4090, A10, L4-24G). Qwen3-8B bf16.

## 0. What you do vs. what is already done

Done (this repo): LoCoMo loader + evidence→session mapping, vLLM engine with
thinking disabled, log-prob answer scoring, shared hybrid retriever, raw /
Mem0-like / LightMem-like hosts, typed probe generator with verbatim-span
filter, orchestrator, analysis with bootstrap CI, smoke test.

You: run the commands below on a GPU machine. Optionally swap the
Mem0-like/LightMem-like hosts for the upstream pipelines later (main table
only; not needed for 4.2-A).

## 1. Setup (10–20 min, mostly model download)

```bash
git clone <this folder>  # or scp certmem/ to the GPU box
cd certmem
bash scripts/setup.sh          # venv, deps, Qwen3-8B, embedding model, LoCoMo json, smoke test
```

If you already have the models cached, just:
```bash
source .venv/bin/activate
python scripts/smoke_test.py data/locomo10.json
```
Expected: `convs: 10 | first: conv-26 | sessions: 19 | qa(non-adv): 152` and
`qa with evidence->session mapping: 150/152`.

## 2. Sanity run (5 min) — raw host, regression must be ≈ 0

```bash
python scripts/run_4_2a.py --host raw --limit_convs 1
python scripts/analyze_4_2a.py runs/4_2a_raw
```
With `raw`, M+ == M-, so `bench_regression` and `probe delta` should both be ~0.
If not, something in retrieval/scoring is nondeterministic — stop and check.

## 3. Real run — Mem0-like host (≈ 2–3 h on a 3090)

```bash
python scripts/run_4_2a.py --host mem0
python scripts/analyze_4_2a.py runs/4_2a_mem0
```
Then LightMem-like:
```bash
python scripts/run_4_2a.py --host lightmem
python scripts/analyze_4_2a.py runs/4_2a_lightmem
```

## 4. Reading the result

`analyze_4_2a.py` prints Spearman ρ between per-session probe risk and
per-session benchmark regression, with a bootstrap 95% CI, plus breakdowns by
LoCoMo category and by probe family.

- ρ ≥ 0.4 → GO. Proceed to calibration (LTT) and the full grid.
- 0.2 ≤ ρ < 0.4 → MARGINAL. Look at the family breakdown; usually `condition`
  and `negation` probes are weak. Regenerate probes with a larger model
  (set `Config.model` to a 32B/72B for `generate_probes` only) and rerun.
- ρ < 0.2 → NO-GO for this framing. Check that `f1_minus > f1_plus` on
  temporal (cat 2) and multi-hop (cat 1) first; if compression is not hurting
  at all, the host is not lossy enough to test — lower `cap` / `n_bullets`.

Also check `bench_f1_minus` vs `bench_f1_plus`: the gap is the total
regression the certificate is supposed to bound. If it is < 1 F1 point, the
paper needs a more aggressive host setting (that is a legitimate knob — λ).

## 5. Outputs

```
runs/4_2a_<host>/
  probes.jsonl    per probe: family, loss on M-, loss on M+, delta
  qa.jsonl        per question: session, category, preds, F1 on M-/M+, regression
  sessions.csv    per session: probe_risk, n_probes, bench_regression, F1s, n_qa
```

## 6. Knobs you may need

| where | what | default |
|---|---|---|
| `certmem/config.py` | `gpu_memory_utilization` | 0.88 (drop to 0.80 if OOM with embedder loaded) |
| `certmem/config.py` | `embed_model` | Qwen3-Embedding-0.6B (4B needs ~8 GB more) |
| `certmem/hosts/mem0_adapter.py` | `cap` (Mem0-like) / `n_bullets` (LightMem-like) | 10 / 6 — this is λ |
| `certmem/config.py` | `context_budget_tokens` | 4096 |
| `certmem/config.py` | `family_weights` | date/negation heavier |

## 7. Design notes (so the numbers are defensible)

- M- and M+ share the host's *past* state; only session t differs. So R_t and
  the benchmark regression are both attributable to compressing session t.
- Probes are generated from the raw session only and must be verbatim spans
  (`grounded` filter). Benchmark questions never touch probe generation.
- Probe loss is mean token log-prob of the verbatim answer under the reader,
  mapped to [0,1]. One forward pass per probe per state; no judge needed.
- Benchmark regression uses LoCoMo token-F1 (official metric); no external
  judge for 4.2-A. The main table later uses the pinned external judge.
- Adversarial (category 5) questions are excluded, matching the protocol.
- Host state advances with its own compressed units only (history-only,
  no gold), matching the paper's ingestion protocol.
