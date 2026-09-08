"""Fact-level 1:1 test (v6b). No LLM needed; embedding only.

For each benchmark QA with evidence turn k: among atomic facts extracted from the
window covering k, find the one most similar to (question + gold answer). Use that
fact's entailment by the compressed block as the predictor:
    fact_lost = 1 - entailed_plus(best fact)
Does fact_lost predict qa_hurt?  This is the exhaustive sensor at its finest grain.

Also reports the result restricted to well-matched pairs (sim >= s_min) and the
top-2 facts (either lost), and prints examples so you can eyeball whether the
matched fact really is the one the QA needs.
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    if labels.all() or (~labels).all() or len(labels) < 6: return float("nan")
    u = mannwhitneyu(scores[labels], scores[~labels], alternative="greater").statistic
    return u / (labels.sum() * (~labels).sum())

def boot(scores, labels, n=1000, seed=0):
    rng = np.random.default_rng(seed); s, l = np.asarray(scores, float), np.asarray(labels).astype(bool); out = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s)); a = auroc(s[i], l[i])
        if not np.isnan(a): out.append(a)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--s_min", type=float, default=0.6)
ap.add_argument("--embed_model", default="Qwen/Qwen3-Embedding-0.6B")
a = ap.parse_args()

qa = pd.read_json(f"{a.run_dir}/qa.jsonl", lines=True)
qa["evidence_turns"] = qa["evidence_turns"].apply(lambda v: v if isinstance(v, list) else (json.loads(v) if isinstance(v, str) else []))
fa = pd.read_json(f"{a.run_dir}/facts.jsonl", lines=True)
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer(a.embed_model, device="cuda")

rows = []
for (cid, t), qg in qa.groupby(["conv_id", "session"]):
    fg = fa[(fa.conv_id == cid) & (fa.session == t)]
    if len(fg) == 0: continue
    E_f = emb.encode(fg.fact.tolist(), normalize_embeddings=True)
    for _, q in qg.iterrows():
        turns = q.evidence_turns
        if not turns: continue
        mask = fg.apply(lambda f: any(f.turn_start <= k <= f.turn_end for k in turns), axis=1).values
        if not mask.any(): continue
        idx = np.where(mask)[0]
        e_q = emb.encode([f"{q.question} {q.gold}"], normalize_embeddings=True)[0]
        sims = E_f[idx] @ e_q
        order = idx[np.argsort(-sims)]
        best = order[0]; s1 = float(np.max(sims))
        top2 = order[:2]
        rows.append({"conv_id": cid, "session": t, "category": q.category,
                     "single_session": q.n_evidence_sessions == 1,
                     "qa_delta": q.delta, "qa_hurt": q.delta > a.tau,
                     "sim": s1,
                     "fact_lost_top1": 1 - int(fg.entailed_plus.values[best]),
                     "fact_lost_top2": 1 - int(fg.entailed_plus.values[top2].min()),   # either of top-2 lost
                     "fact_lost_top2_mean": 1 - float(fg.entailed_plus.values[top2].mean()),
                     "window_risk": 1 - float(fg.entailed_plus.values[idx].mean()),
                     "fact": fg.fact.values[best], "qa_q": q.question, "gold": q.gold,
                     "pred_minus": q.pred_minus, "pred_plus": q.pred_plus})
df = pd.DataFrame(rows)
df.to_csv(f"{a.run_dir}/fact_matches.csv", index=False)
print(f"=== {a.run_dir}: n={len(df)} QA matched to a fact   median sim={df.sim.median():.2f}   P(qa hurt)={df.qa_hurt.mean():.3f}")

def rep(d, col, label):
    if len(d) < 10: print(f"  {label:34s} n={len(d)} too few"); return np.nan
    au = auroc(d[col], d.qa_hurt); lo, hi = boot(d[col], d.qa_hurt)
    x = d[col] > 0.5
    p1 = d.qa_hurt[x].mean() if x.any() else np.nan; p0 = d.qa_hurt[~x].mean() if (~x).any() else np.nan
    print(f"  {label:34s} n={len(d):3d}  AUROC={au:.3f} [{lo:.2f},{hi:.2f}]   P(hurt|lost)={p1:.2f} (n={x.sum()})  P(hurt|kept)={p0:.2f} (n={(~x).sum()})  lift={p1/max(1e-9,p0):.2f}x")
    return au

print("\n[ALL]")
rep(df, "window_risk", "window coverage (v6 baseline)")
au1 = rep(df, "fact_lost_top1", "top-1 matched fact lost")
au2 = rep(df, "fact_lost_top2", "either of top-2 facts lost")
w = df[df.sim >= a.s_min]
print(f"\n[well-matched sim>={a.s_min}]  ({len(w)}/{len(df)})")
rep(w, "window_risk", "window coverage")
au1w = rep(w, "fact_lost_top1", "top-1 matched fact lost")
au2w = rep(w, "fact_lost_top2", "either of top-2 facts lost")
ss = w[w.single_session]
print(f"\n[well-matched + single-session]  ({len(ss)})")
au1s = rep(ss, "fact_lost_top1", "top-1 matched fact lost")
au2s = rep(ss, "fact_lost_top2", "either of top-2 facts lost")

print("\n[by category, well-matched, top-2] (1=multi,2=temporal,3=open,4=single)")
for c, g in w.groupby("category"):
    if len(g) >= 8: print(f"    cat {c}: n={len(g):3d}  AUROC={auroc(g.fact_lost_top2, g.qa_hurt):.3f}")

print("\nexamples -- fact LOST and QA hurt (true positives):")
for _, r in w[(w.fact_lost_top1 == 1) & (w.qa_hurt)].head(4).iterrows():
    print(f"  Q: {r.qa_q} | gold: {r.gold} | raw->{r.pred_minus} | comp->{r.pred_plus}\n     fact: {r.fact}")
print("\nexamples -- fact KEPT but QA hurt (what the sensor misses):")
for _, r in w[(w.fact_lost_top1 == 0) & (w.qa_hurt)].head(4).iterrows():
    print(f"  Q: {r.qa_q} | gold: {r.gold} | raw->{r.pred_minus} | comp->{r.pred_plus}\n     fact: {r.fact}")
print("\nexamples -- fact LOST but QA fine (false alarms):")
for _, r in w[(w.fact_lost_top1 == 1) & (~w.qa_hurt)].head(3).iterrows():
    print(f"  Q: {r.qa_q} | gold: {r.gold} | raw->{r.pred_minus} | comp->{r.pred_plus}\n     fact: {r.fact}")

best = np.nanmax([au1w, au2w, au1s, au2s])
if best >= 0.65: v = "GO (fact-level): the specific fact's survival predicts QA regression"
elif best >= 0.58: v = "MARGINAL: read the 'fact KEPT but QA hurt' examples -- if those are reader/F1 noise, the sensor is fine and the LABEL is the problem"
else: v = "NO-GO (fact-level)"
print(f"\nVERDICT: {v}")
