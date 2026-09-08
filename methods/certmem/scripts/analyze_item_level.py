"""4.2-A item-level analysis (v4).

Question: does failure of the probe that covers the SAME FACT predict regression on
the benchmark question about that fact? This is the unit the repair mechanism
operates on (set cover attaches the span behind a failed probe), so it is the
validity test that matters.

Method:
  For each benchmark QA in session t, find the k nearest probes in the same
  session by embedding similarity of the question text. Define
    probe_fail  = matched probe delta > tau   (compression hurt the probe)
    qa_regress  = QA delta > tau              (compression hurt the QA)
  Report:
    - P(qa_regress | probe_fail) vs P(qa_regress | ~probe_fail)   [the lift]
    - point-biserial / Spearman between matched probe delta and QA delta
    - AUROC of matched-probe delta as a predictor of qa_regress
    - the same restricted to well-matched pairs (similarity >= s_min), since a
      probe about a different fact should NOT predict.
  Also a dose-response view if cost.csv has varying ratio.

Runs on runs/<dir>/probes.jsonl + qa.jsonl. Needs the embedding model (CPU ok).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

def auroc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    if labels.all() or (~labels).all(): return float("nan")
    pos, neg = scores[labels], scores[~labels]
    u = mannwhitneyu(pos, neg, alternative="greater").statistic
    return u / (len(pos) * len(neg))

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--k", type=int, default=1, help="nearest probes per QA")
ap.add_argument("--tau", type=float, default=0.2, help="delta threshold for 'hurt'")
ap.add_argument("--s_min", type=float, default=0.5, help="min cosine sim for 'well matched'")
ap.add_argument("--embed_model", default="Qwen/Qwen3-Embedding-0.6B")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

pr = pd.read_json(f"{a.run_dir}/probes.jsonl", lines=True)
qa = pd.read_json(f"{a.run_dir}/qa.jsonl", lines=True)
print(f"=== {a.run_dir}: {len(pr)} probes, {len(qa)} QA")

from sentence_transformers import SentenceTransformer
emb = SentenceTransformer(a.embed_model, device=a.device)

rows = []
for (cid, t), qg in qa.groupby(["conv_id", "session"]):
    pg = pr[(pr.conv_id == cid) & (pr.session == t)]
    if len(pg) == 0: continue
    # match on question + answer text so probes about the same fact rank first
    qtxt = (qg.question + " " + qg.gold.astype(str)).tolist()
    ptxt = (pg.question + " " + pg.answer.astype(str)).tolist()
    E_q = emb.encode(qtxt, normalize_embeddings=True)
    E_p = emb.encode(ptxt, normalize_embeddings=True)
    S = E_q @ E_p.T
    for i, (_, q) in enumerate(qg.iterrows()):
        order = np.argsort(-S[i])[: a.k]
        sims = S[i][order]
        deltas = pg.delta.values[order]
        # weighted by similarity so a poorly matched second probe counts less
        wdelta = float(np.average(deltas, weights=np.clip(sims, 1e-3, None)))
        rows.append({"conv_id": cid, "session": t, "category": q.category,
                     "qa_delta": q.delta, "qa_f1_minus": q.f1_minus, "qa_f1_plus": q.f1_plus,
                     "probe_delta": wdelta, "sim": float(sims[0]),
                     "probe_family": pg.family.values[order[0]],
                     "probe_q": pg.question.values[order[0]], "qa_q": q.question})
df = pd.DataFrame(rows)
df.to_csv(f"{a.run_dir}/item_matches.csv", index=False)

def report(d, label):
    if len(d) < 10:
        print(f"\n[{label}] n={len(d)} too few"); return
    pf = d.probe_delta > a.tau
    qr = d.qa_delta > a.tau
    p_given_fail = qr[pf].mean() if pf.any() else float("nan")
    p_given_ok   = qr[~pf].mean() if (~pf).any() else float("nan")
    rho = spearmanr(d.probe_delta, d.qa_delta).correlation if d.probe_delta.nunique() > 1 else float("nan")
    au = auroc(d.probe_delta, qr)
    print(f"\n[{label}] n={len(d)}  base rate P(qa hurt)={qr.mean():.3f}")
    print(f"    P(qa hurt | probe hurt)={p_given_fail:.3f}  (n={pf.sum()})")
    print(f"    P(qa hurt | probe ok  )={p_given_ok:.3f}  (n={(~pf).sum()})")
    print(f"    lift={p_given_fail/max(1e-9,p_given_ok):.2f}x   spearman={rho:+.3f}   AUROC={au:.3f}")

report(df, "ALL matched pairs")
report(df[df.sim >= a.s_min], f"WELL-matched (sim>={a.s_min})")
report(df[df.sim <  a.s_min], f"POORLY-matched (sim<{a.s_min}) -- should be ~no signal")

print("\n[by category, well-matched] (1=multi,2=temporal,3=open,4=single)")
w = df[df.sim >= a.s_min]
for c, g in w.groupby("category"):
    if len(g) >= 8:
        pf = g.probe_delta > a.tau; qr = g.qa_delta > a.tau
        print(f"    cat {c}: n={len(g):3d}  P(hurt|probe hurt)={qr[pf].mean() if pf.any() else float('nan'):.2f} "
              f"P(hurt|probe ok)={qr[~pf].mean() if (~pf).any() else float('nan'):.2f}  AUROC={auroc(g.probe_delta, qr):.3f}")

print("\n[by probe family, well-matched]")
for f, g in w.groupby("probe_family"):
    if len(g) >= 8:
        qr = g.qa_delta > a.tau
        print(f"    {f:10s}: n={len(g):3d}  AUROC={auroc(g.probe_delta, qr):.3f}")

print(f"\nsimilarity distribution: median={df.sim.median():.2f}  >= {a.s_min}: {(df.sim>=a.s_min).mean():.0%}")
print("\nexample well-matched pairs (probe -> QA):")
for _, r in w.sort_values("sim", ascending=False).head(5).iterrows():
    print(f"  sim={r.sim:.2f}  probeΔ={r.probe_delta:+.2f}  qaΔ={r.qa_delta:+.2f}\n     P: {r.probe_q}\n     Q: {r.qa_q}")

au_all = auroc(df.probe_delta, df.qa_delta > a.tau)
au_w = auroc(w.probe_delta, w.qa_delta > a.tau) if len(w) >= 10 else float("nan")
if not np.isnan(au_w) and au_w >= 0.65:
    v = "GO (item-level): matched probe failure predicts QA regression"
elif not np.isnan(au_w) and au_w >= 0.58:
    v = "MARGINAL (item-level): signal present; increase probe coverage (probes_per_session 20) and rerun"
else:
    v = "NO-GO (item-level): probes do not cover the facts QA asks about; see similarity distribution"
print(f"\nVERDICT: {v}")
