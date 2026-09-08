"""Sensor comparison at the turn level (v6) -- the paper's Figure 2.

For each benchmark QA with evidence turn k in session t:
  fact_risk(k)  = 1 - fraction of atomic facts from the window covering k that are
                  entailed by the compressed block           (exhaustive sensor)
  probe_risk(k) = max delta of probes whose window covers k   (sampled sensor)
Question: which sensor predicts QA regression?  Report AUROC, lift, and Spearman
for both, on the same QA set, so the comparison is paired.

Also: sanity that raw-block entailment ~1 (extractor is faithful), and
dose-response hooks (fact_risk vs bench delta by session).
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    if labels.all() or (~labels).all() or len(labels) < 6: return float("nan")
    u = mannwhitneyu(scores[labels], scores[~labels], alternative="greater").statistic
    return u / (labels.sum() * (~labels).sum())

def boot_auroc(scores, labels, n=1000, seed=0):
    rng = np.random.default_rng(seed); s, l = np.asarray(scores, float), np.asarray(labels).astype(bool); out = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s)); a = auroc(s[i], l[i])
        if not np.isnan(a): out.append(a)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--tau", type=float, default=0.2, help="QA hurt threshold on F1 delta")
a = ap.parse_args()

qa = pd.read_json(f"{a.run_dir}/qa.jsonl", lines=True)
qa["evidence_turns"] = qa["evidence_turns"].apply(lambda v: v if isinstance(v, list) else (json.loads(v) if isinstance(v, str) else []))
have_facts = os.path.exists(f"{a.run_dir}/facts.jsonl")
have_probes = os.path.exists(f"{a.run_dir}/probes.jsonl")
fa = pd.read_json(f"{a.run_dir}/facts.jsonl", lines=True) if have_facts else None
pr = pd.read_json(f"{a.run_dir}/probes.jsonl", lines=True) if have_probes else None

print(f"=== {a.run_dir}: QA={len(qa)}  facts={len(fa) if fa is not None else 0}  probes={len(pr) if pr is not None else 0}")
if fa is not None:
    print(f"extractor faithfulness: P(fact entailed by RAW block) = {fa.entailed_raw.mean():.3f}   "
          f"coverage by COMPRESSED block = {fa.entailed_plus.mean():.3f}   facts/session = {len(fa)/fa.groupby(['conv_id','session']).ngroups:.1f}")

rows = []
for _, q in qa.iterrows():
    turns = q.evidence_turns
    if not turns: continue
    r = {"conv_id": q.conv_id, "session": q.session, "category": q.category,
         "single_session": q.n_evidence_sessions == 1, "qa_delta": q.delta, "qa_hurt": q.delta > a.tau}
    if fa is not None:
        g = fa[(fa.conv_id == q.conv_id) & (fa.session == q.session)]
        h = g[g.apply(lambda f: any(f.turn_start <= k <= f.turn_end for k in turns), axis=1)]
        if len(h): r["fact_risk"] = 1 - h.entailed_plus.mean(); r["n_facts_on_turn"] = len(h)
    if pr is not None:
        g = pr[(pr.conv_id == q.conv_id) & (pr.session == q.session)]
        h = g[g.apply(lambda p: any(p.turn_start <= k <= p.turn_end for k in turns), axis=1)]
        if len(h): r["probe_risk"] = h.delta.max()
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(f"{a.run_dir}/sensor_matches.csv", index=False)

def report(d, col, label):
    d = d.dropna(subset=[col])
    if len(d) < 10: print(f"  {label:22s} n={len(d)} too few"); return np.nan
    thr = np.nanmedian(d[col]) if d[col].nunique() > 2 else 0.5
    hi = d[col] > thr
    p_hi = d.qa_hurt[hi].mean() if hi.any() else np.nan; p_lo = d.qa_hurt[~hi].mean() if (~hi).any() else np.nan
    au = auroc(d[col], d.qa_hurt); lo, up = boot_auroc(d[col], d.qa_hurt)
    rho = spearmanr(d[col], d.qa_delta).correlation if d[col].nunique() > 1 else np.nan
    print(f"  {label:22s} n={len(d):3d}  AUROC={au:.3f} [{lo:.2f},{up:.2f}]  spearman={rho:+.3f}  "
          f"P(hurt|risk>med)={p_hi:.2f} vs P(hurt|risk<=med)={p_lo:.2f}  lift={p_hi/max(1e-9,p_lo):.2f}x")
    return au

print(f"\n[ALL QA with evidence turn]  base P(qa hurt)={df.qa_hurt.mean():.3f}")
au_f = report(df, "fact_risk", "FACT coverage sensor") if "fact_risk" in df else np.nan
au_p = report(df, "probe_risk", "PROBE sensor") if "probe_risk" in df else np.nan

ss = df[df.single_session]
print(f"\n[single-session evidence only]  base P(qa hurt)={ss.qa_hurt.mean():.3f}")
au_f_ss = report(ss, "fact_risk", "FACT coverage sensor") if "fact_risk" in ss else np.nan
au_p_ss = report(ss, "probe_risk", "PROBE sensor") if "probe_risk" in ss else np.nan

if "fact_risk" in df:
    print("\n[FACT sensor by category] (1=multi,2=temporal,3=open,4=single)")
    for c, g in df.dropna(subset=["fact_risk"]).groupby("category"):
        if len(g) >= 8: print(f"    cat {c}: n={len(g):3d}  AUROC={auroc(g.fact_risk, g.qa_hurt):.3f}")

# session-level dose view
s = pd.read_csv(f"{a.run_dir}/sessions.csv")
if "fact_risk" in s:
    s2 = s.dropna(subset=["fact_risk", "bench_delta_signed"])
    print(f"\n[session-level] Spearman(fact_risk, bench delta) = {spearmanr(s2.fact_risk, s2.bench_delta_signed).correlation:+.3f}  (n={len(s2)})")

au = au_f_ss if not np.isnan(au_f_ss) else au_f
if np.isnan(au): v = "no fact sensor in this run"
elif au >= 0.65: v = "GO: exhaustive fact coverage localizes compression damage (Theorem 1 confirmed)"
elif au >= 0.58: v = "MARGINAL: signal present; tighten extractor (fact_win 2) or entailment prompt"
else: v = "NO-GO: even exhaustive coverage does not predict QA loss -> write-time sensing insufficient; pivot"
print(f"\nVERDICT: {v}")
