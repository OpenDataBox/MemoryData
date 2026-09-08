"""4.2-A turn-matched analysis (v5).

For each benchmark QA with evidence turn k in session t, take the probes whose
window covers turn k. Probe risk for that QA = max delta over those probes
(if ANY probe on that turn failed, the turn lost information).
Test: does turn-level probe failure predict QA regression?

Also reports coverage: fraction of QA evidence turns that have >=1 probe.
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

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--agg", default="max", choices=["max", "mean"])
a = ap.parse_args()

pr = pd.read_json(f"{a.run_dir}/probes.jsonl", lines=True)
qa = pd.read_json(f"{a.run_dir}/qa.jsonl", lines=True)
qa["evidence_turns"] = qa["evidence_turns"].apply(lambda v: v if isinstance(v, list) else json.loads(v) if isinstance(v, str) else [])
print(f"=== {a.run_dir}: {len(pr)} probes, {len(qa)} QA   probes/session={len(pr)/max(1,pr.groupby(['conv_id','session']).ngroups):.1f}")
print(f"probe fail rate (delta>{a.tau}) = {(pr.delta > a.tau).mean():.3f}   QA hurt rate = {(qa.delta > a.tau).mean():.3f}")

rows, covered = [], 0
for _, q in qa.iterrows():
    turns = q.evidence_turns
    if not turns: continue
    pg = pr[(pr.conv_id == q.conv_id) & (pr.session == q.session)]
    hits = pg[pg.apply(lambda p: any(p.turn_start <= k <= p.turn_end for k in turns), axis=1)]
    if len(hits) == 0: continue
    covered += 1
    d = hits.delta.max() if a.agg == "max" else hits.delta.mean()
    rows.append({"conv_id": q.conv_id, "session": q.session, "category": q.category,
                 "single_session": q.n_evidence_sessions == 1,
                 "qa_delta": q.delta, "probe_delta": float(d), "n_probes_on_turn": len(hits),
                 "families": ",".join(sorted(set(hits.family)))})
df = pd.DataFrame(rows)
df.to_csv(f"{a.run_dir}/turn_matches.csv", index=False)
print(f"coverage: {covered}/{len(qa)} QA have >=1 probe on an evidence turn")

def report(d, label):
    if len(d) < 10: print(f"\n[{label}] n={len(d)} too few"); return
    pf, qr = d.probe_delta > a.tau, d.qa_delta > a.tau
    pgf = qr[pf].mean() if pf.any() else float("nan"); pgo = qr[~pf].mean() if (~pf).any() else float("nan")
    rho = spearmanr(d.probe_delta, d.qa_delta).correlation if d.probe_delta.nunique() > 1 else float("nan")
    print(f"\n[{label}] n={len(d)}  P(qa hurt)={qr.mean():.3f}")
    print(f"    P(qa hurt | turn-probe hurt)={pgf:.3f} (n={pf.sum()})   P(qa hurt | turn-probe ok)={pgo:.3f} (n={(~pf).sum()})")
    print(f"    lift={pgf/max(1e-9,pgo):.2f}x   spearman={rho:+.3f}   AUROC={auroc(d.probe_delta, qr):.3f}")
    return auroc(d.probe_delta, qr)

au_all = report(df, "ALL turn-matched")
au_ss  = report(df[df.single_session], "single-session evidence only (cleanest)")
print("\n[by category] (1=multi,2=temporal,3=open,4=single)")
for c, g in df.groupby("category"):
    if len(g) >= 8:
        print(f"    cat {c}: n={len(g):3d}  AUROC={auroc(g.probe_delta, g.qa_delta > a.tau):.3f}  "
              f"P(hurt|probe hurt)={(g.qa_delta>a.tau)[g.probe_delta>a.tau].mean() if (g.probe_delta>a.tau).any() else float('nan'):.2f}")

print("\n[probe delta by family]")
print(pr.groupby("family")["delta"].agg(["mean", lambda x: (x > a.tau).mean(), "size"]).rename(columns={"<lambda_0>": "fail_rate"}).round(3))

au = au_ss if au_ss == au_ss else au_all
if au != au and au_all == au_all: au = au_all
if au == au and au >= 0.65: v = "GO (turn-level)"
elif au == au and au >= 0.58: v = "MARGINAL (turn-level): raise per_win to 4-5, or lower tau"
else: v = "NO-GO (turn-level): self-generated probes do not detect the losses QA measures -- pivot"
print(f"\nVERDICT: {v}")
