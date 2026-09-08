"""4.2-A analysis (v3).

Reports, in order of importance:
  1. Spearman rho between per-session probe risk and benchmark delta (signed, and clipped).
  2. Split-half reliability of the benchmark delta itself (randomly halve each session's
     QA; correlate the halves; Spearman-Brown correct). This is the CEILING any probe
     can reach. If the ceiling is low, low rho is a measurement problem, not a probe problem.
  3. Attenuation-corrected rho = rho / sqrt(reliability) -- the paper-reportable number.
  4. Item-level check: probe delta vs QA delta pooled over all pairs within a session.
  5. Category / family breakdowns and compression ratio.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

def safe_spearman(x, y):
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return spearmanr(x, y).correlation

def boot_ci(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed); rs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        r = safe_spearman(x[i], y[i])
        if not np.isnan(r): rs.append(r)
    return (np.nan, np.nan) if not rs else tuple(np.percentile(rs, [2.5, 97.5]))

def split_half_reliability(qa, min_qa=4, n=200, seed=0):
    """Spearman-Brown corrected split-half reliability of per-session bench delta."""
    rng = np.random.default_rng(seed); rs = []
    groups = [g for _, g in qa.groupby(["conv_id", "session"]) if len(g) >= min_qa]
    if len(groups) < 6:
        return float("nan"), len(groups)
    for _ in range(n):
        a, b = [], []
        for g in groups:
            idx = rng.permutation(len(g)); h = len(g) // 2
            a.append(g.delta.values[idx[:h]].mean()); b.append(g.delta.values[idx[h:2*h]].mean())
        r = safe_spearman(np.array(a), np.array(b))
        if not np.isnan(r): rs.append(r)
    r = float(np.mean(rs)) if rs else float("nan")
    sb = 2*r/(1+r) if not np.isnan(r) and r > -1 else float("nan")
    return sb, len(groups)

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--min_qa", type=int, default=3)
a = ap.parse_args()

df = pd.read_csv(f"{a.run_dir}/sessions.csv")
qa = pd.read_json(f"{a.run_dir}/qa.jsonl", lines=True)
pr = pd.read_json(f"{a.run_dir}/probes.jsonl", lines=True)
df = df[df.n_qa >= a.min_qa]

print(f"=== {a.run_dir}  sessions={len(df)} (n_qa>={a.min_qa})  compression ratio={df.ratio.mean():.2f}")
print(f"F1 raw={df.bench_f1_minus.mean():.3f}  F1 compressed={df.bench_f1_plus.mean():.3f}  "
      f"mean signed delta={df.bench_delta_signed.mean():+.3f}  mean probe delta={df.probe_delta_signed.mean():+.3f}")

# 1. main correlations
x_s, y_s = df.probe_delta_signed.values, df.bench_delta_signed.values
x_c, y_c = df.probe_risk.values, df.bench_regression.values
rho_s = safe_spearman(x_s, y_s); lo_s, hi_s = boot_ci(x_s, y_s)
rho_c = safe_spearman(x_c, y_c); lo_c, hi_c = boot_ci(x_c, y_c)
print(f"\n[1] session-level Spearman")
print(f"    signed  : rho={rho_s:+.3f}  95%CI=[{lo_s:+.3f},{hi_s:+.3f}]")
print(f"    clipped : rho={rho_c:+.3f}  95%CI=[{lo_c:+.3f},{hi_c:+.3f}]")

# 2. reliability ceiling
rel, ng = split_half_reliability(qa)
print(f"\n[2] split-half reliability of bench delta (Spearman-Brown): {rel:.3f}  over {ng} sessions with >=4 QA")
print(f"    -> max attainable rho by ANY predictor ~ sqrt(rel) = {np.sqrt(rel) if rel > 0 else float('nan'):.3f}")

# 3. attenuation-corrected
if rel > 0 and not np.isnan(rho_s):
    print(f"\n[3] attenuation-corrected rho (signed) = {rho_s/np.sqrt(rel):+.3f}")

# 4. item-level pooled
merged = pr.groupby(["conv_id", "session"])["delta"].mean().rename("p").to_frame().join(
         qa.groupby(["conv_id", "session"])["delta"].mean().rename("q")).dropna()
print(f"\n[4] pooled item means (all sessions, no n_qa filter): n={len(merged)}  rho={safe_spearman(merged.p.values, merged.q.values):+.3f}")

# 5. breakdowns
print("\n[5] bench delta by LoCoMo category (1=multi-hop,2=temporal,3=open-domain,4=single-hop):")
print(qa.groupby("category")[["f1_minus", "f1_plus", "delta"]].mean().round(3))
print("\n    probe delta by family (signed; positive = compression hurt):")
print(pr.groupby("family")["delta"].agg(["mean", "size"]).round(3))
print("\n    per-family rho vs bench delta:")
for fam, g in pr.groupby("family"):
    f = g.groupby(["conv_id", "session"])["delta"].mean().rename("p").to_frame().join(
        qa.groupby(["conv_id", "session"])["delta"].mean().rename("q")).dropna()
    print(f"      {fam:10s} n={len(f):3d}  rho={safe_spearman(f.p.values, f.q.values):+.3f}")

# verdict
ceiling = np.sqrt(rel) if rel > 0 else np.nan
if np.isnan(rho_s):
    v = "UNDEFINED (constant input)"
elif not np.isnan(ceiling) and ceiling < 0.3:
    v = f"MEASUREMENT-LIMITED: bench delta unreliable (ceiling {ceiling:.2f}); need more QA per session or conversation-level aggregation"
elif rho_s >= 0.4 or (not np.isnan(ceiling) and rho_s / ceiling >= 0.6):
    v = "GO"
elif rho_s >= 0.2:
    v = "MARGINAL: raise compression (--cap 5), check family breakdown"
else:
    v = "NO-GO at this setting: check compression ratio and family rhos"
print(f"\nVERDICT: {v}")
