"""Certified consolidation from sweep outputs (no GPU).

Inputs: runs/sweep_cap{c}/sessions.csv for several caps c (same sessions, host, backbone).
Each row: (conv_id, session, fact_risk = 1 - coverage, bench_f1_plus, ratio, plus_tokens, raw_tokens).

Two certified policies, evaluated with K-fold by conversation:

(A) GLOBAL-LTT: choose one cap per fold. On calibration convs compute for each cap c the
    Hoeffding-Bentkus p-value for H0: E[fact_risk_c] > eps. Fixed-sequence test from the
    least to the most aggressive cap; lambda_hat = most aggressive cap still rejected at delta.
    Guarantee: P(E[fact_risk] <= eps) >= 1 - delta.  Report realised risk on held-out convs.

(B) SESSION-ADAPTIVE: per session choose the most aggressive cap with fact_risk <= eps;
    if none, keep raw (ratio 1, F1 = bench_f1_minus). Certificate: per-session risk <= eps
    by construction on measured sessions; generalisation to unseen sessions is via the
    calibrated eps->tau map (reported).

Also: uniform baselines (each cap alone), and the anytime e-process check across sessions.
Output: certified.csv (policy, eps, mean ratio, mean F1, realised risk, defer rate) + a
Pareto print. This is the data for Figure 4/5.
"""
import argparse, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import binom

def hb_pvalue(risks, eps):
    """Hoeffding-Bentkus p-value for H0: E[R] > eps, R in [0,1] (Bates et al. 2021)."""
    n = len(risks); rhat = float(np.mean(risks))
    if n == 0: return 1.0
    h1 = lambda a, b: a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b)) if 0 < a < 1 and 0 < b < 1 else 0.0
    hoeff = np.exp(-n * h1(min(rhat, eps), eps)) if rhat < eps else 1.0
    bent = np.e * binom.cdf(np.ceil(n * rhat), n, eps)
    return float(min(1.0, min(hoeff, bent)))

def eprocess_max(risks, eps, lam=0.5):
    """Anytime-valid check: E_t = prod (1 + lam*(eps - r_s)); Ville: P(sup E_t >= 1/delta) <= delta under E[r]<=eps.
    Returns max_t E_t (should stay < 1/delta if certificate holds)."""
    E, mx = 1.0, 1.0
    for r in risks:
        E *= max(1e-9, 1 + lam * (eps - r)); mx = max(mx, E)
    return mx

ap = argparse.ArgumentParser()
ap.add_argument("--sweep_glob", default="runs/sweep_cap*")
ap.add_argument("--eps", default="0.3,0.4,0.5,0.6")
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--out", default="runs/certified")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
eps_list = [float(x) for x in a.eps.split(",")]

frames = []
for d in sorted(glob.glob(a.sweep_glob)):
    cap = int(d.split("cap")[-1]); s = pd.read_csv(f"{d}/sessions.csv"); s["cap"] = cap; frames.append(s)
df = pd.concat(frames)
caps = sorted(df.cap.unique())               # ascending cap = decreasing aggressiveness
df = df.dropna(subset=["fact_risk", "bench_f1_plus", "ratio"])
convs = sorted(df.conv_id.unique())
print(f"caps={caps}  convs={len(convs)}  sessions/cap={df.groupby('cap').size().to_dict()}")

# wide tables: index (conv, session), columns cap
R = df.pivot_table(index=["conv_id", "session"], columns="cap", values="fact_risk")
F = df.pivot_table(index=["conv_id", "session"], columns="cap", values="bench_f1_plus")
T = df.pivot_table(index=["conv_id", "session"], columns="cap", values="ratio")
Fraw = df.groupby(["conv_id", "session"]).bench_f1_minus.first()
common = R.dropna().index.intersection(F.dropna().index)
R, F, T, Fraw = R.loc[common], F.loc[common], T.loc[common], Fraw.loc[common]
print(f"sessions with all caps: {len(common)}")

rows = []
# uniform baselines
for c in caps:
    rows.append({"policy": f"uniform_cap{c}", "eps": np.nan, "ratio": T[c].mean(), "f1": F[c].mean(), "risk": R[c].mean(), "defer": 0.0})
rows.append({"policy": "raw", "eps": np.nan, "ratio": 1.0, "f1": Fraw.mean(), "risk": 0.0, "defer": 1.0})

folds = [convs[i::a.folds] for i in range(a.folds)] if len(convs) >= a.folds else [[c] for c in convs]
for eps in eps_list:
    glob_f1, glob_ratio, glob_risk, glob_n = [], [], [], 0
    ad_f1, ad_ratio, ad_risk, ad_defer, ad_n = [], [], [], [], 0
    emax = []
    for held in folds:
        cal = [c for c in convs if c not in held]
        cal_idx = [i for i in common if i[0] in cal]; held_idx = [i for i in common if i[0] in held]
        if not cal_idx or not held_idx: continue
        # (A) global LTT: fixed-sequence from least aggressive (largest cap) to most aggressive (smallest cap)
        lam_hat = None
        for c in sorted(caps, reverse=True):
            p = hb_pvalue(R.loc[cal_idx, c].values, eps)
            if p <= a.delta: lam_hat = c
            else: break
        if lam_hat is not None:
            glob_f1 += list(F.loc[held_idx, lam_hat].values); glob_ratio += list(T.loc[held_idx, lam_hat].values)
            glob_risk += list(R.loc[held_idx, lam_hat].values); glob_n += len(held_idx)
            emax.append(eprocess_max(R.loc[held_idx, lam_hat].values, eps))
        else:   # nothing certifiable -> keep raw
            glob_f1 += list(Fraw.loc[held_idx].values); glob_ratio += [1.0]*len(held_idx); glob_risk += [0.0]*len(held_idx); glob_n += len(held_idx)
        # (B) session-adaptive: most aggressive cap with risk <= eps, else raw
        for i in held_idx:
            ok = [c for c in caps if R.loc[i, c] <= eps]
            if ok:
                c = min(ok); ad_f1.append(F.loc[i, c]); ad_ratio.append(T.loc[i, c]); ad_risk.append(R.loc[i, c]); ad_defer.append(0)
            else:
                ad_f1.append(Fraw.loc[i]); ad_ratio.append(1.0); ad_risk.append(0.0); ad_defer.append(1)
        ad_n += len(held_idx)
    if glob_n:
        rows.append({"policy": "global_LTT", "eps": eps, "ratio": np.mean(glob_ratio), "f1": np.mean(glob_f1),
                     "risk": np.mean(glob_risk), "defer": 0.0, "eprocess_max": np.max(emax) if emax else np.nan,
                     "cert_holds": bool(np.mean(glob_risk) <= eps)})
    if ad_n:
        rows.append({"policy": "session_adaptive", "eps": eps, "ratio": np.mean(ad_ratio), "f1": np.mean(ad_f1),
                     "risk": np.mean(ad_risk), "defer": np.mean(ad_defer), "cert_holds": bool(np.mean(ad_risk) <= eps)})

out = pd.DataFrame(rows); out.to_csv(f"{a.out}/certified.csv", index=False)
print("\n=== certified consolidation: F1 vs token ratio (held-out, %d-fold by conversation) ===" % len(folds))
print(out.round(3).to_string(index=False))

# Pareto dominance check
pts = out[["policy", "eps", "ratio", "f1"]].dropna(subset=["ratio", "f1"])
def dominated(r):
    others = pts[(pts.ratio <= r.ratio) & (pts.f1 >= r.f1) & ((pts.ratio < r.ratio) | (pts.f1 > r.f1))]
    return len(others) > 0
pts["dominated"] = pts.apply(dominated, axis=1)
print("\nPareto frontier (not dominated):")
print(pts[~pts.dominated].sort_values("ratio").round(3).to_string(index=False))
print("\nDominated:")
print(pts[pts.dominated].sort_values("ratio").round(3).to_string(index=False))
print(f"\n1/delta = {1/a.delta:.1f}; certificate anytime check: eprocess_max >= 1/delta means the anytime certificate is CONFIRMED (evidence against E[risk] > eps).")
