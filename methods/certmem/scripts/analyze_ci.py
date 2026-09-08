"""Paired, conversation-clustered bootstrap for policy comparisons (no GPU).

For items.csv from run_system_v1x / run_read_v9 / run_combined:
  each row = (conv_id, config, policy, question, f1, ...).
For a pair (A, B) we align rows by (conv_id, question), take per-question difference d = f1_A - f1_B,
then bootstrap by resampling CONVERSATIONS (clusters) 5000 times: mean(d) with 95% CI and a two-sided p.
Also reports the sign test and n. Significance gate used in the paper: CI excludes 0 and p < 0.05.

Usage:
  python scripts/analyze_ci.py runs/v12_locomo/items.csv --pairs "full:certified vs full:raw_rag_400" "full:certified vs full:units_only"
Pair syntax: "<config>:<policy> vs <config>:<policy>". Config may be omitted if the file has none.
"""
import argparse, os, sys
import numpy as np, pandas as pd

def parse_side(s):
    s = s.strip()
    return s.split(":", 1) if ":" in s else (None, s)

ap = argparse.ArgumentParser()
ap.add_argument("items_csv")
ap.add_argument("--pairs", nargs="+", required=True)
ap.add_argument("--metric", default="f1")
ap.add_argument("--n_boot", type=int, default=5000)
ap.add_argument("--by_category", action="store_true")
a = ap.parse_args()

df = pd.read_csv(a.items_csv)
if "config" not in df: df["config"] = None
key = ["conv_id", "question"]

def subset(cfg, pol):
    d = df[df.policy == pol]
    if cfg is not None and df.config.notna().any(): d = d[d.config == cfg]
    return d.groupby(key, as_index=False).agg(**{a.metric: (a.metric, "mean"), "category": ("category", "first")})

def boot(d, n, seed=0):
    rng = np.random.default_rng(seed)
    convs = d.conv_id.unique(); groups = {c: d[d.conv_id == c][a.metric + "_diff"].values for c in convs}
    means = []
    for _ in range(n):
        pick = rng.choice(convs, len(convs), replace=True)
        vals = np.concatenate([groups[c] for c in pick]); means.append(vals.mean())
    means = np.array(means); obs = d[a.metric + "_diff"].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return obs, lo, hi, p

print(f"=== {a.items_csv}  metric={a.metric}  clusters=conversations  n_boot={a.n_boot}")
for pair in a.pairs:
    left, right = [parse_side(x) for x in pair.split(" vs ")]
    A, B = subset(*left), subset(*right)
    m = A.merge(B, on=key, suffixes=("_A", "_B"))
    if len(m) < 10: print(f"{pair}: too few paired rows ({len(m)})"); continue
    m[a.metric + "_diff"] = m[a.metric + "_A"] - m[a.metric + "_B"]
    obs, lo, hi, p = boot(m, a.n_boot)
    wins = (m[a.metric + "_diff"] > 0).sum(); losses = (m[a.metric + "_diff"] < 0).sum()
    sig = "SIG" if (lo > 0 or hi < 0) and p < 0.05 else "n.s."
    print(f"\n{pair}\n  n={len(m)} convs={m.conv_id.nunique()}  A={m[a.metric+'_A'].mean():.3f}  B={m[a.metric+'_B'].mean():.3f}  "
          f"diff={obs:+.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]  p={p:.3f}  wins/losses={wins}/{losses}  -> {sig}")
    if a.by_category:
        for c, g in m.groupby("category_A"):
            if len(g) >= 20:
                o, l, h, pp = boot(g, min(a.n_boot, 2000))
                print(f"    cat {c}: n={len(g):4d} diff={o:+.3f} CI=[{l:+.3f},{h:+.3f}] p={pp:.3f}")
