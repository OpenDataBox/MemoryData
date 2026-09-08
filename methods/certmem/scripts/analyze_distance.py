"""F1 by evidence distance (no GPU). Mirrors Fig 10c of 2606.24775.

Distance = (number of sessions in the conversation) - (earliest evidence session index).
Larger = evidence further back. Bins: 1-5, 6-10, 11-15, 16-20, 21-25, 26+.

Works on items.csv from run_read_v9 / run_system_v10 / run_combined (needs 'question','conv_id','policy','f1').
Evidence sessions are recovered from the LoCoMo QA file by question text.
"""
import argparse, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from certmem.locomo import load

ap = argparse.ArgumentParser()
ap.add_argument("items_csv")
ap.add_argument("--locomo", default="data/locomo10.json")
ap.add_argument("--policies", default="", help="comma list; default = all")
ap.add_argument("--config", default="", help="filter config column if present (e.g. full)")
ap.add_argument("--budget", default="", help="keep rows with budget in this comma list (plus rows with budget 0/NaN)")
a = ap.parse_args()

df = pd.read_csv(a.items_csv)
if a.config and "config" in df: df = df[df.config == a.config]
if a.policies: df = df[df.policy.isin(a.policies.split(","))]
if a.budget and "budget" in df:
    keep = [float(x) for x in a.budget.split(",")]
    df = df[df.budget.isin(keep) | (df.budget == 0) | df.budget.isna()]
    df["policy"] = df.policy + "@" + df.budget.fillna(0).astype(str)
convs = load(a.locomo)
n_sess = {cid: len(c["sessions"]) for cid, c in convs.items()}
ev = {}
for cid, c in convs.items():
    for q in c["qa"]:
        s = q.sessions()
        if s: ev[(cid, q.question)] = (min(s), max(s))
df["earliest_ev"] = [ev.get((c, q), (np.nan, np.nan))[0] for c, q in zip(df.conv_id, df.question)]
df["dist"] = [n_sess[c] - e if not np.isnan(e) else np.nan for c, e in zip(df.conv_id, df.earliest_ev)]
df = df.dropna(subset=["dist"])
bins = [0, 5, 10, 15, 20, 25, 100]; labels = ["1-5", "6-10", "11-15", "16-20", "21-25", "26+"]
df["bin"] = pd.cut(df.dist, bins=bins, labels=labels)
tab = df.groupby(["bin", "policy"], observed=True).f1.mean().unstack().round(3)
cnt = df.groupby("bin", observed=True).question.nunique()
tab["n_q"] = cnt
print("=== F1 by evidence distance (sessions back from end) ===")
print(tab.to_string())
# slope: how much each policy loses from nearest to farthest bin
print("\nnear (1-5) -> far (>=16) drop:")
near = df[df.dist <= 5].groupby("policy").f1.mean(); far = df[df.dist >= 16].groupby("policy").f1.mean()
for p in tab.columns:
    if p in near and p in far:
        print(f"  {p:22s} {near[p]:.3f} -> {far[p]:.3f}   drop={near[p]-far[p]:+.3f}")
