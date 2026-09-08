"""Merge our items.csv with external systems' items.csv and run paired conversation-bootstrap comparisons."""
import argparse, subprocess, sys, os
import pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--ours", default="runs/v15_locomo/items.csv")
ap.add_argument("--ours_policy", default="full:certified")
ap.add_argument("--external", nargs="+", required=True, help="items.csv files of external systems")
ap.add_argument("--out", default="runs/compare_baselines.csv")
a = ap.parse_args()
dfs = [pd.read_csv(a.ours)] + [pd.read_csv(x) for x in a.external]
df = pd.concat(dfs, ignore_index=True); df.to_csv(a.out, index=False)
systems = sorted({p for x in a.external for p in pd.read_csv(x).policy.unique()})
pairs = [f"{a.ours_policy} vs full:{s}" for s in systems]
print(df.groupby(["config", "policy"]).f1.mean().round(3).to_string())
subprocess.run([sys.executable, "scripts/analyze_ci.py", a.out, "--by_category", "--pairs", *pairs])
