"""Dose-response across compression levels (v6).
Shows fact coverage, probe delta, QA F1 and tokens vs compression ratio.
The monotone co-movement of fact_risk and bench delta with ratio is the
aggregate link from the certified quantity (coverage) to downstream QA.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
from scipy.stats import spearmanr

frames = []
for d in sys.argv[1:]:
    s = pd.read_csv(f"{d}/sessions.csv"); s["run"] = os.path.basename(d); frames.append(s)
df = pd.concat(frames)
agg = {"ratio": ("ratio", "mean"), "bench_delta": ("bench_delta_signed", "mean"), "f1_plus": ("bench_f1_plus", "mean"), "n": ("session", "size")}
if "fact_risk" in df: agg["fact_risk"] = ("fact_risk", "mean")
if "probe_delta_signed" in df: agg["probe_delta"] = ("probe_delta_signed", "mean")
g = df.groupby("run").agg(**agg).sort_values("ratio")
print("per compression level (ratio = compressed/raw tokens):")
print(g.round(3))
for col in ["fact_risk", "probe_delta"]:
    if col in df:
        print(f"\npooled session-level Spearman({col}, bench delta) across levels (n={len(df)}): "
              f"{spearmanr(df[col], df.bench_delta_signed, nan_policy='omit').correlation:+.3f}")
        print(f"level-wise Spearman({col}, bench delta) over {len(g)} levels: {spearmanr(g[col], g.bench_delta).correlation:+.3f}")
print(f"\nratio vs bench delta: {spearmanr(df.ratio, df.bench_delta_signed).correlation:+.3f}")
