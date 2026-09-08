"""Relabel QA correctness with clean scoring, then re-test the fact-level sensor (v6c).

Three label definitions, reported side by side:
  F1      : original (delta = f1_minus - f1_plus > tau)
  LENIENT : normalized containment (numbers as digits, dates canonical, punctuation
            stripped); correct if gold ⊆ pred or pred ⊆ gold (with >=1 content token)
  JUDGE   : Qwen3-8B yes/no "does the prediction convey the gold answer?"
            (hurt = raw correct AND compressed incorrect)

Uses runs/<dir>/fact_matches.csv from analyze_fact_match.py (needs 'fact' etc.).
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

NUM = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9",
       "ten":"10","eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15","twenty":"20","thirty":"30",
       "a couple of":"2","a few":"3","half":"0.5"}
STOP = {"the","a","an","and","or","of","in","on","at","to","for","is","was","are","were","her","his","their","she","he","they","it","with","about"}

def norm(s):
    s = str(s).lower().strip()
    for k, v in NUM.items(): s = re.sub(rf"\b{k}\b", v, s)
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def content(s): return [t for t in norm(s).split() if t not in STOP]

def lenient(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or p == "unknown": return 0
    if g in p or p in g: return 1
    cg, cp = set(content(gold)), set(content(pred))
    if not cg: return int(g in p)
    return int(len(cg & cp) / len(cg) >= 0.5)   # half of gold content tokens present

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
ap.add_argument("--judge", action="store_true", help="also run Qwen3-8B judge (needs GPU free)")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
a = ap.parse_args()

df = pd.read_csv(f"{a.run_dir}/fact_matches.csv")
df["len_minus"] = [lenient(p, g) for p, g in zip(df.pred_minus, df.gold)]
df["len_plus"]  = [lenient(p, g) for p, g in zip(df.pred_plus,  df.gold)]
df["hurt_f1"] = df.qa_hurt.astype(bool)
df["hurt_lenient"] = (df.len_minus == 1) & (df.len_plus == 0)

if a.judge:
    from certmem.config import Config
    from certmem.llm import Engine
    eng = Engine(Config(model=a.model))
    SYS = ("You judge whether a predicted answer conveys the gold answer to a question. Minor wording, number "
           "format, or extra detail is fine. Answer exactly yes or no.")
    def judge(preds):
        prompts = [f"Question: {q}\nGold answer: {g}\nPredicted answer: {p}\nDoes the prediction convey the gold answer? yes or no:"
                   for q, g, p in zip(df.qa_q, df.gold, preds)]
        outs = eng.generate(SYS, prompts, max_tokens=3)
        return [1 if o.strip().lower().startswith("y") else 0 for o in outs]
    df["j_minus"] = judge(df.pred_minus); df["j_plus"] = judge(df.pred_plus)
    df["hurt_judge"] = (df.j_minus == 1) & (df.j_plus == 0)

df.to_csv(f"{a.run_dir}/fact_matches_relabeled.csv", index=False)

print(f"=== {a.run_dir}  n={len(df)}")
print(f"raw-arm accuracy:  F1>0.2 proxy={ (df.qa_f1_minus>0.2).mean() if 'qa_f1_minus' in df else float('nan'):.3f}  lenient={df.len_minus.mean():.3f}"
      + (f"  judge={df.j_minus.mean():.3f}" if a.judge else ""))
print(f"compressed-arm:                       lenient={df.len_plus.mean():.3f}" + (f"  judge={df.j_plus.mean():.3f}" if a.judge else ""))

labels = ["hurt_f1", "hurt_lenient"] + (["hurt_judge"] if a.judge else [])
w = df[df.sim >= 0.6]
for lab in labels:
    print(f"\n[{lab}]  P(hurt)={df[lab].mean():.3f}")
    for col, name in [("window_risk", "window coverage"), ("fact_lost_top1", "top-1 fact lost"), ("fact_lost_top2", "top-2 either lost")]:
        d = w
        au = auroc(d[col], d[lab]); lo, hi = boot(d[col], d[lab])
        x = d[col] > 0.5
        p1 = d[lab][x].mean() if x.any() else np.nan; p0 = d[lab][~x].mean() if (~x).any() else np.nan
        print(f"    {name:18s} AUROC={au:.3f} [{lo:.2f},{hi:.2f}]  P(hurt|lost)={p1:.2f}  P(hurt|kept)={p0:.2f}  lift={p1/max(1e-9,p0):.2f}x")

# how many F1-hurt were pure formatting?
fmt = df[df.hurt_f1 & ~df.hurt_lenient]
print(f"\nF1-hurt but lenient-OK (formatting noise): {len(fmt)}/{df.hurt_f1.sum()} = {len(fmt)/max(1,df.hurt_f1.sum()):.0%}")
for _, r in fmt.head(5).iterrows():
    print(f"   gold={r.gold!r}  raw={r.pred_minus!r}  comp={r.pred_plus!r}")

best_lab = "hurt_judge" if a.judge else "hurt_lenient"
au = auroc(w.fact_lost_top1, w[best_lab])
print(f"\nVERDICT ({best_lab}, top-1 fact): AUROC={au:.3f} -> " + ("GO" if au >= 0.65 else "MARGINAL" if au >= 0.58 else "NO-GO"))
