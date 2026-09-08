"""Convert an external system's LoCoMo predictions into our items.csv format and score with the SAME
normalized F1 / lenient used for our system, so analyze_ci.py can compare them paired by (conv_id, question).

Supports TriMem's result json ({'detailed_results': [{question, answer, reference, category}, ...]}) and a
generic list of {question, prediction|answer, gold|reference, category} via --fields.
Conversation id is recovered by matching question text against our LoCoMo loader.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
import pandas as pd
from certmem.locomo import load
from run_read_v9 import f1n, lenient

ap = argparse.ArgumentParser()
ap.add_argument("result_json")
ap.add_argument("--system", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fields", default="question,answer,reference,category", help="keys for question,prediction,gold,category")
ap.add_argument("--locomo", default="data/locomo10.json")
ap.add_argument("--tokens", type=float, default=float("nan"), help="mean read tokens reported by the system (for the Pareto table)")
a = ap.parse_args()
qk, pk, gk, ck = a.fields.split(",")
raw = json.load(open(a.result_json))
items = (raw.get("detailed_results") or raw.get("data") or raw) if isinstance(raw, dict) else raw
convs = load(a.locomo)
q2conv, q2cat = {}, {}
for cid, c in convs.items():
    for q in c["qa"]: q2conv[q.question.strip().lower()] = cid; q2cat[q.question.strip().lower()] = q.category
rows, miss = [], 0
for it in items:
    q = str(it.get(qk, "")).strip(); pred = str(it.get(pk, "") or ""); gold = str(it.get(gk, "") or "")
    cid = q2conv.get(q.lower())
    if cid is None: miss += 1; continue
    cat = it.get(ck)
    if cat is None or (isinstance(cat, float) and cat != cat): cat = q2cat.get(q.lower())
    rows.append({"conv_id": cid, "config": "full", "policy": a.system, "category": cat, "question": q,
                 "gold": gold, "pred": pred, "f1": f1n(pred, gold), "lenient": lenient(pred, gold), "read_tokens": a.tokens})
df = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True); df.to_csv(f"{a.out}/items.csv", index=False)
print(f"{a.system}: n={len(df)} (unmatched {miss})  F1={df.f1.mean():.3f}  lenient={df.lenient.mean():.3f}")
print(df.groupby("category").f1.mean().round(3).to_string())
