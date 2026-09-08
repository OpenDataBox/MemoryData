"""v9 — read-time expansion; loss awareness applied to AMOUNT, not ORDER.

v8 finding: multiplying relevance by session loss r_t re-orders candidates toward
lossy-but-irrelevant turns and loses. v9 keeps relevance order and lets r_t decide
HOW MUCH to expand:
  read_uniform  : relevance order, fixed budget B
  read_scaled   : relevance order, per-query budget B_q = B * r_hits / r_mean  (mean over queries == B)
  read_gated    : relevance order, skip candidates whose session loss r_t < r_median (compression was fine there),
                  budget B goes to lossy sessions only
  read_certified: relevance order, expand until residual risk sum(rel*r) <= tau * initial  (variable budget)


Memory for a conversation = all sessions compressed by the host (write-time, sequential),
each compressed unit carrying provenance to its raw turn(s). Each session t has a
certified loss r_t = 1 - fact coverage (write-time entailment).

At read time, for a question q:
  1. retrieve top-K compressed units globally (hybrid BM25+dense)      -- always in context
  2. choose which retrieved units to EXPAND to their raw turns, under an expansion budget B:
       read_uniform   : by relevance only                          (TierMem/Hindsight-style)
       read_lossaware : by relevance * r_t(unit) / cost            (ours)
       read_certified : loss-aware order, stop when residual risk R(E) <= tau * R(empty)  (ours, variable budget)
  3. answer from [retrieved units + expanded raw turns]
References:
  compressed_only : step 1 only
  raw_rag         : retrieve raw turns directly with the SAME total token budget (no compression)
  full_raw        : entire conversation raw (upper ref, budget-free)

Outputs runs/<out>/items.csv, summary.csv. F1 is normalized (dates/numbers canonical).
"""
import argparse, os, sys, re, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from certmem.config import Config
from certmem.locomo import load, token_f1
from certmem.llm import Engine
from certmem.hosts.base import MemoryState
from certmem.hosts.mem0_adapter import get_host
from certmem.facts import extract_facts, check_entailment

READER_SYS = ("You answer questions about a long conversation using only the memory notes provided. "
              "Reply with the shortest exact answer only: a name, a date written like '7 May 2023', a number "
              "written in digits, or a short noun phrase. No sentence, no explanation. "
              "If the notes do not contain the answer, reply: unknown")
def reader_prompt(context, question): return f"Memory notes:\n{context}\n\nQuestion: {question}\nAnswer:"

NUM = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9",
       "ten":"10","eleven":"11","twelve":"12","fifteen":"15","twenty":"20","thirty":"30"}
MON = {"jan":"January","feb":"February","mar":"March","apr":"April","may":"May","jun":"June","jul":"July",
       "aug":"August","sep":"September","sept":"September","oct":"October","nov":"November","dec":"December"}
STOP = {"the","a","an","and","or","of","in","on","at","to","for","is","was","are","were","her","his","their","she","he","they","it","with","about","that","this"}
def normalize_answer(s):
    s = str(s).strip().strip('."\'')
    s = re.sub(r"^(the answer is|answer)[:\s]*", "", s, flags=re.I)
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s)
    for k, v in NUM.items(): s = re.sub(rf"\b{k}\b", v, s, flags=re.I)
    m = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b", s)
    if m and m.group(1)[:3].lower() in MON: s = s.replace(m.group(0), f"{int(m.group(2))} {MON[m.group(1)[:3].lower()]} {m.group(3)}")
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9}),?\s+(\d{4})\b", s)
    if m and m.group(2)[:3].lower() in MON: s = s.replace(m.group(0), f"{int(m.group(1))} {MON[m.group(2)[:3].lower()]} {m.group(3)}")
    return s.strip()
def norm_tokens(s): return [t for t in re.sub(r"[^a-z0-9 ]", " ", normalize_answer(s).lower()).split() if t]
def f1n(p, g): return token_f1(" ".join(norm_tokens(p)), " ".join(norm_tokens(g)))
def lenient(p, g):
    p, g = " ".join(norm_tokens(p)), " ".join(norm_tokens(g))
    if not p or p == "unknown": return 0
    if g in p or p in g: return 1
    cg = {t for t in g.split() if t not in STOP}; cp = {t for t in p.split() if t not in STOP}
    return int(len(cg & cp) / max(1, len(cg)) >= 0.5)
def content(s): return {t for t in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split() if t not in STOP and len(t) > 1}

class Index:
    def __init__(self, emb, texts):
        self.texts = texts
        self.bm25 = BM25Okapi([t.lower().split() for t in texts]) if texts else None
        self.dense = emb.encode(texts, normalize_embeddings=True, batch_size=64) if texts else None
        self.emb = emb
    def search(self, query, k):
        if not self.texts: return [], []
        q = self.emb.encode([query], normalize_embeddings=True)[0]
        d = self.dense @ q; b = np.array(self.bm25.get_scores(query.lower().split()))
        def rank(x): o = np.argsort(-x); r = np.empty_like(o); r[o] = np.arange(len(x)); return r
        rrf = 1.0/(60+rank(d)) + 1.0/(60+rank(b))
        order = np.argsort(-rrf)[:k]
        rel = rrf[order]; rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9) * 0.9 + 0.1   # in (0.1,1]
        return list(order), list(rel)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="mem0"); ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--limit_convs", type=int, default=0)
    ap.add_argument("--k", type=int, default=10, help="retrieved compressed units")
    ap.add_argument("--budgets", default="100,200,400", help="expansion token budgets per query")
    ap.add_argument("--tau", default="0.5,0.3,0.15", help="certified stopping: residual risk fraction")
    ap.add_argument("--out", default="runs/read_v9")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]; taus = [float(x) for x in args.tau.split(",")]
    os.makedirs(args.out, exist_ok=True)

    cfg = Config(host=args.host); convs = load(cfg.locomo_path)
    if args.limit_convs: convs = dict(list(convs.items())[:args.limit_convs])
    engine = Engine(cfg); tok = engine.tok
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(cfg.embed_model, device="cuda")
    host_kw = {"cap": args.cap} if (args.cap and args.host == "mem0") else ({"n_bullets": args.cap} if (args.cap and args.host == "lightmem") else {})
    host = get_host(args.host, engine, **host_kw)
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    rows, t0 = [], time.time()

    for cid, conv in convs.items():
        sessions, qas = conv["sessions"], conv["qa"]
        # ---------- write time: sequential compression with provenance + certified loss ----------
        units = []      # {text, session, turns:[raw turn ids], tok}
        raw_turns = []  # {text, session, turn, tok}
        r_sess = {}
        facts = extract_facts(engine, sessions, win=3)
        host_state = MemoryState()
        for s in tqdm(sessions, desc=f"write {cid}"):
            t = s.num
            for ti, turn in enumerate(s.turns):
                line = f"[session {t}] ({s.date}) {turn.speaker}: {turn.text}"
                raw_turns.append({"text": line, "session": t, "turn": ti + 1, "tok": ntok(line)})
            plus_units = host.compress(s, host_state)
            plus_block = "\n".join(f"[session {t}] {u}" for u in plus_units)
            fs = facts[t-1]
            if fs:
                ent = check_entailment(engine, plus_block, [f["fact"] for f in fs])
                r_sess[t] = 1 - float(np.mean(ent))
            else:
                r_sess[t] = 0.5
            for u in plus_units:
                uc = content(u)
                sims = [(len(uc & content(turn.text)) / max(1, len(uc)), ti + 1) for ti, turn in enumerate(s.turns)]
                sims.sort(reverse=True)
                prov = [ti for sc, ti in sims[:2] if sc > 0] or [sims[0][1]]
                units.append({"text": f"[session {t}] {u}", "session": t, "turns": prov, "tok": ntok(u) + 4})
                host_state.add(u, t, "host")
        r_vals = np.array([r_sess[t] for t in sorted(r_sess)]); r_mean = float(r_vals.mean())
        print(f"[{cid}] units={len(units)} raw_turns={len(raw_turns)} session loss r_t: mean={r_vals.mean():.2f} std={r_vals.std():.2f} min={r_vals.min():.2f} max={r_vals.max():.2f}", flush=True)

        uidx = Index(emb, [u["text"] for u in units])
        ridx = Index(emb, [r["text"] for r in raw_turns])
        turn_lookup = {(r["session"], r["turn"]): r for r in raw_turns}
        full_raw = "\n".join(r["text"] for r in raw_turns)

        # ---------- read time ----------
        prompts, meta = [], []
        def add(policy, budget, ctx, q, exp_tok, base_tok, n_exp):
            prompts.append(reader_prompt(ctx, q.question))
            meta.append({"conv_id": cid, "policy": policy, "budget": budget, "category": q.category,
                         "question": q.question, "gold": q.answer, "exp_tokens": exp_tok,
                         "read_tokens": base_tok + exp_tok, "n_expanded": n_exp, "n_evidence_sessions": len(q.sessions())})
        for q in qas:
            hits, rel = uidx.search(q.question, args.k)
            base_ctx = "\n".join(units[i]["text"] for i in hits)
            base_tok = sum(units[i]["tok"] for i in hits)
            add("compressed_only", 0, base_ctx, q, 0, base_tok, 0)
            # expansion candidates: (turn key) with relevance, loss, cost
            cand = {}
            for i, rl in zip(hits, rel):
                for tt in units[i]["turns"]:
                    key = (units[i]["session"], tt)
                    if key not in turn_lookup: continue
                    c = cand.setdefault(key, {"rel": 0.0, "r": r_sess.get(key[0], 0.5), "cost": turn_lookup[key]["tok"]})
                    c["rel"] = max(c["rel"], rl)
            keys = list(cand)
            def build(sel):
                extra = "\n".join(turn_lookup[k]["text"] for k in sel)
                return f"{base_ctx}\n{extra}".strip(), sum(cand[k]["cost"] for k in sel)
            def fill(order, B):
                sel, used = [], 0
                for k in order:
                    if used + cand[k]["cost"] <= B: sel.append(k); used += cand[k]["cost"]
                return sel
            uni_order = sorted(keys, key=lambda k: -cand[k]["rel"] / cand[k]["cost"])
            r_hits = float(np.mean([cand[k]["r"] for k in keys])) if keys else r_mean
            r_med = float(np.median(list(r_sess.values())))
            gated_order = [k for k in uni_order if cand[k]["r"] >= r_med] + [k for k in uni_order if cand[k]["r"] < r_med]
            for B in budgets:
                sel = fill(uni_order, B); ctx, et = build(sel); add("read_uniform", B, ctx, q, et, base_tok, len(sel))
                Bq = int(B * r_hits / max(1e-6, r_mean))
                sel = fill(uni_order, Bq); ctx, et = build(sel); add("read_scaled", B, ctx, q, et, base_tok, len(sel))
                sel = fill(gated_order, B); ctx, et = build(sel); add("read_gated", B, ctx, q, et, base_tok, len(sel))
                rh, _ = ridx.search(q.question, 40)
                sel_r, used = [], 0
                for i in rh:
                    if used + raw_turns[i]["tok"] <= base_tok + B: sel_r.append(i); used += raw_turns[i]["tok"]
                add("raw_rag", B, "\n".join(raw_turns[i]["text"] for i in sel_r), q, used, 0, len(sel_r))
            R0 = sum(cand[k]["rel"] * cand[k]["r"] for k in keys) + 1e-9
            for tau in taus:
                sel, R = [], R0
                for k in uni_order:
                    if R <= tau * R0: break
                    sel.append(k); R -= cand[k]["rel"] * cand[k]["r"]
                ctx, et = build(sel); add("read_certified", tau, ctx, q, et, base_tok, len(sel))
            add("full_raw", 0, full_raw, q, ntok(full_raw), 0, len(raw_turns))

        preds = engine.generate(READER_SYS, prompts, max_tokens=32)
        for m, p in zip(meta, preds):
            rows.append({**m, "pred": p, "f1": f1n(p, m["gold"]), "lenient": lenient(p, m["gold"])})
        pd.DataFrame(rows).to_csv(f"{args.out}/items.csv", index=False)
        print(f"[{cid}] done {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    g = df.groupby(["policy", "budget"]).agg(f1=("f1", "mean"), lenient=("lenient", "mean"),
                                             read_tokens=("read_tokens", "mean"), exp_tokens=("exp_tokens", "mean"),
                                             n_exp=("n_expanded", "mean"), n=("f1", "size")).reset_index()
    g.to_csv(f"{args.out}/summary.csv", index=False)
    order = {"compressed_only": 0, "read_uniform": 1, "read_scaled": 2, "read_gated": 3, "read_certified": 4, "raw_rag": 5, "full_raw": 6}
    g["o"] = g.policy.map(order); g = g.sort_values(["o", "budget"]).drop(columns="o")
    print("\n=== v9 read-time expansion (normalized F1) ===")
    print(g[["policy", "budget", "read_tokens", "exp_tokens", "n_exp", "f1", "lenient", "n"]].round(3).to_string(index=False))
    print("\nby category (1=multi 2=temporal 3=open 4=single), budget 200 / tau 0.3:")
    d2 = df[((df.policy.isin(["read_uniform", "read_scaled", "read_gated", "raw_rag"])) & (df.budget == 200)) |
            ((df.policy == "read_certified") & (df.budget == 0.3)) | (df.policy.isin(["compressed_only", "full_raw"]))]
    print(d2.groupby(["category", "policy"]).f1.mean().unstack().round(3))
    print("\nmulti-session evidence questions only:")
    d3 = d2[d2.n_evidence_sessions > 1]
    if len(d3): print(d3.groupby("policy").f1.mean().round(3).to_string())

if __name__ == "__main__":
    main()
