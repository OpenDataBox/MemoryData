"""v7 — recover what the summary dropped, as cheaply as possible.

Residual unit = raw SENTENCE (not turn, not paraphrased fact).
Selection policies under budget b (fraction of raw session tokens):
  none    : compressed only
  random  : random sentences (3 seeds)
  lost    : sentences carrying lost atomic facts, ranked by lost-facts-per-token   (task-agnostic, today's policy)
  scorer  : logistic regression on sentence features, trained on OTHER conversations
            with label "sentence lies in a QA evidence turn"  (query-DISTRIBUTION-aware, query-agnostic)
  oracle  : sentences in this session's QA evidence turns          (ceiling)
  raw     : full raw session
All arms share prior-session retrieval, reader, and answer normalization.

Two passes: pass 1 (GPU) compresses, extracts facts, builds sentence features;
pass 2 (CPU) trains leave-one-conv-out scorers; pass 3 (GPU) evaluates.
"""
import argparse, os, sys, re, time, random, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from tqdm import tqdm

from certmem.config import Config
from certmem.locomo import load, token_f1
from certmem.llm import Engine
from certmem.hosts.base import MemoryState, Retriever
from certmem.hosts.mem0_adapter import get_host
from certmem.facts import extract_facts, check_entailment

READER_SYS = ("You answer questions about a long conversation using only the memory notes provided. "
              "Reply with the shortest exact answer only: a name, a date written like '7 May 2023', a number "
              "written in digits, or a short noun phrase. No sentence, no explanation. "
              "If the notes do not contain the answer, reply: unknown")
def reader_prompt(context, question): return f"Memory notes:\n{context}\n\nQuestion: {question}\nAnswer:"

# ---------- answer normalization (applied to predictions AND gold before scoring) ----------
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
    m = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b", s)       # May 7, 2023 -> 7 May 2023
    if m and m.group(1)[:3].lower() in MON: s = s.replace(m.group(0), f"{int(m.group(2))} {MON[m.group(1)[:3].lower()]} {m.group(3)}")
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9}),?\s+(\d{4})\b", s)           # 7 May, 2023 -> 7 May 2023
    if m and m.group(2)[:3].lower() in MON: s = s.replace(m.group(0), f"{int(m.group(1))} {MON[m.group(2)[:3].lower()]} {m.group(3)}")
    return s.strip()
def norm_tokens(s): return [t for t in re.sub(r"[^a-z0-9 ]", " ", normalize_answer(s).lower()).split() if t]
def f1n(pred, gold): return token_f1(" ".join(norm_tokens(pred)), " ".join(norm_tokens(gold)))
def lenient(pred, gold):
    p, g = " ".join(norm_tokens(pred)), " ".join(norm_tokens(gold))
    if not p or p == "unknown": return 0
    if g in p or p in g: return 1
    cg = {t for t in g.split() if t not in STOP}; cp = {t for t in p.split() if t not in STOP}
    return int(len(cg & cp) / max(1, len(cg)) >= 0.5)

def content(s): return {t for t in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split() if t not in STOP and len(t) > 1}
def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]

DATE_RE = re.compile(r"\b(\d{1,2}(st|nd|rd|th)?|yesterday|today|tomorrow|last|next|ago|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|\d{4})\b", re.I)

def sentence_features(sent, turn_idx, n_turns, speaker_is_a, n_lost_here, n_facts_here, tok_len):
    return [n_lost_here, n_facts_here, n_lost_here / max(1, tok_len) * 10, int(bool(DATE_RE.search(sent))),
            len(re.findall(r"\d", sent)) > 0, len(re.findall(r"\b[A-Z][a-z]+", sent)), tok_len / 50.0,
            int("?" in sent), turn_idx / max(1, n_turns), int(speaker_is_a),
            int(bool(re.search(r"\b(not|never|don't|didn't|can't|won't|no)\b", sent, re.I))),
            int(bool(re.search(r"\b(plan|going to|will|next|because|since|only if|unless)\b", sent, re.I)))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="mem0"); ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--limit_convs", type=int, default=0)
    ap.add_argument("--budgets", default="0.10,0.20,0.40")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="runs/recover_v7")
    ap.add_argument("--prior_k", type=int, default=6)
    args = ap.parse_args()
    budgets = [float(b) for b in args.budgets.split(",")]
    os.makedirs(args.out, exist_ok=True)

    cfg = Config(host=args.host); convs = load(cfg.locomo_path)
    if args.limit_convs: convs = dict(list(convs.items())[:args.limit_convs])
    engine = Engine(cfg); retr = Retriever(cfg.embed_model); tok = engine.tok
    host_kw = {"cap": args.cap} if (args.cap and args.host == "mem0") else ({"n_bullets": args.cap} if (args.cap and args.host == "lightmem") else {})
    host = get_host(args.host, engine, **host_kw)
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    t0 = time.time()

    # ================= pass 1: compress, facts, sentence units + features =================
    S = {}   # (cid, t) -> dict
    for cid, conv in convs.items():
        sessions, qas = conv["sessions"], conv["qa"]
        qa_by_session = {}
        for q in qas:
            for s in q.sessions(): qa_by_session.setdefault(s, []).append(q)
        facts = extract_facts(engine, sessions, win=3)
        host_state = MemoryState()
        for s in tqdm(sessions, desc=f"pass1 {cid}"):
            t = s.num; speaker_a = s.turns[0].speaker if s.turns else ""
            plus_units = host.compress(s, host_state)
            plus_block = "\n".join(f"[session {t}] {u}" for u in plus_units)
            # sentences
            units = []   # dict(text, turn, tok, feats, ...)
            for ti, turn in enumerate(s.turns):
                for sent in split_sentences(turn.text):
                    line = f"[session {t}] ({s.date}) {turn.speaker}: {sent}"
                    units.append({"line": line, "turn": ti + 1, "tok": ntok(line), "sent": sent,
                                  "speaker_a": turn.speaker == speaker_a, "n_lost": 0, "n_facts": 0})
            # facts -> lost -> attribute to sentence
            fs = facts[t-1]
            if fs:
                ent = check_entailment(engine, plus_block, [f["fact"] for f in fs])
                for f, e in zip(fs, ent):
                    cands = [u for u in units if f["turn_start"] <= u["turn"] <= f["turn_end"]]
                    if not cands: continue
                    fc = content(f["fact"])
                    best = max(cands, key=lambda u: len(fc & content(u["sent"])))
                    best["n_facts"] += 1
                    if not e: best["n_lost"] += 1
            for u in units:
                u["feats"] = sentence_features(u["sent"], u["turn"], len(s.turns), u["speaker_a"], u["n_lost"], u["n_facts"], u["tok"])
            qs = qa_by_session.get(t, [])
            ev_turns = {k for q in qs for k in q.turns_in(t)}
            for u in units: u["label"] = int(u["turn"] in ev_turns)     # supervision for scorer (used only on OTHER convs)
            raw_tokens = sum(u["tok"] for u in units)
            S[(cid, t)] = {"units": units, "plus_block": plus_block, "plus_units": plus_units, "qs": qs,
                           "raw_tokens": raw_tokens, "ev_turns": ev_turns, "prior_state": host_state.snapshot(),
                           "n_lost_total": sum(u["n_lost"] for u in units), "n_facts_total": len(fs)}
            for u in plus_units: host_state.add(u, t, "host")
        print(f"[pass1 {cid}] done {time.time()-t0:.0f}s", flush=True)

    # ================= pass 2: leave-one-conv-out sentence scorers =================
    from sklearn.linear_model import LogisticRegression
    scorers = {}
    for cid in convs:
        X = [u["feats"] for (c, t), d in S.items() if c != cid for u in d["units"]]
        y = [u["label"] for (c, t), d in S.items() if c != cid for u in d["units"]]
        if len(set(y)) < 2 or len(y) < 50:
            scorers[cid] = None; continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.array(X, float), np.array(y))
        scorers[cid] = clf
    if any(v is not None for v in scorers.values()):
        names = ["n_lost", "n_facts", "lost_per_tok", "has_date", "has_digit", "n_caps", "len", "is_q", "pos", "spk_a", "negation", "plan"]
        c0 = next(v for v in scorers.values() if v is not None)
        print("scorer coefficients:", {n: round(float(w), 2) for n, w in zip(names, c0.coef_[0])})

    # ================= pass 3: evaluate policies =================
    rows = []
    for (cid, t), d in tqdm(S.items(), desc="pass3"):
        qs = d["qs"]
        if not qs: continue
        units, raw_tokens, plus_block = d["units"], d["raw_tokens"], d["plus_block"]
        idx_prior = retr.build(d["prior_state"])
        priors = [retr.retrieve(idx_prior, q.question, args.prior_k, tok, cfg.context_budget_tokens // 2) for q in qs]
        n = len(units)

        def evaluate(sel, policy, budget, seed=0):
            sel = sorted(sel)
            extra = "\n".join(units[i]["line"] for i in sel)
            added = sum(units[i]["tok"] for i in sel)
            block = "\n".join(u["line"] for u in units) if policy == "raw" else f"{plus_block}\n{extra}".strip()
            ctx = [f"{pr}\n{block}".strip() for pr in priors]
            preds = engine.generate(READER_SYS, [reader_prompt(c, q.question) for c, q in zip(ctx, qs)], max_tokens=32)
            for q, p in zip(qs, preds):
                rows.append({"conv_id": cid, "session": t, "policy": policy, "budget": budget, "seed": seed,
                             "category": q.category, "question": q.question, "gold": q.answer, "pred": p,
                             "f1": f1n(p, q.answer), "lenient": lenient(p, q.answer),
                             "added_tokens": added, "raw_tokens": raw_tokens, "plus_tokens": ntok(plus_block)})

        def fill(order, budget):
            sel, used, cap_ = [], 0, budget * raw_tokens
            for i in order:
                if used + units[i]["tok"] <= cap_: sel.append(i); used += units[i]["tok"]
            return sel

        lost_order = sorted(range(n), key=lambda i: -(units[i]["n_lost"] / max(1, units[i]["tok"])))
        lost_order = [i for i in lost_order if units[i]["n_lost"] > 0] + [i for i in lost_order if units[i]["n_lost"] == 0]
        clf = scorers.get(cid)
        if clf is not None:
            sc = clf.predict_proba(np.array([u["feats"] for u in units], float))[:, 1]
            score_order = sorted(range(n), key=lambda i: -(sc[i] / max(1, units[i]["tok"]) ))   # value per token
        else:
            score_order = lost_order
        or_order = [i for i in range(n) if units[i]["turn"] in d["ev_turns"]] + [i for i in range(n) if units[i]["turn"] not in d["ev_turns"]]

        evaluate([], "none", 0.0); evaluate([], "raw", 1.0)
        for b in budgets:
            evaluate(fill(lost_order, b), "lost", b)
            evaluate(fill(score_order, b), "scorer", b)
            evaluate(fill(or_order, b), "oracle", b)
            for sd in range(args.seeds):
                rnd = list(range(n)); random.Random(sd * 1000 + t).shuffle(rnd); evaluate(fill(rnd, b), "random", b, seed=sd)
        pd.DataFrame(rows).to_csv(f"{args.out}/items.csv", index=False)

    df = pd.DataFrame(rows)
    g = df.groupby(["policy", "budget"]).agg(f1=("f1", "mean"), lenient=("lenient", "mean"), n=("f1", "size")).reset_index()
    tot = df.groupby(["policy", "budget"]).apply(lambda d: d.added_tokens.sum() / d.raw_tokens.sum()).rename("added_frac").reset_index()
    g = g.merge(tot, on=["policy", "budget"])
    g.to_csv(f"{args.out}/summary.csv", index=False)
    order = {"none": 0, "random": 1, "lost": 2, "scorer": 3, "oracle": 4, "raw": 5}
    g["o"] = g.policy.map(order); g = g.sort_values(["budget", "o"]).drop(columns="o")
    print("\n=== v7 recovery (normalized F1 / lenient) ===")
    print(g[["policy", "budget", "added_frac", "f1", "lenient", "n"]].round(3).to_string(index=False))
    base = g[g.policy == "none"].f1.iloc[0]; top = g[g.policy == "raw"].f1.iloc[0]
    def rec(p, b):
        d = g[(g.policy == p) & (g.budget == b)]; return (d.f1.iloc[0] - base) / (top - base + 1e-9) if len(d) else float("nan")
    print(f"\nrecovery of (raw - none) gap; none={base:.3f} raw={top:.3f}")
    for b in budgets:
        print(f"  budget {b:.2f}:  random={rec('random',b):+.2f}  lost={rec('lost',b):+.2f}  scorer={rec('scorer',b):+.2f}  oracle={rec('oracle',b):+.2f}")
    print("\nby category (budget 0.20): 1=multi 2=temporal 3=open 4=single")
    d2 = df[df.budget.isin([0.0, 0.2, 1.0])]
    print(d2.groupby(["category", "policy"]).f1.mean().unstack().round(3))

if __name__ == "__main__":
    main()
