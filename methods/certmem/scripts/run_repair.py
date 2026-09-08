"""Repair experiment (Phase 1-1). Figure 3 of the paper.

Setting: session t is compressed by the host (M+). We may attach up to B% of the
session's raw tokens back as verbatim turns ("residual"). Which turns?

  none      : nothing attached (compressed only)                  -- floor
  random    : random turns until budget (mean of 3 seeds)         -- control
  coverage  : turns ranked by (#lost facts attributed to turn / tokens),
              greedy set-cover style, until budget                -- OUR policy (task-agnostic)
  oracle    : the QA evidence turns of this session, until budget -- ceiling (task-conditioned)
  raw       : full raw session                                    -- reference

Lost fact = extracted from raw window, NOT entailed by compressed block.
A lost fact is attributed to the single turn in its window with the highest
content-token overlap.

Outputs runs/<out>/repair.csv with per (policy, budget): F1, lenient acc,
added tokens, coverage after repair. Prior sessions retrieved identically for all arms.
"""
import argparse, os, sys, re, time, random
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
              "Reply with the shortest exact answer (a name, date, number, or short phrase). No explanation. "
              "If the notes do not contain the answer, reply: unknown")
def reader_prompt(context, question): return f"Memory notes:\n{context}\n\nQuestion: {question}\nAnswer:"

NUM = {"one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
STOP = {"the","a","an","and","or","of","in","on","at","to","for","is","was","are","were","her","his","their","she","he","they","it","with","about"}
def norm(s):
    s = str(s).lower()
    for k, v in NUM.items(): s = re.sub(rf"\b{k}\b", v, s)
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s); s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
def content(s): return {t for t in norm(s).split() if t not in STOP and len(t) > 1}
def lenient(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or p == "unknown": return 0
    if g in p or p in g: return 1
    cg, cp = content(gold), content(pred)
    return int(len(cg & cp) / max(1, len(cg)) >= 0.5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="mem0"); ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--limit_convs", type=int, default=2)
    ap.add_argument("--budgets", default="0.05,0.10,0.20,0.40")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="runs/repair")
    ap.add_argument("--prior_k", type=int, default=6)
    args = ap.parse_args()
    budgets = [float(b) for b in args.budgets.split(",")]
    os.makedirs(args.out, exist_ok=True)

    cfg = Config(host=args.host)
    convs = load(cfg.locomo_path)
    if args.limit_convs: convs = dict(list(convs.items())[:args.limit_convs])
    engine = Engine(cfg); retr = Retriever(cfg.embed_model); tok = engine.tok
    host_kw = {"cap": args.cap} if (args.cap and args.host == "mem0") else ({"n_bullets": args.cap} if (args.cap and args.host == "lightmem") else {})
    host = get_host(args.host, engine, **host_kw)
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)

    rows, t0 = [], time.time()
    for cid, conv in convs.items():
        sessions, qas = conv["sessions"], conv["qa"]
        qa_by_session = {}
        for q in qas:
            for s in q.sessions(): qa_by_session.setdefault(s, []).append(q)
        facts = extract_facts(engine, sessions, win=3)
        host_state = MemoryState()
        for s in tqdm(sessions, desc=f"conv {cid}"):
            t = s.num
            qs = qa_by_session.get(t, [])
            raw_units = host.raw_units(s)
            plus_units = host.compress(s, host_state)
            plus_block = "\n".join(f"[session {t}] {u}" for u in plus_units)
            turn_lines = [f"[session {t}] {u}" for u in raw_units]
            turn_tok = [ntok(x) for x in turn_lines]
            raw_tokens = sum(turn_tok)

            # ---- lost facts -> attribute to a turn
            fs = facts[t-1]
            lost_by_turn = np.zeros(len(raw_units))
            lost_facts = []
            if fs:
                ent = check_entailment(engine, plus_block, [f["fact"] for f in fs])
                for f, e in zip(fs, ent):
                    if e: continue
                    cands = range(f["turn_start"]-1, min(f["turn_end"], len(raw_units)))
                    fc = content(f["fact"])
                    best = max(cands, key=lambda i: len(fc & content(s.turns[i].text)), default=None)
                    if best is not None:
                        lost_by_turn[best] += 1; lost_facts.append(f["fact"])
            n_facts, n_lost = len(fs), len(lost_facts)
            # fact-level residual: the lost atomic facts themselves (~15 tok each), dates/numbers first
            def _prio(f): return -(1 if re.search(r"\d", f) else 0)
            lost_sorted = sorted(lost_facts, key=_prio)
            fact_lines = [f"[session {t}] {f}" for f in lost_sorted]
            fact_tok = [ntok(x) for x in fact_lines]

            if not qs:
                for u in plus_units: host_state.add(u, t, "host")
                continue

            idx_prior = retr.build(host_state)
            priors = [retr.retrieve(idx_prior, q.question, args.prior_k, tok, cfg.context_budget_tokens // 2) for q in qs]

            def evaluate(sel_turns, policy, budget, seed=0, sel_facts=None):
                if sel_facts is not None:
                    extra = "\n".join(fact_lines[i] for i in sel_facts)
                    added = sum(fact_tok[i] for i in sel_facts)
                else:
                    extra = "\n".join(turn_lines[i] for i in sorted(sel_turns))
                    added = sum(turn_tok[i] for i in sel_turns)
                block = f"{plus_block}\n{extra}".strip() if policy != "raw" else "\n".join(turn_lines)
                ctx = [f"{pr}\n{block}".strip() for pr in priors]
                preds = engine.generate(READER_SYS, [reader_prompt(c, q.question) for c, q in zip(ctx, qs)], max_tokens=32)
                cov = None
                if lost_facts and policy in ("none", "coverage", "random", "oracle", "facts", "facts_all"):
                    cov = float(np.mean(check_entailment(engine, block, lost_facts))) if policy != "none" else 0.0
                for q, p in zip(qs, preds):
                    rows.append({"conv_id": cid, "session": t, "policy": policy, "budget": budget, "seed": seed,
                                 "category": q.category, "question": q.question, "gold": q.answer, "pred": p,
                                 "f1": token_f1(p, q.answer), "lenient": lenient(p, q.answer),
                                 "added_tokens": added, "raw_tokens": raw_tokens, "plus_tokens": ntok(plus_block),
                                 "n_facts": n_facts, "n_lost": n_lost, "lost_recovered": cov})

            def fill(order, budget):
                sel, used, cap_ = [], 0, budget * raw_tokens
                for i in order:
                    if used + turn_tok[i] <= cap_: sel.append(i); used += turn_tok[i]
                return sel

            evaluate([], "none", 0.0)
            evaluate([], "raw", 1.0)
            ev_turns = sorted({k-1 for q in qs for k in q.turns_in(t) if 0 <= k-1 < len(raw_units)})
            cov_order = sorted(range(len(raw_units)), key=lambda i: -(lost_by_turn[i] / max(1, turn_tok[i])))
            cov_order = [i for i in cov_order if lost_by_turn[i] > 0] + [i for i in cov_order if lost_by_turn[i] == 0]
            def fill_facts(budget):
                sel, used, cap_ = [], 0, budget * raw_tokens
                for i in range(len(fact_lines)):
                    if used + fact_tok[i] <= cap_: sel.append(i); used += fact_tok[i]
                return sel
            if fact_lines:
                evaluate([], "facts_all", 1.0, sel_facts=list(range(len(fact_lines))))
            for b in budgets:
                if fact_lines: evaluate([], "facts", b, sel_facts=fill_facts(b))
                evaluate(fill(cov_order, b), "coverage", b)
                evaluate(fill(ev_turns + [i for i in range(len(raw_units)) if i not in ev_turns], b), "oracle", b)
                for sd in range(args.seeds):
                    rnd = list(range(len(raw_units))); random.Random(sd * 1000 + t).shuffle(rnd)
                    evaluate(fill(rnd, b), "random", b, seed=sd)

            for u in plus_units: host_state.add(u, t, "host")
        pd.DataFrame(rows).to_csv(f"{args.out}/repair_items.csv", index=False)
        print(f"[{cid}] done, elapsed {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    g = df.groupby(["policy", "budget"]).agg(f1=("f1", "mean"), lenient=("lenient", "mean"),
                                             added_frac=("added_tokens", lambda x: x.mean()),
                                             lost_recovered=("lost_recovered", "mean"), n=("f1", "size")).reset_index()
    tot = df.groupby(["policy", "budget"]).apply(lambda d: d.added_tokens.sum() / d.raw_tokens.sum()).rename("added_frac_tok")
    g = g.merge(tot.reset_index(), on=["policy", "budget"])
    g.to_csv(f"{args.out}/repair.csv", index=False)
    order = {"none": 0, "random": 1, "coverage": 2, "facts": 3, "oracle": 4, "facts_all": 5, "raw": 6}
    g["o"] = g.policy.map(order); g = g.sort_values(["budget", "o"]).drop(columns="o")
    print("\n=== repair results (F1 / lenient acc vs added token fraction) ===")
    print(g[["policy", "budget", "added_frac_tok", "f1", "lenient", "lost_recovered", "n"]].round(3).to_string(index=False))
    base = g[g.policy == "none"].f1.iloc[0]; top = g[g.policy == "raw"].f1.iloc[0]
    print(f"\nrecovery = (f1 - none) / (raw - none);  none={base:.3f} raw={top:.3f}")
    def rec(p, b):
        d = g[(g.policy == p) & (g.budget == b)]
        return (d.f1.iloc[0] - base) / (top - base + 1e-9) if len(d) else float("nan")
    for b in budgets:
        print(f"  budget {b:.2f}:  random={rec('random',b):+.2f}  coverage(turns)={rec('coverage',b):+.2f}  "
              f"facts={rec('facts',b):+.2f}  oracle={rec('oracle',b):+.2f}")
    fa_ = g[g.policy == "facts_all"]
    if len(fa_):
        print(f"  facts_all (no budget): added={fa_.added_frac_tok.iloc[0]:.3f} of raw tokens  recovery={rec('facts_all',1.0):+.2f}  f1={fa_.f1.iloc[0]:.3f}")

if __name__ == "__main__":
    main()
