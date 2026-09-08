"""Combined system (C2): adaptive lambda + sentence residual + certified read expansion, with ablations.

WRITE (per session, sequential):
  adaptive : compress at caps [3,6,12,25]; extract atomic facts from raw; entail against each
             candidate; pick the most aggressive cap with fact loss <= eps; if none, keep raw (defer).
  residual : lost facts -> sentences (max content overlap) -> attach up to `res_frac` of raw session
             tokens, ranked by lost-facts-per-token.
  provenance: every memory unit (compressed fact or residual sentence) points to its raw turn.
  r_t      : fact loss of the final session memory (after residual).
READ (per question):
  retrieve top-K memory units (hybrid); expand their raw turns in relevance order until residual
  risk sum(rel * r_t) <= tau * initial (certified stop). Also: compressed_only, uniform-B, raw RAG
  at equal tokens, full raw.
CONFIGS (write x read):
  full         : adaptive + residual + certified expansion
  no_adaptive  : cap=fixed_cap + residual + expansion
  no_residual  : adaptive + expansion
  no_expansion : adaptive + residual, read = units only
Outputs runs/<out>/items.csv, summary.csv, write_stats.csv
"""
import argparse, os, sys, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from tqdm import tqdm

from certmem.config import Config
from certmem.locomo import load
from certmem.llm import Engine
from certmem.hosts.base import MemoryState
from certmem.hosts.mem0_adapter import get_host
from certmem.facts import extract_facts, check_entailment
from run_read_v9 import Index, READER_SYS, reader_prompt, f1n, lenient, content
from run_recover_v7 import split_sentences

def build_memory(engine, host_cls_kw, sessions, facts, tok, ntok, *, adaptive, eps, caps, fixed_cap, residual, res_frac):
    """Returns units(list of dict text/session/turns/tok/kind), raw_turns, r_sess, stats."""
    units, raw_turns, r_sess, stats = [], [], {}, []
    host_state = MemoryState()
    hosts = {c: get_host("mem0", engine, cap=c) for c in (caps if adaptive else [fixed_cap])}
    for s in sessions:
        t = s.num
        turn_lines = []
        for ti, turn in enumerate(s.turns):
            line = f"[session {t}] ({s.date}) {turn.speaker}: {turn.text}"
            raw_turns.append({"text": line, "session": t, "turn": ti + 1, "tok": ntok(line)}); turn_lines.append(line)
        raw_tok = sum(ntok(x) for x in turn_lines)
        fs = facts[t-1]; fact_txt = [f["fact"] for f in fs]

        # ---- choose compression
        chosen, chosen_cap, chosen_risk, deferred = None, None, None, 0
        cand_caps = sorted(hosts) if adaptive else [fixed_cap]
        for c in cand_caps:                      # ascending cap = most aggressive first
            pu = hosts[c].compress(s, host_state)
            blk = "\n".join(f"[session {t}] {u}" for u in pu)
            ent = check_entailment(engine, blk, fact_txt) if fact_txt else []
            risk = 1 - float(np.mean(ent)) if ent else 0.0
            if (not adaptive) or risk <= eps:
                chosen, chosen_cap, chosen_risk, chosen_ent = pu, c, risk, ent; break
        if chosen is None:                       # defer: keep raw turns as memory
            deferred = 1; chosen_cap = 0; chosen_risk = 0.0
            for ti, turn in enumerate(s.turns):
                units.append({"text": turn_lines[ti], "session": t, "turns": [ti + 1], "tok": ntok(turn_lines[ti]), "kind": "raw"})
                host_state.add(turn_lines[ti], t, "raw")
            r_sess[t] = 0.0
            stats.append({"session": t, "cap": 0, "risk_pre": None, "risk_post": 0.0, "deferred": 1, "mem_tok": raw_tok, "raw_tok": raw_tok, "res_tok": 0})
            continue

        # ---- compressed units with provenance
        for u in chosen:
            uc = content(u)
            sims = sorted(((len(uc & content(turn.text)) / max(1, len(uc)), ti + 1) for ti, turn in enumerate(s.turns)), reverse=True)
            prov = [ti for sc, ti in sims[:2] if sc > 0] or [sims[0][1]]
            units.append({"text": f"[session {t}] {u}", "session": t, "turns": prov, "tok": ntok(u) + 4, "kind": "fact"})
            host_state.add(u, t, "host")
        mem_tok = sum(ntok(u) + 4 for u in chosen); res_tok = 0; risk_post = chosen_risk

        # ---- residual sentences from lost facts
        if residual and fact_txt:
            sents = []
            for ti, turn in enumerate(s.turns):
                for sent in split_sentences(turn.text):
                    line = f"[session {t}] ({s.date}) {turn.speaker}: {sent}"
                    sents.append({"line": line, "turn": ti + 1, "tok": ntok(line), "sent": sent, "n_lost": 0})
            lost = []
            for f, e in zip(fs, chosen_ent):
                if e: continue
                cands = [x for x in sents if f["turn_start"] <= x["turn"] <= f["turn_end"]]
                if not cands: continue
                fc = content(f["fact"]); best = max(cands, key=lambda x: len(fc & content(x["sent"])))
                best["n_lost"] += 1; lost.append(f["fact"])
            order = sorted([x for x in sents if x["n_lost"] > 0], key=lambda x: -x["n_lost"] / max(1, x["tok"]))
            budget = res_frac * raw_tok; sel = []
            for x in order:
                if res_tok + x["tok"] <= budget: sel.append(x); res_tok += x["tok"]
            for x in sel:
                units.append({"text": x["line"], "session": t, "turns": [x["turn"]], "tok": x["tok"], "kind": "residual"})
                host_state.add(x["line"], t, "residual")
            if lost and sel:
                blk2 = "\n".join(f"[session {t}] {u}" for u in chosen) + "\n" + "\n".join(x["line"] for x in sel)
                ent2 = check_entailment(engine, blk2, lost)
                risk_post = (len(fact_txt) - sum(chosen_ent) - sum(ent2)) / len(fact_txt)
        r_sess[t] = risk_post
        stats.append({"session": t, "cap": chosen_cap, "risk_pre": chosen_risk, "risk_post": risk_post, "deferred": 0,
                      "mem_tok": mem_tok + res_tok, "raw_tok": raw_tok, "res_tok": res_tok})
    return units, raw_turns, r_sess, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_convs", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.4); ap.add_argument("--caps", default="3,6,12,25")
    ap.add_argument("--fixed_cap", type=int, default=10); ap.add_argument("--res_frac", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--configs", default="full,no_adaptive,no_residual")
    ap.add_argument("--out", default="runs/combined")
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")]; os.makedirs(args.out, exist_ok=True)
    cfg = Config(); convs = load(cfg.locomo_path)
    if args.limit_convs: convs = dict(list(convs.items())[:args.limit_convs])
    engine = Engine(cfg); tok = engine.tok; ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(cfg.embed_model, device="cuda")
    rows, wstats, t0 = [], [], time.time()
    CONF = {"full": dict(adaptive=True, residual=True), "no_adaptive": dict(adaptive=False, residual=True),
            "no_residual": dict(adaptive=True, residual=False)}

    for cid, conv in convs.items():
        sessions, qas = conv["sessions"], conv["qa"]
        facts = extract_facts(engine, sessions, win=3)
        for cname in args.configs.split(","):
            cf = CONF[cname]
            units, raw_turns, r_sess, stats = build_memory(engine, None, sessions, facts, tok, ntok, adaptive=cf["adaptive"], eps=args.eps,
                                                           caps=caps, fixed_cap=args.fixed_cap, residual=cf["residual"], res_frac=args.res_frac)
            for st in stats: wstats.append({"conv_id": cid, "config": cname, **st})
            mem_total = sum(u["tok"] for u in units); raw_total = sum(r["tok"] for r in raw_turns)
            print(f"[{cid}][{cname}] units={len(units)} mem/raw={mem_total/raw_total:.2f} defer={np.mean([s['deferred'] for s in stats]):.2f} "
                  f"r_t mean={np.mean(list(r_sess.values())):.2f}", flush=True)
            uidx = Index(emb, [u["text"] for u in units]); ridx = Index(emb, [r["text"] for r in raw_turns])
            tl = {(r["session"], r["turn"]): r for r in raw_turns}
            prompts, meta = [], []
            def add(policy, ctx, q, read_tok, n_exp):
                prompts.append(reader_prompt(ctx, q.question))
                meta.append({"conv_id": cid, "config": cname, "policy": policy, "category": q.category, "question": q.question,
                             "gold": q.answer, "read_tokens": read_tok, "n_expanded": n_exp, "n_evidence_sessions": len(q.sessions()),
                             "mem_ratio": mem_total / raw_total})
            for q in qas:
                hits, rel = uidx.search(q.question, args.k)
                base_ctx = "\n".join(units[i]["text"] for i in hits); base_tok = sum(units[i]["tok"] for i in hits)
                add("units_only", base_ctx, q, base_tok, 0)
                cand = {}
                for i, rl in zip(hits, rel):
                    for tt in units[i]["turns"]:
                        key = (units[i]["session"], tt)
                        if key in tl:
                            c = cand.setdefault(key, {"rel": 0.0, "r": r_sess.get(key[0], 0.0), "cost": tl[key]["tok"]}); c["rel"] = max(c["rel"], rl)
                keys = sorted(cand, key=lambda k: -cand[k]["rel"] / cand[k]["cost"])
                def build(sel): return f"{base_ctx}\n" + "\n".join(tl[k]["text"] for k in sel), sum(cand[k]["cost"] for k in sel)
                R0 = sum(cand[k]["rel"] * cand[k]["r"] for k in keys) + 1e-9
                sel, R = [], R0
                for k in keys:
                    if R <= args.tau * R0: break
                    sel.append(k); R -= cand[k]["rel"] * cand[k]["r"]
                ctx, et = build(sel); add("certified", ctx, q, base_tok + et, len(sel))
                sel, used = [], 0
                for k in keys:
                    if used + cand[k]["cost"] <= args.budget: sel.append(k); used += cand[k]["cost"]
                ctx, et = build(sel); add("uniform", ctx, q, base_tok + et, len(sel))
                if cname == "full":
                    rh, _ = ridx.search(q.question, 40); sel_r, used = [], 0
                    for i in rh:
                        if used + raw_turns[i]["tok"] <= base_tok + args.budget: sel_r.append(i); used += raw_turns[i]["tok"]
                    add("raw_rag", "\n".join(raw_turns[i]["text"] for i in sel_r), q, used, len(sel_r))
                    fr = "\n".join(r["text"] for r in raw_turns); add("full_raw", fr, q, ntok(fr), len(raw_turns))
            preds = engine.generate(READER_SYS, prompts, max_tokens=32)
            for m, p in zip(meta, preds): rows.append({**m, "pred": p, "f1": f1n(p, m["gold"]), "lenient": lenient(p, m["gold"])})
            pd.DataFrame(rows).to_csv(f"{args.out}/items.csv", index=False); pd.DataFrame(wstats).to_csv(f"{args.out}/write_stats.csv", index=False)
        print(f"[{cid}] done {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    g = df.groupby(["config", "policy"]).agg(f1=("f1", "mean"), lenient=("lenient", "mean"), read_tokens=("read_tokens", "mean"),
                                             mem_ratio=("mem_ratio", "mean"), n=("f1", "size")).reset_index()
    g.to_csv(f"{args.out}/summary.csv", index=False)
    print("\n=== combined system & ablations (normalized F1) ===")
    print(g.round(3).to_string(index=False))
    print("\nrows of interest:")
    for cfg_, pol, label in [("full", "certified", "OURS (adaptive+residual+certified read)"), ("no_adaptive", "certified", "- adaptive"),
                              ("no_residual", "certified", "- residual"), ("full", "units_only", "- expansion"),
                              ("full", "raw_rag", "raw RAG (equal tokens)"), ("full", "full_raw", "full raw")]:
        d = g[(g.config == cfg_) & (g.policy == pol)]
        if len(d): print(f"  {label:42s} F1={d.f1.iloc[0]:.3f}  read_tok={d.read_tokens.iloc[0]:.0f}  mem/raw={d.mem_ratio.iloc[0]:.2f}")
    print("\nby category (full/certified vs raw_rag):")
    d2 = df[((df.config == "full") & (df.policy.isin(["certified", "raw_rag", "units_only", "full_raw"])))]
    print(d2.groupby(["category", "policy"]).f1.mean().unstack().round(3))
    d3 = d2[d2.n_evidence_sessions > 1]
    if len(d3): print("\nmulti-session:"); print(d3.groupby("policy").f1.mean().round(3).to_string())

if __name__ == "__main__":
    main()
