"""System v13 = v11 + (a) top-K 25, (b) SENTENCE-level expansion: an expanded turn contributes only its
sentences most similar to the search string, so the same token budget covers 2-3x more turns,
(c) --think: Qwen3 reasoning mode for the reader (applied to every policy; think tokens counted).

v11 = v10 + two components adopted from TriMem (Sun et al., 2605.19952), cited:
  * query reformulation: one LLM call turns the question into a structured search string
    (entities, time expressions, paraphrase) used for BOTH indexes; the reader still sees the original question.
  * entity profiles: per-speaker profile synthesized from the session's compressed facts, updated each session
    (1 call per speaker per session); profiles of speakers named in the query (else all) are prepended to the read context.
  flags --no_reform / --no_profile for ablation.

v10 notes: union retrieval (compressed units + raw turns) with certified expansion.

v10 changes vs run_combined:
  * READ retrieves over BOTH the memory-unit index and the raw-turn index (RRF union), so the
    candidate set contains everything raw RAG would see plus consolidated cross-session facts.
  * Certified stop never expands fewer than --min_exp turns (fixes temporal under-expansion).
  * --dataset locomo|longmemeval ; --eps default 0.5.

Original combined-system notes: adaptive lambda + sentence residual + certified read expansion, with ablations.

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
from certmem.locomo import load as load_locomo
from certmem.longmemeval import load as load_lme
from certmem.llm import Engine
from certmem.hosts.base import MemoryState
from certmem.hosts.mem0_adapter import get_host
from certmem.facts import extract_facts, check_entailment
from run_read_v9 import Index, READER_SYS, reader_prompt, f1n, lenient, content
from run_recover_v7 import split_sentences

REFORM_SYS = ("You rewrite a question about a long personal conversation into a search string for a memory index. "
              "Output ONE line containing: the key people, places, objects, dates or time expressions, and a short paraphrase "
              "of what is being asked. No explanation.")
PROFILE_SYS = ("You maintain a concise profile of a person from conversation facts. Merge the new facts into the existing profile. "
               "Keep: identity, relationships, work/study, interests, notable events with dates, plans, preferences and dislikes. "
               "Resolve updates (newer facts override older). At most 80 words. Plain text.")

def build_memory(engine, host_cls_kw, sessions, facts, tok, ntok, *, adaptive, eps, caps, fixed_cap, residual, res_frac, profile=True):
    """Returns units(list of dict text/session/turns/tok/kind), raw_turns, r_sess, stats."""
    units, raw_turns, r_sess, stats = [], [], {}, []
    profiles = {}
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
        if profile:
            speakers = sorted({turn.speaker for turn in s.turns})
            prompts_p = [f"Person: {sp}\nExisting profile:\n{profiles.get(sp, '(none)')}\n\nNew facts (session {t}, {s.date}):\n" +
                         "\n".join(u for u in chosen if sp.lower() in u.lower()) + "\n\nUpdated profile:" for sp in speakers]
            outs = engine.generate(PROFILE_SYS, prompts_p, max_tokens=120)
            for sp, o in zip(speakers, outs):
                if o.strip(): profiles[sp] = o.strip()

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
    return units, raw_turns, r_sess, stats, profiles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_convs", type=int, default=0)
    ap.add_argument("--dataset", default="locomo", choices=["locomo", "longmemeval"])
    ap.add_argument("--lme_path", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--min_exp", type=int, default=2)
    ap.add_argument("--no_reform", action="store_true")
    ap.add_argument("--no_profile", action="store_true")
    ap.add_argument("--no_raw", action="store_true", help="raw turns are deleted after write (privacy regime): no expansion, RAG impossible")
    ap.add_argument("--eps", type=float, default=0.5); ap.add_argument("--caps", default="3,6,12,25")
    ap.add_argument("--fixed_cap", type=int, default=10); ap.add_argument("--res_frac", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--sent_frac", type=float, default=0.5, help="fraction of a turn's sentences kept when expanding (by similarity)"); ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--budgets", default="400,1600,4000,8000,16000", help="read expansion budgets (tokens); RAG uses base_tok+B")
    ap.add_argument("--configs", default="full,no_adaptive,no_residual")
    ap.add_argument("--out", default="runs/system_v13")
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")]; budgets = [int(b) for b in args.budgets.split(",")]; os.makedirs(args.out, exist_ok=True)
    cfg = Config()
    if args.dataset == "locomo":
        convs = load_locomo(cfg.locomo_path)
        if args.limit_convs: convs = dict(list(convs.items())[:args.limit_convs])
    else:
        convs = load_lme(args.lme_path, limit=args.limit_convs)
    engine = Engine(cfg); tok = engine.tok; ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(cfg.embed_model, device="cuda")
    rows, wstats, t0 = [], [], time.time()
    CONF = {"full": dict(adaptive=True, residual=True), "no_adaptive": dict(adaptive=False, residual=True),
            "no_residual": dict(adaptive=True, residual=False), "plain": dict(adaptive=False, residual=False)}

    for cid, conv in convs.items():
        sessions, qas = conv["sessions"], conv["qa"]
        facts = extract_facts(engine, sessions, win=3)
        for cname in args.configs.split(","):
            cf = CONF[cname]
            units, raw_turns, r_sess, stats, profiles = build_memory(engine, None, sessions, facts, tok, ntok, adaptive=cf["adaptive"], eps=args.eps,
                                                           caps=caps, fixed_cap=args.fixed_cap, residual=cf["residual"], res_frac=args.res_frac, profile=not args.no_profile)
            for st in stats: wstats.append({"conv_id": cid, "config": cname, **st})
            mem_total = sum(u["tok"] for u in units); raw_total = sum(r["tok"] for r in raw_turns)
            print(f"[{cid}][{cname}] units={len(units)} mem/raw={mem_total/raw_total:.2f} defer={np.mean([s['deferred'] for s in stats]):.2f} "
                  f"r_t mean={np.mean(list(r_sess.values())):.2f}", flush=True)
            uidx = Index(emb, [u["text"] for u in units]); ridx = Index(emb, [r["text"] for r in raw_turns])
            tl = {(r["session"], r["turn"]): r for r in raw_turns}
            # sentence split of every raw turn for budget-efficient expansion
            for r in raw_turns:
                head, _, body = r["text"].partition(": ")
                sents = split_sentences(body) if body else [r["text"]]
                r["sents"] = [f"{head}: {x}" for x in sents] if body else sents
                r["sent_tok"] = [ntok(x) for x in r["sents"]]
                r["sent_emb"] = emb.encode(r["sents"], normalize_embeddings=True) if len(r["sents"]) > 1 else None
            prompts, meta = [], []
            def add(policy, ctx, q, read_tok, n_exp):
                qd = getattr(q, "question_date", "")
                qtext = f"(Today's date: {qd}) {q.question}" if qd else q.question
                prompts.append(reader_prompt(ctx, qtext))
                meta.append({"conv_id": cid, "config": cname, "policy": policy, "category": q.category, "qtype": getattr(q, "qtype", ""), "question": q.question,
                             "gold": q.answer, "read_tokens": read_tok, "n_expanded": n_exp, "n_evidence_sessions": len(q.sessions()),
                             "mem_ratio": mem_total / raw_total})
            if args.no_reform:
                squeries = [q.question for q in qas]
            else:
                squeries = engine.generate(REFORM_SYS, [f"Question: {q.question}\nSearch string:" for q in qas], max_tokens=60)
                squeries = [(sq.strip() + " " + q.question) if sq.strip() else q.question for sq, q in zip(squeries, qas)]
            for q, sq in zip(qas, squeries):
                hits, rel = uidx.search(sq, args.k)
                prof_block = ""
                if profiles:
                    named = [sp for sp in profiles if sp.lower() in q.question.lower()]
                    use = named
                    prof_block = "\n".join(f"[profile] {sp}: {profiles[sp]}" for sp in use)
                base_ctx = (prof_block + "\n" if prof_block else "") + "\n".join(units[i]["text"] for i in hits)
                base_tok = sum(units[i]["tok"] for i in hits) + (ntok(prof_block) if prof_block else 0)
                add("units_only", base_ctx, q, base_tok, 0)
                if args.no_raw:
                    continue
                cand = {}
                for i, rl in zip(hits, rel):
                    for tt in units[i]["turns"]:
                        key = (units[i]["session"], tt)
                        if key in tl:
                            c = cand.setdefault(key, {"rel": 0.0, "r": r_sess.get(key[0], 0.0), "cost": tl[key]["tok"]}); c["rel"] = max(c["rel"], rl)
                # UNION: raw-turn hits join the candidate pool (what raw RAG would see)
                rhits, rrel = ridx.search(sq, args.k)
                for i, rl in zip(rhits, rrel):
                    key = (raw_turns[i]["session"], raw_turns[i]["turn"])
                    c = cand.setdefault(key, {"rel": 0.0, "r": r_sess.get(key[0], 0.0), "cost": tl[key]["tok"]}); c["rel"] = max(c["rel"], rl)
                q_emb = emb.encode([sq], normalize_embeddings=True)[0]
                def turn_snippet(k):
                    r = tl[k]
                    if r["sent_emb"] is None or len(r["sents"]) <= 2: return r["text"], r["tok"]
                    sims = r["sent_emb"] @ q_emb
                    keep = max(1, int(round(len(r["sents"]) * args.sent_frac)))
                    idx = sorted(np.argsort(-sims)[:keep])
                    txt = " ".join(r["sents"][i] for i in idx); return txt, sum(r["sent_tok"][i] for i in idx)
                snip = {k: turn_snippet(k) for k in cand}
                for k in cand: cand[k]["cost"] = max(1, snip[k][1])
                keys = sorted(cand, key=lambda k: -cand[k]["rel"] / cand[k]["cost"])
                def build(sel): return f"{base_ctx}\n" + "\n".join(snip[k][0] for k in sel), sum(cand[k]["cost"] for k in sel)
                R0 = sum(cand[k]["rel"] * cand[k]["r"] for k in keys) + 1e-9
                sel, R = [], R0
                for k in keys:
                    if R <= args.tau * R0 and len(sel) >= args.min_exp: break
                    sel.append(k); R -= cand[k]["rel"] * cand[k]["r"]
                ctx, et = build(sel); add("certified", ctx, q, base_tok + et, len(sel))
                for B in budgets:
                    sel, used = [], 0
                    for k in keys:
                        if used + cand[k]["cost"] <= B: sel.append(k); used += cand[k]["cost"]
                    ctx, et = build(sel); add(f"uniform_{B}", ctx, q, base_tok + et, len(sel))
                    # OURS@B: budget-scaled candidate pool (K grows with B), union of units + raw, fill to B in relevance order.
                    K_B = max(args.k, int(B / 60))
                    bh, brel = uidx.search(sq, K_B); rh2, rrel2 = ridx.search(sq, K_B)
                    cB = {}
                    for i, rl in zip(bh, brel):
                        for tt in units[i]["turns"]:
                            key = (units[i]["session"], tt)
                            if key in tl:
                                c = cB.setdefault(key, {"rel": 0.0, "r": r_sess.get(key[0], 0.0), "cost": tl[key]["tok"]}); c["rel"] = max(c["rel"], rl)
                    for i, rl in zip(rh2, rrel2):
                        key = (raw_turns[i]["session"], raw_turns[i]["turn"])
                        c = cB.setdefault(key, {"rel": 0.0, "r": r_sess.get(key[0], 0.0), "cost": tl[key]["tok"]}); c["rel"] = max(c["rel"], rl)
                    snipB = {k: (snip[k] if k in snip else turn_snippet(k)) for k in cB}
                    for k in cB: cB[k]["cost"] = max(1, snipB[k][1])
                    keysB = sorted(cB, key=lambda k: -cB[k]["rel"] / cB[k]["cost"])
                    selB, usedB = [], 0
                    for k in keysB:
                        if usedB + cB[k]["cost"] <= B: selB.append(k); usedB += cB[k]["cost"]
                    ctxB = f"{base_ctx}\n" + "\n".join(snipB[k][0] for k in selB)
                    add(f"ours_{B}", ctxB, q, base_tok + usedB, len(selB))
                    # certified with a cap at this budget (stop rule OR budget, whichever first) -> budget-matched curve
                    selc, R, used = [], R0, 0
                    for k in keys:
                        if (R <= args.tau * R0 and len(selc) >= args.min_exp) or used + cand[k]["cost"] > B: break
                        selc.append(k); R -= cand[k]["rel"] * cand[k]["r"]; used += cand[k]["cost"]
                    ctx, et = build(selc); add(f"certified_{B}", ctx, q, base_tok + et, len(selc))
                    if cname == "full":
                        rh, _ = ridx.search(sq, 400); sel_r, used = [], 0
                        for i in rh:
                            if used + raw_turns[i]["tok"] <= base_tok + B: sel_r.append(i); used += raw_turns[i]["tok"]
                            if used > 38000: break
                        add(f"raw_rag_{B}", "\n".join(raw_turns[i]["text"] for i in sel_r), q, used, len(sel_r))
                if cname == "full":
                    fr = "\n".join(r["text"] for r in raw_turns)
                    if ntok(fr) < 39000: add("full_raw", fr, q, ntok(fr), len(raw_turns))
            preds = engine.generate(READER_SYS, prompts, max_tokens=32, think=args.think)
            think_toks = getattr(engine, "last_think_tokens", [0]*len(prompts))
            for m, p, tt in zip(meta, preds, think_toks): rows.append({**m, "pred": p, "f1": f1n(p, m["gold"]), "lenient": lenient(p, m["gold"]), "think_tokens": tt})
            pd.DataFrame(rows).to_csv(f"{args.out}/items.csv", index=False); pd.DataFrame(wstats).to_csv(f"{args.out}/write_stats.csv", index=False)
        print(f"[{cid}] done {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    g = df.groupby(["config", "policy"]).agg(f1=("f1", "mean"), lenient=("lenient", "mean"), read_tokens=("read_tokens", "mean"),
                                             mem_ratio=("mem_ratio", "mean"), n=("f1", "size")).reset_index()
    g.to_csv(f"{args.out}/summary.csv", index=False)
    print("\n=== system v13 (k10 + sentence expansion + reform + profiles) & ablations (normalized F1) ===")
    print(g.round(3).to_string(index=False))
    print("\nrows of interest:")
    for cfg_, pol, label in [("full", "certified", "OURS (adaptive+residual+certified read)"), ("no_adaptive", "certified", "- adaptive"),
                              ("no_residual", "certified", "- residual"), ("full", "units_only", "- expansion"),
                              ("full", "full_raw", "full raw")] + [("full", f"raw_rag_{B}", f"raw RAG  B={B}") for B in budgets] \
                             + [("full", f"ours_{B}", f"OURS fill B={B}") for B in budgets]:
        d = g[(g.config == cfg_) & (g.policy == pol)]
        if len(d): print(f"  {label:42s} F1={d.f1.iloc[0]:.3f}  read_tok={d.read_tokens.iloc[0]:.0f}  mem/raw={d.mem_ratio.iloc[0]:.2f}")
    print("\nby category (full/certified vs raw_rag):")
    d2 = df[((df.config == "full") & (df.policy.isin(["certified", "raw_rag_400", "raw_rag_4000", "certified_4000", "units_only", "full_raw"])))]
    print(d2.groupby(["category", "policy"]).f1.mean().unstack().round(3))
    d3 = d2[d2.n_evidence_sessions > 1]
    if len(d3): print("\nmulti-session:"); print(d3.groupby("policy").f1.mean().round(3).to_string())
    print("\nBUDGET SWEEP (full config): read tokens -> F1, RAG vs ours")
    sw = df[(df.config == "full") & (df.policy.str.startswith(("raw_rag_", "certified_", "uniform_", "ours_")))].copy()
    sw["B"] = sw.policy.str.extract(r"_(\d+)$").astype(int); sw["kind"] = sw.policy.str.replace(r"_\d+$", "", regex=True)
    print(sw.groupby(["B", "kind"]).agg(f1=("f1", "mean"), read_tok=("read_tokens", "mean")).unstack().round(3))
    if "qtype" in df and df.qtype.astype(str).str.len().max() > 0:
        print("\nby LongMemEval question type (full config):")
        print(df[df.config == "full"].groupby(["qtype", "policy"]).f1.mean().unstack().round(3))

if __name__ == "__main__":
    main()
