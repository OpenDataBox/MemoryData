"""Experiment 4.2-A (v3): does probe risk predict benchmark regression?

Design fixes vs v1:
  (1) Retrieval bias removed. The session-t block is placed in the context IN FULL:
      raw for M-, compressed for M+. Prior sessions (< t) come from the host's own
      compressed memory and are retrieved IDENTICALLY for both arms. The only
      difference between arms is what compression kept vs dropped in session t.
  (2) Same instrument for probes and QA. Both are answered by GENERATION and scored
      by token-F1 against the reference (probe: grounded span; QA: gold answer).
      Log-prob of a verbatim span penalises paraphrase, not information loss.
  (3) Signed deltas kept (f1_minus - f1_plus); clipping is done in analysis.
  (4) --cap controls host aggressiveness (lambda). Lower cap = more compression.

Outputs (out_dir): probes.jsonl, qa.jsonl, cost.csv, sessions.csv
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from tqdm import tqdm

from certmem.config import Config
from certmem.locomo import load, token_f1
from certmem.llm import Engine
from certmem.hosts.base import MemoryState, Retriever
from certmem.hosts.mem0_adapter import get_host
from certmem.probes import generate_probes
from certmem.facts import extract_facts, check_entailment

READER_SYS = ("You answer questions about a long conversation using only the memory notes provided. "
              "Reply with the shortest exact answer (a name, date, number, or short phrase). No explanation. "
              "If the notes do not contain the answer, reply: unknown")

def reader_prompt(context, question):
    return f"Memory notes:\n{context}\n\nQuestion: {question}\nAnswer:"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="mem0", choices=["raw", "mem0", "lightmem"])
    ap.add_argument("--cap", type=int, default=None, help="compression knob: facts cap (mem0) / bullets (lightmem)")
    ap.add_argument("--locomo", default="data/locomo10.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit_convs", type=int, default=0)
    ap.add_argument("--probes_per_session", type=int, default=10)
    ap.add_argument("--prior_k", type=int, default=6, help="retrieved prior-memory units (identical both arms)")
    ap.add_argument("--win", type=int, default=3, help="turns per probe window")
    ap.add_argument("--per_win", type=int, default=3, help="probes per window")
    ap.add_argument("--sensors", default="facts,probes", help="comma list: facts, probes")
    ap.add_argument("--fact_win", type=int, default=3)
    args = ap.parse_args()

    cfg = Config(host=args.host, locomo_path=args.locomo, probes_per_session=args.probes_per_session)
    tag = f"{args.host}" + (f"_cap{args.cap}" if args.cap else "")
    out_dir = args.out or f"runs/4_2a_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    convs = load(cfg.locomo_path)
    if args.limit_convs:
        convs = dict(list(convs.items())[:args.limit_convs])

    engine = Engine(cfg)
    retr = Retriever(cfg.embed_model)
    host_kw = {}
    if args.cap:
        host_kw = {"cap": args.cap} if args.host == "mem0" else ({"n_bullets": args.cap} if args.host == "lightmem" else {})
    host = get_host(args.host, engine, **host_kw)
    tok = engine.tok

    probe_rows, qa_rows, cost_rows, fact_rows = [], [], [], []
    sensors = set(x.strip() for x in args.sensors.split(','))
    t0 = time.time()

    for cid, conv in convs.items():
        sessions, qas = conv["sessions"], conv["qa"]
        qa_by_session = {}
        for q in qas:
            for s in q.sessions():
                qa_by_session.setdefault(s, []).append(q)

        probes = generate_probes(engine, sessions, win=args.win, per_win=args.per_win) if "probes" in sensors else [[] for _ in sessions]
        facts  = extract_facts(engine, sessions, win=args.fact_win) if "facts" in sensors else [[] for _ in sessions]
        print(f"[{cid}] probes={sum(len(p) for p in probes)}  facts={sum(len(f) for f in facts)}  over {len(sessions)} sessions", flush=True)

        host_state = MemoryState()
        for s in tqdm(sessions, desc=f"conv {cid}"):
            t = s.num
            idx_prior = retr.build(host_state)
            raw_units  = host.raw_units(s)
            plus_units = host.compress(s, host_state)
            raw_block  = "\n".join(f"[session {t}] {u}" for u in raw_units)
            plus_block = "\n".join(f"[session {t}] {u}" for u in plus_units)
            n_raw  = len(tok(raw_block, add_special_tokens=False).input_ids)
            n_plus = len(tok(plus_block, add_special_tokens=False).input_ids)
            cost_rows.append({"conv_id": cid, "session": t, "raw_tokens": n_raw, "plus_tokens": n_plus,
                              "ratio": n_plus / max(1, n_raw), "n_raw_units": len(raw_units), "n_plus_units": len(plus_units)})

            # ---- fact-coverage sensor: entailment of each raw fact by the COMPRESSED block
            fs = facts[t-1]
            if fs:
                ent_plus = check_entailment(engine, plus_block, [f["fact"] for f in fs])
                ent_raw  = check_entailment(engine, raw_block,  [f["fact"] for f in fs])   # sanity: should be ~1
                for f, ep, er in zip(fs, ent_plus, ent_raw):
                    fact_rows.append({**f, "entailed_plus": ep, "entailed_raw": er})

            def prior_for(query):
                return retr.retrieve(idx_prior, query, args.prior_k, tok, cfg.context_budget_tokens // 2)

            # ---- probes: generate + F1 vs grounded span (same instrument as QA)
            ps = probes[t-1]
            if ps:
                priors = [prior_for(p["question"]) for p in ps]
                cm = [f"{pr}\n{raw_block}".strip()  for pr in priors]
                cp = [f"{pr}\n{plus_block}".strip() for pr in priors]
                am = engine.generate(READER_SYS, [reader_prompt(c, p["question"]) for c, p in zip(cm, ps)], max_tokens=24)
                ap_ = engine.generate(READER_SYS, [reader_prompt(c, p["question"]) for c, p in zip(cp, ps)], max_tokens=24)
                for p, a, b in zip(ps, am, ap_):
                    fm, fp = token_f1(a, p["answer"]), token_f1(b, p["answer"])
                    probe_rows.append({**p, "pred_minus": a, "pred_plus": b,
                                       "f1_minus": fm, "f1_plus": fp, "delta": fm - fp})

            # ---- benchmark QA whose evidence includes session t
            qs = qa_by_session.get(t, [])
            if qs:
                priors = [prior_for(q.question) for q in qs]
                cm = [f"{pr}\n{raw_block}".strip()  for pr in priors]
                cp = [f"{pr}\n{plus_block}".strip() for pr in priors]
                am = engine.generate(READER_SYS, [reader_prompt(c, q.question) for c, q in zip(cm, qs)], max_tokens=32)
                ap_ = engine.generate(READER_SYS, [reader_prompt(c, q.question) for c, q in zip(cp, qs)], max_tokens=32)
                for q, a, b in zip(qs, am, ap_):
                    fm, fp = token_f1(a, q.answer), token_f1(b, q.answer)
                    qa_rows.append({"conv_id": cid, "session": t, "category": q.category,
                                    "evidence_turns": q.turns_in(t), "n_evidence_sessions": len(q.sessions()),
                                    "question": q.question, "gold": q.answer,
                                    "pred_minus": a, "pred_plus": b,
                                    "f1_minus": fm, "f1_plus": fp, "delta": fm - fp})

            for u in plus_units:
                host_state.add(u, t, "host")

        pd.DataFrame(probe_rows).to_json(f"{out_dir}/probes.jsonl", orient="records", lines=True)
        pd.DataFrame(qa_rows).to_json(f"{out_dir}/qa.jsonl", orient="records", lines=True)
        pd.DataFrame(cost_rows).to_csv(f"{out_dir}/cost.csv", index=False)
        if fact_rows: pd.DataFrame(fact_rows).to_json(f"{out_dir}/facts.jsonl", orient="records", lines=True)
        print(f"[{cid}] done, elapsed {time.time()-t0:.0f}s", flush=True)

    # ---- per-session aggregation
    pr = pd.DataFrame(probe_rows); qa = pd.DataFrame(qa_rows); cost = pd.DataFrame(cost_rows); fa = pd.DataFrame(fact_rows)
    w = cfg.family_weights
    if len(pr):
        def risk_w(g):
            fam = g.groupby("family")["delta"].mean().clip(lower=0)
            return sum(w.get(f, 0) * v for f, v in fam.items()) / max(1e-9, sum(w.get(f, 0) for f in fam.index))
        R = pr.groupby(["conv_id", "session"]).apply(risk_w).rename("probe_risk")
        Ru = pr.groupby(["conv_id", "session"])["delta"].mean().rename("probe_delta_signed")
        Rn = pr.groupby(["conv_id", "session"]).size().rename("n_probes")
    else:
        idx = pd.MultiIndex.from_tuples([], names=["conv_id", "session"])
        R = pd.Series(dtype=float, index=idx, name="probe_risk"); Ru = R.rename("probe_delta_signed"); Rn = R.rename("n_probes")
    if len(fa):
        F = fa.groupby(["conv_id", "session"]).agg(fact_coverage=("entailed_plus", "mean"),
                                                    fact_coverage_raw=("entailed_raw", "mean"), n_facts=("fact", "size"))
        F["fact_risk"] = 1 - F["fact_coverage"]
    else:
        F = pd.DataFrame(columns=["fact_coverage", "fact_coverage_raw", "n_facts", "fact_risk"])
    Q = qa.groupby(["conv_id", "session"]).agg(bench_delta_signed=("delta", "mean"),
                                                bench_f1_minus=("f1_minus", "mean"),
                                                bench_f1_plus=("f1_plus", "mean"),
                                                n_qa=("delta", "size"))
    Q["bench_regression"] = Q["bench_delta_signed"].clip(lower=0)
    C = cost.set_index(["conv_id", "session"])[["ratio", "raw_tokens", "plus_tokens"]]
    df = pd.concat([R, Ru, Rn, F, Q, C], axis=1).dropna(subset=["bench_delta_signed"]).reset_index()
    df.to_csv(f"{out_dir}/sessions.csv", index=False)
    cols = [c for c in ["probe_delta_signed", "fact_risk", "fact_coverage_raw", "bench_delta_signed", "bench_f1_minus", "bench_f1_plus", "ratio"] if c in df.columns]
    print(df[cols].describe().round(3))
    print(f"wrote {out_dir}/sessions.csv ({len(df)} sessions)  compression ratio mean={cost.ratio.mean():.2f}")

if __name__ == "__main__":
    main()
