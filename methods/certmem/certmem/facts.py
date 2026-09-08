"""Exhaustive fact-coverage sensor (v6).

Instead of sampling probe questions, enumerate every atomic fact in each window
of raw turns, then ask whether each fact is entailed by the compressed block.
Coverage(turn window) = fraction of its facts entailed.  Risk = 1 - coverage.

Theorem-1 intuition: sampled probes are independent of which fact a downstream
question needs; exhaustive enumeration is not. This module is the exhaustive arm.
"""
import json, re

EXTRACT_SYS = ("You extract atomic facts from a few conversation turns. An atomic fact is ONE self-contained "
               "statement with the speaker's name, e.g. 'Caroline attended an LGBTQ support group on 7 May 2023', "
               "'Melanie does not like coffee', 'Gina got her tattoo 3 years ago'. Resolve relative dates using the "
               "session date. Include: events, dates, numbers, names, preferences, negations, plans, reasons. "
               "Return strict JSON: a list of strings. No prose, no markdown.")

ENTAIL_SYS = ("You check whether a fact is supported by memory notes. Answer with exactly one word: "
              "yes (the fact is stated or clearly implied by the notes), or no (it is missing, contradicted, "
              "or the notes are too vague to support it).")

def _norm(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def build_extract_prompt(session, turns):
    txt = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
    return f"Session date: {session.date}\n\nTurns:\n{txt}\n\nList every atomic fact:"

def parse_facts(out):
    m = re.search(r"\[.*\]", out, re.S)
    if not m: return []
    try: items = json.loads(m.group(0))
    except Exception: return []
    facts = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict): it = it.get("fact") or it.get("text") or ""
        s = str(it).strip()
        if 3 <= len(s.split()) <= 40:
            facts.append(s)
    # dedupe
    seen, out_ = set(), []
    for f in facts:
        k = _norm(f)
        if k in seen: continue
        seen.add(k); out_.append(f)
    return out_

def extract_facts(engine, sessions, win=3, max_per_window=12):
    """Returns per-session list of dicts: fact, turn_start, turn_end."""
    prompts, meta = [], []
    for s in sessions:
        n = len(s.turns)
        for i in range(0, n, win):
            prompts.append(build_extract_prompt(s, s.turns[i:i+win]))
            meta.append((s, i+1, min(n, i+win)))
    outs = engine.generate(EXTRACT_SYS, prompts, max_tokens=700)
    per = {s.num: [] for s in sessions}
    for (s, a, b), o in zip(meta, outs):
        for f in parse_facts(o)[:max_per_window]:
            per[s.num].append({"conv_id": s.conv_id, "session": s.num, "turn_start": a, "turn_end": b, "fact": f})
    return [per[s.num] for s in sessions]

def entail_prompt(block, fact):
    return f"Memory notes:\n{block}\n\nFact: {fact}\n\nIs the fact supported by the notes? Answer yes or no:"

def check_entailment(engine, block, facts):
    """Returns list of 1/0 for each fact against one memory block."""
    if not facts: return []
    outs = engine.generate(ENTAIL_SYS, [entail_prompt(block, f) for f in facts], max_tokens=4)
    return [1 if o.strip().lower().startswith("y") else 0 for o in outs]
