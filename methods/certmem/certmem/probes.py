"""Turn-dense diagnostic probes (v5).

Every window of `win` consecutive turns gets `per_win` probes, so coverage is
uniform across the session instead of 10 probes for ~18 turns. Each probe records
the turn index its answer comes from, so benchmark QA (which carries evidence
turn ids) can be matched by TURN, not by question embedding.

Answers are the resolved fact (e.g. an absolute date computed from the session
date), not a verbatim span. Grounding: the answer or its key tokens must appear
in the window, OR the family is 'date' (dates may be resolved).
"""
import json, re

FAMILIES = {
    "date":      "a question whose answer is a date/time/duration; RESOLVE relative dates using the session date (e.g. 'yesterday' -> the actual date)",
    "negation":  "a question whose answer depends on something a speaker said they did NOT do, like, want, or have",
    "entity":    "a question whose answer is a specific person, place, organization, title, or object",
    "condition": "a question whose answer is a condition, exception, plan, or reason ('only if', 'unless', 'because', 'planning to')",
    "number":    "a question whose answer is a number, count, price, or quantity",
    "event":     "a question about what happened, what someone did, or what someone said they experienced",
}

SYS = ("You write diagnostic questions that test whether a memory system retained the information in a few "
       "conversation turns. Each question must be answerable from the given turns, must name the speaker, and must "
       "have a SHORT answer (1-8 words). Prefer questions about specific details a summary might drop. "
       "Return strict JSON: a list of objects with keys question, answer, family. No prose, no markdown.")

def _norm(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _grounded(answer, window_text, family):
    a, w = _norm(answer), _norm(window_text)
    if not a: return False
    if a in w: return True
    toks = [t for t in a.split() if len(t) > 2 and t not in {"the","and","for","with","that","this","she","him","her","his","they"}]
    if not toks: return False
    hit = sum(t in w for t in toks) / len(toks)
    if family == "date":
        # resolved dates won't appear verbatim; accept if the window has any temporal expression
        temporal = re.search(r"\b(yesterday|today|tomorrow|last|next|ago|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                             r"january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}(st|nd|rd|th)?|\d{4})\b", w)
        return hit >= 0.34 or temporal is not None
    return hit >= 0.6

def build_prompt(session, turns, per_win):
    fams = "\n".join(f"- {k}: {v}" for k, v in FAMILIES.items())
    txt = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
    return (f"Session date: {session.date}\n\nTurns:\n{txt}\n\n"
            f"Write {per_win} questions total, each from a DIFFERENT family where possible:\n{fams}")

def parse(out, window_text):
    m = re.search(r"\[.*\]", out, re.S)
    if not m: return []
    try: items = json.loads(m.group(0))
    except Exception: return []
    res = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict): continue
        q, a, f = str(it.get("question","")).strip(), str(it.get("answer","")).strip(), it.get("family","")
        if not q or not a or f not in FAMILIES or len(a.split()) > 10: continue
        res.append({"question": q, "answer": a, "family": f, "grounded": _grounded(a, window_text, f)})
    return res

def generate_probes(engine, sessions, per_session=None, win=3, per_win=3):
    """Returns list (per session) of probe dicts with turn_start/turn_end (1-based, inclusive)."""
    prompts, meta = [], []
    for s in sessions:
        n = len(s.turns)
        for i in range(0, n, win):
            turns = s.turns[i:i+win]
            prompts.append(build_prompt(s, turns, per_win))
            meta.append((s, i+1, min(n, i+win), "\n".join(t.text for t in turns)))
    outs = engine.generate(SYS, prompts, max_tokens=600)
    per = {s.num: [] for s in sessions}
    seen = set()
    for (s, a, b, wtext), o in zip(meta, outs):
        for p in parse(o, wtext):
            if not p["grounded"]: continue
            k = (s.num, _norm(p["question"]))
            if k in seen: continue
            seen.add(k)
            p.update({"conv_id": s.conv_id, "session": s.num, "turn_start": a, "turn_end": b})
            per[s.num].append(p)
    return [per[s.num] for s in sessions]
