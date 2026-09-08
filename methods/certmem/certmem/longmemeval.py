"""LongMemEval-S loader -> same structures as certmem.locomo (Session, Turn, QA).

Each LongMemEval question has its own haystack of ~40-50 sessions. We treat each
question as one "conversation" (conv_id = question_id) with one QA whose evidence
sessions are `answer_session_ids`. Sessions are user/assistant turns with a date.

File: longmemeval_s_cleaned.json (or longmemeval_s.json). Fields used:
  question_id, question_type, question, answer, question_date,
  haystack_session_ids, haystack_dates, haystack_sessions (list of list of {role, content}),
  answer_session_ids
"""
import json
from certmem.locomo import Turn, Session, QA

TYPE_TO_CAT = {"single-session-user": 4, "single-session-assistant": 4, "single-session-preference": 3,
               "multi-session": 1, "temporal-reasoning": 2, "knowledge-update": 5}
# note: category 5 here = knowledge-update (NOT adversarial as in LoCoMo); loaders keep it.

def load(path, limit=0, types=None):
    data = json.load(open(path))
    convs = {}
    for ex in data:
        qid = ex["question_id"]
        qtype = ex.get("question_type", "")
        if types and qtype not in types: continue
        if qid.endswith("_abs"):          # abstention variants: skip for QA accuracy
            continue
        sids = ex["haystack_session_ids"]; dates = ex.get("haystack_dates", [""] * len(sids))
        sessions = []
        for n, (sid, date, turns) in enumerate(zip(sids, dates, ex["haystack_sessions"]), start=1):
            ts = []
            for i, t in enumerate(turns):
                spk = "User" if t.get("role") == "user" else "Assistant"
                ts.append(Turn(n, i, spk, t.get("content", ""), f"D{n}:{i+1}"))
            if not ts: ts = [Turn(n, 0, "User", "(empty session)", f"D{n}:1")]   # keep numbering stable
            sessions.append(Session(qid, n, date, ts))
        ev_sessions = {sids.index(a) + 1 for a in ex.get("answer_session_ids", []) if a in sids}
        # evidence turns unknown at turn granularity -> mark all turns of evidence sessions
        evidence = [f"D{n}:{i+1}" for n in ev_sessions for i in range(len(sessions[n-1].turns))]
        qa = QA(qid, ex["question"], str(ex["answer"]), evidence, TYPE_TO_CAT.get(qtype, 0))
        qa.qtype = qtype; qa.question_date = ex.get("question_date", "")
        convs[qid] = {"sessions": sessions, "qa": [qa]}
        if limit and len(convs) >= limit: break
    return convs
