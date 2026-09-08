"""LoCoMo loader. Expects the official locomo10.json.
Each sample: {"sample_id", "conversation": {"speaker_a","speaker_b","session_1":[turns], "session_1_date_time":..., ...},
              "qa": [{"question","answer","evidence":["D1:3",...],"category":int}]}
"""
import json, re
from dataclasses import dataclass

@dataclass
class Turn:
    session: int
    idx: int
    speaker: str
    text: str
    dia_id: str

@dataclass
class Session:
    conv_id: str
    num: int
    date: str
    turns: list

    def text(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)

@dataclass
class QA:
    conv_id: str
    question: str
    answer: str
    evidence: list        # ["D3:5", ...]
    category: int
    def sessions(self):
        out = set()
        for e in self.evidence:
            m = re.match(r"D(\d+):", e)
            if m: out.add(int(m.group(1)))
        return sorted(out)

    def turns_in(self, session_num):
        """1-based turn indices of evidence inside a given session."""
        out = []
        for e in self.evidence:
            m = re.match(r"D(\d+):(\d+)", e)
            if m and int(m.group(1)) == session_num:
                out.append(int(m.group(2)))
        return sorted(out)

def load(path):
    data = json.load(open(path))
    convs = {}
    for sample in data:
        cid = str(sample["sample_id"])
        conv = sample["conversation"]
        sessions = []
        n = 1
        while f"session_{n}" in conv:
            turns = []
            for i, t in enumerate(conv[f"session_{n}"]):
                text = t.get("text", "")
                if t.get("blip_caption"):
                    text = f"{text} [image: {t['blip_caption']}]"
                turns.append(Turn(n, i, t["speaker"], text, t.get("dia_id", f"D{n}:{i+1}")))
            sessions.append(Session(cid, n, conv.get(f"session_{n}_date_time", ""), turns))
            n += 1
        qas = []
        for q in sample["qa"]:
            if q.get("category") == 5:      # adversarial: excluded from 4.2-A
                continue
            ans = q.get("answer", "")
            qas.append(QA(cid, q["question"], str(ans), q.get("evidence", []), q.get("category", 0)))
        convs[cid] = {"sessions": sessions, "qa": qas}
    return convs

def normalize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return [w for w in s.split() if w not in {"a","an","the"}]

def token_f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if not p or not g: return float(p == g)
    common = {}
    for w in p: common[w] = common.get(w, 0) + 1
    hit = sum(min(c, g.count(w)) for w, c in common.items())
    if hit == 0: return 0.0
    prec, rec = hit/len(p), hit/len(g)
    return 2*prec*rec/(prec+rec)
