"""Host memory interface.

A host is anything that turns sessions into a memory state and can build a
retrieval context for a question. 4.2-A needs two states at every session
boundary t:
  M_minus[t] = memory with session t appended verbatim (no compression)
  M_plus[t]  = memory after the host's own compression of session t

Retrieval is shared (hybrid BM25 + dense over "memory units") so that the only
difference between M- and M+ is WHAT is stored, not HOW it is retrieved.
"""
import copy
import numpy as np
from rank_bm25 import BM25Okapi


class MemoryState:
    """A list of memory units (strings) plus metadata. Deep-copyable snapshots."""
    def __init__(self):
        self.units = []          # list[dict(text=str, session=int, kind="raw"|"fact"|...)]

    def add(self, text, session, kind):
        self.units.append({"text": text, "session": session, "kind": kind})

    def snapshot(self):
        return copy.deepcopy(self)

    def n_tokens(self, tok):
        return sum(len(tok(u["text"], add_special_tokens=False).input_ids) for u in self.units)


class Retriever:
    def __init__(self, embed_model_name, device="cuda"):
        from sentence_transformers import SentenceTransformer
        self.emb = SentenceTransformer(embed_model_name, device=device)

    def build(self, state: MemoryState):
        texts = [u["text"] for u in state.units]
        if not texts:
            return None
        bm25 = BM25Okapi([t.lower().split() for t in texts])
        dense = self.emb.encode(texts, normalize_embeddings=True, batch_size=64)
        return {"texts": texts, "bm25": bm25, "dense": dense, "units": state.units}

    def retrieve(self, index, query, k=8, tok=None, budget=None):
        if index is None:
            return ""
        q = self.emb.encode([query], normalize_embeddings=True)[0]
        d = index["dense"] @ q
        b = np.array(index["bm25"].get_scores(query.lower().split()))
        def rank(x):
            order = np.argsort(-x); r = np.empty_like(order); r[order] = np.arange(len(x)); return r
        rrf = 1.0/(60+rank(d)) + 1.0/(60+rank(b))
        order = np.argsort(-rrf)[:k]
        chosen, used = [], 0
        for i in order:
            u = index["units"][i]
            n = len(tok(u["text"], add_special_tokens=False).input_ids) if tok else 0
            if budget and used + n > budget:
                continue
            used += n
            chosen.append(f"[session {u['session']}] {u['text']}")
        return "\n".join(chosen)


class Host:
    """Subclass and implement `compress`. Returns list of memory-unit strings
    for the session (the host's compressed representation)."""
    name = "base"
    def __init__(self, engine):
        self.engine = engine

    def compress(self, session, prior_state: MemoryState) -> list:
        raise NotImplementedError

    def raw_units(self, session):
        # one unit per turn, verbatim, with date
        return [f"({session.date}) {t.speaker}: {t.text}" for t in session.turns]


class RawHost(Host):
    """No compression: M+ == M-. Used as a sanity check (regression must be ~0)."""
    name = "raw"
    def compress(self, session, prior_state):
        return self.raw_units(session)
