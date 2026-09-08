"""Mem0-style fact extraction using the same Qwen3-8B engine.

This is a *prompt-level reimplementation* of Mem0's extraction step so that
4.2-A runs without external services. It is NOT the upstream Mem0 code.
For the paper's main table you will swap in the vendored Mem0/LightMem
pipelines (see TODO in `compress`). For 4.2-A this is enough: what we need
is a realistic lossy compression whose regression we can measure.
"""
from .base import Host

SYS = ("You extract long-term memory facts from a conversation session. "
       "Output one fact per line, each a self-contained sentence with the speaker name, "
       "keeping dates, negations, numbers, and conditions exactly as stated. "
       "Do not add facts that are not in the text. Output at most {cap} lines.")

class Mem0LikeHost(Host):
    name = "mem0like"
    def __init__(self, engine, cap=10):
        super().__init__(engine)
        self.cap = cap                     # lambda: aggressiveness knob (lower cap = more compression)

    def compress(self, session, prior_state):
        # TODO(paper main table): replace with upstream Mem0 `Memory.add()` calls.
        user = f"Session date: {session.date}\n\n{session.text()}\n\nFacts:"
        out = self.engine.generate(SYS.format(cap=self.cap), [user], max_tokens=512)[0]
        facts = [l.strip("-• ").strip() for l in out.split("\n") if l.strip()]
        return [f"({session.date}) {f}" for f in facts[:self.cap]]


class LightMemLikeHost(Host):
    """Summary-style compression (topic segments -> short summary). Stand-in for LightMem."""
    name = "lightmemlike"
    SYS = ("Summarize the session into {n} short bullet points that preserve who said what, "
           "dates, numbers, negations, and any plans or conditions. No commentary.")
    def __init__(self, engine, n_bullets=6):
        super().__init__(engine)
        self.n = n_bullets

    def compress(self, session, prior_state):
        # TODO(paper main table): replace with upstream LightMem pipeline.
        user = f"Session date: {session.date}\n\n{session.text()}"
        out = self.engine.generate(self.SYS.format(n=self.n), [user], max_tokens=400)[0]
        lines = [l.strip("-• ").strip() for l in out.split("\n") if l.strip()]
        return [f"({session.date}) {l}" for l in lines[:self.n]]


def get_host(name, engine, **kw):
    from .base import RawHost
    return {"raw": RawHost, "mem0": Mem0LikeHost, "lightmem": LightMemLikeHost}[name](engine, **kw)
