"""A deterministic software demonstration, never a scientific model/result."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .backends import Tokenizer, tokens
from .engine import RetentionMemory
from .features import RuleRisk
from .storage import MemoryStore
from .types import Candidate, Event, Probe, SourceSpan


class DemoEmbedder:
    fingerprint = "test-only-hashed-embedding"

    def encode(self, texts, stage="embedding"):
        values = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for term in tokens(text):
                values[i, int(hashlib.sha256(term.encode()).hexdigest()[:8], 16) % 64] += 1
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1)


class DemoGenerator:
    def generate(self, units, events, target_tokens):
        current = events[1]
        quote = "The current shipping address is Busan."
        start = current.text.index(quote)
        probe = Probe("current_address", "What is the current shipping address?", "Busan",
                      (SourceSpan(current.id, start, start+len(quote)),), "current")
        return Candidate("A customer changed their shipping address and is planning a business trip to Daejeon.", (probe,))


def run_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"memory.sqlite").exists():
        raise FileExistsError("Use a new demo directory")
    store = MemoryStore(output/"memory.sqlite")
    tokenizer = Tokenizer()
    memory = RetentionMemory(store, DemoEmbedder(), tokenizer, DemoGenerator(), RuleRisk(),
                             mode="heuristic_residual", buffer_size=99)
    try:
        filler = " The notebook has a blue cover and plain pages."*25
        memory.observe(Event("march", "The previous shipping address was Seoul."+filler, "2025-03-01"))
        memory.observe(Event("june", "The current shipping address is Busan."+filler, "2025-06-01"))
        memory.observe(Event("july", "Next week I visit Daejeon for a business trip. This is unrelated to delivery."+filler, "2025-07-01"))
        result = memory.maintain()
        retrieval = memory.retrieve("Current shipping address?", 128)
        text = "\n\n".join(r.text for r in retrieval)
        assert result.action == "residual_merge" and "Busan" in text and "2025-06-01" in text
        assert tokenizer.count(text) <= 128 and len(store.events()) == 3
        report = {"test_artifact": True, "policy": "lexical control, not trained", "decision": result.to_dict(),
                  "retrieved_text": text, "stats": memory.stats(), "scientific_accuracy": None}
        (output/"demo.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        store.close()
