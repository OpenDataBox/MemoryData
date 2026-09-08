"""Frozen embedding features plus explicit retention indicators."""
from __future__ import annotations

import re
import numpy as np

from .backends import tokens
from .generation import render_span

FEATURE_VERSION = "retention-v2-projection64-source-time"
NEGATIONS = {"not", "never", "no", "except", "unless", "without", "cannot", "must", "before", "after", "아니", "않", "금지"}


def coverage(needles, haystack):
    needles = set(needles)
    return len(needles & set(haystack))/len(needles) if needles else 1.0


def lexical_features(probe, source, text, before_tokens, token_count):
    src, dst = tokens(source), tokens(text)
    answer = tokens(probe.answer)
    source_numbers = re.findall(r"\d+(?:[./:-]\d+)*", source)
    target_numbers = re.findall(r"\d+(?:[./:-]\d+)*", text)
    return np.asarray([
        coverage(answer, dst), coverage(src, dst), coverage(source_numbers, target_numbers),
        coverage(set(src) & NEGATIONS, dst), float(probe.answer.casefold() in text.casefold()),
        min(token_count/max(before_tokens, 1), 4.0), min(len(probe.spans)/8, 1),
        float(probe.scope == "current"), float(probe.scope == "historical"),
        float(probe.scope == "atemporal"), float(probe.scope == "action"),
    ], dtype=np.float32)


class FeatureExtractor:
    def __init__(self, embedder, tokenizer):
        self.embedder, self.tokenizer = embedder, tokenizer
        self._projection = None

    @property
    def fingerprint(self):
        return {"version": FEATURE_VERSION, "embedder": self.embedder.fingerprint,
                "tokenizer": self.tokenizer.encoding, "projection_seed": 23, "projection_dim": 64}

    def extract(self, probes, events, text, before_tokens):
        if not probes:
            return np.empty((0, 267), dtype=np.float32)
        sources = ["\n".join(render_span(s, events) for s in p.spans) for p in probes]
        strings = [text] + [s for p, src in zip(probes, sources) for s in (p.question, p.answer, src)]
        vectors = self.embedder.encode(strings, stage="policy_embedding")
        if self._projection is None:
            rng = np.random.default_rng(23)
            self._projection = rng.choice(np.array([-1., 1.], dtype=np.float32), (vectors.shape[1], 64))/8
        if vectors.shape[1] != self._projection.shape[0]:
            raise ValueError("Embedding dimension changed within a run")
        projected = vectors @ self._projection
        rows = []
        for i, (probe, source) in enumerate(zip(probes, sources)):
            q, answer, src = projected[1+3*i:4+3*i]
            lexical = lexical_features(probe, source, text, before_tokens, self.tokenizer.count(text))
            rows.append(np.concatenate((q, answer, src-projected[0], src*projected[0], lexical)))
        return np.stack(rows).astype(np.float32)


class RuleRisk:
    """Named lexical control; never represented as a learned policy."""
    threshold = 0.35
    mode = "heuristic"

    def predict(self, x):
        # Preserve answers, numbers and negative/conditional qualifiers.
        lex = x[:, -11:]
        return np.maximum.reduce((1-lex[:, 0], 1-lex[:, 2], 1-lex[:, 3]))
