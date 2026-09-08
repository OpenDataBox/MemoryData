"""RetentionMem public API: observe, maintain, retrieve.

Only active units are read by retrieve(). The archive is consulted during maintenance.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import math
import time
import hashlib
import numpy as np

from .backends import tokens
from .features import FeatureExtractor
from .generation import render_events, render_span
from .types import Event, MemoryUnit, MaintenanceResult, RetrievedMemory, SourceSpan

MODES = {"raw", "keep", "always_merge", "heuristic", "heuristic_residual", "criteria_verifier", "learned", "learned_residual"}


def cost_delta(after, before):
    stages = {}
    for stage, values in after.get("by_stage", {}).items():
        stages[stage] = {k: v-before.get("by_stage", {}).get(stage, {}).get(k, 0) for k, v in values.items()}
    keys = ("calls", "input_tokens", "output_tokens", "cache_hits", "missing_usage_calls", "service_seconds", "wall_seconds", "policy_seconds")
    return {"by_stage": stages, **{k: after.get(k, 0)-before.get(k, 0) for k in keys}, "usd": None}


def coalesce_spans(spans):
    """Merge overlapping quotes so one source phrase is not paid for repeatedly."""
    grouped = {}
    for s in spans:
        grouped.setdefault(s.event_id, []).append((s.start, s.end))
    result = []
    for event_id, intervals in grouped.items():
        merged = []
        for start, end in sorted(set(intervals)):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        result.extend(SourceSpan(event_id, start, end) for start, end in merged)
    return result


class RetentionMemory:
    def __init__(self, store, embedder, tokenizer, generator=None, policy=None, *, mode="learned_residual",
                 verifier=None, region_size=8, buffer_size=8, compression_ratio=4,
                 max_source_tokens=24000, sparse_weight=0.5):
        if mode not in MODES:
            raise ValueError(f"Unknown memory mode {mode}")
        if not 1 <= region_size <= 8 or not 1 <= buffer_size or compression_ratio <= 1:
            raise ValueError("region_size must be 1..8, buffer_size positive, compression_ratio >1")
        if not 0 <= sparse_weight <= 1:
            raise ValueError("sparse_weight must be within [0,1]")
        if mode.startswith("learned") and (policy is None or getattr(policy, "mode", None) != "learned"):
            raise ValueError("Learned mode requires a trained policy; there is no heuristic fallback")
        if mode.startswith("heuristic") and policy is None:
            raise ValueError("Heuristic mode requires its explicitly named rule policy")
        if mode == "criteria_verifier" and verifier is None:
            raise ValueError("criteria_verifier mode requires a verifier")
        if mode not in {"raw", "keep"} and generator is None:
            raise ValueError("Consolidation requires a candidate generator")
        self.store, self.embedder, self.tokenizer = store, embedder, tokenizer
        self.generator, self.policy, self.verifier = generator, policy, verifier
        self.mode, self.region_size, self.buffer_size = mode, region_size, buffer_size
        self.compression_ratio, self.max_source_tokens = compression_ratio, max_source_tokens
        self.sparse_weight = sparse_weight
        self.features = FeatureExtractor(embedder, tokenizer)

    def observe(self, event: Event):
        start = time.perf_counter()
        added = self.store.observe(event)
        if added:
            pending = self.store.meta("pending", [])
            self.store.cost({"stage": "write", "seconds": time.perf_counter()-start})
            if len(pending) >= self.buffer_size:
                self.maintain()
        return added

    def finalize(self):
        """Call at the session boundary, before any evaluation question is exposed."""
        results = []
        while self.store.meta("pending", []):
            old = self.store.meta("pending", [])
            results.append(self.maintain())
            if self.store.meta("pending", []) == old:
                raise RuntimeError("Finalization made no progress")
        return results

    def rank(self, query, units, stage="retrieval_embedding"):
        if not units:
            return []
        matrix = self.embedder.encode([query]+[u.text for u in units], stage=stage)
        norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        normalized = matrix/norms
        dense = normalized[1:] @ normalized[0]
        query_terms = set(tokens(query))
        corpora = [Counter(tokens(u.text)) for u in units]
        lengths = [sum(c.values()) for c in corpora]
        avg_length = sum(lengths)/max(len(lengths), 1)
        sparse = []
        for corpus, length in zip(corpora, lengths):
            score = 0.0
            for term in query_terms:
                df = sum(term in c for c in corpora)
                idf = math.log(1+(len(units)-df+.5)/(df+.5))
                tf = corpus[term]
                score += idf*tf*2.5/(tf+1.5*(.25+.75*length/max(avg_length, 1))) if tf else 0
            sparse.append(score)
        def normalize(values):
            values = np.asarray(values, dtype=float)
            width = np.ptp(values)
            return (values-values.min())/width if width > 1e-12 else np.zeros_like(values)
        scores = self.sparse_weight*normalize(sparse)+(1-self.sparse_weight)*normalize(dense)
        return sorted(zip(units, scores.tolist()), key=lambda x: (-x[1], x[0].id))

    def select_region(self):
        units = self.store.units()
        pending = set(self.store.meta("pending", []))
        anchors = [u for u in units if u.id in pending]
        if not anchors:
            return []
        anchor = anchors[-1]
        if len(units) <= self.region_size:
            return units
        ranked = self.rank(anchor.text, [u for u in units if u.id != anchor.id], "region_embedding")
        selected = {anchor.id, *(u.id for u, _ in ranked[:self.region_size-1])}
        return [u for u in units if u.id in selected]

    def maintain(self, region=None, budget=None):
        """budget is a proposed replacement ceiling. KEEP reports an unmet ceiling.

        No information is deleted simply to satisfy a maintenance budget. Query budgets are hard.
        """
        costs_before = self.store.costs()
        start = time.perf_counter()
        all_units = {u.id: u for u in self.store.units()}
        if region is None:
            units = self.select_region()
        else:
            if len(region) > self.region_size or len(set(region)) != len(region):
                raise ValueError("Region exceeds configured bound or contains duplicate IDs")
            units = [all_units[x] for x in region]
        before_text = "\n\n".join(u.text for u in units)
        before_tokens = self.tokenizer.count(before_text)
        if budget is not None and budget < 1:
            raise ValueError("Maintenance budget must be positive")
        ceiling = before_tokens-1 if budget is None else min(budget, before_tokens-1)
        event_ids = list(dict.fromkeys(e for u in units for e in u.event_ids))
        source_ids = list(dict.fromkeys(s for u in units for s in u.source_ids))
        result = MaintenanceResult("keep", "unchanged", [u.id for u in units], source_ids,
                                   before_tokens, before_tokens, budget_met=budget is None or before_tokens <= budget)
        replacement = None
        try:
            if not units:
                result.reason = "empty_region"
            elif self.mode in {"raw", "keep"}:
                result.reason = self.mode + "_control"
            else:
                events = self.store.events(event_ids)
                if self.tokenizer.count(render_events(events)) > self.max_source_tokens:
                    result.reason = "source_prefix_exceeds_maintenance_limit"
                else:
                    candidate = self.generator.generate(units, events, max(16, before_tokens//self.compression_ratio))
                    result.summary_tokens = self.tokenizer.count(candidate.summary)
                    if result.summary_tokens >= before_tokens:
                        result.reason = "no_compression"
                    else:
                        replacement = self._choose(candidate, units, events, ceiling, result)
            self.store.cost({"stage": "maintenance_wall", "seconds": time.perf_counter()-start})
            result.costs = cost_delta(self.store.costs(), costs_before)
            if replacement is not None:
                self.store.commit(units, replacement, result.to_dict())
            else:
                self.store.audit(result.to_dict(), result.region)
            return result
        except Exception:
            # Archive and previous active region survive a failed proposal unchanged.
            self.store.cost({"stage": "maintenance_failed_wall", "seconds": time.perf_counter()-start})
            raise

    def _choose(self, candidate, units, events, ceiling, result):
        text, spans = candidate.summary, []
        event_map = {e.id: e for e in events}
        for probe in candidate.probes:
            for span in probe.spans:
                span.extract(event_map)
        if self.mode == "always_merge":
            risk = None
            accept = self.tokenizer.count(text) <= ceiling
        elif self.mode == "criteria_verifier":
            risk = self.verifier.risk("\n\n".join(u.text for u in units), text, events)
            accept = risk == 0 and self.tokenizer.count(text) <= ceiling
        elif not candidate.probes:
            result.reason = "no_valid_source_anchored_probes"
            return None
        else:
            threshold = self.policy.threshold
            risks = self._risks(candidate.probes, event_map, text, result.before_tokens)
            risk = float(max(risks))
            if self.mode.endswith("_residual") and threshold >= 0:
                for _ in range(len(candidate.probes)):
                    if risk <= threshold:
                        break
                    best = None
                    base_loss = float(np.maximum(risks-threshold, 0).sum())
                    for probe, probe_risk in zip(candidate.probes, risks):
                        if probe_risk <= threshold:
                            continue
                        proposed_spans = coalesce_spans(spans+list(probe.spans))
                        if proposed_spans == spans:
                            continue
                        proposed_text = candidate.summary+"\n\nRetained details:\n"+"\n".join(render_span(s, event_map) for s in proposed_spans)
                        count = self.tokenizer.count(proposed_text)
                        if count > ceiling:
                            continue
                        next_risks = self._risks(candidate.probes, event_map, proposed_text, result.before_tokens)
                        gain = base_loss-float(np.maximum(next_risks-threshold, 0).sum())
                        utility = gain/max(count-self.tokenizer.count(text), 1)
                        if gain > 1e-8 and (best is None or utility > best[0]):
                            best = (utility, proposed_spans, proposed_text, next_risks)
                    if best is None:
                        break
                    _, spans, text, risks = best
                    risk = float(max(risks))
            accept = risk <= threshold and self.tokenizer.count(text) <= ceiling
        result.predicted_risk = risk
        if not accept:
            result.reason = "risk_or_replacement_budget"
            return None
        result.action = "residual_merge" if spans else "merge"
        result.reason = "selected"
        result.after_tokens = self.tokenizer.count(text)
        result.residual_tokens = result.after_tokens-result.summary_tokens
        result.residual_spans = [asdict(s) for s in spans]
        result.budget_met = True
        result.version = 1+max(u.version for u in units)
        identity = hashlib.sha256((str(result.version)+"\0"+"\0".join(u.id for u in units)+"\0"+text).encode()).hexdigest()
        return MemoryUnit("merged:"+identity, text, tuple(e.id for e in events),
                          tuple(result.retained_source_ids), result.version, tuple(spans))

    def _risks(self, probes, event_map, text, before_tokens):
        x = self.features.extract(probes, event_map, text, before_tokens)
        start = time.perf_counter()
        result = np.asarray(self.policy.predict(x))
        self.store.cost({"stage": "policy", "calls": 1, "seconds": time.perf_counter()-start})
        if result.shape != (len(probes),) or not np.isfinite(result).all():
            raise ValueError("Policy returned invalid probabilities")
        return result

    def retrieve(self, query, budget):
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("Query budget must be a positive integer")
        start = time.perf_counter()
        ranked = self.rank(query, self.store.units())  # Never fetch archive content here.
        result, rendered = [], ""
        for unit, score in ranked:
            separator = "\n\n" if result else ""
            remaining = budget-self.tokenizer.count(rendered+separator)
            if remaining <= 0:
                break
            text = self.tokenizer.truncate(unit.text, remaining)
            while text and self.tokenizer.count(rendered+separator+text) > budget:
                text = self.tokenizer.truncate(text, self.tokenizer.count(text)-1)
            if not text:
                continue
            result.append(RetrievedMemory(unit.id, text, unit.source_ids, self.tokenizer.count(text), score,
                                          unit.version, text != unit.text))
            rendered += separator+text
        self.store.cost({"stage": "retrieval_wall", "seconds": time.perf_counter()-start})
        assert self.tokenizer.count(rendered) <= budget
        return result

    def stats(self):
        return {"mode": self.mode, "storage": self.store.storage_stats(), "costs": self.store.costs(),
                "active_tokens": sum(self.tokenizer.count(u.text) for u in self.store.units()),
                "maintenance_operations": len(self.store.audits())}
