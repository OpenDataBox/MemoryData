import json
import sqlite3
from dataclasses import replace
import numpy as np
import pytest

from methods.retentionmem.backends import Tokenizer
from methods.retentionmem.demo import DemoEmbedder, DemoGenerator, run_demo
from methods.retentionmem.engine import RetentionMemory, coalesce_spans
from methods.retentionmem.features import FeatureExtractor, RuleRisk
from methods.retentionmem.generation import CandidateGenerator
from methods.retentionmem.policy import RetentionPolicy, train_policy, calibrate
from methods.retentionmem.storage import MemoryStore
from methods.retentionmem.training import validate_training_streams, answer_probes
from methods.retentionmem.types import Candidate, Event, MemoryUnit, Probe, SourceSpan


@pytest.fixture
def store(tmp_path):
    value = MemoryStore(tmp_path/"memory.sqlite")
    yield value
    value.close()


def engine(store, mode="heuristic_residual", generator=None, **kw):
    return RetentionMemory(store, DemoEmbedder(), Tokenizer(), generator or DemoGenerator(), RuleRisk(),
                           mode=mode, buffer_size=99, **kw)


def populate(memory):
    filler = " Unrelated packing material details are on the receipt."*30
    memory.observe(Event("old", "The old address was Seoul."+filler, "2025-03-01"))
    memory.observe(Event("new", "The current shipping address is Busan."+filler, "2025-06-01"))


def test_immutable_idempotent_archive(store):
    event = Event("a", "text")
    assert store.observe(event) and not store.observe(event)
    with pytest.raises(ValueError): store.observe(replace(event, text="changed"))
    with pytest.raises(sqlite3.IntegrityError): store.db.execute("UPDATE events SET payload='bad'")
    store.db.rollback()
    with pytest.raises(sqlite3.IntegrityError): store.db.execute("DELETE FROM events")
    store.db.rollback()
    assert store.events() == [event]


def test_residual_preserves_current_scope_and_active_only(store, monkeypatch):
    memory = engine(store)
    populate(memory)
    result = memory.maintain()
    assert result.action == "residual_merge" and result.after_tokens < result.before_tokens
    assert result.residual_spans == [{"event_id": "new", "start": 0, "end": 38}]
    assert len(store.events()) == 2
    monkeypatch.setattr(store, "events", lambda *args: pytest.fail("Archive accessed during query"))
    context = "\n\n".join(r.text for r in memory.retrieve("current address", 100))
    assert "Busan" in context and "2025-06-01" in context and "Seoul" not in context


def test_failure_keeps_memory_and_pending(store):
    class Broken:
        def generate(self, *args): raise RuntimeError("server down")
    memory = engine(store, generator=Broken())
    populate(memory)
    before = store.units()
    with pytest.raises(RuntimeError): memory.maintain()
    assert store.units() == before and len(store.meta("pending")) == 2


def test_budget_rejects_residual_without_deleting(store):
    memory = engine(store)
    populate(memory)
    result = memory.maintain(budget=20)
    assert result.action == "keep" and not result.budget_met and len(store.units()) == 2


def test_missing_probes_cannot_claim_safe_merge(store):
    class Empty:
        def generate(self, *args): return Candidate("Short summary", ())
    memory = engine(store, generator=Empty())
    populate(memory)
    result = memory.maintain()
    assert result.action == "keep" and result.reason == "no_valid_source_anchored_probes"


def test_token_packing_and_finalize_progress(store):
    memory = engine(store, mode="keep")
    for i in range(21): memory.observe(Event(str(i), "A city with details and numbers 138."*5))
    memory.finalize()
    assert store.meta("pending") == [] and len(store.units()) == 21
    for budget in (1, 2, 7, 30, 100):
        context = "\n\n".join(r.text for r in memory.retrieve("city", budget))
        assert 0 < memory.tokenizer.count(context) <= budget


def test_commit_rejects_stale_proposal_atomically(store):
    store.observe(Event("a", "Alpha")); store.observe(Event("b", "Beta"))
    region = store.units()
    successor = MemoryUnit("m", "summary", ("a", "b"), ("a", "b"))
    store.commit(region, successor, {"action": "merge"})
    assert store.meta("pending") == []
    with pytest.raises(RuntimeError): store.commit(region, replace(successor, id="m2"), {})
    assert store.units() == [successor] and len(store.audits()) == 1


def test_source_anchor_validation_and_isolated_writer(store):
    calls = []
    class Endpoint:
        def chat(self, messages, **kw):
            calls.append((messages, kw["stage"]))
            if kw["stage"] == "candidate": return {"summary": "short"}
            return {"probes": [
                {"id": "p1", "question": "Where?", "answer": "Busan", "scope": "current", "unambiguous": True,
                 "sources": [{"event_id": "a", "quote": "SECRET_ARCHIVE Busan"}]},
                {"id": "p2", "question": "Where?", "answer": "Seoul", "scope": "historical", "unambiguous": True,
                 "sources": [{"event_id": "a", "quote": "absent quote"}]}]}
    generator = CandidateGenerator(Endpoint(), store)
    candidate = generator.generate([MemoryUnit("m", "ACTIVE ONLY", ("a",), ("a",))], [Event("a", "SECRET_ARCHIVE Busan")], 10)
    assert len(candidate.probes) == 1
    assert "SECRET_ARCHIVE" not in str(calls[0]) and "ACTIVE ONLY" in str(calls[0])
    assert "SECRET_ARCHIVE" in str(calls[1])


def test_source_group_leakage_even_when_ids_renamed():
    a = {"group_id": "a", "origin": "synthetic", "split": "train", "events": [Event("a", "same source").to_dict()]}
    b = {"group_id": "b", "origin": "synthetic", "split": "validation", "events": [Event("b", "same source").to_dict()]}
    with pytest.raises(ValueError, match="leakage"): validate_training_streams([a, b])
    with pytest.raises(ValueError): validate_training_streams([{**a, "origin": "locomo"}])
    with pytest.raises(ValueError): validate_training_streams([{**a, "questions": ["test QA"]}])


def test_no_learned_fallback(store):
    with pytest.raises(ValueError, match="trained policy"): engine(store, mode="learned")
    with pytest.raises(FileNotFoundError): RetentionPolicy.load("missing-policy.npz")


def test_actual_mlp_training_calibration_and_checkpoint(tmp_path):
    rng = np.random.default_rng(33)
    x = rng.standard_normal((180, 12)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    model = train_policy(x[:140], y[:140], x[140:], y[140:], list(range(40)), fingerprint={"demo": True},
                         epochs=35, min_accepted=5, test_artifact=True)
    assert np.mean((model.predict(x[:140]) > .5) == y[:140]) > .95
    path = tmp_path/"policy.npz"; model.save(path)
    with pytest.raises(ValueError, match="test/demo"): RetentionPolicy.load(path)
    loaded = RetentionPolicy.load(path, {"demo": True}, allow_test_artifact=True)
    assert np.allclose(model.predict(x), loaded.predict(x))
    with pytest.raises(ValueError, match="configuration"): RetentionPolicy.load(path, {"demo": False}, True)
    threshold, report = calibrate([.1, .2], [1, 1], ["a", "b"], min_accepted=1)
    assert threshold == -1 and report["accepted_candidates"] == 0


def test_overlap_deduplication_and_cost_accounting(store):
    assert coalesce_spans([SourceSpan("a", 1, 5), SourceSpan("a", 3, 8)]) == [SourceSpan("a", 1, 8)]
    store.cost({"stage": "candidate", "kind": "llm", "calls": 1, "seconds": 2, "usage_reported": False})
    store.cost({"stage": "maintenance_wall", "seconds": 3})
    totals = store.costs()
    assert totals["service_seconds"] == 2 and totals["wall_seconds"] == 3 and totals["missing_usage_calls"] == 1
    assert "seconds" not in totals


def test_missing_reader_output_is_not_regression(store):
    class Reader:
        def chat(self, *args, **kwargs): return {"answers": []}
    with pytest.raises(ValueError, match="parser failures"):
        answer_probes(Reader(), [Probe("a", "Where", "Here", (SourceSpan("a", 0, 1),))], "context", store, "test")


def test_failed_api_attempts_remain_in_cost_ledger(store, monkeypatch):
    from methods.retentionmem.backends import OpenAIEndpoint
    endpoint = OpenAIEndpoint("http://localhost:1/v1", "test")
    def fail(*args):
        endpoint.last_attempts = 2
        raise RuntimeError("unavailable")
    monkeypatch.setattr(endpoint, "post", fail)
    with pytest.raises(RuntimeError): endpoint.chat([], ledger=store, stage="candidate")
    assert store.costs()["calls"] == 2 and store.costs()["missing_usage_calls"] == 2


def test_durable_resume_manifest_and_demo(tmp_path):
    from methods.retentionmem.config import build_runtime
    config = {"mode": "raw", "llm": {"model": "fake", "base_url": "http://localhost:1/v1"},
              "embedding": {"model": "fake", "base_url": "http://localhost:1/v1"}}
    first = build_runtime(config, tmp_path/"saved")
    first.observe(Event("a", "Persistent text")); first.finalize(); first.store.close()
    second = build_runtime(config, tmp_path/"saved")
    assert second.store.events()[0].text == "Persistent text"
    second.store.close()
    with pytest.raises(ValueError, match="different configuration"):
        build_runtime({**config, "mode": "keep"}, tmp_path/"saved")
    assert run_demo(tmp_path/"demo")["test_artifact"] is True
