"""Protocol tests that never start a model or access the network."""

import json
from pathlib import Path

import pytest

from research.future_memory_pilot.run_pilot import (
    LocalClient,
    NoRedirect,
    grade,
    reader_messages,
    run_case,
    select_candidates,
    summarize,
    validate_cases,
    writer_messages,
)


@pytest.fixture
def branch():
    return {"future": "The application was denied.", "question": "Which city?",
            "gold": {"city": "Busan"}, "aliases": {"city": ["Busan", "Pusan"]}}


@pytest.mark.parametrize("answer, score", [
    ('{"city": "Busan"}', 1.0), ('{"city": "PUSAN"}', 1.0),
    ('```json\n{"city": "Busan"}\n```', 1.0),
    ('{"city": "not Busan"}', 0.0), ('{"city": ""}', 0.0),
    ('{"city": ["Busan"]}', 0.0), ('["Busan"]', 0.0), ('Busan', 0.0),
])
def test_exact_grading(branch, answer, score):
    assert grade(answer, branch)["score"] == score


def test_mean_and_worst_can_disagree_without_ties():
    assert select_candidates([[1.0, 0.25], [0.5, 0.5], [0.0, 0.0]], [1.0, 1.0]) == {
        "plain": 0, "mean": 0, "worst": 1}


def test_ties_and_improvements_over_source_are_handled():
    assert select_candidates([[0.5, 1.0], [1.0, 1.0]], [0.5, 0.5]) == {
        "plain": 0, "mean": 0, "worst": 0}


@pytest.mark.parametrize("scores, source", [([], [1.0]), ([[1.0]], []), ([[1.0]], [1.0, 1.0])])
def test_invalid_selection(scores, source):
    with pytest.raises(ValueError):
        select_candidates(scores, source)


def test_writer_and_reader_do_not_receive_gold(branch):
    messages = writer_messages("A prefix sentinel.", 96)
    assert messages[1]["content"] == "A prefix sentinel."
    serialized = json.dumps(reader_messages("A memory sentinel.", branch))
    assert "Busan" not in serialized
    assert "Pusan" not in serialized
    assert branch["future"] in serialized


def test_refuse_paid_endpoint(tmp_path):
    with pytest.raises(ValueError, match="local"):
        LocalClient("https://api.openai.com/v1", "unused", tmp_path / "trace.jsonl")


def test_refuse_redirect_and_external_post(tmp_path):
    client = LocalClient("http://127.0.0.1:9911/v1", "unused", tmp_path / "trace.jsonl")
    with pytest.raises(ValueError, match="local"):
        client.post("https://example.com", {})
    assert NoRedirect().redirect_request(None, None, 307, "redirect", {}, "https://example.com") is None


def test_non_roundtrip_tokenizer_cannot_break_budget():
    class FakeClient:
        root_url = "http://127.0.0.1"

        def chat(self, *args, **kwargs):
            return {"answer": "memory", "finish_reason": "stop"}

        def tokenize(self, text):
            return list(range(100))

        def post(self, *args, **kwargs):
            return {"content": "still too long"}

    with pytest.raises(ValueError, match="token budget"):
        run_case(FakeClient(), {"id": "test", "prefix": "test"}, 96, 1, 1)


def test_fixture_protocol():
    cases = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))["cases"]
    validate_cases(cases)
    assert len(cases) == 6
    for case in cases:
        assert len({b["id"] for b in case["branches"]}) == 3
        for branch in case["branches"]:
            assert len(branch["gold"]) == 4
            assert set(branch["gold"]) == set(case["answer_fields"])
            assert grade(json.dumps(branch["gold"]), branch)["score"] == 1.0


def test_no_source_solved_cases_is_not_reported_as_zero():
    result = {"heldout_id": "h", "controls": {"h": {"source": {"score": 0.5},
              "future_only": {"score": 0.0}}},
              "heldout": {name: {"score": 0.0} for name in ("plain", "mean", "worst")},
              "selected_indices": {"plain": 0, "mean": 0, "worst": 0},
              "source_selection_scores": [1.0, 1.0],
              "candidates": [{"memory": "test", "selection": [{"score": 0.0}, {"score": 0.0}]}]}
    summary = summarize([result])
    assert summary["source_solved_cases"] == 0
    assert summary["heldout_accuracy_on_source_solved"]["plain"] is None
    assert summary["strict_selector_disagreements"] == 0


@pytest.mark.parametrize("scores, expected", [
    ([[1.0, 0.25], [0.5, 0.5]], 1),
    ([[0.0, 0.0], [0.0, 0.75]], 0),
])
def test_strict_reversals_exclude_tie_driven_disagreement(scores, expected):
    result = {"heldout_id": "h", "controls": {"h": {"source": {"score": 1.0},
              "future_only": {"score": 0.0}}},
              "heldout": {name: {"score": 1.0} for name in ("plain", "mean", "worst")},
              "selected_indices": select_candidates(scores, [1.0, 1.0]),
              "source_selection_scores": [1.0, 1.0],
              "candidates": [{"memory": str(i), "selection": [{"score": value} for value in row]}
                             for i, row in enumerate(scores)]}
    assert summarize([result])["strict_selector_disagreements"] == expected
