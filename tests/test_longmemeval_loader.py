import json

import pytest

from benchmark.longmemeval.loader import load_longmemeval_eval_data


def test_history_excludes_labels_and_preserves_dates_and_roles(tmp_path):
    path = tmp_path / "longmemeval.json"
    path.write_text(json.dumps([{
        "question_id": "q1_abs", "question_type": "knowledge-update",
        "question": "Where do I live?", "answer": "GOLD_ONLY",
        "question_date": "2025-02-03", "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s1"], "haystack_dates": ["2025-02-01"],
        "haystack_sessions": [[
            {"role": "user", "content": "I moved to Seoul.", "has_answer": True},
            {"role": "assistant", "content": "You moved from Busan."},
        ]],
    }]), encoding="utf-8")
    sample = load_longmemeval_eval_data({"test_files": str(path)})["data"][0]
    history = "\n".join(chunk["text"] for chunk in sample["context_chunks"])
    assert "2025-02-01" in history
    assert "user: I moved to Seoul." in history
    assert "assistant: You moved from Busan." in history
    assert "GOLD_ONLY" not in history
    assert "has_answer" not in history
    assert "Where do I live?" not in history
    assert sample["eval_metadata"][0]["question_id"] == "q1_abs"
    assert sample["question_dates"] == ["2025-02-03"]

def test_misaligned_session_metadata_fails_instead_of_losing_history(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{
        "question_id": "q1", "haystack_sessions": [[], []],
        "haystack_dates": ["2025-01-01"], "haystack_session_ids": ["s1"],
    }]), encoding="utf-8")
    with pytest.raises(ValueError, match="q1"):
        load_longmemeval_eval_data({"test_files": str(path)})

