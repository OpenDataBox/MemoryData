import json

from benchmark.longmemeval.loader import _load_json_records, load_longmemeval_eval_data


def test_load_json_records_stops_at_requested_limit(tmp_path):
    source = tmp_path / "records.json"
    source.write_text(
        json.dumps([{"id": 1}, {"id": 2}, {"id": 3}]),
        encoding="utf-8",
    )

    assert _load_json_records(source, limit=2) == [{"id": 1}, {"id": 2}]


def test_longmemeval_loader_can_limit_context_chunks_for_smoke_runs(tmp_path):
    source = tmp_path / "longmemeval.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question_date": "2024/01/03",
                    "question": "Where does Alice live?",
                    "answer": "Seoul",
                    "answer_session_ids": ["s1"],
                    "haystack_session_ids": ["s1", "s2"],
                    "haystack_dates": ["2024/01/01", "2024/01/02"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "Alice lives in Seoul."}],
                        [{"role": "user", "content": "Bob lives in Busan."}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_longmemeval_eval_data(
        {
            "test_files": str(source),
            "max_test_samples": 1,
            "max_context_chunks": 1,
            "chunk_size": 4096,
        }
    )

    assert len(loaded["data"][0]["context_chunks"]) == 1
    assert "Alice lives in Seoul" in loaded["data"][0]["context_chunks"][0]["text"]
