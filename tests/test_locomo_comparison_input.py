import json
from pathlib import Path

import pytest

from benchmark.locomo.loader import load_locomo_eval_data


def _write_locomo_sample(tmp_path):
    source = tmp_path / "locomo.json"
    source.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample-1",
                    "conversation": {
                        "session_1_date_time": "2023/05/20 (Sat) 12:00",
                        "session_1": [
                            {"speaker": "Alice", "text": "TURN_ONE", "dia_id": "D1"},
                            {
                                "speaker": "Bob",
                                "text": "TURN_TWO",
                                "blip_caption": "CAPTION_TWO",
                                "dia_id": "D2",
                            },
                            {"speaker": "Alice", "text": "TURN_THREE", "dia_id": "D3"},
                        ],
                    },
                    "qa": [
                        {
                            "question_id": "sample-1_qa0",
                            "question": "What happened?",
                            "answer": "Three turns.",
                            "category": 1,
                            "evidence": ["D1", "D2", "D3"],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return source


def test_repeat_session_header_marks_every_mid_session_chunk_without_losing_turns(tmp_path):
    source = _write_locomo_sample(tmp_path)
    sample = load_locomo_eval_data(
        {
            "test_files": str(source),
            "chunk_size": 45,
            "locomo_repeat_session_header": True,
        }
    )["data"][0]

    chunks = sample["context_chunks"]
    assert len(chunks) == 3
    assert all(
        chunk["text"].startswith("Session 1 (2023/05/20 (Sat) 12:00)\n")
        for chunk in chunks
    )

    combined = "\n".join(chunk["text"] for chunk in chunks)
    for raw_turn in (
        "Alice: TURN_ONE",
        "Bob: [Image: CAPTION_TWO] TURN_TWO",
        "Alice: TURN_THREE",
    ):
        assert combined.count(raw_turn) == 1

    source_ids = [source_id for chunk in chunks for source_id in chunk["source_ids"]]
    assert source_ids == ["D1", "D2", "D3"]


def test_default_does_not_repeat_session_header(tmp_path):
    source = _write_locomo_sample(tmp_path)
    base_config = {"test_files": str(source), "chunk_size": 45}

    implicit_default = load_locomo_eval_data(base_config)["data"][0]["context_chunks"]
    explicit_default = load_locomo_eval_data(
        {**base_config, "locomo_repeat_session_header": False}
    )["data"][0]["context_chunks"]

    assert implicit_default == explicit_default
    combined = "\n".join(chunk["text"] for chunk in implicit_default)
    assert combined.count("Session 1 (2023/05/20 (Sat) 12:00)") == 1


def test_public_locomo_chunk_size_one_preserves_all_5882_whole_turns():
    source = Path(__file__).resolve().parents[2] / "HiGMem" / "data" / "locomo10.json"
    if not source.is_file():
        pytest.skip("The supplied public HiGMem LoCoMo dataset is not checked out")

    raw_samples = json.loads(source.read_text(encoding="utf-8"))
    loaded_samples = load_locomo_eval_data(
        {
            "test_files": str(source),
            "chunk_size": 1,
            "locomo_repeat_session_header": True,
        }
    )["data"]

    expected_by_sample = {}
    for sample in raw_samples:
        expected_turns = []
        conversation = sample["conversation"]
        session_keys = sorted(
            (
                key
                for key, value in conversation.items()
                if key.startswith("session_")
                and not key.endswith("_date_time")
                and isinstance(value, list)
            ),
            key=lambda key: int(key.split("_")[1]),
        )
        for session_key in session_keys:
            session_label = session_key.replace("_", " ").title()
            session_time = str(
                conversation.get(f"{session_key}_date_time", "") or ""
            ).strip()
            header = session_label + (f" ({session_time})" if session_time else "")
            for turn in conversation[session_key]:
                text = str(turn.get("text", "") or "").strip()
                caption = str(turn.get("blip_caption", "") or "").strip()
                if caption:
                    text = f"[Image: {caption}] {text}".strip()
                speaker = str(turn.get("speaker", "Unknown") or "Unknown").strip()
                expected_turns.append(
                    (
                        f"{header}\n{speaker}: {text}".strip(),
                        str(turn.get("dia_id", "")).strip(),
                    )
                )
        expected_by_sample[str(sample["sample_id"])] = expected_turns

    assert sum(len(turns) for turns in expected_by_sample.values()) == 5882
    assert sum(len(sample["context_chunks"]) for sample in loaded_samples) == 5882
    for sample in loaded_samples:
        expected_turns = expected_by_sample[sample["sample_id"]]
        chunks = sample["context_chunks"]
        assert [chunk["text"] for chunk in chunks] == [text for text, _ in expected_turns]
        assert [chunk["source_ids"] for chunk in chunks] == [
            [source_id] for _, source_id in expected_turns
        ]
