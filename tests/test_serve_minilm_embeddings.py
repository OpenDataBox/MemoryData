"""Embedding protocol and token accounting without downloading a model."""

import base64

import numpy as np
import pytest

from scripts.serve_minilm_embeddings import MODEL, encode_request


class FakeEncoder:
    max_seq_length = 4

    def tokenizer(self, texts, **kwargs):
        assert kwargs == {"truncation": False, "padding": False}
        return {"input_ids": [[101, *range(len(text)), 102] for text in texts]}

    def tokenize(self, texts):
        lengths = [min(len(text) + 2, self.max_seq_length) for text in texts]
        return {"attention_mask": np.array([[1] * n + [0] * (4 - n) for n in lengths])}

    def encode(self, texts, **kwargs):
        assert kwargs == {"convert_to_numpy": True, "show_progress_bar": False}
        return np.array([[float(index)] * 384 for index, _ in enumerate(texts)], dtype=np.float32)


def test_float_vectors_and_native_truncation_are_accounted():
    result = encode_request(FakeEncoder(), {"model": MODEL, "input": ["a", "abcde"]})
    assert len(result["data"]) == 2
    assert result["data"][1]["embedding"] == [1.0] * 384
    assert result["usage"] == {
        "prompt_tokens": 7, "total_tokens": 7,
        "prompt_tokens_details": {
            "tokens_before_truncation": 10, "encoded_tokens": 7,
            "truncated_inputs": 1, "max_seq_length": 4,
        },
    }


def test_base64_is_little_endian_float32():
    result = encode_request(FakeEncoder(), {
        "model": MODEL, "input": ["x", "y"], "encoding_format": "base64",
    })
    raw = base64.b64decode(result["data"][1]["embedding"])
    assert len(raw) == 384 * 4
    assert np.frombuffer(raw, dtype="<f4").tolist() == [1.0] * 384


def test_string_input_and_empty_string():
    result = encode_request(FakeEncoder(), {"model": MODEL, "input": ""})
    assert len(result["data"]) == 1
    assert result["usage"]["prompt_tokens"] == 2


@pytest.mark.parametrize("change", [
    {"model": "other"}, {"input": []}, {"input": [[1, 2]]},
    {"input": None}, {"dimensions": 768}, {"encoding_format": "int8"},
])
def test_invalid_requests_are_rejected(change):
    with pytest.raises(ValueError):
        encode_request(FakeEncoder(), {"model": MODEL, "input": "hello", **change})
