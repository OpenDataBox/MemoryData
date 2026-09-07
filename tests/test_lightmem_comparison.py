"""Controlled LightMem must use its full pipeline without hiding failed work."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from methods.lightmem import lightmem_adapter as module


@pytest.fixture
def make_adapter(monkeypatch, tmp_path):
    created = []

    def factory(config):
        memory = SimpleNamespace(
            manager=SimpleNamespace(
                client=Mock(),
                generate_response=Mock(return_value=('{"data": []}', {"total_tokens": 1})),
                meta_text_extract=Mock(return_value=[]),
                _call_update_llm=Mock(),
            ),
            compressor=SimpleNamespace(inner_compressor=SimpleNamespace(
                compress_prompt=Mock(return_value={"compressed_prompt": "test"}),
                model=SimpleNamespace(
                    config=SimpleNamespace(_commit_hash="test-revision"), dtype="float32",
                    register_forward_pre_hook=Mock(return_value=Mock()),
                    register_forward_hook=Mock(return_value=Mock()),
                ),
            )),
            senmem_buffer_manager=SimpleNamespace(
                buffer=[], big_buffer=[], token_count=0,
                cut_with_segmenter=Mock(return_value=[]),
            ),
            shortmem_buffer_manager=SimpleNamespace(buffer=[]),
            embedding_retriever=SimpleNamespace(
                client=Mock(), search=Mock(return_value=[]), update=Mock(),
                delete=Mock(), insert=Mock(), get_all=Mock(return_value=[]),
            ),
            text_embedder=SimpleNamespace(embed=Mock(return_value=[1.0])),
            add_memory=Mock(return_value={}),
            construct_update_queue_all_entries=Mock(),
            offline_update_all_entries=Mock(),
            offline_update=Mock(),
        )
        created.append((deepcopy(config), memory))
        return memory

    monkeypatch.setattr(module.LightMemory, "from_config", factory)

    def build(**kwargs):
        adapter = module.LightMemAdapter(
            api_key="test", model="Qwen/Qwen3.5-9B",
            base_url="http://127.0.0.1:18180/v1",
            embedding_base_url="http://127.0.0.1:18181/v1",
            db_path=str(tmp_path), collection_name="test",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dims=384, retrieve_num=60, **kwargs,
        )
        return adapter, created[-1][0], created[-1][1]

    return build


def test_comparison_profile_enables_all_native_stages(make_adapter):
    adapter, config, memory = make_adapter(comparison_mode=True)
    assert adapter.ingest_mode == "pipeline"
    assert all(config[key] for key in (
        "pre_compress", "topic_segment", "precomp_topic_shared",
        "metadata_generate", "text_summary",
    ))
    assert config["pre_compressor"]["configs"]["compress_config"]["rate"] == 0.6
    assert config["pre_compressor"]["configs"]["llmlingua_config"]["device_map"] == "cpu"
    assert config["memory_manager"]["configs"]["max_tokens"] == 16000
    assert config["extract_threshold"] == 0.1
    assert config["extraction_mode"] == "flat"
    assert config["update"] == "offline"
    assert config["text_embedder"]["configs"]["openai_base_url"] == "http://127.0.0.1:18181/v1"
    assert memory.manager.client is adapter.client


def test_legacy_direct_mode_remains_unchanged(make_adapter):
    adapter, config, memory = make_adapter()
    adapter.add_chunk("Alex: Hello", "2023-05-25 13:14:00")
    adapter.finalize()
    assert adapter.ingest_mode == "direct"
    assert config["pre_compress"] is False
    memory.offline_update.assert_called_once()
    memory.add_memory.assert_not_called()
    memory.construct_update_queue_all_entries.assert_not_called()


def test_pipeline_preserves_every_turn_and_session_dates_across_chunks(make_adapter):
    adapter, _, memory = make_adapter(comparison_mode=True)
    adapter.add_chunk("Session 1 (1:14 pm on 25 May, 2023)\nAlex: First\nSam: Second")
    adapter.add_chunk("Alex: Third")
    calls = memory.add_memory.call_args_list
    assert len(calls) == 3
    assert [call.args[0][0]["content"] for call in calls] == ["First", "Second", "Third"]
    assert [call.args[0][0]["speaker_name"] for call in calls] == ["Alex", "Sam", "Alex"]
    for call in calls:
        assert [message["role"] for message in call.args[0]] == ["user", "assistant"]
        assert call.args[0][1]["content"] == ""
        assert call.args[0][0]["time_stamp"] == "2023-05-25 13:14:00"
        assert call.kwargs["force_segment"] is False
        assert call.kwargs["force_extract"] is False
        assert call.kwargs["preserve_source_metadata"] is True
        assert "Personal Information Extractor" in call.kwargs["METADATA_GENERATE_PROMPT"]


def test_comparison_refuses_missing_session_timestamp(make_adapter):
    adapter, _, _ = make_adapter(comparison_mode=True)
    with pytest.raises(ValueError, match="timestamp"):
        adapter.add_chunk("Alex: No date")


def test_finalize_flushes_then_consolidates_exactly_once(make_adapter):
    adapter, _, memory = make_adapter(comparison_mode=True)
    adapter.add_chunk("Alex: End", "2023-05-25 13:14:00")
    events = []
    memory.add_memory.side_effect = lambda *a, **kw: events.append(("flush", a, kw))
    memory.construct_update_queue_all_entries.side_effect = lambda **kw: events.append(("queue", kw))
    memory.offline_update_all_entries.side_effect = lambda **kw: events.append(("update", kw))
    adapter.finalize()
    adapter.finalize()
    assert [event[0] for event in events] == ["flush", "queue", "update"]
    assert events[0][1] == ([],)
    assert events[0][2]["force_segment"] is True
    assert events[0][2]["force_extract"] is True
    assert events[1][1] == {"propagate_errors": True}
    assert events[2][1] == {"score_threshold": 0.9, "propagate_errors": True}


def test_failed_consolidation_cannot_be_marked_finalized(make_adapter):
    adapter, _, memory = make_adapter(comparison_mode=True)
    adapter.add_chunk("Alex: End", "2023-05-25 13:14:00")
    memory.offline_update_all_entries.side_effect = RuntimeError("update failed")
    with pytest.raises(RuntimeError, match="update failed"):
        adapter.finalize()
    assert adapter._finalized is False


def test_suppressed_compression_exception_is_not_success(make_adapter):
    adapter, _, memory = make_adapter(comparison_mode=True)

    def broken_compression(**kwargs):
        raise RuntimeError("compressor failed")

    # Reinstall around a failing edge, as the upstream compressor catches exceptions.
    memory.compressor.inner_compressor.compress_prompt = broken_compression
    adapter._install_comparison_guards()

    def swallowed(*args, **kwargs):
        try:
            memory.compressor.inner_compressor.compress_prompt(context="ignored")
        except RuntimeError:
            pass
        return {}

    memory.add_memory.side_effect = swallowed
    with pytest.raises(RuntimeError, match="compressor failed"):
        adapter.add_chunk("Alex: End", "2023-05-25 13:14:00")


def test_invalid_extraction_json_is_not_silent_empty_memory(make_adapter):
    adapter, _, memory = make_adapter(comparison_mode=True)
    memory.manager.generate_response = Mock(return_value=("broken JSON", {"total_tokens": 3}))
    adapter._install_comparison_guards()
    with pytest.raises(ValueError, match="JSON"):
        memory.manager.generate_response([], response_format={"type": "json_object"})


def test_finalize_does_not_allow_new_memory_after_consolidation(make_adapter):
    adapter, _, _ = make_adapter(comparison_mode=True)
    adapter.add_chunk("Alex: End", "2023-05-25 13:14:00")
    adapter.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        adapter.add_chunk("Alex: Another", "2023-05-25 13:14:00")


def test_native_final_segmenter_clears_only_verified_emitted_turns(make_adapter):
    from lightmem.factory.memory_buffer.sensory_memory import SenMemBufferManager

    adapter, _, memory = make_adapter(comparison_mode=True)
    sensory = SenMemBufferManager(tokenizer=SimpleNamespace(encode=lambda value: value.split()))
    before = [
        {"role": role, "content": f"turn {index}" if role == "user" else ""}
        for index in range(3) for role in ("user", "assistant")
    ]
    sensory.buffer = list(before)
    memory.senmem_buffer_manager = sensory
    adapter._install_comparison_guards()
    segmenter = SimpleNamespace(propose_cut=lambda messages: [1])
    embedder = SimpleNamespace(embed=Mock(side_effect=[[1, 0], [0, 1], [1, 0]]))
    segments = sensory.cut_with_segmenter(segmenter, embedder, True)
    assert [id(item) for segment in segments for item in segment] == [id(item) for item in before]
    assert sensory.buffer == []
    assert sensory.token_count == 0


@pytest.mark.parametrize("result", [
    None,
    {"usage": None, "cleaned_result": []},
    {"usage": {"total_tokens": 1}, "cleaned_result": [{"source_id": 7, "fact": "No source"}]},
    {"usage": {"total_tokens": 1}, "cleaned_result": [17]},
])
def test_failed_or_invalid_native_extraction_cannot_pass(make_adapter, result):
    adapter, _, memory = make_adapter(comparison_mode=True)
    memory.manager.meta_text_extract = Mock(return_value=[result])
    adapter._install_comparison_guards()
    batch = [[[{"role": "user", "sequence_number": 0}]]]
    with pytest.raises((ValueError, RuntimeError)):
        memory.manager.meta_text_extract(extract_list=batch)


def test_auxiliary_forward_tokens_logged_once_without_content(make_adapter, monkeypatch, tmp_path):
    import json
    import torch

    class Model(torch.nn.Module):
        config = SimpleNamespace(_commit_hash="pinned-revision")
        dtype = torch.float32

        def forward(self, input_ids, attention_mask=None):
            return input_ids

    adapter, _, memory = make_adapter(comparison_mode=True)
    model = Model()
    memory.compressor.inner_compressor.model = model
    adapter._install_auxiliary_meter("pinned-local-snapshot", "cpu")
    model(torch.tensor([[1, 2, 0], [3, 0, 0]]), attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]))
    model(input_ids=torch.tensor([[1, 2]]))
    stats = adapter.comparison_auxiliary_statistics
    assert stats["actual_forward_calls"] == 2
    assert stats["attention_mask_tokens"] == 5
    assert stats["padded_token_slots"] == 8
    assert stats["forward_wall_seconds"] >= 0
    journal = tmp_path / "auxiliary.jsonl"
    monkeypatch.setenv("METER_AUXILIARY_JOURNAL", str(journal))
    adapter.add_chunk("Alex: Private conversation must not reach metrics", "2023-05-25 13:14:00")
    adapter.finalize()
    adapter.close()
    rows = journal.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert "Private conversation" not in rows[0]
    assert json.loads(rows[0])["revision"] == "pinned-revision"
    assert not model._forward_hooks
    assert not model._forward_pre_hooks


def test_background_vector_update_failure_is_surfaced(make_adapter):
    adapter, _, memory = make_adapter(comparison_mode=True)
    memory.embedding_retriever.update = Mock(side_effect=RuntimeError("disk full"))
    adapter._install_comparison_guards()

    def swallowed(**kwargs):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(1) as executor:
            executor.map(lambda _: memory.embedding_retriever.update(), [1])

    memory.construct_update_queue_all_entries.side_effect = swallowed
    adapter.add_chunk("Alex: End", "2023-05-25 13:14:00")
    with pytest.raises(RuntimeError, match="disk full"):
        adapter.finalize()
    memory.offline_update_all_entries.assert_not_called()


def source_conversion_inputs(first_empty=False):
    """Two actual extraction batches have globally numbered user/assistant pairs."""
    import logging
    from lightmem.memory.utils import assign_sequence_numbers_with_timestamps

    batches = []
    for day, speaker in ((1, "Alex"), (2, "Sam")):
        batches.append([[{
            "role": role, "content": speaker if role == "user" else "",
            "session_time": f"2023-01-0{day} 12:00:00", "weekday": "Sun",
            "speaker_id": speaker.lower(), "speaker_name": speaker,
        } for role in ("user", "assistant")]])
    batches, timestamps, weekdays, speakers, topics = assign_sequence_numbers_with_timestamps(
        batches, topic_id_mapping=[[10], [20]],
    )
    arguments = dict(
        extracted_results=[
            {"cleaned_result": [] if first_empty else [{"source_id": 0, "fact": "First fact"}]},
            {"cleaned_result": [{"source_id": 1, "fact": "Second fact"}]},
        ],
        timestamps_list=timestamps, weekday_list=weekdays, speaker_list=speakers,
        topic_id_map=topics, max_source_ids=[0, 0],
        logger=logging.getLogger("lightmem-source-test"),
    )
    return batches, arguments


@pytest.mark.parametrize("first_empty", [False, True])
def test_global_source_id_keeps_same_topic_timestamp_and_speaker(first_empty):
    from lightmem.memory.utils import convert_extraction_results_to_memory_entries

    _, arguments = source_conversion_inputs(first_empty)
    entries = convert_extraction_results_to_memory_entries(
        **arguments, source_ids_by_batch=[[0], [1]],
    )
    assert len(entries) == (1 if first_empty else 2)
    second = entries[-1]
    assert second.memory == "Second fact"
    assert second.topic_id == 20
    assert second.time_stamp.startswith("2023-01-02T12:00:00")
    assert second.speaker_id == "sam"
    assert second.speaker_name == "Sam"


def test_legacy_source_conversion_retains_default_behavior():
    from lightmem.memory.utils import convert_extraction_results_to_memory_entries

    _, arguments = source_conversion_inputs()
    entries = convert_extraction_results_to_memory_entries(**arguments)
    assert entries[-1].topic_id == 10
    assert entries[-1].speaker_id == "sam"


def test_comparison_rejects_source_from_another_extraction_batch():
    from lightmem.memory.utils import convert_extraction_results_to_memory_entries

    _, arguments = source_conversion_inputs(first_empty=True)
    arguments["extracted_results"][1]["cleaned_result"][0]["source_id"] = 0
    with pytest.raises(ValueError, match="source_id"):
        convert_extraction_results_to_memory_entries(**arguments, source_ids_by_batch=[[0], [1]])


def test_comparison_rejects_missing_original_source_metadata():
    from lightmem.memory.utils import convert_extraction_results_to_memory_entries

    _, arguments = source_conversion_inputs(first_empty=True)
    arguments["timestamps_list"] = []
    with pytest.raises(ValueError, match="metadata"):
        convert_extraction_results_to_memory_entries(**arguments, source_ids_by_batch=[[0], [1]])


@pytest.mark.parametrize("operation", ["construct_update_queue_all_entries", "offline_update_all_entries"])
def test_native_background_worker_exception_is_propagated(operation):
    import logging

    memory = module.LightMemory.__new__(module.LightMemory)
    memory.logger = logging.getLogger("lightmem-worker-test")
    # This fails inside the worker before any guarded LLM or vector API call.
    memory.embedding_retriever = SimpleNamespace(get_all=lambda: [{"payload": {}}])
    with pytest.raises(KeyError, match="id"):
        getattr(memory, operation)(propagate_errors=True)


@pytest.mark.parametrize("first_empty", [False, True])
def test_native_add_memory_preserves_global_source_metadata(first_empty, monkeypatch):
    import logging
    import lightmem.memory.lightmem as native

    batches, arguments = source_conversion_inputs(first_empty)
    memory = module.LightMemory.__new__(module.LightMemory)
    memory.logger = logging.getLogger("lightmem-native-add-test")
    memory.config = SimpleNamespace(
        extraction_mode="flat", pre_compress=False, topic_segment=True,
        messages_use="user_only", metadata_generate=True, text_summary=True, update="offline",
    )
    memory.senmem_buffer_manager = SimpleNamespace(add_messages=lambda *args: [[{}]])
    memory.shortmem_buffer_manager = SimpleNamespace(add_segments=lambda *args: (2, batches))
    memory.segmenter = object()
    memory.text_embedder = object()
    extracted = deepcopy(arguments["extracted_results"])
    for result in extracted:
        result.update({"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                       "input_prompt": [], "output_prompt": "{}"})
    memory.manager = SimpleNamespace(meta_text_extract=lambda **kwargs: extracted)
    memory.token_stats = {key: 0 for key in (
        "add_memory_calls", "add_memory_prompt_tokens", "add_memory_completion_tokens", "add_memory_total_tokens",
    )}
    memory.offline_update = Mock()
    monkeypatch.setattr(native, "GLOBAL_TOPIC_IDX", 100)
    memory.add_memory([], preserve_source_metadata=True)
    entries = memory.offline_update.call_args.args[0]
    assert entries[-1].topic_id == 101
    assert entries[-1].speaker_name == "Sam"
    assert entries[-1].time_stamp.startswith("2023-01-02T12:00:00")


def test_worker_body_typeerror_is_strict_only():
    import logging

    memory = module.LightMemory.__new__(module.LightMemory)
    memory.logger = logging.getLogger("lightmem-worker-score-test")
    memory.embedding_retriever = SimpleNamespace(
        get_all=lambda: [{"id": "target", "payload": {"float_time_stamp": 1.0}, "vector": [1.0]}],
        search=lambda **kwargs: [{"id": "a", "score": None}, {"id": "b", "score": None}],
        update=Mock(),
    )
    assert memory.construct_update_queue_all_entries() is None
    with pytest.raises(TypeError):
        memory.construct_update_queue_all_entries(propagate_errors=True)
    memory.embedding_retriever.update.assert_not_called()
