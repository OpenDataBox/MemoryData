import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from utils import request_metering


@pytest.fixture(autouse=True)
def reset_request_metering(monkeypatch):
    request_metering._reset_for_tests()
    for name in (
        "METER_RUN_ID",
        "METER_METHOD",
        "METER_LLM_PROXY_ORIGIN",
        "METER_EMBEDDING_PROXY_ORIGIN",
        "METER_TIMING_JOURNAL",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    request_metering._reset_for_tests()


def _enable(monkeypatch, journal=None):
    monkeypatch.setenv("METER_RUN_ID", "run-7")
    monkeypatch.setenv("METER_METHOD", "langmem")
    monkeypatch.setenv("METER_LLM_PROXY_ORIGIN", "http://127.0.0.1:9001/v1")
    monkeypatch.setenv("METER_EMBEDDING_PROXY_ORIGIN", "http://127.0.0.1:9002/v1")
    if journal is not None:
        monkeypatch.setenv("METER_TIMING_JOURNAL", str(journal))
    assert request_metering.install_request_metering()


def test_internal_threads_receive_an_immutable_operation_snapshot(monkeypatch):
    _enable(monkeypatch)
    captured = []

    def handler(request):
        captured.append(dict(request.headers))
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with request_metering.meter_operation(
            "qa", sample_id="sample-3", question_id="12"
        ):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        client.post,
                        "http://127.0.0.1:9001/v1/chat/completions",
                        json={"request": index},
                    )
                    for index in range(8)
                ]
                for future in futures:
                    assert future.result().status_code == 200

    assert len(captured) == 8
    for headers in captured:
        assert headers["x-meter-run-id"] == "run-7"
        assert headers["x-meter-method"] == "langmem"
        assert headers["x-meter-phase"] == "qa"
        assert headers["x-meter-sample-id"] == "sample-3"
        assert headers["x-meter-question-id"] == "12"


def test_async_httpx_send_is_metered(monkeypatch):
    _enable(monkeypatch)
    captured = []

    async def handler(request):
        captured.append(dict(request.headers))
        return httpx.Response(200, json={"ok": True})

    async def run_request():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with request_metering.meter_operation(
                "memory_add", sample_id="history-4"
            ):
                response = await client.post(
                    "http://127.0.0.1:9002/v1/embeddings",
                    json={"input": "not inspected by the meter"},
                )
                assert response.status_code == 200

    asyncio.run(run_request())

    assert captured[0]["x-meter-phase"] == "memory_add"
    assert captured[0]["x-meter-sample-id"] == "history-4"
    assert "x-meter-question-id" not in captured[0]


def test_headers_are_not_added_outside_the_proxy_allowlist(monkeypatch):
    _enable(monkeypatch)
    captured = []

    def handler(request):
        captured.append(dict(request.headers))
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with request_metering.meter_operation("qa", sample_id="3", question_id="9"):
            client.post("https://huggingface.co/api/models/example", content=b"opaque")

    headers = captured[0]
    assert not any(name.startswith("x-meter-") for name in headers)


def test_exception_restores_parent_scope_and_writes_timing_events(monkeypatch, tmp_path):
    journal = tmp_path / "meter" / "timing.jsonl"
    _enable(monkeypatch, journal=journal)

    with request_metering.meter_operation("initialize", sample_id="real-sample"):
        assert request_metering.current_sample_id("context-2") == "real-sample"
        with pytest.raises(RuntimeError, match="boom"):
            with request_metering.meter_operation("memory_finalize"):
                raise RuntimeError("boom")
        snapshot = request_metering.current_operation_snapshot()
        assert snapshot.phase == "initialize"
        assert snapshot.sample_id == "real-sample"

    assert request_metering.current_operation_snapshot() is None
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [(row["event"], row["phase"]) for row in rows] == [
        ("begin", "initialize"),
        ("begin", "memory_finalize"),
        ("end", "memory_finalize"),
        ("end", "initialize"),
    ]
    assert rows[2]["status"] == "error"
    assert rows[2]["error_type"] == "RuntimeError"
    assert rows[2]["duration_s"] >= 0
    assert "boom" not in journal.read_text(encoding="utf-8")


def test_begin_journal_failure_does_not_leak_operation_scope(monkeypatch):
    _enable(monkeypatch)
    original_journal_event = request_metering._journal_event

    def fail_begin(snapshot, event, **extra):
        if event == "begin":
            raise OSError("journal unavailable")
        return original_journal_event(snapshot, event, **extra)

    monkeypatch.setattr(request_metering, "_journal_event", fail_begin)
    with pytest.raises(OSError, match="journal unavailable"):
        with request_metering.meter_operation("qa", sample_id="sample-a"):
            pass

    assert request_metering.current_operation_snapshot() is None
    monkeypatch.setattr(request_metering, "_journal_event", original_journal_event)
    with request_metering.meter_operation("qa", sample_id="sample-b"):
        assert request_metering.current_sample_id() == "sample-b"


def test_query_ids_prefer_dataset_metadata_and_normalize_qa_suffix(monkeypatch):
    _enable(monkeypatch)
    assert request_metering.resolve_sample_id(2, {"sample_id": "locomo-8"}) == "locomo-8"
    assert request_metering.resolve_question_id(
        99, {"question_id": "locomo-8_qa_17"}
    ) == "17"
    assert request_metering.resolve_question_id(
        99, {"question_id": "dataset-native-id"}
    ) == "dataset-native-id"
    assert request_metering.resolve_question_id(99, {}) == "99"


def test_partial_environment_configuration_fails_loudly(monkeypatch):
    monkeypatch.setenv("METER_RUN_ID", "run-without-origins")
    with pytest.raises(ValueError, match="Incomplete request metering configuration"):
        request_metering.install_request_metering()


def test_concurrent_outer_operations_are_rejected(monkeypatch):
    _enable(monkeypatch)
    entered = threading.Event()
    release = threading.Event()

    def hold_outer_operation():
        with request_metering.meter_operation("qa", sample_id="sample-a"):
            entered.set()
            assert release.wait(timeout=5)

    thread = threading.Thread(target=hold_outer_operation)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="one serial outer operation"):
            with request_metering.meter_operation("qa", sample_id="sample-b"):
                pass
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
