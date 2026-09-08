"""Exercise the proxy against an in-process upstream; no external model calls."""

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from scripts.metered_openai_proxy import COMPARISON_MODEL, MeteredProxy, ProxyHandler, ResponseMeter, usage_status


USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
         "prompt_tokens_details": {"cached_tokens": 4}}


def answer(response_id="response-1"):
    return {"id": response_id, "model": "test-model", "usage": USAGE,
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": None, "tool_calls": [{"id": "call-1", "type": "function",
                    "function": {"name": "remember", "arguments": "{\"city\":\"Seoul\"}"}}]}}]}


def json_response(handler, status, payload):
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("X-Upstream", "preserved")
    handler.end_headers()
    handler.wfile.write(body)


@contextmanager
def running_proxy(tmp_path, respond=None, comparison_policy=False):
    received = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            json_response(self, 200, {"data": [{"id": "test-model"}]})

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append({"path": self.path, "body": body, "headers": dict(self.headers)})
            if respond:
                respond(self, json.loads(body))
            else:
                json_response(self, 200, answer())

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream.daemon_threads = True
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = MeteredProxy(("127.0.0.1", 0), f"http://127.0.0.1:{upstream.server_port}/v1",
                         tmp_path / "usage.jsonl", "run-1", "a_mem", timeout=5,
                         comparison_policy=comparison_policy)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    client = httpx.Client(base_url=f"http://127.0.0.1:{proxy.server_port}",
                          trust_env=False, timeout=10)
    try:
        yield proxy, client, received
    finally:
        client.close()
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()
        proxy_thread.join(timeout=2)
        upstream_thread.join(timeout=2)


def records(proxy, expected=1):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with proxy.journal_lock:
            rows = [json.loads(line) for line in proxy.journal_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) >= expected:
            return rows
        time.sleep(0.01)
    pytest.fail(f"Expected {expected} accounting records; found {len(rows)}")


def test_json_tools_bytes_headers_and_usage_preserved_without_content_logging(tmp_path):
    body = b'{ "model": "test-model", "temperature": 0, "messages": [{"role":"user","content":"PRIVATE PROMPT"}], "tools": [] }'
    with running_proxy(tmp_path) as (proxy, client, received):
        result = client.post("/v1/chat/completions", content=body,
                             headers={"Authorization": "Bearer PRIVATE KEY", "Content-Type": "application/json",
                                      "X-Meter-Phase": "qa", "X-Meter-Sample-Id": "conv-26",
                                      "X-Meter-Question-Id": "7"})
        assert result.status_code == 200
        assert result.json() == answer()
        assert result.headers["X-Upstream"] == "preserved"
        assert received[0]["body"] == body
        assert received[0]["headers"]["Authorization"] == "Bearer PRIVATE KEY"
        assert not any(key.lower().startswith("x-meter-") for key in received[0]["headers"])
        row = records(proxy)[0]
        assert row["usage"] == USAGE and row["usage_status"] == "reported"
        assert row["finish_reasons"] == ["tool_calls"] and row["success"]
        assert (row["phase"], row["sample_id"], row["question_id"]) == ("qa", "conv-26", "7")
        assert row["seconds"] >= 0 and row["completed_at"] >= row["timestamp"]
        journal = proxy.journal_path.read_text(encoding="utf-8")
        assert "PRIVATE" not in journal and "remember" not in journal and "Seoul" not in journal


@pytest.mark.parametrize("status,payload,error,usage_state", [
    (400, {"error": {"message": "PRIVATE INVALID CONTENT"}}, "http_error", "missing"),
    (400, {"error": {"message": "Error after token consumption"}, "usage": USAGE}, "http_error", "reported"),
    (200, {"id": "r", "choices": [{"finish_reason": "stop"}]}, None, "missing"),
    (200, {"id": "r", "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 7}}, None, "invalid"),
])
def test_errors_and_absent_or_invalid_usage_are_explicit(tmp_path, status, payload, error, usage_state):
    with running_proxy(tmp_path, lambda h, p: json_response(h, status, payload)) as (proxy, client, _):
        response = client.post("/v1/chat/completions", json={"model": "test-model"})
        assert response.status_code == status and response.json() == payload
        row = records(proxy)[0]
        assert row["error_type"] == error and row["usage_status"] == usage_state
        assert row["usage"] == payload.get("usage")
        assert row["phase"] is None


def test_embeddings_and_scoped_attribution(tmp_path):
    payload = {"data": [{"embedding": [0.1, 0.2]}], "model": "MiniLM",
               "usage": {"prompt_tokens": 11, "total_tokens": 11}}
    with running_proxy(tmp_path, lambda h, p: json_response(h, 200, payload)) as (proxy, client, received):
        response = client.post("/meter/run-2/mem0/memory_add/conv-26/none/v1/embeddings",
                               json={"model": "MiniLM", "input": "PRIVATE EMBEDDING TEXT"},
                               headers={"X-Meter-Question-Id": "override"})
        assert response.json() == payload
        assert received[0]["path"] == "/v1/embeddings"
        row = records(proxy)[0]
        assert row["request_kind"] == "embedding" and row["response_id"] is None
        assert row["usage_status"] == "reported" and row["usage"] == payload["usage"]
        assert (row["run_id"], row["method"], row["phase"], row["sample_id"], row["question_id"]) == (
            "run-2", "mem0", "memory_add", "conv-26", "override")


def test_chunked_request_preserves_json_body(tmp_path):
    parts = [b'{"model":', b'"test-model","temperature":0}']
    with running_proxy(tmp_path) as (proxy, client, received):
        result = client.post("/v1/chat/completions", content=iter(parts),
                             headers={"Content-Type": "application/json"})
        assert result.status_code == 200
        assert received[0]["body"] == b"".join(parts)
        assert records(proxy)[0]["usage_status"] == "reported"


@pytest.mark.parametrize("usage", ["PRIVATE GENERATED RESPONSE", {"prompt_tokens": "PRIVATE GENERATED RESPONSE"},
                                  {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
                                   "prompt_tokens_details": {"text": "PRIVATE GENERATED RESPONSE"}}])
def test_malformed_metadata_cannot_log_generated_content(tmp_path, usage):
    payload = {"id": {"text": "PRIVATE RESPONSE ID"}, "model": ["PRIVATE MODEL"], "usage": usage,
               "choices": [{"finish_reason": {"text": "PRIVATE FINISH REASON"}}]}
    with running_proxy(tmp_path, lambda h, p: json_response(h, 200, payload)) as (proxy, client, _):
        response = client.post("/v1/chat/completions", json={"model": {"text": "PRIVATE REQUEST MODEL"}})
        assert response.json() == payload
        row = records(proxy)[0]
        assert row["usage"] is None and row["usage_status"] == "invalid"
        assert row["model"] is None and row["response_id"] is None and row["response_model"] is None
        assert row["finish_reasons"] == []
        assert "PRIVATE" not in proxy.journal_path.read_text(encoding="utf-8")


def test_concurrent_requests_keep_independent_attribution(tmp_path):
    with running_proxy(tmp_path, lambda h, p: json_response(h, 200, answer(p["tag"]))) as (proxy, client, _):
        def request(index):
            return client.post("/v1/chat/completions", json={"model": "test-model", "tag": str(index)},
                               headers={"X-Meter-Phase": "qa", "X-Meter-Question-Id": str(index)}).status_code
        with ThreadPoolExecutor(max_workers=8) as executor:
            assert list(executor.map(request, range(24))) == [200] * 24
        rows = records(proxy, 24)
        assert len({row["request_id"] for row in rows}) == 24
        assert all(row["response_id"] == row["question_id"] for row in rows)
        assert sum(row["usage"]["total_tokens"] for row in rows) == 240


def stream_bytes():
    chunks = [
        {"id": "stream-1", "model": "test-model", "usage": None,
         "choices": [{"delta": {"content": "서울 PRIVATE"}, "finish_reason": None}]},
        {"id": "stream-1", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"id": "stream-1", "choices": [], "usage": USAGE},
    ]
    return b"".join(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\r\n\r\n"
                    for chunk in chunks) + b"data: [DONE]\r\n\r\n"


def test_sse_is_transparent_and_requests_terminal_usage(tmp_path):
    expected = stream_bytes()

    def respond(handler, payload):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Content-Length", str(len(expected)))
        handler.end_headers()
        for index in range(0, len(expected), 7):
            handler.wfile.write(expected[index:index + 7])
            handler.wfile.flush()

    with running_proxy(tmp_path, respond) as (proxy, client, received):
        response = client.post("/v1/chat/completions", json={"model": "test-model", "stream": True,
                               "stream_options": {"include_usage": False, "continuous_usage_stats": True}})
        assert response.content == expected
        sent = json.loads(received[0]["body"])
        assert sent["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
        row = records(proxy)[0]
        assert row["usage"] == USAGE and row["stream_complete"] and row["success"]
        assert row["finish_reasons"] == ["stop"]
        assert "PRIVATE" not in proxy.journal_path.read_text(encoding="utf-8")


def test_sse_parser_handles_every_byte_boundary():
    row = {"http_status": 200, "error_type": None, "usage": None, "finish_reasons": []}
    meter = ResponseMeter(row)
    for byte in stream_bytes():
        meter.feed(bytes([byte]))
    assert row["usage"] == USAGE and row["stream_complete"]
    assert row["finish_reasons"] == ["stop"] and row["error_type"] is None


@pytest.mark.parametrize("body,error,complete", [
    (stream_bytes().removesuffix(b"data: [DONE]\r\n\r\n"), "incomplete_stream", False),
    (b"data: {broken}\n\ndata: [DONE]\n\n", "sse_parse_error", True),
    (b'data: {"error":{"message":"PRIVATE STREAM ERROR"}}\n\ndata: [DONE]\n\n',
     "upstream_stream_error", True),
])
def test_incomplete_or_invalid_stream_is_not_reported_successful(tmp_path, body, error, complete):
    def respond(handler, payload):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    with running_proxy(tmp_path, respond) as (proxy, client, _):
        response = client.post("/v1/chat/completions", json={"model": "test-model", "stream": True})
        assert response.content == body
        row = records(proxy)[0]
        assert not row["success"] and row["error_type"] == error
        assert row["stream_complete"] is complete
        assert "PRIVATE" not in proxy.journal_path.read_text(encoding="utf-8")


def test_client_disconnect_still_drains_stream_and_records_consumed_tokens(tmp_path):
    release = threading.Event()

    def respond(handler, payload):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(b": ready\n\n")
        handler.wfile.flush()
        assert release.wait(5)
        for _ in range(4):
            handler.wfile.write(b": " + b"x" * 65536 + b"\n\n")
            handler.wfile.flush()
        handler.wfile.write(stream_bytes())
        handler.wfile.flush()
        handler.close_connection = True

    with running_proxy(tmp_path, respond) as (proxy, _, _):
        peer = socket.create_connection(("127.0.0.1", proxy.server_port), timeout=5)
        body = b'{"model":"test-model","stream":true}'
        peer.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                     + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        assert peer.recv(4096)
        peer.shutdown(socket.SHUT_RDWR)
        peer.close()
        release.set()
        row = records(proxy)[0]
        assert row["client_disconnected"] and row["error_type"] == "client_disconnected"
        assert row["usage"] == USAGE and row["stream_complete"] and not row["success"]


def test_disconnect_before_sse_headers_still_accounts_terminal_usage(tmp_path, monkeypatch):
    original = ProxyHandler._response_headers

    def disconnected_headers(handler, response, length=None):
        if "text/event-stream" in response.headers.get("content-type", ""):
            raise BrokenPipeError("Client closed before headers")
        return original(handler, response, length)

    def respond(handler, payload):
        body = stream_bytes()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    monkeypatch.setattr(ProxyHandler, "_response_headers", disconnected_headers)
    with running_proxy(tmp_path, respond) as (proxy, client, _):
        with pytest.raises(httpx.RemoteProtocolError):
            client.post("/v1/chat/completions", json={"model": "test-model", "stream": True})
        row = records(proxy)[0]
        assert row["client_disconnected"] and row["usage"] == USAGE
        assert row["stream_complete"] and row["error_type"] == "client_disconnected"


def test_transport_failure_is_an_attempt_without_invented_tokens(tmp_path):
    with running_proxy(tmp_path) as (proxy, client, received):
        def fail(request):
            raise httpx.ConnectError("PRIVATE TRANSPORT DETAILS", request=request)
        old_client = proxy.client
        proxy.client = httpx.Client(transport=httpx.MockTransport(fail))
        old_client.close()
        response = client.post("/v1/chat/completions", json={"model": "test-model"},
                               headers={"X-Meter-Phase": "memory_add"})
        assert response.status_code == 502 and not received
        row = records(proxy)[0]
        assert row["request_id"] and row["phase"] == "memory_add"
        assert row["http_status"] is None and row["usage"] is None
        assert row["error_type"] == "upstream_transport_error" and not row["success"]


def test_health_models_and_rejected_paths_do_not_infer(tmp_path):
    with running_proxy(tmp_path) as (proxy, client, received):
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").json() == {"data": [{"id": "test-model"}]}
        assert client.post("/v1/files", json={}).status_code == 404
        assert not received and proxy.journal_path.read_text() == ""
        proxy.journal_error = "OSError"
        assert client.get("/health").status_code == 503
        assert client.post("/v1/chat/completions", json={}).status_code == 503
        assert not received


def test_journal_durability_failure_blocks_new_inference(tmp_path, monkeypatch):
    with running_proxy(tmp_path) as (proxy, client, received):
        def fail_fsync(fd):
            raise OSError("Disk unavailable")
        monkeypatch.setattr("scripts.metered_openai_proxy.os.fsync", fail_fsync)
        assert client.post("/v1/chat/completions", json={"model": "test-model"}).status_code == 200
        records(proxy)
        assert proxy.journal_error == "OSError"
        assert client.get("/health").status_code == 503
        assert client.post("/v1/chat/completions", json={"model": "test-model"}).status_code == 503
        assert len(received) == 1


def test_health_exposes_durable_drain_state(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def respond(handler, payload):
        entered.set()
        assert release.wait(5)
        json_response(handler, 200, answer())

    with running_proxy(tmp_path, respond) as (proxy, client, _):
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(client.post, "/v1/chat/completions", json={"model": "test-model"})
            try:
                assert entered.wait(5)
                assert client.get("/health").json() == {"status": "ok", "in_flight": 1, "drained": False}
            finally:
                release.set()
            assert pending.result(timeout=5).status_code == 200
        records(proxy)
        assert client.get("/health").json() == {"status": "ok", "in_flight": 0, "drained": True}


def test_server_close_waits_for_active_accounting(tmp_path):
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()

    def respond(handler, payload):
        entered.set()
        assert release.wait(5)
        json_response(handler, 200, answer())

    with running_proxy(tmp_path, respond) as (proxy, client, _):
        def close_proxy():
            closing.set()
            proxy.server_close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = executor.submit(client.post, "/v1/chat/completions", json={"model": "test-model"})
            assert entered.wait(5)
            proxy.shutdown()
            closed = executor.submit(close_proxy)
            try:
                assert closing.wait(5)
                assert not closed.done() and not proxy.journal.closed
            finally:
                release.set()
            assert pending.result(timeout=5).status_code == 200
            closed.result(timeout=5)
        assert proxy.journal.closed and proxy.in_flight == 0
        assert records(proxy)[0]["usage"] == USAGE


def test_comparison_policy_changes_only_agreed_inference_fields_and_logs_changes(tmp_path):
    payload = {
        "model": COMPARISON_MODEL, "temperature": 0.8, "top_p": 0.9, "seed": 42,
        "max_tokens": 2048, "max_completion_tokens": 4096, "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True, "other_option": "PRIVATE TEMPLATE"},
        "messages": [{"role": "user", "content": "PRIVATE PROMPT"}],
        "tools": [{"type": "function", "function": {"name": "remember", "parameters": {"type": "object"}}}],
        "tool_choice": "auto", "response_format": {"type": "json_schema", "json_schema": {"name": "result"}},
    }
    with running_proxy(tmp_path, comparison_policy=True) as (proxy, client, received):
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200 and response.json() == answer()
        effective = json.loads(received[0]["body"])
        for key in ("messages", "tools", "tool_choice", "response_format", "top_p", "seed", "model"):
            assert effective[key] == payload[key]
        assert effective["temperature"] == 0 and effective["enable_thinking"] is False
        assert effective["chat_template_kwargs"] == {"enable_thinking": False, "other_option": "PRIVATE TEMPLATE"}
        assert "max_tokens" not in effective and "max_completion_tokens" not in effective
        row = records(proxy)[0]
        policy = row["comparison_policy"]
        assert policy["applied"] and policy["name"] == "qwen35_9b_comparison_v1"
        assert policy["requested"]["temperature"] == 0.8 and policy["requested"]["max_tokens"] == 2048
        assert policy["requested"]["max_completion_tokens"] == 4096
        assert policy["requested"]["enable_thinking"] is True
        assert policy["requested"]["chat_template_enable_thinking"] is True
        assert policy["effective"]["temperature"] == 0 and policy["effective"]["top_p"] == 0.9
        assert policy["effective"]["enable_thinking"] is False
        assert policy["effective"]["chat_template_enable_thinking"] is False
        assert policy["effective"]["max_tokens"] is None and policy["effective"]["max_completion_tokens"] is None
        assert policy["removed_output_caps"] == ["max_tokens", "max_completion_tokens"]
        assert row["usage"] == USAGE
        assert "PRIVATE" not in proxy.journal_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("payload,rejection", [
    ({"model": "different-model"}, "qwen35_9b_model_required"),
    ({}, "qwen35_9b_model_required"),
    ([], "json_object_required"),
    ({"model": COMPARISON_MODEL, "stream": True}, "streaming_not_allowed"),
    ({"model": COMPARISON_MODEL, "stream": "true"}, "streaming_not_allowed"),
    ({"model": COMPARISON_MODEL, "chat_template_kwargs": "PRIVATE INVALID TEMPLATE"},
     "chat_template_kwargs_object_required"),
])
def test_comparison_rejects_wrong_model_or_stream_before_inference(tmp_path, payload, rejection):
    with running_proxy(tmp_path, comparison_policy=True) as (proxy, client, received):
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 400 and not received
        row = records(proxy)[0]
        assert row["http_status"] is None and row["client_http_status"] == 400
        assert row["error_type"] == "comparison_policy_rejected" and row["usage"] is None
        assert row["comparison_policy"]["rejection"] == rejection
        assert row["comparison_policy"]["applied"] is False
        assert "PRIVATE" not in proxy.journal_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", ["/v1/embeddings", "/v1/completions"])
def test_comparison_leaves_non_chat_requests_untouched(tmp_path, path):
    body = b'{ "model":"MiniLM", "input":"PRIVATE TEXT", "temperature":0.7, "max_tokens":10 }'
    with running_proxy(tmp_path, comparison_policy=True) as (proxy, client, received):
        assert client.post(path, content=body, headers={"Content-Type": "application/json"}).status_code == 200
        assert received[0]["body"] == body
        assert "comparison_policy" not in records(proxy)[0]


@pytest.mark.parametrize("reason", ["length", "content_filter"])
@pytest.mark.parametrize("enabled", [False, True])
def test_incomplete_output_is_durably_recorded_before_policy_failure(tmp_path, monkeypatch, reason, enabled):
    payload = answer()
    payload["choices"][0].update(finish_reason=reason, message={"content": "PRIVATE PARTIAL RESPONSE"})
    reply_checks = []
    original = ProxyHandler._json

    def checked_error(handler, status, body):
        if status == 502 and body.get("error", {}).get("type") == "comparison_incomplete_output":
            prior = [json.loads(line) for line in handler.server.journal_path.read_text(encoding="utf-8").splitlines()]
            assert len(prior) == 1 and prior[0]["usage"] == USAGE
            assert prior[0]["http_status"] == 200 and prior[0]["client_http_status"] == 502
            assert handler.server.in_flight == 1
            reply_checks.append(True)
        return original(handler, status, body)

    monkeypatch.setattr(ProxyHandler, "_json", checked_error)
    with running_proxy(tmp_path, lambda h, p: json_response(h, 200, payload), enabled) as (proxy, client, _):
        response = client.post("/v1/chat/completions", json={"model": COMPARISON_MODEL})
        row = records(proxy)[0]
        assert row["usage"] == USAGE and row["finish_reasons"] == [reason]
        if enabled:
            assert response.status_code == 502 and reply_checks == [True]
            assert row["error_type"] == "comparison_incomplete_output" and not row["success"]
            assert "PRIVATE" not in response.text
        else:
            assert response.status_code == 200 and response.json() == payload and not reply_checks
            assert row["success"] and "comparison_policy" not in row
        assert "PRIVATE" not in proxy.journal_path.read_text(encoding="utf-8")


def test_comparison_unexpected_upstream_stream_is_drained_and_rejected(tmp_path):
    def respond(handler, payload):
        body = stream_bytes()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    with running_proxy(tmp_path, respond, True) as (proxy, client, _):
        response = client.post("/v1/chat/completions", json={"model": COMPARISON_MODEL})
        assert response.status_code == 502
        row = records(proxy)[0]
        assert row["usage"] == USAGE and row["stream_complete"]
        assert row["error_type"] == "comparison_unexpected_stream" and not row["success"]


def test_comparison_missing_usage_remains_unknown_for_final_gate(tmp_path):
    payload = answer()
    del payload["usage"]
    with running_proxy(tmp_path, lambda h, p: json_response(h, 200, payload), True) as (proxy, client, _):
        assert client.post("/v1/chat/completions", json={"model": COMPARISON_MODEL}).status_code == 200
        row = records(proxy)[0]
        assert row["usage"] is None and row["usage_status"] == "missing"
        assert row["comparison_policy"]["applied"]


@pytest.mark.parametrize("value", [True, -1, 1.0, "1", None])
def test_invalid_numeric_usage_does_not_look_measured(value):
    assert usage_status({"prompt_tokens": value, "completion_tokens": 0, "total_tokens": 1},
                        "chat_completion") == "invalid"
