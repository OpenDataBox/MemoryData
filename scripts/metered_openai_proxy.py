"""Local OpenAI pass-through with a content-free, per-attempt usage journal.

Attribution: X-Meter-Run-Id/Method/Phase/Sample-Id/Question-Id headers, or
/meter/<run>/<method>/<phase>/<sample>/<question>/v1 as the API base URL.
Headers override path attribution and are never forwarded upstream.
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx


ATTRIBUTION = {
    "run_id": "X-Meter-Run-Id", "method": "X-Meter-Method",
    "phase": "X-Meter-Phase", "sample_id": "X-Meter-Sample-Id",
    "question_id": "X-Meter-Question-Id",
}
POST_PATHS = {
    "/v1/chat/completions": "chat_completion",
    "/v1/completions": "completion", "/v1/embeddings": "embedding",
}
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
}
COMPARISON_MODEL = "Qwen/Qwen3.5-9B"
NUMERIC_SETTINGS = (
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
    "presence_penalty", "frequency_penalty", "seed", "n", "best_of",
    "max_tokens", "max_completion_tokens",
)


class ComparisonPolicyError(ValueError):
    """A request cannot use the explicitly selected comparison policy."""


def sampling_metadata(payload):
    values, invalid = {}, []
    for key in NUMERIC_SETTINGS:
        if key not in payload:
            continue
        value = payload[key]
        if value is None or (type(value) in (int, float) and math.isfinite(value)):
            values[key] = value
        else:
            invalid.append(key)
    thinking = {"enable_thinking": payload.get("enable_thinking")}
    template = payload.get("chat_template_kwargs")
    if isinstance(template, dict):
        thinking["chat_template_enable_thinking"] = template.get("enable_thinking")
    for key, value in thinking.items():
        if value is None or type(value) is bool:
            values[key] = value
        else:
            invalid.append(key)
    return values, invalid


def comparison_payload(payload, record):
    """Change only agreed inference settings; leave all content and tools intact."""
    policy = {"name": "qwen35_9b_comparison_v1", "applied": False}
    record["comparison_policy"] = policy
    if not isinstance(payload, dict):
        raise ComparisonPolicyError("json_object_required")
    policy["requested"], policy["invalid_requested_fields"] = sampling_metadata(payload)
    if payload.get("model") != COMPARISON_MODEL:
        raise ComparisonPolicyError("qwen35_9b_model_required")
    if payload.get("stream") is not None and payload["stream"] is not False:
        raise ComparisonPolicyError("streaming_not_allowed")
    template = payload.get("chat_template_kwargs")
    if template is not None and not isinstance(template, dict):
        raise ComparisonPolicyError("chat_template_kwargs_object_required")
    effective = dict(payload)
    effective.update(temperature=0, enable_thinking=False,
                     chat_template_kwargs={**(template or {}), "enable_thinking": False})
    policy["removed_output_caps"] = [key for key in ("max_tokens", "max_completion_tokens")
                                      if key in effective]
    for key in policy["removed_output_caps"]:
        del effective[key]
    policy["effective"], _ = sampling_metadata(effective)
    policy["effective"].update(max_tokens=None, max_completion_tokens=None)
    policy["applied"] = True
    return effective


def usage_status(usage, kind):
    if usage is None:
        return "missing"
    names = ["prompt_tokens", "total_tokens"]
    if kind != "embedding":
        names.append("completion_tokens")
    if not isinstance(usage, dict) or any(
        type(usage.get(name)) is not int or usage[name] < 0 for name in names
    ):
        return "invalid"
    completion = usage.get("completion_tokens", 0)
    if type(completion) is not int or completion < 0:
        return "invalid"
    return ("reported" if usage["prompt_tokens"] + completion == usage["total_tokens"]
            else "invalid")


def metadata_id(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,512}", value) else None


def numeric_usage(value):
    """Accept numeric usage metadata, never arbitrary response text."""
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", key)
               and (item is None or type(item) is int or numeric_usage(item))
               for key, item in value.items())


class ResponseMeter:
    """Parse accounting metadata, retaining no generated text."""

    def __init__(self, record):
        self.record = record
        self.buffer = b""

    def accept(self, payload):
        if not isinstance(payload, dict):
            self.record["error_type"] = "invalid_json_response"
            return
        for source, target in (("id", "response_id"), ("model", "response_model")):
            if payload.get(source) is not None:
                self.record[target] = metadata_id(payload[source])
                if self.record[target] is None:
                    self.record["error_type"] = "invalid_json_response"
        if payload.get("usage") is not None:
            if numeric_usage(payload["usage"]):
                self.record["usage"] = payload["usage"]
            else:
                self.record["usage_invalid"] = True
        for choice in payload.get("choices", []):
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason in ("stop", "length", "tool_calls", "content_filter", "function_call"):
                    if reason not in self.record["finish_reasons"]:
                        self.record["finish_reasons"].append(reason)
                else:
                    self.record["error_type"] = "invalid_json_response"
        if payload.get("error") is not None and self.record["http_status"] < 400:
            self.record["error_type"] = "upstream_stream_error"

    def feed(self, data):
        self.buffer += data
        while True:
            boundary = re.search(br"\r\n\r\n|\n\n|\r\r", self.buffer)
            if boundary is None:
                return
            event, self.buffer = self.buffer[:boundary.start()], self.buffer[boundary.end():]
            data = b"\n".join(line[5:].removeprefix(b" ") for line in event.splitlines()
                              if line.startswith(b"data:"))
            if not data:
                continue
            if data.strip() == b"[DONE]":
                self.record["stream_complete"] = True
                continue
            try:
                self.accept(json.loads(data))
            except (ValueError, TypeError, AttributeError):
                self.record["error_type"] = "sse_parse_error"


class MeteredProxy(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, upstream_base_url, journal_path, run_id, method, timeout=600,
                 comparison_policy=False):
        parsed = urlsplit(upstream_base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Use an absolute HTTP(S) upstream URL without credentials or query.")
        self.upstream = upstream_base_url.rstrip("/")
        if not parsed.path.rstrip("/").endswith("/v1"):
            self.upstream += "/v1"
        self.defaults = {"run_id": run_id, "method": method}
        self.comparison_policy = comparison_policy
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal = self.journal_path.open("a", encoding="utf-8")
        self.journal_lock = threading.Lock()
        self.activity = threading.Condition(self.journal_lock)
        self.in_flight = 0
        self.closing = False
        self.journal_error = None
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10)),
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=32),
            follow_redirects=False, trust_env=False,
        )
        super().__init__(address, ProxyHandler)

    def begin_attempt(self):
        with self.activity:
            if self.closing or self.journal_error:
                return False
            self.in_flight += 1
            return True

    def health(self):
        with self.activity:
            return {"status": "journal_error" if self.journal_error else
                    "draining" if self.closing else "ok",
                    "in_flight": self.in_flight, "drained": self.in_flight == 0}

    def record(self, record):
        with self.activity:
            try:
                self.journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.journal.flush()
                os.fsync(self.journal.fileno())
            except OSError as exc:
                self.journal_error = type(exc).__name__
                print(f"Usage journal failed: {self.journal_error}; rejecting new inference.",
                      file=sys.stderr, flush=True)

    def end_attempt(self):
        with self.activity:
            self.in_flight -= 1
            self.activity.notify_all()

    def server_close(self):
        with self.activity:
            self.closing = True
            self.activity.wait_for(lambda: self.in_flight == 0)
        super().server_close()
        self.client.close()
        self.journal.close()


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Avoid logging query strings or user-supplied headers.
        pass

    def _route(self):
        path = urlsplit(self.path).path
        attribution = dict.fromkeys(ATTRIBUTION)
        attribution.update(self.server.defaults)
        if path.startswith("/meter/"):
            parts = path.split("/")
            if len(parts) < 9 or parts[7] != "v1":
                return None, attribution
            attribution.update(zip(ATTRIBUTION, (unquote(value) for value in parts[2:7])))
            path = "/" + "/".join(parts[7:])
        for field, header in ATTRIBUTION.items():
            if header in self.headers:
                attribution[field] = self.headers[header]
        return path, attribution

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        transfer = self.headers.get("Transfer-Encoding", "").lower()
        if transfer == "chunked":
            if "Content-Length" in self.headers:
                raise ValueError("Conflicting body framing")
            chunks = []
            while True:
                size = int(self.rfile.readline().split(b";", 1)[0], 16)
                if size < 0:
                    raise ValueError("Invalid chunk length")
                if size == 0:
                    while self.rfile.readline() not in {b"\r\n", b"\n", b""}:
                        pass
                    return b"".join(chunks)
                chunk = self.rfile.read(size)
                if len(chunk) != size or self.rfile.read(2) != b"\r\n":
                    raise ValueError("Incomplete chunk")
                chunks.append(chunk)
        if transfer:
            raise ValueError("Unsupported transfer encoding")
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0:
            raise ValueError("Invalid Content-Length")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ConnectionError("Incomplete request body")
        return body

    def do_GET(self):
        path, _ = self._route()
        if path == "/health":
            health = self.server.health()
            self._json(200 if health["status"] == "ok" else 503, health)
        elif path == "/v1/models":
            try:
                response = self.server.client.get(self.server.upstream + "/models",
                                                  headers=self._request_headers())
                self._response_headers(response, len(response.content))
                self.wfile.write(response.content)
            except httpx.HTTPError:
                self._json(502, {"error": {"message": "Upstream transport error"}})
        else:
            self._json(404, {"error": {"message": "Unsupported endpoint"}})

    def _request_headers(self):
        excluded = HOP_HEADERS | {"host", "accept-encoding"}
        excluded.update(value.strip().lower() for value in self.headers.get("Connection", "").split(","))
        return {key: value for key, value in self.headers.items()
                if key.lower() not in excluded and not key.lower().startswith("x-meter-")}

    def _response_headers(self, response, length=None):
        self.send_response(response.status_code)
        excluded = HOP_HEADERS | {"content-encoding"}
        excluded.update(value.strip().lower() for value in response.headers.get("connection", "").split(","))
        for key, value in response.headers.multi_items():
            if key.lower() not in excluded:
                self.send_header(key, value)
        self.send_header("Content-Length" if length is not None else "Transfer-Encoding",
                         str(length) if length is not None else "chunked")
        self.end_headers()

    def _record_attempt(self, record, started):
        if record["client_disconnected"] and record["error_type"] is None:
            record["error_type"] = "client_disconnected"
        record["completed_at"] = time.time()
        record["seconds"] = time.monotonic() - started
        record["usage_status"] = ("invalid" if record["usage_invalid"] else
                                  usage_status(record["usage"], record["request_kind"]))
        record["success"] = (record["http_status"] is not None
                             and 200 <= record["http_status"] < 300
                             and record["error_type"] is None)
        self.server.record(record)

    def do_POST(self):
        path, attribution = self._route()
        supported = path in POST_PATHS
        if not supported or not self.server.begin_attempt():
            # Drain valid request framing before closing, otherwise Windows can
            # reset the connection before the client receives the error response.
            try:
                self._body()
            except (ValueError, ConnectionError):
                pass
            self.close_connection = True
            status = 503 if supported else 404
            self._json(status, {"error": {"message": "Usage journal unavailable" if status == 503
                                         else "Unsupported endpoint"}})
            return
        started = time.monotonic()
        record = {
            "schema_version": 1, "request_id": str(uuid.uuid4()),
            "timestamp": time.time(), **attribution, "endpoint": path,
            "request_kind": POST_PATHS[path], "model": None, "response_model": None,
            "response_id": None, "http_status": None, "usage": None,
            "usage_status": "missing", "usage_invalid": False,
            "finish_reasons": [], "stream": False,
            "stream_complete": None, "client_disconnected": False,
            "success": False, "error_type": None,
        }
        headers_sent = recorded = False
        strict_chat = self.server.comparison_policy and path == "/v1/chat/completions"
        try:
            body = self._body()
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                payload = None
            if isinstance(payload, dict):
                record["model"] = metadata_id(payload.get("model"))
                record["stream"] = payload.get("stream") is True
            if strict_chat:
                payload = comparison_payload(payload, record)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if isinstance(payload, dict):
                if record["stream"]:
                    options = payload.get("stream_options")
                    if options is None or isinstance(options, dict):
                        payload["stream_options"] = {**(options or {}), "include_usage": True}
                        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            url = self.server.upstream + path.removeprefix("/v1")
            query = urlsplit(self.path).query
            if query:
                url += "?" + query
            with self.server.client.stream("POST", url, content=body,
                                           headers=self._request_headers()) as response:
                record["http_status"] = response.status_code
                if response.status_code >= 300:
                    record["error_type"] = "http_error"
                meter = ResponseMeter(record)
                if "text/event-stream" in response.headers.get("content-type", ""):
                    record["stream_complete"] = False
                    if strict_chat:
                        for chunk in response.iter_bytes():
                            meter.feed(chunk)
                        record.update(error_type="comparison_unexpected_stream", client_http_status=502,
                                      recorded_before_error_response=True)
                        self._record_attempt(record, started)
                        recorded = True
                        self._json(502, {"error": {"message": "Comparison requires non-streaming model output",
                                                   "type": "comparison_unexpected_stream"}})
                        return
                    try:
                        self._response_headers(response)
                    except (BrokenPipeError, ConnectionError, OSError):
                        record["client_disconnected"] = True
                        self.close_connection = True
                    headers_sent = True
                    for chunk in response.iter_bytes():
                        meter.feed(chunk)
                        if not record["client_disconnected"] and chunk:
                            try:
                                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionError, OSError):
                                record["client_disconnected"] = True
                    if not record["stream_complete"] and record["error_type"] is None:
                        record["error_type"] = "incomplete_stream"
                    if not record["client_disconnected"]:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                else:
                    content = response.read()
                    try:
                        meter.accept(json.loads(content))
                    except (ValueError, TypeError, AttributeError):
                        if record["error_type"] is None:
                            record["error_type"] = "invalid_json_response"
                    if strict_chat and any(reason in {"length", "content_filter"}
                                           for reason in record["finish_reasons"]):
                        record.update(error_type="comparison_incomplete_output", client_http_status=502,
                                      recorded_before_error_response=True)
                        self._record_attempt(record, started)
                        recorded = True
                        self._json(502, {"error": {"message": "Comparison rejected incomplete model output",
                                                   "type": "comparison_incomplete_output"}})
                        return
                    self._response_headers(response, len(content))
                    headers_sent = True
                    self.wfile.write(content)
        except ComparisonPolicyError as exc:
            record.update(error_type="comparison_policy_rejected", client_http_status=400)
            record["comparison_policy"]["rejection"] = str(exc)
            self._json(400, {"error": {"message": str(exc), "type": "comparison_policy_rejected"}})
        except httpx.HTTPError as exc:
            record.update(error_type="upstream_transport_error", error_class=type(exc).__name__)
            self.close_connection = True
            if not headers_sent:
                self._json(502, {"error": {"message": "Upstream transport error"}})
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            record.update(client_disconnected=True, error_class=type(exc).__name__)
            self.close_connection = True
        except ValueError as exc:
            record.update(error_type="proxy_error", error_class=type(exc).__name__)
            self.close_connection = True
            if not headers_sent:
                self._json(400, {"error": {"message": "Invalid request body framing"}})
        finally:
            try:
                if not recorded:
                    self._record_attempt(record, started)
            finally:
                self.server.end_attempt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--comparison-policy", action="store_true",
                        help="Enforce Qwen3.5-9B non-thinking chat comparison settings; reject truncated output")
    args = parser.parse_args()
    with MeteredProxy((args.host, args.port), args.upstream_base_url,
                      args.journal, args.run_id, args.method, args.timeout,
                      comparison_policy=args.comparison_policy) as server:
        print(f"Metered API listening on {args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
