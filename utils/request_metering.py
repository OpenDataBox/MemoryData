"""Opt-in request attribution for OpenAI-compatible HTTP traffic.

The benchmark runner executes one outer operation at a time while individual
memory implementations may fan that operation out across threads. This module
uses a process-wide immutable operation snapshot so those internal threads
inherit attribution without framework-specific patches. Concurrent outer
operations in one process are rejected.
"""

from __future__ import annotations

import functools
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TypeVar

import httpx


_ENV_NAMES = (
    "METER_RUN_ID",
    "METER_METHOD",
    "METER_LLM_PROXY_ORIGIN",
    "METER_EMBEDDING_PROXY_ORIGIN",
    "METER_TIMING_JOURNAL",
)
_REQUIRED_ENV_NAMES = _ENV_NAMES[:4]
_QA_SUFFIX_RE = re.compile(r"(?:^|_)qa_?(?P<index>\d+)$", flags=re.IGNORECASE)
_HEADER_NAMES = {
    "run_id": "X-Meter-Run-Id",
    "method": "X-Meter-Method",
    "phase": "X-Meter-Phase",
    "sample_id": "X-Meter-Sample-Id",
    "question_id": "X-Meter-Question-Id",
}

_STATE_LOCK = threading.RLock()
_JOURNAL_LOCK = threading.Lock()
_PATCH_LOCK = threading.Lock()
_SCOPE_OWNER_THREAD_ID: int | None = None
_SCOPE_STACK: list["OperationSnapshot"] = []
_PATCHED = False
_ORIGINAL_SYNC_SEND = httpx.Client.send
_ORIGINAL_ASYNC_SEND = httpx.AsyncClient.send

_T = TypeVar("_T")


@dataclass(frozen=True)
class MeterConfig:
    """Immutable process-level metering configuration."""

    enabled: bool = False
    run_id: str | None = None
    method: str | None = None
    proxy_origins: frozenset[tuple[str, str, int]] = frozenset()
    timing_journal: Path | None = None


@dataclass(frozen=True)
class OperationSnapshot:
    """Attribution shared by every HTTP request in one outer operation."""

    operation_id: str
    phase: str
    sample_id: str | None = None
    question_id: str | None = None


_CONFIG = MeterConfig()


def _normalized_origin(raw_url: str) -> tuple[str, str, int]:
    url = httpx.URL(raw_url)
    if url.scheme not in {"http", "https"} or not url.host:
        raise ValueError(f"Invalid HTTP proxy origin: {raw_url!r}")
    default_port = 443 if url.scheme == "https" else 80
    return url.scheme, url.host.lower(), url.port or default_port


def _load_config(environ: Mapping[str, str]) -> MeterConfig:
    values = {name: str(environ.get(name, "") or "").strip() for name in _ENV_NAMES}
    configured_names = [name for name, value in values.items() if value]
    if not configured_names:
        return MeterConfig()

    missing = [name for name in _REQUIRED_ENV_NAMES if not values[name]]
    if missing:
        raise ValueError(
            "Incomplete request metering configuration; missing "
            + ", ".join(missing)
        )

    origins = frozenset(
        {
            _normalized_origin(values["METER_LLM_PROXY_ORIGIN"]),
            _normalized_origin(values["METER_EMBEDDING_PROXY_ORIGIN"]),
        }
    )
    journal = (
        Path(values["METER_TIMING_JOURNAL"]).expanduser()
        if values["METER_TIMING_JOURNAL"]
        else None
    )
    return MeterConfig(
        enabled=True,
        run_id=values["METER_RUN_ID"],
        method=values["METER_METHOD"],
        proxy_origins=origins,
        timing_journal=journal,
    )


def _request_is_allowlisted(request: httpx.Request, config: MeterConfig) -> bool:
    url = request.url
    default_port = 443 if url.scheme == "https" else 80
    origin = (url.scheme, (url.host or "").lower(), url.port or default_port)
    return origin in config.proxy_origins


def _request_headers() -> dict[str, str]:
    with _STATE_LOCK:
        config = _CONFIG
        snapshot = _SCOPE_STACK[-1] if _SCOPE_STACK else None

    if not config.enabled:
        return {}

    values = {
        "run_id": config.run_id,
        "method": config.method,
        "phase": snapshot.phase if snapshot else None,
        "sample_id": snapshot.sample_id if snapshot else None,
        "question_id": snapshot.question_id if snapshot else None,
    }
    return {
        _HEADER_NAMES[key]: str(value)
        for key, value in values.items()
        if value is not None
    }


def _metered_sync_send(client: httpx.Client, request: httpx.Request, *args, **kwargs):
    with _STATE_LOCK:
        config = _CONFIG
    if config.enabled and _request_is_allowlisted(request, config):
        request.headers.update(_request_headers())
    return _ORIGINAL_SYNC_SEND(client, request, *args, **kwargs)


async def _metered_async_send(
    client: httpx.AsyncClient,
    request: httpx.Request,
    *args,
    **kwargs,
):
    with _STATE_LOCK:
        config = _CONFIG
    if config.enabled and _request_is_allowlisted(request, config):
        request.headers.update(_request_headers())
    return await _ORIGINAL_ASYNC_SEND(client, request, *args, **kwargs)


def install_request_metering(environ: Mapping[str, str] | None = None) -> bool:
    """Load opt-in settings and install process-wide HTTPX hooks.

    Supplying any metering environment variable requires the complete run,
    method, LLM-origin, and embedding-origin tuple. With no variables set this
    is a strict no-op.
    """

    global _CONFIG, _PATCHED
    config = _load_config(os.environ if environ is None else environ)
    with _STATE_LOCK:
        _CONFIG = config
    if not config.enabled:
        return False

    if config.timing_journal is not None:
        config.timing_journal.parent.mkdir(parents=True, exist_ok=True)

    with _PATCH_LOCK:
        if not _PATCHED:
            httpx.Client.send = _metered_sync_send
            httpx.AsyncClient.send = _metered_async_send
            _PATCHED = True
    return True


def current_operation_snapshot() -> OperationSnapshot | None:
    """Return the current immutable operation snapshot."""

    with _STATE_LOCK:
        return _SCOPE_STACK[-1] if _SCOPE_STACK else None


def current_sample_id(fallback: Any = None) -> str | None:
    """Prefer the enclosing operation's real sample ID over a context index."""

    snapshot = current_operation_snapshot()
    if snapshot is not None and snapshot.sample_id is not None:
        return snapshot.sample_id
    return None if fallback is None else str(fallback)


def resolve_sample_id(context_id: Any, eval_metadata: Mapping[str, Any] | None) -> str | None:
    """Resolve a dataset-native sample ID, then inherited ID, then context ID."""

    metadata_sample_id = (eval_metadata or {}).get("sample_id")
    if metadata_sample_id is not None and str(metadata_sample_id).strip():
        return str(metadata_sample_id)
    return current_sample_id(context_id)


def resolve_question_id(query_id: Any, eval_metadata: Mapping[str, Any] | None) -> str | None:
    """Prefer a dataset question ID and normalize a trailing QA index."""

    metadata_question_id = (eval_metadata or {}).get("question_id")
    if metadata_question_id is not None and str(metadata_question_id).strip():
        raw_question_id = str(metadata_question_id)
        match = _QA_SUFFIX_RE.search(raw_question_id)
        return match.group("index") if match else raw_question_id
    return None if query_id is None else str(query_id)


def _journal_event(snapshot: OperationSnapshot, event: str, **extra: Any) -> None:
    with _STATE_LOCK:
        config = _CONFIG
    if not config.enabled or config.timing_journal is None:
        return

    row = {
        "event": event,
        "operation_id": snapshot.operation_id,
        "run_id": config.run_id,
        "method": config.method,
        "phase": snapshot.phase,
        "sample_id": snapshot.sample_id,
        "question_id": snapshot.question_id,
        "monotonic_s": time.perf_counter(),
    }
    row.update(extra)
    serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    with _JOURNAL_LOCK:
        with config.timing_journal.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")


@contextmanager
def meter_operation(
    phase: str,
    *,
    sample_id: Any = None,
    question_id: Any = None,
) -> Iterator[OperationSnapshot | None]:
    """Attribute and time one serial outer benchmark operation.

    Nested scopes on the owning thread are supported. A second outer operation
    on another thread is rejected because internal worker threads consume the
    process-global immutable snapshot.
    """

    global _SCOPE_OWNER_THREAD_ID
    with _STATE_LOCK:
        config = _CONFIG
    if not config.enabled:
        yield None
        return

    owner_thread_id = threading.get_ident()
    with _STATE_LOCK:
        if (
            _SCOPE_OWNER_THREAD_ID is not None
            and _SCOPE_OWNER_THREAD_ID != owner_thread_id
        ):
            raise RuntimeError(
                "Request metering supports one serial outer operation per process"
            )
        parent = _SCOPE_STACK[-1] if _SCOPE_STACK else None
        inherited_sample_id = parent.sample_id if parent is not None else None
        inherited_question_id = parent.question_id if parent is not None else None
        snapshot = OperationSnapshot(
            operation_id=uuid.uuid4().hex,
            phase=str(phase),
            sample_id=(
                str(sample_id)
                if sample_id is not None
                else inherited_sample_id
            ),
            question_id=(
                str(question_id)
                if question_id is not None
                else inherited_question_id
            ),
        )
        _SCOPE_OWNER_THREAD_ID = owner_thread_id
        _SCOPE_STACK.append(snapshot)

    started_at = time.perf_counter()
    try:
        _journal_event(snapshot, "begin")
        yield snapshot
    except BaseException as exc:
        _journal_event(
            snapshot,
            "end",
            status="error",
            error_type=type(exc).__name__,
            duration_s=max(0.0, time.perf_counter() - started_at),
        )
        raise
    else:
        _journal_event(
            snapshot,
            "end",
            status="success",
            duration_s=max(0.0, time.perf_counter() - started_at),
        )
    finally:
        with _STATE_LOCK:
            if not _SCOPE_STACK or _SCOPE_STACK[-1] is not snapshot:
                raise RuntimeError("Request metering operation scopes exited out of order")
            _SCOPE_STACK.pop()
            if not _SCOPE_STACK:
                _SCOPE_OWNER_THREAD_ID = None


def metered_phase(phase: str) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorate a synchronous method with an inherited metering phase."""

    def decorator(function: Callable[..., _T]) -> Callable[..., _T]:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with meter_operation(phase):
                return function(*args, **kwargs)

        return wrapper

    return decorator


def _reset_for_tests() -> None:
    """Restore global HTTPX state. Intended only for isolated unit tests."""

    global _CONFIG, _PATCHED, _SCOPE_OWNER_THREAD_ID
    with _PATCH_LOCK:
        if _PATCHED:
            httpx.Client.send = _ORIGINAL_SYNC_SEND
            httpx.AsyncClient.send = _ORIGINAL_ASYNC_SEND
            _PATCHED = False
    with _STATE_LOCK:
        _CONFIG = MeterConfig()
        _SCOPE_STACK.clear()
        _SCOPE_OWNER_THREAD_ID = None


install_request_metering()
