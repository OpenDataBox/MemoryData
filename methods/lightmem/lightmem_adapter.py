"""Repository adapter for the vendored LightMem source."""

from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from openai import OpenAI
from utils.locomo_utils import (
    build_locomo_storage_text,
    parse_locomo_metadata,
    parse_locomo_source_ids,
    strip_locomo_metadata,
)


CURRENT_DIR = Path(__file__).resolve().parent
LIGHTMEM_SRC = CURRENT_DIR / "source"
if str(LIGHTMEM_SRC) not in sys.path:
    sys.path.insert(0, str(LIGHTMEM_SRC))

from lightmem.memory.lightmem import LightMemory
from lightmem.memory.utils import MemoryEntry


class LightMemAdapter:
    """Compatibility wrapper for using LightMem in repository evaluations."""

    _SESSION_HEADER_RE = re.compile(
        r"^Session\s+\S+(?:\s*\((?P<session_time>.+)\))?$",
        flags=re.IGNORECASE,
    )
    _SPEAKER_LINE_RE = re.compile(
        r"^(?P<speaker>[A-Za-z][A-Za-z0-9 _/-]{0,63}):\s*(?P<content>.*)$"
    )
    _ASSISTANT_SPEAKERS = {
        "assistant",
        "bot",
        "ai",
        "agent",
        "system",
        "gpt",
        "chatgpt",
    }

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: Optional[str],
        embedding_base_url: Optional[str],
        db_path: str,
        collection_name: str,
        embedding_model: str,
        embedding_dims: int,
        retrieve_num: int,
        ingest_mode: str = "direct",
        memory_manager_backend: str = "openai",
        embedding_backend: str = "openai",
        qdrant_on_disk: bool = True,
        messages_use: str = "user_only",
        metadata_generate: bool = False,
        text_summary: bool = False,
        pre_compress: bool = False,
        topic_segment: bool = False,
        index_strategy: str = "embedding",
        retrieve_strategy: str = "embedding",
        update_mode: str = "offline",
        comparison_mode: bool = False,
        compressor_model: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        compression_device: str = "cpu",
    ) -> None:
        self.comparison_mode = comparison_mode
        self._finalized = False
        self._pipeline_dirty = False
        self._last_session_time = ""
        self._comparison_errors: list[Exception] = []
        self._comparison_error_lock = threading.Lock()
        self._auxiliary_hooks: list[Any] = []
        self._auxiliary_written = False
        if comparison_mode:
            if memory_manager_backend != "openai" or embedding_backend != "openai":
                raise ValueError("Comparison mode requires metered OpenAI-compatible backends")
            if not base_url or not embedding_base_url:
                raise ValueError("Comparison mode requires explicit LLM and embedding endpoints")
            ingest_mode = "pipeline"
            pre_compress = topic_segment = metadata_generate = text_summary = True
            messages_use = "user_only"
            index_strategy = retrieve_strategy = "embedding"
            update_mode = "offline"
            self._extraction_prompt = self._load_locomo_extraction_prompt()
        self.model = model
        self.retrieve_num = retrieve_num
        self.ingest_mode = ingest_mode
        self.messages_use = messages_use
        self._memory_counter = 0

        db_root = Path(db_path)
        db_root.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(trust_env=False),
        )
        config_payload = {
            "pre_compress": pre_compress,
            "topic_segment": topic_segment,
            "metadata_generate": metadata_generate,
            "text_summary": text_summary,
            "messages_use": messages_use,
            "index_strategy": index_strategy,
            "retrieve_strategy": retrieve_strategy,
            "update": update_mode,
            "history_db_path": str(db_root / "history.db"),
            "kv_cache_path": str(db_root / "kv_cache.db"),
            "memory_manager": {
                "model_name": memory_manager_backend,
                "configs": {
                    "model": model,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                    "temperature": 0.0,
                    "max_tokens": 1024,
                },
            },
            "text_embedder": {
                "model_name": embedding_backend,
                "configs": {
                    "model": embedding_model,
                    "api_key": api_key,
                    "embedding_dims": embedding_dims,
                    "openai_base_url": embedding_base_url or base_url,
                    # Default local sentence-transformer models to CPU so LongBench
                    # runs do not contend for the chat GPU and OOM during init.
                    "model_kwargs": {"device": "cpu"} if embedding_backend == "huggingface" else {},
                },
            },
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": collection_name,
                    "embedding_model_dims": embedding_dims,
                    "path": str(db_root / "qdrant"),
                    "on_disk": qdrant_on_disk,
                },
            },
        }
        if pre_compress:
            config_payload["pre_compressor"] = {"model_name": "llmlingua-2"}
        if comparison_mode:
            config_payload.update({
                "precomp_topic_shared": True,
                "topic_segmenter": {"model_name": "llmlingua-2"},
                "extract_threshold": 0.1,
                "extraction_mode": "flat",
            })
            config_payload["memory_manager"]["configs"]["max_tokens"] = 16000
            config_payload["pre_compressor"]["configs"] = {
                "llmlingua_config": {
                    "model_name": compressor_model,
                    "device_map": compression_device,
                    "use_llmlingua2": True,
                },
                "compress_config": {"instruction": "", "rate": 0.6, "target_token": -1},
            }
        try:
            self.lightmem = LightMemory.from_config(config_payload)
        except Exception as exc:
            if embedding_backend == "huggingface" and "out of memory" in str(exc).lower():
                fallback_payload = deepcopy(config_payload)
                fallback_payload["text_embedder"]["configs"]["model_kwargs"] = {"device": "cpu"}
                self.lightmem = LightMemory.from_config(fallback_payload)
            else:
                raise
        if comparison_mode:
            # The upstream manager prioritizes OPENROUTER_API_KEY over its config.
            # Bind the explicitly supplied endpoint, never an ambient paid provider.
            old_client = self.lightmem.manager.client
            self.lightmem.manager.client = self.client
            old_client.close()
            self._install_comparison_guards()
            self._install_auxiliary_meter(compressor_model, compression_device)

    def add_chunk(self, content: str, timestamp: Optional[str] = None) -> None:
        if self.comparison_mode and self._finalized:
            raise RuntimeError("LightMem comparison memory is already finalized")
        parsed_chunk = self._parse_chunk_messages(content, timestamp)
        if self.ingest_mode == "pipeline":
            payload = [
                {
                    "role": message["role"],
                    "content": message["content"],
                    "time_stamp": message["time_stamp"],
                    "speaker_id": message["speaker_id"],
                    "speaker_name": message["speaker_name"],
                }
                for message in parsed_chunk["messages"]
            ]
            if self.comparison_mode:
                # The official LoCoMo builder supplies one user turn plus an empty
                # assistant placeholder. Native segmentation indexes paired turns.
                for message in payload:
                    if not message["time_stamp"]:
                        raise ValueError("LoCoMo comparison requires a session timestamp")
                    message["time_stamp"] = self._parse_timestamp(message["time_stamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    message["role"] = "user"
                    placeholder = dict(message, role="assistant", content="")
                    self._run_pipeline([message, placeholder], force=False)
                    self._memory_counter += 1
                    self._pipeline_dirty = True
                return
            self.lightmem.add_memory(payload, force_segment=True, force_extract=True)
            return

        selected_messages = self._select_messages_for_direct_ingest(parsed_chunk["messages"])
        rendered_text = self._render_messages(selected_messages)
        if not rendered_text:
            rendered_text = strip_locomo_metadata(content).strip() or str(content or "").strip()
        memory_text = self._attach_locomo_metadata(
            rendered_text,
            parsed_chunk["chunk_id"],
            parsed_chunk["source_ids"],
        )

        preferred_timestamp = (
            selected_messages[-1]["time_stamp"]
            if selected_messages
            else parsed_chunk["messages"][-1]["time_stamp"]
        )
        dt = self._parse_timestamp(preferred_timestamp)
        entry = MemoryEntry(
            time_stamp=dt.isoformat(timespec="seconds"),
            float_time_stamp=dt.timestamp(),
            weekday=dt.strftime("%a"),
            category="benchmark_context",
            subcategory="chunk",
            memory_class="verbatim_chunk",
            memory=memory_text,
            original_memory=memory_text,
            compressed_memory=memory_text,
            topic_id=self._memory_counter,
            topic_summary="Benchmark chunk",
            speaker_id="benchmark",
            speaker_name="Benchmark",
        )
        self._memory_counter += 1
        self.lightmem.offline_update([entry])

    def retrieve(self, question: str) -> list[str]:
        query_vector = self.lightmem.text_embedder.embed(question)
        results = self.lightmem.embedding_retriever.search(
            query_vector=query_vector,
            limit=self.retrieve_num,
            filters=None,
            return_full=True,
        )
        retrieved_memories = []
        for result in results:
            payload = result.get("payload", {})
            time_stamp = str(payload.get("time_stamp", "") or "").strip()
            weekday = str(payload.get("weekday", "") or "").strip()
            memory = str(payload.get("memory", "") or "").strip()
            prefix = " ".join(part for part in (time_stamp, weekday) if part).strip()
            if memory:
                if parse_locomo_source_ids(memory):
                    formatted_memory = f"{prefix}\n{memory}".strip() if prefix else memory
                else:
                    formatted_memory = f"{prefix} {memory}".strip()
                retrieved_memories.append(formatted_memory)
            elif prefix:
                retrieved_memories.append(prefix)
        return retrieved_memories

    def ask(self, question: str) -> str:
        retrieved_context = "\n".join(self.retrieve(question))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the question strictly based on the retrieved memory. "
                        "If the question is multiple-choice, reply with exactly one uppercase letter: A, B, C, or D. "
                        "Do not explain your answer. If the memory is insufficient, say that briefly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Retrieved memory:\n{retrieved_context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.0,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            } if "qwen3" in str(self.model or "").lower() else None,
        )
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    text_parts.append(item.strip())
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("content")
                    if isinstance(text_value, str) and text_value.strip():
                        text_parts.append(text_value.strip())
            if text_parts:
                return "\n".join(text_parts)
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
        return ""

    def finalize(self) -> None:
        if not self.comparison_mode or self._finalized or not self._pipeline_dirty:
            return
        # A separate empty-input flush retains any segments emitted by the final
        # real turn; upstream add_memory otherwise overwrites them when forced.
        self._run_pipeline([], force=True)
        self.lightmem.construct_update_queue_all_entries(propagate_errors=True)
        self._raise_comparison_errors()
        self.lightmem.offline_update_all_entries(score_threshold=0.9, propagate_errors=True)
        self._raise_comparison_errors()
        self._finalized = True
        self._write_auxiliary_statistics("complete")

    @staticmethod
    def _load_locomo_extraction_prompt() -> str:
        path = CURRENT_DIR / "upstream" / "experiments" / "locomo" / "prompts.py"
        # Read only the upstream literal; do not import benchmark scripts or run
        # arbitrary top-level code from a prompt file.
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "METADATA_GENERATE_PROMPT_locomo"
                for target in node.targets
            ):
                value = ast.literal_eval(node.value)
                if isinstance(value, str) and value.strip():
                    return value
        raise ValueError("Official LightMem LoCoMo extraction prompt is missing")

    def _raise_comparison_errors(self) -> None:
        with self._comparison_error_lock:
            errors = list(self._comparison_errors)
        if errors:
            raise RuntimeError(f"LightMem internal operation failed: {errors[0]}") from errors[0]

    def _run_pipeline(self, messages: list[dict[str, Any]], *, force: bool) -> None:
        self._raise_comparison_errors()
        self.lightmem.add_memory(
            messages,
            METADATA_GENERATE_PROMPT=self._extraction_prompt,
            force_segment=force,
            force_extract=force,
            preserve_source_metadata=True,
        )
        self._raise_comparison_errors()

    def _install_comparison_guards(self) -> None:
        """Surface errors the native compressor or background update can swallow."""
        def guard(function, *, validate_json=False):
            def wrapped(*args, **kwargs):
                try:
                    result = function(*args, **kwargs)
                    if validate_json and kwargs.get("response_format", {}).get("type") == "json_object":
                        raw, usage = result
                        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw).strip())
                        try:
                            parsed = json.loads(cleaned)
                        except json.JSONDecodeError as exc:
                            raise ValueError("LightMem returned invalid extraction/update JSON") from exc
                        if not isinstance(parsed, dict) or not usage:
                            raise ValueError("LightMem requires an object response with actual usage")
                        if "data" in parsed and not isinstance(parsed["data"], list):
                            raise ValueError("LightMem extraction data must be a list")
                        if "data" not in parsed and parsed.get("action") not in {"ignore", "update", "delete"}:
                            raise ValueError("LightMem response lacks extraction data or a valid update action")
                        if parsed.get("action") == "update" and not str(parsed.get("new_memory") or "").strip():
                            raise ValueError("LightMem update response lacks replacement memory")
                    return result
                except Exception as exc:
                    with self._comparison_error_lock:
                        self._comparison_errors.append(exc)
                    raise
            return wrapped

        manager = self.lightmem.manager
        manager.generate_response = guard(manager.generate_response, validate_json=True)
        manager._call_update_llm = guard(manager._call_update_llm)
        original_extract = manager.meta_text_extract

        def checked_extract(*args, **kwargs):
            results = original_extract(*args, **kwargs)
            batches = kwargs["extract_list"]
            if not isinstance(results, list) or len(results) != len(batches):
                raise ValueError("LightMem extraction returned an incomplete batch list")
            for batch, result in zip(batches, results):
                if not isinstance(result, dict) or not result.get("usage"):
                    raise RuntimeError("LightMem extraction failed without usage")
                allowed_sources = {
                    message["sequence_number"] // 2
                    for segment in batch for message in segment
                    if message["role"] == "user"
                }
                facts = result.get("cleaned_result")
                if not isinstance(facts, list):
                    raise ValueError("LightMem extraction facts must be a list")
                for fact in facts:
                    if not isinstance(fact, dict) or not str(fact.get("fact", "")).strip():
                        raise ValueError("LightMem extraction returned a malformed fact")
                    if int(fact.get("source_id", -1)) not in allowed_sources:
                        raise ValueError("LightMem extraction returned an invalid source_id")
            return results

        manager.meta_text_extract = checked_extract
        compressor = self.lightmem.compressor.inner_compressor
        compressor.compress_prompt = guard(compressor.compress_prompt)
        self.lightmem.text_embedder.embed = guard(self.lightmem.text_embedder.embed)
        retriever = self.lightmem.embedding_retriever
        for name in ("search", "insert", "update", "delete", "get_all"):
            setattr(retriever, name, guard(getattr(retriever, name)))

        sensory = self.lightmem.senmem_buffer_manager
        original_cut = sensory.cut_with_segmenter

        def cut_preserving_final_buffer(segmenter, embedder, force_segment=False):
            before = list(sensory.buffer)
            if force_segment and not before:
                return []
            segments = original_cut(segmenter, embedder, force_segment)
            if force_segment:
                emitted = [message for segment in segments for message in segment]
                if [id(message) for message in emitted] != [id(message) for message in before]:
                    raise RuntimeError("LightMem final segmentation did not preserve every buffered message")
                # Native forced cut deletes len(boundaries), not the emitted
                # message count. All original messages are verified emitted above.
                sensory.buffer.clear()
                sensory.token_count = 0
            return segments

        sensory.cut_with_segmenter = cut_preserving_final_buffer

    def _install_auxiliary_meter(self, model_path: str, device: str) -> None:
        model = self.lightmem.compressor.inner_compressor.model
        revision = getattr(model.config, "_commit_hash", None)
        if not revision and re.fullmatch(r"[0-9a-f]{40}", Path(model_path).name):
            revision = Path(model_path).name
        self.comparison_auxiliary_statistics = {
            "model": model_path,
            "revision": revision,
            "device": device,
            "dtype": str(model.dtype),
            "actual_forward_calls": 0,
            "attention_mask_tokens": 0,
            "padded_token_slots": 0,
            "forward_wall_seconds": 0.0,
            "scope": "shared_llmlingua_compression_and_topic_segmentation",
        }
        starts = []

        def before_forward(module, args, kwargs):
            starts.append(time.perf_counter())
            stats = self.comparison_auxiliary_statistics
            stats["actual_forward_calls"] += 1
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            mask = kwargs.get("attention_mask")
            if ids is not None:
                stats["padded_token_slots"] += int(ids.numel())
                stats["attention_mask_tokens"] += int(mask.sum().item()) if mask is not None else int(ids.numel())

        def after_forward(module, args, kwargs, output):
            self.comparison_auxiliary_statistics["forward_wall_seconds"] += time.perf_counter() - starts.pop()

        # The same module powers compression and segmentation: register once.
        self._auxiliary_hooks = [
            model.register_forward_pre_hook(before_forward, with_kwargs=True),
            model.register_forward_hook(after_forward, with_kwargs=True, always_call=True),
        ]

    def _write_auxiliary_statistics(self, status: str) -> None:
        journal = os.getenv("METER_AUXILIARY_JOURNAL")
        if not journal or self._auxiliary_written:
            return
        from utils.request_metering import current_sample_id

        row = {
            "schema_version": 1,
            "run_id": os.getenv("METER_RUN_ID"),
            "method": os.getenv("METER_METHOD", "lightmem"),
            "sample_id": current_sample_id(),
            "status": status,
            "conversation_turns_processed": self._memory_counter,
            **self.comparison_auxiliary_statistics,
        }
        with Path(journal).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        self._auxiliary_written = True

    def close(self) -> None:
        """Release the local Qdrant lock after all queries finish."""
        try:
            if getattr(self, "comparison_mode", False):
                self._write_auxiliary_statistics("complete" if self._finalized else "incomplete")
        finally:
            if getattr(self, "comparison_mode", False):
                for hook in self._auxiliary_hooks:
                    hook.remove()
                self._auxiliary_hooks.clear()
                self.client.close()
            retriever = getattr(self.lightmem, "embedding_retriever", None)
            client = getattr(retriever, "client", None)
            if callable(getattr(client, "close", None)):
                client.close()

    @staticmethod
    def _parse_timestamp(timestamp: Optional[str]) -> datetime:
        if not timestamp:
            return datetime.now()

        for fmt in (
            "%Y/%m/%d (%a) %H:%M:%S",
            "%Y/%m/%d (%a) %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%I:%M %p on %d %B, %Y",
        ):
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
        return datetime.fromisoformat(timestamp)

    @classmethod
    def _infer_role(cls, speaker_name: str) -> str:
        normalized = str(speaker_name or "").strip().lower()
        if normalized in cls._ASSISTANT_SPEAKERS or normalized.startswith("assistant"):
            return "assistant"
        return "user"

    def _parse_chunk_messages(self, content: str, timestamp: Optional[str]) -> dict[str, Any]:
        metadata_entries = parse_locomo_metadata(content)
        chunk_id = metadata_entries[0]["chunk_id"] if metadata_entries else None
        source_ids = metadata_entries[0]["source_ids"] if metadata_entries else parse_locomo_source_ids(content)
        body = strip_locomo_metadata(content).strip()
        if getattr(self, "comparison_mode", False):
            default_session_time = self._last_session_time or timestamp or ""
        else:
            default_session_time = timestamp or datetime.now().strftime("%Y/%m/%d (%a) %H:%M:%S")

        messages = []
        current_message = None
        current_session_label = ""
        current_session_time = default_session_time

        def flush_current_message():
            nonlocal current_message
            if current_message and str(current_message.get("content", "")).strip():
                messages.append(current_message)
            current_message = None

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            session_match = self._SESSION_HEADER_RE.match(line)
            if session_match:
                flush_current_message()
                current_session_label = line.split("(", 1)[0].strip()
                current_session_time = (
                    str(session_match.group("session_time") or "").strip()
                    or default_session_time
                )
                if getattr(self, "comparison_mode", False):
                    self._last_session_time = current_session_time
                continue

            speaker_match = self._SPEAKER_LINE_RE.match(line)
            if speaker_match:
                flush_current_message()
                speaker_name = speaker_match.group("speaker").strip()
                current_message = {
                    "role": self._infer_role(speaker_name),
                    "content": speaker_match.group("content").strip(),
                    "time_stamp": current_session_time,
                    "speaker_id": re.sub(r"[^A-Za-z0-9._-]+", "_", speaker_name).strip("_").lower() or "benchmark",
                    "speaker_name": speaker_name,
                    "session_label": current_session_label,
                }
                continue

            if current_message is None:
                current_message = {
                    "role": "user",
                    "content": line,
                    "time_stamp": current_session_time,
                    "speaker_id": "benchmark",
                    "speaker_name": "Benchmark",
                    "session_label": current_session_label,
                }
                continue

            current_message["content"] = f"{current_message['content']}\n{line}".strip()

        flush_current_message()
        if not messages:
            messages = [
                {
                    "role": "user",
                    "content": body or str(content or "").strip(),
                    "time_stamp": default_session_time,
                    "speaker_id": "benchmark",
                    "speaker_name": "Benchmark",
                    "session_label": "",
                }
            ]

        return {
            "chunk_id": chunk_id,
            "source_ids": source_ids,
            "messages": messages,
        }

    def _select_messages_for_direct_ingest(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.messages_use == "hybrid":
            return list(messages)
        if self.messages_use == "assistant_only":
            selected = [message for message in messages if message.get("role") == "assistant"]
            return selected or list(messages)
        selected = [message for message in messages if message.get("role") == "user"]
        return selected or list(messages)

    @staticmethod
    def _render_messages(messages: list[dict[str, Any]]) -> str:
        rendered_lines = []
        last_session_key = None
        for message in messages:
            session_label = str(message.get("session_label", "") or "").strip()
            session_time = str(message.get("time_stamp", "") or "").strip()
            session_key = (session_label, session_time)
            if session_label and session_key != last_session_key:
                header = session_label
                if session_time:
                    header = f"{header} ({session_time})"
                rendered_lines.append(header)
                last_session_key = session_key

            speaker_name = str(message.get("speaker_name", "") or "").strip()
            content = str(message.get("content", "") or "").strip()
            if not content:
                continue
            rendered_lines.append(f"{speaker_name}: {content}" if speaker_name else content)
        return "\n".join(rendered_lines).strip()

    @staticmethod
    def _attach_locomo_metadata(text: str, chunk_id: Optional[str], source_ids: list[str]) -> str:
        if chunk_id and source_ids:
            return build_locomo_storage_text(text, chunk_id, source_ids)
        return text
