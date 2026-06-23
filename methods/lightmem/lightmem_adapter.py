"""Repository adapter for the vendored LightMem source."""

from __future__ import annotations

from copy import deepcopy
import re
import sys
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
        r"^Session\s+\d+(?:\s*\((?P<session_time>.+)\))?$",
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
    ) -> None:
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
        try:
            self.lightmem = LightMemory.from_config(config_payload)
        except Exception as exc:
            if embedding_backend == "huggingface" and "out of memory" in str(exc).lower():
                fallback_payload = deepcopy(config_payload)
                fallback_payload["text_embedder"]["configs"]["model_kwargs"] = {"device": "cpu"}
                self.lightmem = LightMemory.from_config(fallback_payload)
            else:
                raise

    def add_chunk(self, content: str, timestamp: Optional[str] = None) -> None:
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
        return None

    @staticmethod
    def _parse_timestamp(timestamp: Optional[str]) -> datetime:
        if not timestamp:
            return datetime.now()

        for fmt in ("%Y/%m/%d (%a) %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
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
