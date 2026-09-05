"""Repository adapter for the EverOS HTTP API."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx


class EverOSAdapter:
    """Thin synchronous wrapper around the official EverOS REST API."""

    def __init__(
        self,
        *,
        base_url: str,
        group_id: str,
        group_name: str,
        api_key: Optional[str] = None,
        scene: str = "assistant",
        default_timezone: str = "UTC",
        retrieve_method: str = "rrf",
        memory_types: Optional[List[str]] = None,
        top_k: int = 10,
        timeout_seconds: float = 60.0,
        sync_mode: bool = True,
        chunk_time_gap_minutes: int = 360,
        user_id: str = "benchmark_user",
        user_name: str = "Benchmark User",
        assistant_id: str = "benchmark_assistant",
        assistant_name: str = "Benchmark Assistant",
    ) -> None:
        if not base_url:
            raise ValueError("EverOS base_url is required.")

        self.base_url = self._normalize_memories_url(base_url)
        self.search_url = f"{self.base_url}/search"
        self.conversation_meta_url = f"{self.base_url}/conversation-meta"
        self.group_id = group_id
        self.group_name = group_name
        self.scene = scene
        self.default_timezone = default_timezone
        self.retrieve_method = retrieve_method
        self.memory_types = memory_types or ["episodic_memory"]
        self.top_k = max(1, int(top_k))
        self.timeout_seconds = float(timeout_seconds)
        self.sync_mode = bool(sync_mode)
        self.chunk_time_gap_minutes = max(1, int(chunk_time_gap_minutes))
        self.user_id = user_id
        self.user_name = user_name
        self.assistant_id = assistant_id
        self.assistant_name = assistant_name

        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            timeout=self.timeout_seconds,
            headers=self._headers,
            trust_env=False,
        )
        self._message_counter = 0
        self._base_timestamp = datetime.now(timezone.utc)
        self._conversation_meta_saved = False

    @staticmethod
    def _normalize_memories_url(base_url: str) -> str:
        url = (base_url or "").rstrip("/")
        if url.endswith("/api/v1/memories"):
            return url
        return f"{url}/api/v1/memories"

    @staticmethod
    def build_group_id(agent_name: str, sub_dataset: str, entropy: str) -> str:
        base = "-".join(part for part in [agent_name, sub_dataset] if part)
        base = re.sub(r"[^a-zA-Z0-9_-]+", "-", base).strip("-").lower() or "everos"
        digest = hashlib.md5(entropy.encode("utf-8")).hexdigest()[:10]
        return f"{base[:48]}-{digest}"

    def close(self) -> None:
        self._client.close()

    def prepare(self) -> None:
        if self._conversation_meta_saved:
            return
        payload = {
            "scene": self.scene,
            "scene_desc": {"description": "Benchmark evaluation conversation"},
            "name": self.group_name,
            "description": "Auto-generated EverOS benchmark session",
            "group_id": self.group_id,
            "created_at": self._base_timestamp.isoformat(),
            "default_timezone": self.default_timezone,
            "user_details": {
                self.user_id: {
                    "full_name": self.user_name,
                    "role": "user",
                    "custom_role": "benchmark_user",
                    "extra": {},
                },
                self.assistant_id: {
                    "full_name": self.assistant_name,
                    "role": "assistant",
                    "custom_role": "benchmark_assistant",
                    "extra": {},
                },
            },
            "tags": ["benchmark", "everos"],
        }
        self._request_json("post", self.conversation_meta_url, json=payload)
        self._conversation_meta_saved = True

    def add_chunk(self, content: str, *, timestamp: Optional[str] = None, role: str = "user") -> Dict[str, Any]:
        self.prepare()
        self._message_counter += 1
        payload = {
            "message_id": f"{self.group_id}_msg_{self._message_counter}",
            "create_time": timestamp or self._next_timestamp().isoformat(),
            "sender": self.user_id if role != "assistant" else self.assistant_id,
            "sender_name": self.user_name if role != "assistant" else self.assistant_name,
            "role": role,
            "content": content,
            "group_id": self.group_id,
            "group_name": self.group_name,
            "refer_list": [],
        }
        params = {"sync_mode": "true"} if self.sync_mode else None
        return self._request_json("post", self.base_url, json=payload, params=params)

    def search(self, query: str, *, top_k: Optional[int] = None) -> Dict[str, Any]:
        self.prepare()
        params = {
            "query": query,
            "group_id": self.group_id,
            "retrieve_method": self.retrieve_method,
            "memory_types": ",".join(self.memory_types),
            "top_k": int(top_k or self.top_k),
            "include_metadata": "true",
        }
        data = self._request_json("get", self.search_url, params=params)
        result = (data or {}).get("result") or {}
        return {
            "memory_entries": self._flatten_memories(result.get("memories"), result.get("scores")),
            "pending_entries": self._flatten_pending_messages(result.get("pending_messages")),
            "raw_result": result,
            "response": data,
        }

    def _request_json(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        response = self._client.request(method.upper(), url, **kwargs)
        if response.status_code >= 400:
            detail = response.text[:800]
            raise RuntimeError(f"EverOS API error {response.status_code} for {url}: {detail}")
        if not response.text:
            return {}
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"EverOS returned non-JSON response for {url}: {response.text[:400]}") from exc

    def _next_timestamp(self) -> datetime:
        return self._base_timestamp + timedelta(
            minutes=self._message_counter * self.chunk_time_gap_minutes
        )

    @staticmethod
    def _flatten_memories(memories: Any, scores: Any) -> List[Dict[str, Any]]:
        if not isinstance(memories, list):
            return []

        flattened: List[Dict[str, Any]] = []
        for group_index, group in enumerate(memories):
            score_group = scores[group_index] if isinstance(scores, list) and group_index < len(scores) else {}
            if not isinstance(group, dict):
                continue
            for memory_type, items in group.items():
                if not isinstance(items, list):
                    continue
                score_items = score_group.get(memory_type, []) if isinstance(score_group, dict) else []
                for item_index, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    score = score_items[item_index] if item_index < len(score_items) else 0.0
                    flattened.append(
                        {
                            "memory_type": memory_type,
                            "score": float(score or 0.0),
                            "text": EverOSAdapter._format_memory_text(memory_type, item),
                            "raw": item,
                        }
                    )

        flattened.sort(key=lambda entry: entry["score"], reverse=True)
        deduped: List[Dict[str, Any]] = []
        seen_texts = set()
        for entry in flattened:
            text = entry["text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            deduped.append(entry)
        return deduped

    @staticmethod
    def _flatten_pending_messages(pending_messages: Any) -> List[Dict[str, Any]]:
        if not isinstance(pending_messages, list):
            return []

        flattened: List[Dict[str, Any]] = []
        for item in pending_messages:
            if not isinstance(item, dict):
                continue
            flattened.append(
                {
                    "text": EverOSAdapter._format_pending_text(item),
                    "raw": item,
                }
            )
        return flattened

    @staticmethod
    def _format_memory_text(memory_type: str, item: Dict[str, Any]) -> str:
        lines = [f"[{memory_type}]"]
        if item.get("timestamp"):
            lines.append(f"Time: {item['timestamp']}")
        if item.get("subject"):
            lines.append(f"Subject: {item['subject']}")
        if item.get("summary"):
            lines.append(f"Summary: {item['summary']}")
        if item.get("episode"):
            lines.append(f"Episode: {item['episode']}")
        if item.get("content"):
            lines.append(f"Content: {item['content']}")
        if len(lines) == 1:
            lines.append(json.dumps(item, ensure_ascii=False, default=str))
        return "\n".join(lines).strip()

    @staticmethod
    def _format_pending_text(item: Dict[str, Any]) -> str:
        lines = ["[pending_message]"]
        if item.get("message_create_time"):
            lines.append(f"Time: {item['message_create_time']}")
        if item.get("sender_name") or item.get("sender"):
            lines.append(f"Sender: {item.get('sender_name') or item.get('sender')}")
        if item.get("content"):
            lines.append(f"Content: {item['content']}")
        if len(lines) == 1:
            lines.append(json.dumps(item, ensure_ascii=False, default=str))
        return "\n".join(lines).strip()
