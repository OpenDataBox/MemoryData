"""Upstream-inspired MemoChat runtime adapted to the repository lifecycle."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from utils.memory_runtime_support import (
    ChatBackend,
    EmbeddingBackend,
    build_summary,
    cosine_similarity,
    extract_keywords,
    keyword_overlap,
)


PROMPTS = {
    "chatting": {
        "system": (
            "You are an intelligent dialog bot. You will be shown Related Evidences "
            "supporting for User Input, and Recent Dialogs between user and you. "
            "Please read, memorize, and understand given materials, then generate one "
            "concise, coherent and helpful response. Note that you should just give a "
            "consice and direct answer without any explanations or extra information.\n\n"
        ),
        "instruction": "",
    },
    "writing_dialogsum": {
        "system": (
            "You will be shown a LINE-line Task Conversation between user and bot. "
            "Please read, memorize, and understand Task Conversation, then complete "
            "the task under the guidance of Task Introduction."
        ),
        "instruction": (
            "\n\n```\nTask Introduction:\n"
            "Based on the Task Conversation, perform the following actions:\n"
            "1 - Conclude all possible topics in the conversation with concise spans.\n"
            "2 - Determine the chat range of each topic. These ranges should be a set "
            "of non-intersecting, sequentially connected end-to-end intervals.\n"
            "3 - Conclude a summary of each chat with brief sentences.\n"
            "4 - Report topic, summary and range resutls in JSON format only with the "
            "assigned keys: 'topic', 'summary', 'start', 'end'. For example, assuming "
            "an M-line conversation talks about 'banana' from line 1 to line N, then "
            "turns to talk about 'mango' from line N+1 to line M. Thus, its task result "
            "could be: [{'topic': 'banana', 'summary': 'user talks banana with bot.', "
            "'start': 1, 'end': N}, {'topic': 'mango', 'summary': 'bot brings mango for "
            "user.', 'start': N+1, 'end': M}].\n"
            "Besides, following notations are provides:\n"
            "1 - For each element of Task Conversation's JSON result, the value of 'end' "
            "should be smaller than the value of 'start', while both values should be "
            "larger than 0 but not exceed the total num of Task Conversation lines LINE.\n"
            "2 - Intersecting intervals such as {'topic': 'apple', 'summary': 'user and "
            "bot share apples.', 'start': K, 'end': N} and {'topic': 'pear', 'summary': "
            "'bot sends pear to user.', 'start': N-2, 'end': M} are illegal.\n"
            "```\n\nTask Result:"
        ),
    },
    "retrieval": {
        "system": (
            "You will be shown 1 Query Sentence and OPTION Topic Options. Please read, "
            "memorize, and understand given materials, then complete the task under the "
            "guidance of Task Introduction.\n\n"
        ),
        "instruction": (
            "\n\n```\nTask Introduction:\n"
            "Select one or more topics from Topic Options that relevant with Query "
            "Sentence. Note that there is a NOTO option, select it if all other topic "
            "options are not related to Query Sentence. Do not report the option content, "
            "but only report selected option numbers in a string separated with '#'. For "
            "example, if topic option N and M are chosen, then the output is: N#M. For "
            "Query Sentence in the task, any chosen option numbers should be larger than 0 "
            "but not exceed the total num of Topic Options OPTION.\n"
            "```\n\nTask Result:"
        ),
    },
}


def normalize_model_outputs(model_text: str) -> list[dict]:
    extracted_elements = [
        re.sub(r"\s+", " ", match.replace('"', "").replace("'", ""))
        for match in re.findall(r"'[^']*'|\"[^\"]*\"|\d+", model_text or "")
    ]
    model_outputs = []
    index = 0
    while index + 7 < len(extracted_elements):
        if (
            extracted_elements[index] == "topic"
            and extracted_elements[index + 2] == "summary"
            and extracted_elements[index + 4] == "start"
            and extracted_elements[index + 6] == "end"
        ):
            try:
                model_outputs.append(
                    {
                        "topic": extracted_elements[index + 1],
                        "summary": extracted_elements[index + 3],
                        "start": int(extracted_elements[index + 5]),
                        "end": int(extracted_elements[index + 7]),
                    }
                )
            except Exception:
                pass
        index += 1
    return model_outputs


@dataclass
class MemoTopicRecord:
    topic_id: str
    topic: str
    summary: str
    dialogs: list[str]
    start_order: int
    end_order: int


class MemoChat:
    """MemoChat runtime that keeps the original summarize-then-select-topic flow."""

    def __init__(
        self,
        *,
        state_path: str,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str],
        embedding_model: str,
        embedding_provider: Optional[str],
        embedding_base_url: Optional[str],
        embedding_api_key: Optional[str],
        embedding_dimensions: Optional[int],
        retrieve_num: int,
        summary_trigger_chunks: int,
        keep_recent_chunks: int,
        max_topics_per_window: int,
        summary_chars: int,
        keyword_limit: int,
        topic_top_k: int,
        recent_top_k: int,
        dialogs_per_topic: int,
        llm_max_tokens: int,
        use_llm_topic_segmentation: bool,
    ) -> None:
        self.state_path = Path(state_path)
        self.model = model
        self.retrieve_num = retrieve_num
        self.summary_trigger_chunks = summary_trigger_chunks
        self.keep_recent_chunks = keep_recent_chunks
        self.max_topics_per_window = max_topics_per_window
        self.summary_chars = summary_chars
        self.keyword_limit = keyword_limit
        self.topic_top_k = topic_top_k
        self.recent_top_k = recent_top_k
        self.dialogs_per_topic = dialogs_per_topic
        self.llm_max_tokens = llm_max_tokens
        self.use_llm_topic_segmentation = use_llm_topic_segmentation
        self.base_timestamp = datetime(2024, 1, 1, 0, 0, 0)
        self.next_dialog_index = 1
        self.next_topic_index = 1
        self.recent_dialogs: list[str] = []
        self.memo: dict[str, list[MemoTopicRecord]] = {
            "NOTO": [
                MemoTopicRecord(
                    topic_id="topic_NOTO",
                    topic="NOTO",
                    summary="None of the others.",
                    dialogs=[],
                    start_order=-1,
                    end_order=-1,
                )
            ]
        }
        self.chat = ChatBackend(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        # Keep an embedding backend only as a fallback selector when LLM retrieval is unavailable.
        self.embedder = EmbeddingBackend(
            model=embedding_model,
            provider=embedding_provider,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            embedding_dimensions=embedding_dimensions,
        )

    def _next_timestamp(self, order: int) -> str:
        return (self.base_timestamp + timedelta(seconds=order)).strftime("%Y-%m-%d %H:%M:%S")

    def _compose_dialog_line(self, content: str, timestamp: str, dialog_id: str) -> str:
        return f"MemoryChunk: {content} [Time: {timestamp}, ID: {dialog_id}]"

    def _gen_model_output(self, input_prompt: str, target_len: int) -> str:
        if not self.chat.available:
            raise RuntimeError("MemoChat requires an LLM backend for its core summarize/retrieve flow.")
        return self.chat.complete(
            messages=[{"role": "system", "content": input_prompt}],
            temperature=0.2,
            max_tokens=target_len,
        )

    def _fallback_summary_outputs(self) -> list[dict]:
        if not self.recent_dialogs:
            return []
        topic = ", ".join(extract_keywords(" ".join(self.recent_dialogs), 3)) or "conversation memory"
        summary = build_summary(" ".join(self.recent_dialogs), self.summary_chars)
        return [{"topic": topic, "summary": summary, "start": 1, "end": len(self.recent_dialogs)}]

    def _run_summary(self) -> None:
        if not self.recent_dialogs:
            return

        if self.use_llm_topic_segmentation and self.chat.available:
            system_instruction = PROMPTS["writing_dialogsum"]["system"]
            task_instruction = PROMPTS["writing_dialogsum"]["instruction"]
            history_log = "\n\n```\nTask Conversation:\n" + "\n".join(
                f"(line {index + 1}) {dialog.replace(chr(10), ' ')}"
                for index, dialog in enumerate(self.recent_dialogs)
            )
            query = (
                system_instruction.replace("LINE", str(len(self.recent_dialogs)))
                + history_log
                + "\n```"
                + task_instruction.replace("LINE", str(len(self.recent_dialogs)))
            )
            try:
                sum_history = normalize_model_outputs(
                    self._gen_model_output(query, target_len=self.llm_max_tokens)
                )
            except Exception:
                sum_history = self._fallback_summary_outputs()
        else:
            sum_history = self._fallback_summary_outputs()

        if not sum_history:
            sum_history = self._fallback_summary_outputs()

        added_records = False
        if sum_history:
            for summary_item in sum_history[: self.max_topics_per_window]:
                start = max(1, int(summary_item.get("start", 1)))
                end = min(len(self.recent_dialogs), int(summary_item.get("end", len(self.recent_dialogs))))
                if start > end:
                    continue
                dialogs = self.recent_dialogs[start - 1:end]
                topic = " ".join(str(summary_item.get("topic", "")).split()).strip() or "conversation memory"
                summary = " ".join(str(summary_item.get("summary", "")).split()).strip()
                if not summary:
                    summary = build_summary(" ".join(dialogs), self.summary_chars)
                topic_id = f"topic_{self.next_topic_index:06d}"
                self.next_topic_index += 1
                record = MemoTopicRecord(
                    topic_id=topic_id,
                    topic=topic,
                    summary=build_summary(summary, self.summary_chars),
                    dialogs=dialogs,
                    start_order=self.next_dialog_index - len(self.recent_dialogs) + start - 1,
                    end_order=self.next_dialog_index - len(self.recent_dialogs) + end - 1,
                )
                self.memo.setdefault(topic, []).append(record)
                added_records = True
        if not added_records:
            dialogs = list(self.recent_dialogs)
            fallback_summary = build_summary(" ".join(dialogs), self.summary_chars)
            topic_id = f"topic_{self.next_topic_index:06d}"
            self.next_topic_index += 1
            self.memo["NOTO"].append(
                MemoTopicRecord(
                    topic_id=topic_id,
                    topic="NOTO",
                    summary=fallback_summary,
                    dialogs=dialogs,
                    start_order=self.next_dialog_index - len(self.recent_dialogs),
                    end_order=self.next_dialog_index - 1,
                )
            )

        self.recent_dialogs = (
            self.recent_dialogs[-self.keep_recent_chunks:]
            if len(self.recent_dialogs) >= self.keep_recent_chunks
            else list(self.recent_dialogs)
        )

    def _should_summarize(self) -> bool:
        recent_word_count = len(" ### ".join(self.recent_dialogs).split())
        return recent_word_count > 1024 or len(self.recent_dialogs) >= max(10, self.summary_trigger_chunks)

    def add_chunk(self, content: str, timestamp: Optional[str] = None) -> str:
        order = self.next_dialog_index - 1
        if timestamp is None:
            timestamp = self._next_timestamp(order)
        dialog_id = f"dialog_{self.next_dialog_index:06d}"
        self.next_dialog_index += 1

        self.recent_dialogs.append(self._compose_dialog_line(content, timestamp, dialog_id))
        if self._should_summarize():
            self._run_summary()
        return dialog_id

    def _topic_candidates(self) -> list[tuple[str, str, list[str]]]:
        candidates = []
        for topic, items in self.memo.items():
            for item in items:
                candidates.append((topic, item.summary, item.dialogs))
        return candidates

    def _fallback_retrieve(self, query: str, topics: list[tuple[str, str, list[str]]]) -> list[tuple[str, str, list[str]]]:
        if not topics:
            return []
        query_embedding = self.embedder.embed_text(query)
        query_keywords = set(extract_keywords(query, self.keyword_limit))
        scored_topics = []
        for topic, summary, dialogs in topics:
            if topic == "NOTO":
                continue
            combined = f"{topic}. {summary}"
            score = (0.8 * cosine_similarity(query_embedding, self.embedder.embed_text(combined))) + (
                0.2 * keyword_overlap(query_keywords, extract_keywords(combined, self.keyword_limit))
            )
            scored_topics.append((score, (topic, summary, dialogs)))
        scored_topics.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored_topics[: max(1, self.topic_top_k)]]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> dict:
        top_k = top_k or self.retrieve_num
        topics = self._topic_candidates()
        chosen_topics: list[tuple[str, str, list[str]]] = []

        if topics and self.chat.available:
            system_instruction = PROMPTS["retrieval"]["system"]
            task_instruction = PROMPTS["retrieval"]["instruction"]
            task_case = (
                "```\nQuery Sentence:\n"
                + query
                + "\nTopic Options:\n"
                + "\n".join(
                    f"({index + 1}) {topic}. {summary}"
                    for index, (topic, summary, _dialogs) in enumerate(topics)
                )
                + "\n```"
            )
            retrieval_prompt = (
                system_instruction.replace("OPTION", str(len(topics)))
                + task_case
                + task_instruction.replace("OPTION", str(len(topics)))
            )
            try:
                outputs = self._gen_model_output(retrieval_prompt, target_len=32).split("#")
            except Exception:
                outputs = []
            for output in outputs:
                try:
                    topic_index = int(output) - 1
                except Exception:
                    continue
                if 0 <= topic_index < len(topics) and topics[topic_index][0] != "NOTO":
                    chosen_topics.append(topics[topic_index])

        if not chosen_topics:
            chosen_topics = self._fallback_retrieve(query, topics)

        related_topics = [topic for topic, _summary, _dialogs in chosen_topics]
        related_summaries = [summary for _topic, summary, _dialogs in chosen_topics]
        related_dialogs = [" ### ".join(dialogs[-self.dialogs_per_topic:]) for _topic, _summary, dialogs in chosen_topics]

        combined_texts = []
        for topic, summary in zip(related_topics, related_summaries):
            combined_texts.append(f"[topic-summary] {topic}: {summary}")
        for dialogs in related_dialogs:
            combined_texts.append(f"[topic-dialogs] {dialogs}")
        for dialog in self.recent_dialogs[-self.recent_top_k:]:
            combined_texts.append(f"[recent-dialog] {dialog}")

        return {
            "related_topics": related_topics,
            "related_summaries": related_summaries,
            "related_dialogs": related_dialogs,
            "combined_texts": combined_texts[:top_k],
        }

    def save(self) -> None:
        if self.recent_dialogs:
            self._run_summary()
        payload = {
            "base_timestamp": self.base_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "next_dialog_index": self.next_dialog_index,
            "next_topic_index": self.next_topic_index,
            "recent_dialogs": self.recent_dialogs,
            "memo": {
                key: [asdict(item) for item in items]
                for key, items in self.memo.items()
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.state_path.exists():
            raise FileNotFoundError(f"MemoChat state file not found at {self.state_path}")

        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.base_timestamp = datetime.strptime(payload["base_timestamp"], "%Y-%m-%d %H:%M:%S")
        self.next_dialog_index = int(payload.get("next_dialog_index", 1))
        self.next_topic_index = int(payload.get("next_topic_index", 1))
        self.recent_dialogs = list(payload.get("recent_dialogs", []))
        self.memo = {
            key: [MemoTopicRecord(**item) for item in items]
            for key, items in payload.get("memo", {}).items()
        }
        if "NOTO" not in self.memo:
            self.memo["NOTO"] = [
                MemoTopicRecord(
                    topic_id="topic_NOTO",
                    topic="NOTO",
                    summary="None of the others.",
                    dialogs=[],
                    start_order=-1,
                    end_order=-1,
                )
            ]

    def memory_count(self) -> int:
        return sum(len(items) for items in self.memo.values()) + len(self.recent_dialogs)
