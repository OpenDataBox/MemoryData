"""Upstream-inspired MemoryOS runtime adapted to the repository lifecycle."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from utils.memory_runtime_support import (
    ChatBackend,
    EmbeddingBackend,
    build_summary,
    extract_keywords,
    load_json_from_model_output,
    normalize_embedding,
)


def compute_time_decay(session_timestamp: str, current_timestamp: str, tau: float = 3600.0) -> float:
    fmt = "%Y-%m-%d %H:%M:%S"
    left = datetime.strptime(session_timestamp, fmt)
    right = datetime.strptime(current_timestamp, fmt)
    delta = (right - left).total_seconds()
    return math.exp(-delta / tau)


def compute_segment_heat(session: "SessionRecord", alpha: float = 0.8, beta: float = 0.8, gamma: float = 0.0001) -> float:
    return (
        alpha * session.N_visit
        + beta * session.L_interaction
        + gamma * session.R_recency
    )


@dataclass
class ShortTermMessage:
    message_id: str
    user_input: str
    agent_response: str
    timestamp: str


@dataclass
class PageRecord:
    page_id: str
    user_input: str
    agent_response: str
    timestamp: str
    page_embedding: list[float]
    page_keywords: list[str]
    preloaded: bool = False
    analyzed: bool = False
    pre_page: Optional[str] = None
    next_page: Optional[str] = None
    meta_info: Optional[str] = None


@dataclass
class SessionRecord:
    id: str
    summary: str
    summary_keywords: list[str]
    summary_embedding: list[float]
    details: list[PageRecord] = field(default_factory=list)
    L_interaction: int = 0
    R_recency: float = 1.0
    N_visit: int = 0
    H_segment: float = 0.0
    timestamp: str = ""
    access_count: int = 0
    last_visit_time: str = ""


@dataclass
class KnowledgeEntry:
    knowledge: str
    timestamp: str
    knowledge_embedding: list[float]


@dataclass
class UserProfileRecord:
    data: str
    last_updated: str


class MemoryOS:
    """MemoryOS runtime with short/mid/long-term memory and heat-driven updates."""

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
        short_term_capacity: int,
        mid_term_capacity: int,
        queue_capacity: int,
        topic_similarity_threshold: float,
        heat_threshold: float,
        summary_chars: int,
        keyword_limit: int,
        segment_threshold: float,
        page_threshold: float,
        knowledge_threshold: float,
        llm_max_tokens: int,
    ) -> None:
        self.state_path = Path(state_path)
        self.model = model
        self.retrieve_num = retrieve_num
        self.short_term_capacity = max(1, short_term_capacity)
        self.mid_term_capacity = max(1, mid_term_capacity)
        self.queue_capacity = max(1, queue_capacity)
        self.topic_similarity_threshold = topic_similarity_threshold
        self.heat_threshold = heat_threshold
        self.summary_chars = summary_chars
        self.keyword_limit = keyword_limit
        self.segment_threshold = segment_threshold
        self.page_threshold = page_threshold
        self.knowledge_threshold = knowledge_threshold
        self.llm_max_tokens = llm_max_tokens
        self.base_timestamp = datetime(2024, 1, 1, 0, 0, 0)
        self.next_message_index = 1
        self.next_page_index = 1
        self.next_session_index = 1
        self.short_term_memory: list[ShortTermMessage] = []
        self.sessions: dict[str, SessionRecord] = {}
        self.access_frequency: dict[str, int] = {}
        self.user_profiles: dict[str, UserProfileRecord] = {}
        self.knowledge_base: list[KnowledgeEntry] = []
        self.assistant_knowledge: list[KnowledgeEntry] = []
        self.last_evicted_page_id: Optional[str] = None
        self.chat = ChatBackend(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        self.embedder = EmbeddingBackend(
            model=embedding_model,
            provider=embedding_provider,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            embedding_dimensions=embedding_dimensions,
        )

    def _next_timestamp(self, order: int) -> str:
        return (self.base_timestamp + timedelta(seconds=order)).strftime("%Y-%m-%d %H:%M:%S")

    def _timestamp_now(self) -> str:
        return datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")

    def _next_message_id(self) -> str:
        value = f"message_{self.next_message_index:06d}"
        self.next_message_index += 1
        return value

    def _next_page_id(self) -> str:
        value = f"page_{self.next_page_index:06d}"
        self.next_page_index += 1
        return value

    def _next_session_id(self) -> str:
        value = f"session_{self.next_session_index:06d}"
        self.next_session_index += 1
        return value

    def _embed_text(self, text: str) -> list[float]:
        return normalize_embedding(self.embedder.embed_text(text))

    def _chat_generate(self, *, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        if not self.chat.available:
            raise RuntimeError("MemoryOS requires a chat backend for its core update/retrieval flow.")
        return self.chat.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        ).strip()

    def _llm_extract_keywords(self, text: str) -> list[str]:
        if not text.strip():
            return []
        if self.chat.available:
            prompt = (
                "Please extract the keywords of the conversation topic from the following "
                "dialogue, separated by commas, and do not exceed three:\n"
                + text
            )
            try:
                output = self._chat_generate(
                    prompt=prompt,
                    system="You are a keyword extraction expert. Please extract the keywords of the conversation topic.",
                    temperature=0.0,
                    max_tokens=64,
                )
                keywords = [item.strip() for item in output.split(",") if item.strip()]
                if keywords:
                    return keywords[:3]
            except Exception:
                pass
        return extract_keywords(text, min(self.keyword_limit, 3))

    def _generate_multi_summary(self, text: str) -> list[dict]:
        if not text.strip():
            return []
        if self.chat.available:
            prompt = (
                "Please analyze the following dialogue and generate multiple subtopic summaries "
                "(if applicable), with a maximum of two themes.\n"
                "Each summary should include the subtopic name, keywords (separated by commas), "
                "and the summary text, formatted as a JSON array, with an example format as follows:\n"
                "[\n"
                '  {"theme": "Business trip", "keywords": ["Business trip", "Itinerary", "Work"], "content": "User mentioned the troubles related to business trips."},\n'
                '  {"theme": "Health", "keywords": ["Cold", "Uncomfortable", "Sick"], "content": "User reported feeling unwell due to a cold."}\n'
                "]\n"
                "Please directly output the JSON array, without adding any other content.\n"
                "Conversation content:\n"
                + text
            )
            try:
                output = self._chat_generate(
                    prompt=prompt,
                    system="You are an expert in analyzing dialogue topics. No more than two topics.",
                    temperature=0.0,
                    max_tokens=self.llm_max_tokens,
                )
                parsed = load_json_from_model_output(output)
                if isinstance(parsed, list):
                    normalized = []
                    for item in parsed[:2]:
                        if not isinstance(item, dict):
                            continue
                        theme = " ".join(str(item.get("theme", "")).split()).strip()
                        content = " ".join(str(item.get("content", "")).split()).strip()
                        keywords = item.get("keywords", [])
                        if isinstance(keywords, str):
                            keywords = [part.strip() for part in keywords.split(",") if part.strip()]
                        if not isinstance(keywords, list):
                            keywords = []
                        if content:
                            normalized.append(
                                {
                                    "theme": theme or "conversation memory",
                                    "content": build_summary(content, self.summary_chars),
                                    "keywords": [str(keyword).strip() for keyword in keywords if str(keyword).strip()][:3],
                                }
                            )
                    if normalized:
                        return normalized
            except Exception:
                pass

        fallback_summary = build_summary(text, self.summary_chars)
        return [
            {
                "theme": ", ".join(extract_keywords(text, 3)) or "conversation memory",
                "content": fallback_summary,
                "keywords": extract_keywords(text, 3),
            }
        ]

    def _analyze_assistant_knowledge(self, pages: list[PageRecord]) -> str:
        if not pages:
            return "None"
        conversation = "\n".join(
            f"User: {page.user_input}\nAI: {page.agent_response}\nTime: {page.timestamp}\n"
            for page in pages
        )
        prompt = (
            "# Assistant Knowledge Extraction Task\n"
            "Analyze the conversation and extract any fact or identity traits about the assistant.\n"
            'If no traits can be extracted, reply with "None".\n\n'
            "Conversation:\n"
            + conversation
        )
        try:
            output = self._chat_generate(
                prompt=prompt,
                system=(
                    "You are an assistant knowledge extraction engine. Extract only explicit "
                    "statements about the assistant's identity or knowledge. Use concise and "
                    "factual statements in the first person."
                ),
                temperature=0.0,
                max_tokens=256,
            )
            cleaned = output.replace("【Assistant Knowledge】", "").strip()
            return cleaned or "None"
        except Exception:
            return "None"

    def _personality_analysis(self, pages: list[PageRecord]) -> dict:
        if not pages:
            return {"profile": "", "private": "None", "assistant_knowledge": "None"}

        conversation = "\n".join(
            f"User: {page.user_input}\nAssistant: {page.agent_response}\nTime:{page.timestamp}"
            for page in pages
        )
        prompt = (
            "# Personality and User Data Analysis Task\n"
            "Analyze the conversation and output in EXACTLY this format:\n\n"
            "【User Profile】\n"
            "1. Core Psychological Traits:\n"
            "2. Content Preferences:\n"
            "3. Interaction Style:\n"
            "4. Value Alignment:\n\n"
            "【User Data】\n"
            "[Fact 1]: [Details]\n"
            '(Include events, dates, locations, preferences, or other general or private information explicitly mentioned in the conversation. If none, write "None.")\n\n'
            "Conversation:\n"
            + conversation
        )

        try:
            output = self._chat_generate(
                prompt=prompt,
                system=(
                    "You are a personality and user data analysis engine. Extract only observable "
                    "traits and data with direct evidence. Include general user data such as events, "
                    "dates, locations, and preferences."
                ),
                temperature=0.0,
                max_tokens=self.llm_max_tokens,
            )
            if "【User Data】" in output:
                profile, user_data = output.split("【User Data】", 1)
            else:
                profile, user_data = output, "None"
            return {
                "profile": profile.replace("【User Profile】", "").strip(),
                "private": user_data.strip(),
                "assistant_knowledge": self._analyze_assistant_knowledge(pages),
            }
        except Exception:
            joined = " ".join(page.user_input for page in pages)
            return {
                "profile": build_summary(joined, self.summary_chars),
                "private": "None",
                "assistant_knowledge": self._analyze_assistant_knowledge(pages),
            }

    def _update_profile(self, old_profile: str, new_profile: str) -> str:
        if not old_profile.strip():
            return new_profile.strip()
        if not new_profile.strip():
            return old_profile.strip()
        prompt = (
            "# Profile Merge Task\n"
            "Consolidate these profiles while preserving valid observations, resolving conflicts, "
            "and adding new dimensions.\n\n"
            "## Current Profile\n"
            f"{old_profile}\n\n"
            "## New Data\n"
            f"{new_profile}\n\n"
            "Output ONLY the merged profile (no commentary)."
        )
        try:
            output = self._chat_generate(
                prompt=prompt,
                system=(
                    "You are a profile integration system. Never discard verified information. "
                    "Resolve conflicts conservatively and keep the structure concise."
                ),
                temperature=0.0,
                max_tokens=self.llm_max_tokens,
            )
            return output.strip() or new_profile.strip()
        except Exception:
            return f"{old_profile.strip()}\n\n--- Updated ---\n{new_profile.strip()}".strip()

    def _is_conversation_continuing(self, previous_page: Optional[PageRecord], current_page: PageRecord) -> bool:
        if previous_page is None:
            return False
        prompt = (
            "Determine if these two conversation pages are continuous (true continuation without topic shift).\n"
            'Return ONLY "true" or "false".\n\n'
            f"Previous Page:\nUser: {previous_page.user_input}\nAssistant: {previous_page.agent_response}\n\n"
            f"Current Page:\nUser: {current_page.user_input}\nAssistant: {current_page.agent_response}\n\n"
            "Continuous?"
        )
        try:
            output = self._chat_generate(
                prompt=prompt,
                system="You are a conversation continuity detector. Return ONLY 'true' or 'false'.",
                temperature=0.0,
                max_tokens=10,
            )
            return output.strip().lower() == "true"
        except Exception:
            previous_keywords = set(self._llm_extract_keywords(previous_page.user_input or previous_page.agent_response))
            current_keywords = set(self._llm_extract_keywords(current_page.user_input or current_page.agent_response))
            return bool(previous_keywords & current_keywords)

    def _generate_meta_info(self, last_page_meta: Optional[str], current_page: PageRecord) -> str:
        current_conversation = (
            f"User: {current_page.user_input}\nAssistant: {current_page.agent_response}"
        )
        prompt = (
            "Update the conversation meta-summary by incorporating the new dialogue while maintaining continuity.\n\n"
            "Previous Meta-summary: "
            + (last_page_meta or "None")
            + "\nNew Dialogue:\n"
            + current_conversation
            + "\n\nUpdated Meta-summary:"
        )
        try:
            output = self._chat_generate(
                prompt=prompt,
                system=(
                    "You are a conversation meta-summary updater. Preserve relevant context "
                    "from the previous summary and integrate the new dialogue in 1-2 sentences."
                ),
                temperature=0.3,
                max_tokens=128,
            )
            return output.strip() or build_summary(current_conversation, self.summary_chars)
        except Exception:
            base = f"{last_page_meta}\n{current_conversation}" if last_page_meta else current_conversation
            return build_summary(base, self.summary_chars)

    def _get_page_by_id(self, page_id: Optional[str]) -> Optional[PageRecord]:
        if not page_id:
            return None
        for session in self.sessions.values():
            for page in session.details:
                if page.page_id == page_id:
                    return page
        return None

    def _update_connected_pages(self, page_id: str, new_meta_info: str) -> None:
        current_page = self._get_page_by_id(page_id)
        if current_page is None:
            return

        connected_pages: list[PageRecord] = []
        prev_page_id = current_page.pre_page
        while prev_page_id:
            prev_page = self._get_page_by_id(prev_page_id)
            if prev_page is None:
                break
            connected_pages.insert(0, prev_page)
            prev_page_id = prev_page.pre_page

        next_page_id = current_page.next_page
        while next_page_id:
            next_page = self._get_page_by_id(next_page_id)
            if next_page is None:
                break
            connected_pages.append(next_page)
            next_page_id = next_page.next_page

        for page in connected_pages:
            page.meta_info = new_meta_info

    def _add_knowledge_entry(self, collection: list[KnowledgeEntry], text: str) -> None:
        if not text or text.strip() in {"", "- None", "- None.", "None"}:
            return
        collection.append(
            KnowledgeEntry(
                knowledge=text.strip(),
                timestamp=self._timestamp_now(),
                knowledge_embedding=self._embed_text(text.strip()),
            )
        )

    def _evict_lfu(self) -> None:
        if not self.access_frequency:
            return
        lfu_session_id = min(self.access_frequency, key=self.access_frequency.get)
        if lfu_session_id not in self.sessions:
            self.access_frequency.pop(lfu_session_id, None)
            return
        del self.sessions[lfu_session_id]
        self.access_frequency.pop(lfu_session_id, None)

    def _add_session(self, summary: str, details: list[PageRecord]) -> str:
        session_id = self._next_session_id()
        summary_keywords = self._llm_extract_keywords(summary)
        session = SessionRecord(
            id=session_id,
            summary=summary,
            summary_keywords=summary_keywords,
            summary_embedding=self._embed_text(summary),
            details=details,
            L_interaction=len(details),
            R_recency=1.0,
            N_visit=0,
            H_segment=0.0,
            timestamp=self._timestamp_now(),
            access_count=0,
            last_visit_time=self._timestamp_now(),
        )
        session.H_segment = compute_segment_heat(session)
        self.sessions[session_id] = session
        self.access_frequency[session_id] = 0
        if len(self.sessions) > self.mid_term_capacity:
            self._evict_lfu()
        return session_id

    def _insert_pages_into_session(self, summary: str, keywords: list[str], pages: list[PageRecord]) -> None:
        new_summary_embedding = self._embed_text(summary)
        best_session_id = None
        best_similarity = -1.0

        for session_id, session in self.sessions.items():
            similarity = float(sum(left * right for left, right in zip(session.summary_embedding, new_summary_embedding)))
            if similarity > best_similarity:
                best_similarity = similarity
                best_session_id = session_id

        merged = False
        if best_similarity >= 0.0 and best_session_id is not None:
            session = self.sessions[best_session_id]
            session_keywords = set(session.summary_keywords)
            new_keywords = set(keywords)
            if session_keywords and new_keywords:
                overlap = session_keywords & new_keywords
                topic_overlap = 0.5 * (
                    len(overlap) / max(len(session_keywords), 1)
                    + len(overlap) / max(len(new_keywords), 1)
                )
            else:
                topic_overlap = 0.0
            overall_score = best_similarity + topic_overlap
            if overall_score >= self.topic_similarity_threshold:
                session.details.extend(pages)
                session.timestamp = self._timestamp_now()
                session.L_interaction += len(pages)
                session.H_segment = compute_segment_heat(session)
                merged = True

        if not merged:
            self._add_session(summary, pages)

    def _bulk_evict_and_update_mid_term(self) -> None:
        evicted_messages: list[ShortTermMessage] = []
        while len(self.short_term_memory) >= self.short_term_capacity:
            evicted_messages.append(self.short_term_memory.pop(0))
            if len(self.short_term_memory) < self.short_term_capacity:
                break

        if not evicted_messages:
            return

        pages: list[PageRecord] = []
        for message in evicted_messages:
            page = PageRecord(
                page_id=self._next_page_id(),
                user_input=message.user_input,
                agent_response=message.agent_response,
                timestamp=message.timestamp,
                page_embedding=self._embed_text(
                    f"User: {message.user_input}\nAssistant: {message.agent_response}".strip()
                ),
                page_keywords=self._llm_extract_keywords(
                    f"User: {message.user_input}\nAssistant: {message.agent_response}".strip()
                ),
            )

            previous_page = self._get_page_by_id(self.last_evicted_page_id)
            is_continuous = self._is_conversation_continuing(previous_page, page)
            if is_continuous and previous_page is not None:
                page.pre_page = previous_page.page_id
                previous_page.next_page = page.page_id
                new_meta_info = self._generate_meta_info(previous_page.meta_info, page)
                page.meta_info = new_meta_info
                self._update_connected_pages(page.pre_page, new_meta_info)
            else:
                page.meta_info = self._generate_meta_info(None, page)

            pages.append(page)
            self.last_evicted_page_id = page.page_id

        input_text = "\n".join(
            f"User: {page.user_input}\nAssistant: {page.agent_response}"
            for page in pages
        )
        for summary_item in self._generate_multi_summary(input_text):
            sub_summary = summary_item.get("content", "")
            sub_keywords = summary_item.get("keywords") or self._llm_extract_keywords(sub_summary)
            self._insert_pages_into_session(sub_summary, sub_keywords, pages)

    def _update_user_profile_from_top_segment(self) -> None:
        if not self.sessions:
            return
        top_session = max(self.sessions.values(), key=lambda item: item.H_segment)
        if top_session.H_segment < self.heat_threshold:
            return

        un_analyzed_pages = [page for page in top_session.details if not page.analyzed]
        if not un_analyzed_pages:
            return

        sample_key = "global_profile"
        analysis = self._personality_analysis(un_analyzed_pages)
        old_profile = self.user_profiles.get(sample_key)
        if old_profile is not None:
            merged_profile = self._update_profile(old_profile.data, analysis["profile"])
        else:
            merged_profile = analysis["profile"]
        self.user_profiles[sample_key] = UserProfileRecord(
            data=merged_profile,
            last_updated=self._timestamp_now(),
        )

        private_data = analysis.get("private", "")
        if private_data and private_data not in {"None", "- None", "- None."}:
            facts = [line.strip() for line in private_data.split("\n") if line.strip()]
            for fact in facts:
                self._add_knowledge_entry(self.knowledge_base, fact)

        assistant_knowledge = analysis.get("assistant_knowledge", "")
        if assistant_knowledge and assistant_knowledge != "None":
            self._add_knowledge_entry(self.assistant_knowledge, assistant_knowledge)

        for page in top_session.details:
            page.analyzed = True
        top_session.N_visit = 0
        top_session.L_interaction = 0
        top_session.R_recency = 1.0
        top_session.H_segment = 0.0
        top_session.last_visit_time = self._timestamp_now()

    def add_chunk(self, content: str, timestamp: Optional[str] = None) -> str:
        order = self.next_message_index - 1
        if timestamp is None:
            timestamp = self._next_timestamp(order)
        message = ShortTermMessage(
            message_id=self._next_message_id(),
            user_input=content,
            agent_response="",
            timestamp=timestamp,
        )
        self.short_term_memory.append(message)
        if len(self.short_term_memory) >= self.short_term_capacity:
            self._bulk_evict_and_update_mid_term()
            self._update_user_profile_from_top_segment()
        return message.message_id

    def _search_knowledge(self, query: str) -> list[KnowledgeEntry]:
        if not self.knowledge_base:
            return []
        query_embedding = self._embed_text(query)
        scored = []
        for entry in self.knowledge_base:
            score = float(sum(left * right for left, right in zip(query_embedding, entry.knowledge_embedding)))
            if score >= self.knowledge_threshold:
                scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _score, entry in scored[: self.retrieve_num]]

    def _search_sessions_by_summary(self, query: str) -> list[dict]:
        if not self.sessions:
            return []

        query_embedding = self._embed_text(query)
        query_keywords = set(self._llm_extract_keywords(query))
        current_timestamp = self._timestamp_now()
        matched_sessions = []

        for session_id, session in self.sessions.items():
            summary_similarity = float(sum(left * right for left, right in zip(query_embedding, session.summary_embedding)))
            session_keywords = set(session.summary_keywords)
            if query_keywords and session_keywords:
                overlap = query_keywords & session_keywords
                topic_overlap = 0.5 * (
                    len(overlap) / max(len(query_keywords), 1)
                    + len(overlap) / max(len(session_keywords), 1)
                )
            else:
                topic_overlap = 0.0
            overall_score = summary_similarity + topic_overlap
            if overall_score < self.segment_threshold:
                continue

            matched_pages = []
            for page in session.details:
                page_similarity = float(sum(left * right for left, right in zip(query_embedding, page.page_embedding)))
                if page_similarity >= self.page_threshold:
                    matched_pages.append((page, page_similarity))

            if matched_pages:
                self.access_frequency[session_id] = self.access_frequency.get(session_id, 0) + 1
                session.N_visit += 1
                session.last_visit_time = current_timestamp
                session.R_recency = compute_time_decay(session.timestamp or current_timestamp, current_timestamp)
                session.access_count += 1
                session.H_segment = compute_segment_heat(session)
                matched_sessions.append(
                    {
                        "session": session,
                        "matched_pages": matched_pages,
                        "session_similarity": overall_score,
                    }
                )

        matched_sessions.sort(key=lambda item: item["session_similarity"], reverse=True)
        return matched_sessions

    def retrieve(self, query: str, top_k: Optional[int] = None) -> dict:
        top_k = top_k or self.retrieve_num
        matched_sessions = self._search_sessions_by_summary(query)
        top_pages_heap: list[tuple[float, PageRecord]] = []
        for session_item in matched_sessions:
            for page, overall_score in session_item["matched_pages"]:
                top_pages_heap.append((overall_score, page))
        top_pages_heap.sort(key=lambda item: item[0], reverse=True)
        top_pages = top_pages_heap[: self.queue_capacity]
        long_term_items = self._search_knowledge(query)

        combined_texts = []
        seen = set()
        for profile in self.user_profiles.values():
            text = f"[user-profile | updated={profile.last_updated}] {profile.data}"
            if text not in seen:
                combined_texts.append(text)
                seen.add(text)

        for item in self.assistant_knowledge:
            text = f"[assistant-knowledge | ts={item.timestamp}] {item.knowledge}"
            if text not in seen:
                combined_texts.append(text)
                seen.add(text)

        for item in long_term_items:
            text = f"[long-term-knowledge | ts={item.timestamp}] {item.knowledge}"
            if text not in seen:
                combined_texts.append(text)
                seen.add(text)

        for score, page in top_pages:
            parts = [
                f"[historical-memory | score={score:.3f}]",
                f"User: {page.user_input}",
            ]
            if page.agent_response:
                parts.append(f"Assistant: {page.agent_response}")
            parts.append(f"Time: {page.timestamp}")
            if page.meta_info:
                parts.append(f"Conversation chain overview: {page.meta_info}")
            text = "\n".join(parts)
            if text not in seen:
                combined_texts.append(text)
                seen.add(text)

        return {
            "retrieval_queue": [
                {
                    "page_id": page.page_id,
                    "user_input": page.user_input,
                    "agent_response": page.agent_response,
                    "timestamp": page.timestamp,
                    "meta_info": page.meta_info,
                    "score": score,
                }
                for score, page in top_pages
            ],
            "long_term_knowledge": [
                {
                    "knowledge": item.knowledge,
                    "timestamp": item.timestamp,
                }
                for item in long_term_items
            ],
            "user_profiles": [
                {
                    "data": profile.data,
                    "last_updated": profile.last_updated,
                }
                for profile in self.user_profiles.values()
            ],
            "assistant_knowledge": [
                {
                    "knowledge": item.knowledge,
                    "timestamp": item.timestamp,
                }
                for item in self.assistant_knowledge
            ],
            "combined_texts": combined_texts[:top_k],
        }

    def save(self) -> None:
        payload = {
            "base_timestamp": self.base_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "next_message_index": self.next_message_index,
            "next_page_index": self.next_page_index,
            "next_session_index": self.next_session_index,
            "short_term_memory": [asdict(item) for item in self.short_term_memory],
            "sessions": {
                key: {
                    **asdict(session),
                    "details": [asdict(page) for page in session.details],
                }
                for key, session in self.sessions.items()
            },
            "access_frequency": self.access_frequency,
            "user_profiles": {key: asdict(value) for key, value in self.user_profiles.items()},
            "knowledge_base": [asdict(item) for item in self.knowledge_base],
            "assistant_knowledge": [asdict(item) for item in self.assistant_knowledge],
            "last_evicted_page_id": self.last_evicted_page_id,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.state_path.exists():
            raise FileNotFoundError(f"MemoryOS state file not found at {self.state_path}")

        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.base_timestamp = datetime.strptime(payload["base_timestamp"], "%Y-%m-%d %H:%M:%S")
        self.next_message_index = int(payload.get("next_message_index", 1))
        self.next_page_index = int(payload.get("next_page_index", 1))
        self.next_session_index = int(payload.get("next_session_index", 1))
        self.short_term_memory = [ShortTermMessage(**item) for item in payload.get("short_term_memory", [])]
        self.sessions = {}
        for key, value in payload.get("sessions", {}).items():
            details = [PageRecord(**item) for item in value.get("details", [])]
            session_data = dict(value)
            session_data["details"] = details
            self.sessions[key] = SessionRecord(**session_data)
        self.access_frequency = {str(key): int(value) for key, value in payload.get("access_frequency", {}).items()}
        self.user_profiles = {
            key: UserProfileRecord(**value)
            for key, value in payload.get("user_profiles", {}).items()
        }
        self.knowledge_base = [KnowledgeEntry(**item) for item in payload.get("knowledge_base", [])]
        self.assistant_knowledge = [KnowledgeEntry(**item) for item in payload.get("assistant_knowledge", [])]
        self.last_evicted_page_id = payload.get("last_evicted_page_id")

    def memory_count(self) -> int:
        return sum(len(session.details) for session in self.sessions.values())
