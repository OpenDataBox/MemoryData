"""Repository adapter for the vendored A-MEM source."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from utils.locomo_utils import dedupe_preserve_order, parse_locomo_source_ids


CURRENT_DIR = Path(__file__).resolve().parent
A_MEM_SOURCE_ROOT = CURRENT_DIR / "source" / "a_mem"
if str(A_MEM_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(A_MEM_SOURCE_ROOT))

from memory_layer import AgenticMemorySystem, MemoryNote, SimpleEmbeddingRetriever


class HybridBM25EmbeddingRetriever:
    """Adapter-local hybrid retriever that preserves A-MEM's embedding backend."""

    def __init__(
        self,
        *,
        model_name: str,
        provider: Optional[str],
        api_key: Optional[str],
        api_base: Optional[str],
        bm25_weight: float,
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.bm25_weight = self._normalize_weight(bm25_weight)
        self.semantic = SimpleEmbeddingRetriever(
            model_name,
            provider=provider,
            api_key=api_key,
            api_base=api_base,
        )
        self.bm25 = None
        self.corpus = []
        self.document_ids = {}

    @staticmethod
    def _normalize_weight(raw_weight: float) -> float:
        try:
            value = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid hybrid BM25 weight: {raw_weight!r}") from exc
        return max(0.0, min(1.0, value))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        normalized = str(text or "").strip().lower()
        return normalized.split() or [""]

    @staticmethod
    def _normalize_scores(scores) -> np.ndarray:
        score_array = np.asarray(scores, dtype=np.float32)
        if score_array.size == 0:
            return score_array
        score_min = float(score_array.min())
        score_max = float(score_array.max())
        score_range = score_max - score_min
        if score_range <= 1e-6:
            return np.ones_like(score_array) if score_max > 0 else np.zeros_like(score_array)
        return (score_array - score_min) / (score_range + 1e-6)

    def _rebuild_bm25(self) -> None:
        if not self.corpus:
            self.bm25 = None
            return
        tokenized_corpus = [self._tokenize(document) for document in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_documents(self, documents) -> None:
        normalized_documents = [str(document) for document in (documents or []) if str(document).strip()]
        if not normalized_documents:
            return
        self.semantic.add_documents(normalized_documents)
        self.corpus = list(self.semantic.corpus)
        self.document_ids = dict(self.semantic.document_ids)
        self._rebuild_bm25()

    def search(self, query: str, k: int = 5) -> list[int]:
        if not self.corpus:
            return []

        semantic_embeddings = self.semantic.embeddings
        if semantic_embeddings is None or len(semantic_embeddings) == 0:
            return []

        query_embedding = self.semantic.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], semantic_embeddings)[0]
        semantic_scores = self._normalize_scores(semantic_scores)

        bm25_scores = np.zeros(len(self.corpus), dtype=np.float32)
        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(self._tokenize(query))
            bm25_scores = self._normalize_scores(bm25_scores)

        hybrid_scores = self.bm25_weight * bm25_scores + (1.0 - self.bm25_weight) * semantic_scores
        top_k = min(int(k), len(self.corpus))
        if top_k <= 0:
            return []
        return np.argsort(hybrid_scores)[-top_k:][::-1].tolist()


class AMemAdapter:
    """Thin wrapper around A-MEM for the benchmark harness."""

    def __init__(
        self,
        *,
        model: str,
        backend: str,
        retrieve_k: int,
        embedding_model: str,
        embedding_provider: Optional[str],
        embedding_api_key: Optional[str],
        embedding_api_base: Optional[str],
        api_key: Optional[str],
        api_base: Optional[str],
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        state_path: Optional[str] = None,
        retriever_type: str = "dense",
        hybrid_bm25_weight: float = 0.5,
    ) -> None:
        self.retrieve_k = retrieve_k
        self.state_path = Path(state_path) if state_path else None
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_api_key = embedding_api_key
        self.embedding_api_base = embedding_api_base
        normalized_retriever_type = str(retriever_type or "dense").strip().lower()
        if normalized_retriever_type not in {"dense", "hybrid"}:
            raise ValueError(
                f"Unsupported A-MEM retriever type: {retriever_type!r}. Supported values are 'dense' and 'hybrid'."
            )
        self.retriever_type = normalized_retriever_type
        self.hybrid_bm25_weight = HybridBM25EmbeddingRetriever._normalize_weight(hybrid_bm25_weight)
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.memory_system = AgenticMemorySystem(
            model_name=embedding_model,
            llm_backend=backend,
            llm_model=model,
            api_key=api_key,
            api_base=api_base,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        if self.retriever_type == "hybrid":
            self._install_hybrid_retriever()

    def _should_disable_qwen3_thinking(self) -> bool:
        model_name = str(self.model or "").strip().lower()
        return "qwen3" in model_name

    def _build_hybrid_retriever(self) -> HybridBM25EmbeddingRetriever:
        return HybridBM25EmbeddingRetriever(
            model_name=self.embedding_model,
            provider=self.embedding_provider,
            api_key=self.embedding_api_key,
            api_base=self.embedding_api_base,
            bm25_weight=self.hybrid_bm25_weight,
        )

    @staticmethod
    def _memory_to_document(memory: MemoryNote) -> str:
        return (
            "content:" + str(memory.content or "")
            + " context:" + str(memory.context or "")
            + " keywords: " + ", ".join(memory.keywords or [])
            + " tags: " + ", ".join(memory.tags or [])
        )

    def _rebuild_hybrid_retriever(self) -> None:
        retriever = self._build_hybrid_retriever()
        documents = [self._memory_to_document(memory) for memory in self.memory_system.memories.values()]
        if documents:
            retriever.add_documents(documents)
        self.memory_system.retriever = retriever

    def _install_hybrid_retriever(self) -> None:
        def _consolidate_memories_override():
            self._rebuild_hybrid_retriever()

        self.memory_system.consolidate_memories = _consolidate_memories_override
        self._rebuild_hybrid_retriever()

    def add_chunk(self, content: str, timestamp: Optional[str] = None) -> str:
        return self.memory_system.add_note(content, time=timestamp)

    def retrieve(self, question: str) -> str:
        retrieved_context, _ = self.retrieve_with_source_groups(question)
        return retrieved_context

    def retrieve_items(self, question: str) -> list[str]:
        if not self.memory_system.memories:
            return []

        indices = self.memory_system.retriever.search(question, self.retrieve_k)
        all_memories = list(self.memory_system.memories.values())
        retrieved_items = []
        for index in indices:
            if 0 <= index < len(all_memories):
                memory_text = str(all_memories[index].content or "").strip()
                if memory_text:
                    retrieved_items.append(memory_text)
        return retrieved_items

    def retrieve_with_source_groups(self, question: str) -> tuple[str, list[list[str]]]:
        if not self.memory_system.memories:
            return "", []

        formatted_lines = []
        source_id_groups = []
        memories_by_id = self.memory_system.memories
        all_memories = list(memories_by_id.values())
        indices = self.memory_system.retriever.search(question, self.retrieve_k)

        for index in indices:
            if index < 0 or index >= len(all_memories):
                continue

            root_memory = all_memories[index]
            grouped_memories = [root_memory]
            neighbor_count = 0

            for neighbor in root_memory.links or []:
                if isinstance(neighbor, int):
                    if neighbor < 0 or neighbor >= len(all_memories):
                        continue
                    neighbor_memory = all_memories[neighbor]
                else:
                    neighbor_memory = memories_by_id.get(neighbor)
                    if neighbor_memory is None:
                        continue

                grouped_memories.append(neighbor_memory)
                if neighbor_count >= self.retrieve_k:
                    break
                neighbor_count += 1

            grouped_source_ids = []
            for memory in grouped_memories:
                grouped_source_ids.extend(parse_locomo_source_ids(memory.content))
                formatted_lines.append(self._format_memory_note(memory))

            normalized_source_ids = dedupe_preserve_order(grouped_source_ids)
            if normalized_source_ids:
                source_id_groups.append(normalized_source_ids)

        return "\n".join(formatted_lines), source_id_groups

    def ask(self, question: str) -> str:
        return self.ask_with_retrieved_context(question, self.retrieve(question))

    def ask_with_retrieved_context(self, question: str, retrieved_context: str) -> str:
        if not retrieved_context.strip():
            return "I could not find enough relevant memory to answer the question."

        max_context_chars = 60000
        if len(retrieved_context) > max_context_chars:
            retrieved_context = retrieved_context[:max_context_chars]

        request_kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the question strictly based on the retrieved memory. "
                        "If the memory is insufficient, say that briefly. Keep the answer concise."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Retrieved memory:\n{retrieved_context}\n\nQuestion: {question}",
                },
            ],
            "temperature": 0.0,
            "max_tokens": 1000,
        }
        if self._should_disable_qwen3_thinking():
            request_kwargs["extra_body"] = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }

        response = self.client.chat.completions.create(
            **request_kwargs,
        )
        if not getattr(response, "choices", None):
            return "The model returned no choices."

        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            return "The model returned no message."

        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_value = part.get("text")
                    if text_value:
                        text_parts.append(text_value)
                else:
                    text_value = getattr(part, "text", None)
                    if text_value:
                        text_parts.append(text_value)
            if text_parts:
                return "\n".join(text_parts)

        reasoning_content = getattr(message, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return reasoning_content

        return "The model returned an empty response."

    @staticmethod
    def _format_memory_note(memory: MemoryNote) -> str:
        return (
            "talk start time:" + str(memory.timestamp or "")
            + "memory content: " + str(memory.content or "")
            + "memory context: " + str(memory.context or "")
            + "memory keywords: " + str(memory.keywords or [])
            + "memory tags: " + str(memory.tags or [])
        )

    def save(self) -> None:
        if not self.state_path:
            return

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memories": [
                {
                    "content": memory.content,
                    "id": memory.id,
                    "keywords": memory.keywords,
                    "links": memory.links,
                    "importance_score": memory.importance_score,
                    "retrieval_count": memory.retrieval_count,
                    "timestamp": memory.timestamp,
                    "last_accessed": memory.last_accessed,
                    "context": memory.context,
                    "evolution_history": memory.evolution_history,
                    "category": memory.category,
                    "tags": memory.tags,
                }
                for memory in self.memory_system.memories.values()
            ]
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            raise FileNotFoundError(f"A-MEM state file not found at {self.state_path}")

        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        restored = {}
        for memory_payload in payload.get("memories", []):
            note = MemoryNote(
                content=memory_payload["content"],
                id=memory_payload.get("id"),
                keywords=memory_payload.get("keywords"),
                links=memory_payload.get("links"),
                importance_score=memory_payload.get("importance_score"),
                retrieval_count=memory_payload.get("retrieval_count"),
                timestamp=memory_payload.get("timestamp"),
                last_accessed=memory_payload.get("last_accessed"),
                context=memory_payload.get("context"),
                evolution_history=memory_payload.get("evolution_history"),
                category=memory_payload.get("category"),
                tags=memory_payload.get("tags"),
            )
            restored[note.id] = note

        self.memory_system.memories = restored
        if self.retriever_type == "hybrid":
            self._rebuild_hybrid_retriever()
        else:
            self.memory_system.consolidate_memories()

    def memory_count(self) -> int:
        return len(self.memory_system.memories)
