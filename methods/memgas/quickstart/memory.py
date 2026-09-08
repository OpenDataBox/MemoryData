from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import torch

from .config import MemoryConfig
from .embedder import TextEmbedder
from .llm_client import OpenAICompatibleLLM
from .retriever import MultiGranRetriever
from .schemas import MemoryRecord, RetrievalHit
from .store import MemoryStore


class MemGASMemory:
    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or MemoryConfig()
        self.embedder = TextEmbedder(
            backend=self.config.embedder,
            device=self.config.device,
            batch_size=self.config.batch_size,
        )
        self.llm = OpenAICompatibleLLM(
            api_key=self.config.resolve_api_key(),
            base_url=self.config.resolve_base_url(),
            model=self.config.llm_model,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            max_retries=self.config.llm_max_retries,
            retry_wait_sec=self.config.llm_retry_wait_sec,
        )
        self.store = MemoryStore(self.config.storage_dir)
        self.retriever = MultiGranRetriever(self.config)

    def __len__(self) -> int:
        return len(self.store)

    def add(
        self,
        session: List[str],
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        normalized_session = self._normalize_session(session)
        summary, keywords = self._generate_multigran_info(normalized_session)
        vectors = self._build_vectors(normalized_session, summary, keywords)

        now = self._utc_now()
        memory_id = f"mem_{uuid4().hex[:12]}"
        record = MemoryRecord(
            memory_id=memory_id,
            conversation_id=conversation_id or "default",
            session=normalized_session,
            summary=summary,
            keywords=keywords,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self.store.add(record, vectors)
        if self.config.auto_save:
            self.save()
        return memory_id

    def retrieve(
        self,
        query: str,
        topk: int = 5,
        conversation_id: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        if topk <= 0:
            return []

        indices = self.store.filter_indices(conversation_id=conversation_id)
        if not indices:
            return []

        query_emb = self.embedder.encode([query])[0]
        subset_embeddings = self.store.slice_embeddings(indices)
        use_mode = mode or self.config.default_mode
        rankings, fused_scores, granularity_scores = self.retriever.rank(
            query_emb=query_emb,
            embeddings=subset_embeddings,
            mode=use_mode,
        )

        hits: List[Dict[str, Any]] = []
        for local_idx in rankings[:topk]:
            global_idx = indices[local_idx]
            record = self.store.records[global_idx]
            hit = RetrievalHit(
                memory_id=record.memory_id,
                conversation_id=record.conversation_id,
                score=float(fused_scores[local_idx].item()),
                session=record.session,
                summary=record.summary,
                keywords=record.keywords,
                metadata=record.metadata,
                granularity_scores={
                    key: float(score_list[local_idx].item())
                    for key, score_list in granularity_scores.items()
                },
            )
            hits.append(hit.to_dict())
        return hits

    def save(self) -> None:
        self.store.save()

    def load(self) -> None:
        self.store.load()

    def _generate_multigran_info(self, session: List[str]) -> tuple[str, List[str]]:
        session_text = "\n".join(session)
        summary, keywords = self.llm.summarize_and_keywords(session_text)
        return summary, keywords

    def _build_vectors(
        self, session: List[str], summary: str, keywords: List[str]
    ) -> Dict[str, torch.Tensor]:
        session_text = "\n".join(session)
        keyword_text = "; ".join(keywords)
        hybrid_text = f"{summary}\n{keyword_text}\n{session_text}".strip()

        session_vec = self.embedder.encode([session_text])[0]
        turn_vecs = self.embedder.encode(session)
        turn_vec = turn_vecs.mean(dim=0)
        summary_vec = self.embedder.encode([summary])[0]
        keyword_vec = self.embedder.encode([keyword_text])[0]
        hybrid_vec = self.embedder.encode([hybrid_text])[0]

        return {
            "session": session_vec,
            "turn": turn_vec,
            "summary": summary_vec,
            "keyword": keyword_vec,
            "hybrid": hybrid_vec,
        }

    @staticmethod
    def _normalize_session(session: List[str]) -> List[str]:
        if not isinstance(session, list):
            raise TypeError("session must be a list of strings")
        normalized = [item.strip() for item in session if isinstance(item, str) and item.strip()]
        if not normalized:
            raise ValueError("session must contain at least one non-empty string")
        return normalized

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
