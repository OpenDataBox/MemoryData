"""Repository adapter inspired by the upstream MemTree implementation.

This adapter keeps the core tree-structured memory update logic while adapting it
to the incremental memorize/query/save/load lifecycle used in this repository.
"""

from __future__ import annotations

import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np
from openai import AzureOpenAI, BadRequestError, OpenAI

from .prompt import AGGREGATE_PROMPT
from utils.locomo_utils import dedupe_preserve_order
from utils.provider_utils import ApproximateTokenizer, load_local_hf_tokenizer


def _cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarities between one vector and a matrix."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return np.asarray([], dtype=np.float32)

    vector_norm = np.linalg.norm(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denominator = np.clip(vector_norm * matrix_norms, a_min=1e-12, a_max=None)
    return np.dot(matrix, vector) / denominator


class _BaseEmbedder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


def _is_embedding_context_length_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "maximum context length" in message
        or "requested 0 output tokens" in message
        or "parameter=input_tokens" in message
    )


class _SentenceTransformerEmbedder(_BaseEmbedder):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "MemTree requires sentence-transformers for local embedding models. "
                "Install project requirements before running MemTree."
            ) from exc

        self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        embeddings = model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings


class _OpenAIEmbedder(_BaseEmbedder):
    def __init__(
        self,
        *,
        model_name: str,
        provider: Optional[str],
        base_url: Optional[str],
        api_key: Optional[str],
        azure_endpoint: Optional[str],
        azure_api_version: Optional[str],
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key or "EMPTY"
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self._client = self._create_client()
        self._tokenizer = load_local_hf_tokenizer(model_name) or ApproximateTokenizer()
        self._max_input_tokens = int(os.getenv("MEMTREE_EMBED_MAX_TOKENS", "8192"))
        self._token_safety_margin = int(
            os.getenv("MEMTREE_EMBED_TOKEN_SAFETY_MARGIN", "256")
        )

    def _create_client(self):
        if self.provider == "azure_openai":
            return AzureOpenAI(
                api_key=self.api_key,
                api_version=self.azure_api_version,
                azure_endpoint=self.azure_endpoint,
            )

        if self.base_url:
            return OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                http_client=httpx.Client(trust_env=False),
            )

        return OpenAI(api_key=self.api_key)

    def _fit_text_to_token_budget(self, text: str, max_input_tokens: int) -> str:
        normalized_text = str(text or "")
        if not normalized_text:
            return ""

        tokens = self._tokenizer.encode(normalized_text, disallowed_special=())
        if len(tokens) <= max_input_tokens:
            return normalized_text

        if max_input_tokens <= 32:
            return self._tokenizer.decode(tokens[:max_input_tokens])

        head = max_input_tokens // 2
        tail = max_input_tokens - head
        fitted_tokens = tokens[:head] + tokens[-tail:]
        return self._tokenizer.decode(fitted_tokens)

    def _encode_with_budget(self, texts: Sequence[str], max_input_tokens: int) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self.model_name,
            input=[
                self._fit_text_to_token_budget(text, max_input_tokens)
                for text in list(texts)
            ],
        )
        ordered_items = sorted(response.data, key=lambda item: item.index)
        embeddings = np.asarray([item.embedding for item in ordered_items], dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        primary_budget = max(256, self._max_input_tokens - self._token_safety_margin)
        retry_budget = max(256, primary_budget - max(512, self._token_safety_margin * 2))
        budgets = []
        for budget in (primary_budget, retry_budget):
            if budget not in budgets:
                budgets.append(budget)

        last_error = None
        for index, budget in enumerate(budgets):
            try:
                return self._encode_with_budget(texts, budget)
            except BadRequestError as exc:
                last_error = exc
                if index == len(budgets) - 1 or not _is_embedding_context_length_error(exc):
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("MemTree embedding request failed without a captured exception.")


def _should_use_openai_embeddings(
    model_name: Optional[str],
    provider: Optional[str],
    base_url: Optional[str],
    azure_endpoint: Optional[str],
) -> bool:
    if provider in {"openai", "openai_compatible", "azure_openai"}:
        return True
    if base_url or azure_endpoint:
        return True

    normalized_model = (model_name or "").strip().lower()
    if normalized_model.startswith(("text-embedding-", "azure-text-embedding-")):
        return True
    return False


def _build_embedder(
    *,
    model_name: str,
    provider: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    azure_endpoint: Optional[str],
    azure_api_version: Optional[str],
) -> _BaseEmbedder:
    if _should_use_openai_embeddings(model_name, provider, base_url, azure_endpoint):
        return _OpenAIEmbedder(
            model_name=model_name,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )

    return _SentenceTransformerEmbedder(model_name=model_name)


@dataclass
class _LocalVectorStore:
    vectors: Dict[int, List[float]] = field(default_factory=dict)

    def insert(self, items: Sequence[Dict[str, object]]) -> None:
        for item in items:
            self.vectors[int(item["id"])] = list(item["vector"])

    def upsert(self, items: Sequence[Dict[str, object]]) -> None:
        self.insert(items)

    def get(self, ids: Sequence[int]) -> List[Dict[str, object]]:
        results = []
        for item_id in ids:
            if item_id in self.vectors:
                results.append({"id": item_id, "vector": self.vectors[item_id]})
        return results

    def search(self, query_vector: np.ndarray, limit: int) -> List[Tuple[int, float]]:
        if not self.vectors:
            return []

        ids = list(self.vectors.keys())
        matrix = np.asarray([self.vectors[item_id] for item_id in ids], dtype=np.float32)
        similarities = _cosine_similarity(query_vector, matrix)
        top_indices = np.argsort(-similarities)[:limit]
        return [(ids[index], float(similarities[index])) for index in top_indices]


@dataclass
class MemTreeNode:
    node_id: int
    content: str
    parent_id: Optional[int]
    depth: int


@dataclass
class MemTreeState:
    root_id: int = 0
    next_node_id: int = 1
    nodes: Dict[int, MemTreeNode] = field(default_factory=dict)
    children: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def create(cls) -> "MemTreeState":
        state = cls()
        state.nodes[state.root_id] = MemTreeNode(
            node_id=state.root_id,
            content="Root",
            parent_id=None,
            depth=0,
        )
        return state


class MemTreeAdapter:
    """Incremental MemTree memory for the benchmark runner."""

    def __init__(
        self,
        *,
        state_path: str,
        llm_model: str,
        llm_provider: Optional[str],
        llm_base_url: Optional[str],
        llm_api_key: Optional[str],
        llm_azure_endpoint: Optional[str],
        llm_azure_api_version: Optional[str],
        llm_temperature: float,
        summary_max_tokens: int,
        embedding_model: str,
        embedding_provider: Optional[str],
        embedding_base_url: Optional[str],
        embedding_api_key: Optional[str],
        embedding_azure_endpoint: Optional[str],
        embedding_azure_api_version: Optional[str],
        retrieve_num: int,
        base_threshold: float,
        rate: float,
        max_depth: int,
    ) -> None:
        self.state_path = state_path
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key or "EMPTY"
        self.llm_azure_endpoint = llm_azure_endpoint
        self.llm_azure_api_version = llm_azure_api_version
        self.llm_temperature = llm_temperature
        self.summary_max_tokens = summary_max_tokens
        self.retrieve_num = retrieve_num
        self.base_threshold = base_threshold
        self.rate = rate
        self.max_depth = max(1, max_depth)

        self.embedder = _build_embedder(
            model_name=embedding_model,
            provider=embedding_provider,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            azure_endpoint=embedding_azure_endpoint,
            azure_api_version=embedding_azure_api_version,
        )
        self.summary_client = self._build_summary_client()

        self.tree = MemTreeState.create()
        self.vector_store = _LocalVectorStore()
        self.node_source_ids: Dict[int, List[str]] = {}

    def _build_summary_client(self):
        if self.llm_provider == "azure_openai":
            return AzureOpenAI(
                api_key=self.llm_api_key,
                api_version=self.llm_azure_api_version,
                azure_endpoint=self.llm_azure_endpoint,
            )

        if self.llm_base_url:
            return OpenAI(
                base_url=self.llm_base_url,
                api_key=self.llm_api_key,
                http_client=httpx.Client(trust_env=False),
            )

        if self.llm_provider in {None, "openai", "openai_compatible"} and self.llm_api_key:
            return OpenAI(api_key=self.llm_api_key)

        return None

    def add_chunk(self, content: str, source_ids: Optional[Sequence[str]] = None) -> None:
        normalized_content = (content or "").strip()
        if not normalized_content:
            return

        embedding = self.embedder.encode([normalized_content])[0]
        parent_id, traversed_node_ids = self._find_parent_path(embedding)
        normalized_source_ids = dedupe_preserve_order(source_ids or [])
        self._add_leaf_node(
            normalized_content,
            parent_id,
            embedding,
            source_ids=normalized_source_ids,
        )
        self._update_traversed_nodes(
            traversed_node_ids,
            normalized_content,
            new_source_ids=normalized_source_ids,
        )

    def search(self, query: str, top_k: Optional[int] = None) -> List[str]:
        return [entry["content"] for entry in self.search_entries(query, top_k=top_k)]

    def search_entries(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, object]]:
        normalized_query = (query or "").strip()
        if not normalized_query or not self.vector_store.vectors:
            return []

        query_embedding = self.embedder.encode([normalized_query])[0]
        results = self.vector_store.search(query_embedding, limit=top_k or self.retrieve_num)

        retrieved_entries: List[Dict[str, object]] = []
        seen_contents = set()
        for node_id, _score in results:
            node = self.tree.nodes.get(node_id)
            if not node:
                continue
            content = node.content.strip()
            if not content or content in seen_contents:
                continue
            seen_contents.add(content)
            retrieved_entries.append(
                {
                    "node_id": node_id,
                    "content": content,
                    "source_ids": self.node_source_ids.get(node_id, []),
                }
            )
        return retrieved_entries

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "wb") as file:
            pickle.dump(
                {
                    "tree": self.tree,
                    "vectors": self.vector_store.vectors,
                    "node_source_ids": self.node_source_ids,
                },
                file,
            )

    def load(self) -> None:
        with open(self.state_path, "rb") as file:
            data = pickle.load(file)

        self.tree = data["tree"]
        if not isinstance(self.tree.children, defaultdict):
            self.tree.children = defaultdict(list, self.tree.children)
        self.vector_store = _LocalVectorStore(vectors=data["vectors"])
        self.node_source_ids = data.get("node_source_ids", {})

    def memory_count(self) -> int:
        return max(0, len(self.tree.nodes) - 1)

    def _find_parent_path(self, new_embedding: np.ndarray) -> Tuple[int, List[int]]:
        current_parent_id = self.tree.root_id
        traversed_node_ids: List[int] = []

        while True:
            child_ids = self.tree.children.get(current_parent_id, [])
            if not child_ids:
                break

            child_vectors = self.vector_store.get(child_ids)
            if not child_vectors:
                break

            matrix = np.asarray([item["vector"] for item in child_vectors], dtype=np.float32)
            similarities = _cosine_similarity(new_embedding, matrix)
            if similarities.size == 0:
                break

            current_depth = self.tree.nodes[current_parent_id].depth + 1
            threshold = self._calculate_threshold(current_depth)
            best_index = int(np.argmax(similarities))
            best_score = float(similarities[best_index])

            if best_score <= threshold:
                break

            current_parent_id = int(child_vectors[best_index]["id"])
            traversed_node_ids.append(current_parent_id)

        return current_parent_id, traversed_node_ids

    def _add_leaf_node(
        self,
        content: str,
        parent_id: int,
        embedding: np.ndarray,
        source_ids: Optional[Sequence[str]] = None,
    ) -> int:
        node_id = self.tree.next_node_id
        self.tree.next_node_id += 1

        node = MemTreeNode(
            node_id=node_id,
            content=content,
            parent_id=parent_id,
            depth=self.tree.nodes[parent_id].depth + 1,
        )
        self.tree.nodes[node_id] = node
        self.tree.children[parent_id].append(node_id)
        self.vector_store.insert([{"id": node_id, "vector": embedding.tolist()}])
        normalized_source_ids = dedupe_preserve_order(source_ids or [])
        if normalized_source_ids:
            self.node_source_ids[node_id] = normalized_source_ids
        return node_id

    def _update_traversed_nodes(
        self,
        traversed_node_ids: Sequence[int],
        new_content: str,
        new_source_ids: Optional[Sequence[str]] = None,
    ) -> None:
        if not traversed_node_ids:
            return

        updates: List[Tuple[int, str, str]] = []
        original_source_ids_by_node: Dict[int, List[str]] = {}
        normalized_new_source_ids = dedupe_preserve_order(new_source_ids or [])
        for node_id in traversed_node_ids:
            node = self.tree.nodes[node_id]
            existing_source_ids = self.node_source_ids.get(node_id, [])
            merged_source_ids = dedupe_preserve_order(
                list(existing_source_ids) + normalized_new_source_ids
            )
            if merged_source_ids:
                self.node_source_ids[node_id] = merged_source_ids
            updated_content = self._summarize_node(
                current_content=node.content,
                new_content=new_content,
                n_children=len(self.tree.children.get(node_id, [])),
            )
            if not updated_content or updated_content == node.content:
                continue

            original_source_ids_by_node[node_id] = list(existing_source_ids)
            updates.append((node_id, node.content, updated_content))
            node.content = updated_content

        if not updates:
            return

        updated_embeddings = self.embedder.encode([item[2] for item in updates])
        self.vector_store.upsert(
            [
                {"id": updates[index][0], "vector": updated_embeddings[index].tolist()}
                for index in range(len(updates))
            ]
        )

        deepest_node_id, deepest_original_content, _deepest_updated_content = updates[-1]
        if deepest_original_content.strip():
            original_embedding = self.embedder.encode([deepest_original_content])[0]
            self._add_leaf_node(
                deepest_original_content,
                deepest_node_id,
                original_embedding,
                source_ids=original_source_ids_by_node.get(deepest_node_id, []),
            )

    def _summarize_node(self, *, current_content: str, new_content: str, n_children: int) -> str:
        if not current_content.strip():
            return new_content.strip()

        prompt = AGGREGATE_PROMPT.format(
            new_content=new_content,
            n_children=str(max(1, n_children)),
            current_content=current_content,
        )

        if self.summary_client is None:
            return self._fallback_summary(current_content, new_content)

        try:
            response = self.summary_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.llm_temperature,
                max_tokens=self.summary_max_tokens,
            )
            output = (response.choices[0].message.content or "").strip()
            return output or self._fallback_summary(current_content, new_content)
        except Exception:
            return self._fallback_summary(current_content, new_content)

    def _fallback_summary(self, current_content: str, new_content: str) -> str:
        if not current_content.strip():
            return new_content.strip()
        if not new_content.strip():
            return current_content.strip()

        merged = f"{current_content.strip()} | {new_content.strip()}"
        return merged[:1024].strip()

    def _calculate_threshold(self, current_depth: int) -> float:
        return self.base_threshold * math.exp(self.rate * current_depth / self.max_depth)
