from __future__ import annotations

import math
from typing import Dict, List, Tuple

import igraph as ig
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from .config import MemoryConfig


class MultiGranRetriever:
    MODE_TO_KEY = {
        "session_level": "session",
        "turn_level": "turn",
        "summary_level": "summary",
        "keyword_level": "keyword",
        "hybrid_level": "hybrid",
    }

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config

    def rank(
        self,
        query_emb: torch.Tensor,
        embeddings: Dict[str, torch.Tensor],
        mode: str,
    ) -> Tuple[List[int], torch.Tensor, Dict[str, torch.Tensor]]:
        if not embeddings:
            return [], torch.empty(0), {}

        scores_by_granularity = {
            key: torch.mv(matrix, query_emb)
            for key, matrix in embeddings.items()
        }

        if mode in self.MODE_TO_KEY:
            fused_scores = scores_by_granularity[self.MODE_TO_KEY[mode]]
        elif mode == "lite":
            fused_scores = self._lite_fuse(scores_by_granularity)
        elif mode == "memgas":
            fused_scores = self._memgas_fuse(query_emb, embeddings, scores_by_granularity)
        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. "
                "Choose from session_level/turn_level/summary_level/keyword_level/"
                "hybrid_level/lite/memgas."
            )

        rankings = torch.argsort(fused_scores, descending=True).tolist()
        return rankings, fused_scores, scores_by_granularity

    @staticmethod
    def _lite_fuse(scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Session and summary usually contribute more stable semantics.
        return (
            0.35 * scores["session"]
            + 0.15 * scores["turn"]
            + 0.30 * scores["summary"]
            + 0.20 * scores["keyword"]
        )

    def _memgas_fuse(
        self,
        query_emb: torch.Tensor,
        embeddings: Dict[str, torch.Tensor],
        scores: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        granular_keys = ["session", "turn", "summary", "keyword"]
        router_weights = self._router_weights(scores, granular_keys)
        node_embeddings = self._flatten_nodes(embeddings, granular_keys, router_weights)
        node_scores = torch.mv(node_embeddings, query_emb)
        reset_prob = self._build_reset_prob(node_scores)

        graph = self._build_graph(node_embeddings)
        page_rank = graph.personalized_pagerank(
            damping=self.config.damping,
            directed=False,
            reset=reset_prob.tolist(),
            implementation="prpack",
        )

        page_rank = np.asarray(page_rank, dtype=np.float32)
        per_entry_scores = []
        step = len(granular_keys)
        for start in range(0, len(page_rank), step):
            per_entry_scores.append(float(np.sum(page_rank[start : start + step])))
        return torch.tensor(per_entry_scores, dtype=torch.float32)

    def _router_weights(
        self, scores: Dict[str, torch.Tensor], granular_keys: List[str]
    ) -> torch.Tensor:
        weights = []
        temp = max(self.config.router_temp, 1e-6)
        for key in granular_keys:
            similarity = scores[key]
            prob_dist = F.softmax(similarity / temp, dim=0)
            entropy = -(prob_dist * torch.log(prob_dist + 1e-12)).sum().item()
            max_entropy = math.log(max(len(similarity), 2))
            norm_entropy = entropy / max_entropy
            weight = max(1e-6, 1.0 - norm_entropy)
            weights.append(weight)

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor /= weights_tensor.sum()
        return weights_tensor

    @staticmethod
    def _flatten_nodes(
        embeddings: Dict[str, torch.Tensor],
        granular_keys: List[str],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        weighted_embeddings = [
            weight * embeddings[key]
            for key, weight in zip(granular_keys, weights)
        ]
        nodes = []
        for row_idx in range(weighted_embeddings[0].shape[0]):
            for emb in weighted_embeddings:
                nodes.append(emb[row_idx])
        return torch.stack(nodes, dim=0)

    def _build_reset_prob(self, node_scores: torch.Tensor) -> np.ndarray:
        if node_scores.numel() == 0:
            return np.array([], dtype=np.float32)

        seed_count = min(self.config.num_seednodes, node_scores.numel())
        seed_scores = node_scores.clone()
        if seed_count > 0:
            threshold = torch.topk(seed_scores, k=seed_count).values[-1]
            seed_scores[seed_scores < threshold] = 0

        reset_prob = seed_scores.detach().cpu().numpy()
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)

        if reset_prob.sum() <= 0:
            reset_prob = np.ones_like(reset_prob, dtype=np.float32)
        reset_prob = reset_prob / reset_prob.sum()
        return reset_prob.astype(np.float32)

    def _build_graph(self, node_embeddings: torch.Tensor) -> ig.Graph:
        node_matrix = node_embeddings.detach().cpu().numpy()
        node_count = node_matrix.shape[0]
        if node_count == 0:
            return ig.Graph(0)

        edges: List[Tuple[int, int]] = []
        for node_idx in range(1, node_count):
            previous = node_matrix[:node_idx]
            sim_scores = previous @ node_matrix[node_idx].T
            selected = self._select_edges(sim_scores)
            edges.extend((node_idx, prev_idx) for prev_idx in selected)

        return ig.Graph(n=node_count, edges=edges, directed=False)

    def _select_edges(self, sim_scores: np.ndarray) -> List[int]:
        if sim_scores.size == 0:
            return []
        n_candidates = min(max(self.config.mem_threshold, 1), sim_scores.shape[0])
        top_indices = np.argsort(sim_scores)[::-1][:n_candidates]

        if top_indices.size <= 1:
            return top_indices.tolist()

        n_components = min(max(self.config.n_components, 1), top_indices.size)
        if n_components == 1:
            return top_indices.tolist()

        candidate_scores = sim_scores[top_indices].reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=n_components, random_state=0)
            gmm.fit(candidate_scores)
            labels = gmm.predict(candidate_scores)
            means = gmm.means_.flatten()
            keep_label = int(np.argmax(means))
            selected = top_indices[labels == keep_label]
            if selected.size == 0:
                return [int(top_indices[0])]
            return [int(idx) for idx in selected]
        except Exception:
            return [int(top_indices[0])]
