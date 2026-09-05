"""
Hybrid Router for KV Cache Memory System.

Combines three scoring methods for robust memory block selection:
1. Summary embedding similarity
2. Original text embedding similarity (chunked)
3. BM25 keyword scoring

Supports both KV Cache mode and Text Storage mode.
"""

import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.agent.base import BaseAgent
from src.memory.router.bm25_scorer import BM25Scorer, create_multilingual_tokenizer
from src.memory.router.embedding_factory import (
    BaseEmbeddingModel,
    EmbeddingModelFactory,
    cosine_similarity,
)
from src.utils.parser import limit_memory_segments
from src.utils.prompt import ROUTER_SYS_PROMPT

if TYPE_CHECKING:
    from src.memory.memory_agent.agent import MemoryAgent
    from src.memory.memory_agent.text_agent import TextMemoryAgent

logger = logging.getLogger(__name__)


def _empty_cuda_cache() -> None:
    """Clear CUDA state only when KV mode has already imported torch."""
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        torch_module.cuda.empty_cache()


def chunk_text(
    text: str,
    max_chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into smaller chunks while preserving meaning.

    Args:
        text: Text to split.
        max_chunk_size: Maximum chunk size in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    # Try to split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Handle very long sentences
            if len(sentence) > max_chunk_size:
                # Split by words
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chunk_size:
                        current_chunk += (" " if current_chunk else "") + word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class HybridRouter(BaseAgent):
    """
    Hybrid Router that combines multiple scoring methods for block selection.

    Supports three scoring components:
    1. Summary embedding similarity: Match query against block summaries
    2. Text embedding similarity: Match query against chunked original text
    3. BM25 keyword scoring: Traditional keyword matching

    Works with both KV Cache mode and Text Storage mode.
    """

    def __init__(
        self,
        openai_config: Optional[Dict[str, Any]] = None,
        system_prompt: str = ROUTER_SYS_PROMPT,
        max_memory_segments: Optional[int] = None,
        max_blocks: int = 5,
        max_parallel_cache_loads: int = 8,
        query_batch_size: int = 4,
        enable_router: bool = True,
        # Hybrid router specific settings
        embedding_provider: str = "huggingface",
        embedding_model: Optional[str] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        # Scoring weights
        summary_weight: float = 0.3,
        text_weight: float = 0.4,
        bm25_weight: float = 0.3,
        # Top-k settings
        summary_top_k: int = 10,
        text_top_k: int = 20,
        bm25_top_k: int = 10,
        # Text chunking settings
        text_chunk_size: int = 512,
        text_chunk_overlap: int = 50,
        # Fallback to LLM routing
        use_llm_fallback: bool = False,
        # BM25 tokenizer settings
        bm25_use_jieba: bool = True,  # Use jieba for Chinese text support
        # BM25 boost threshold - blocks with BM25 score above this are auto-selected
        bm25_boost_threshold: Optional[float] = None,  # e.g., 0.8 means top 20% BM25 scores
    ) -> None:
        """
        Initialize HybridRouter.

        Args:
            openai_config: OpenAI API configuration for LLM fallback.
            system_prompt: System prompt for LLM routing (fallback).
            max_memory_segments: Maximum memory segments per query result.
            max_blocks: Maximum blocks to query.
            max_parallel_cache_loads: Max parallel KV cache loads to GPU.
            query_batch_size: Queries to batch together.
            enable_router: If False, skip routing and query ALL blocks.
            embedding_provider: Embedding provider ('huggingface' or 'openai').
            embedding_model: Embedding model name (provider-specific).
            embedding_config: Additional embedding configuration.
            summary_weight: Weight for summary embedding score (0-1).
            text_weight: Weight for text embedding score (0-1).
            bm25_weight: Weight for BM25 score (0-1).
            summary_top_k: Top-k summaries to consider.
            text_top_k: Top-k text chunks to consider.
            bm25_top_k: Top-k BM25 results to consider.
            text_chunk_size: Maximum chunk size for text embedding.
            text_chunk_overlap: Overlap between text chunks.
            use_llm_fallback: Use LLM for routing when embeddings fail.
            bm25_use_jieba: Use jieba tokenizer for Chinese text support.
            bm25_boost_threshold: If a block's normalized BM25 score exceeds this
                threshold (0-1), it will be auto-selected regardless of combined score.
                This is useful for exact keyword matches. None to disable.
        """
        # Initialize OpenAI client only if needed for fallback
        if openai_config and use_llm_fallback:
            super().__init__(openai_config, system_prompt)
        else:
            self.openai_config = openai_config
            self.system_prompt = system_prompt

        self.name = "hybrid_router"
        self.agent: List[Any] = []  # Memory agents (blocks)
        self.max_memory_segments = max_memory_segments
        self.max_blocks = max_blocks
        self.max_parallel_cache_loads = max_parallel_cache_loads
        self.query_batch_size = query_batch_size
        self.enable_router = enable_router

        # Scoring weights (normalize to sum to 1)
        total_weight = summary_weight + text_weight + bm25_weight
        self.summary_weight = summary_weight / total_weight
        self.text_weight = text_weight / total_weight
        self.bm25_weight = bm25_weight / total_weight

        # Top-k settings
        self.summary_top_k = summary_top_k
        self.text_top_k = text_top_k
        self.bm25_top_k = bm25_top_k

        # Text chunking settings
        self.text_chunk_size = text_chunk_size
        self.text_chunk_overlap = text_chunk_overlap

        # LLM fallback
        self.use_llm_fallback = use_llm_fallback

        # BM25 tokenizer settings
        self.bm25_use_jieba = bm25_use_jieba

        # BM25 boost threshold for auto-selection
        self.bm25_boost_threshold = bm25_boost_threshold

        # Initialize embedding model
        self._embedding_model: Optional[BaseEmbeddingModel] = None
        self._embedding_provider = embedding_provider
        self._embedding_model_name = embedding_model
        self._embedding_config = embedding_config or {}

        # Cache for embeddings (lazy initialized)
        self._summary_embeddings: Optional[np.ndarray] = None
        self._text_chunks_per_block: List[List[str]] = []
        self._text_chunk_embeddings: Optional[np.ndarray] = None
        self._chunk_to_block_map: List[int] = []  # Maps chunk index to block index
        self._embeddings_dirty = True  # Flag to rebuild embeddings

        # BM25 scorer (lazy initialized)
        self._bm25_scorer: Optional[BM25Scorer] = None
        self._bm25_documents: List[str] = []

        router_status = "enabled (hybrid)" if enable_router else "DISABLED (querying all blocks)"
        logger.info(
            f"HybridRouter initialized (status={router_status}, "
            f"weights=[summary={self.summary_weight:.2f}, text={self.text_weight:.2f}, bm25={self.bm25_weight:.2f}], "
            f"max_blocks={max_blocks})"
        )

    def _get_embedding_model(self) -> BaseEmbeddingModel:
        """Get or create embedding model (lazy initialization)."""
        if self._embedding_model is None:
            # For OpenAI embeddings, pass openai_config if not in embedding_config
            config = self._embedding_config.copy()
            if self._embedding_provider == "openai" and "api_key" not in config:
                if self.openai_config and "api_key" in self.openai_config:
                    config["api_key"] = self.openai_config["api_key"]
                    if "base_url" in self.openai_config:
                        config["base_url"] = self.openai_config["base_url"]

            self._embedding_model = EmbeddingModelFactory.create(
                provider=self._embedding_provider,
                model_name=self._embedding_model_name,
                config=config,
            )
        return self._embedding_model

    def add_blocks(self, memory_agent: Union["MemoryAgent", "TextMemoryAgent"]) -> None:
        """
        Add inactive memory agent to router for querying.

        Args:
            memory_agent: Memory agent to add.
        """
        if memory_agent.is_active:
            logger.warning("Attempted to add active memory agent to router")
            return
        self.agent.append(memory_agent)
        self._embeddings_dirty = True  # Mark embeddings for rebuild
        logger.info(f"Added memory block to router, total blocks: {len(self.agent)}")

    def _rebuild_embeddings(self) -> None:
        """Rebuild all embedding caches."""
        if not self.agent or not self._embeddings_dirty:
            return

        logger.info(f"Rebuilding embeddings for {len(self.agent)} blocks")
        embedding_model = self._get_embedding_model()

        # Part 1: Build summary embeddings
        summaries = []
        for agent in self.agent:
            summary = agent.summary or ""
            summaries.append(summary)

        if summaries and any(summaries):
            self._summary_embeddings = embedding_model.embed(summaries)
        else:
            self._summary_embeddings = None

        # Part 2: Build text chunk embeddings
        self._text_chunks_per_block = []
        self._chunk_to_block_map = []
        all_chunks = []

        for block_idx, agent in enumerate(self.agent):
            # Get original texts from agent
            original_texts = self._get_agent_texts(agent)
            block_text = "\n".join(original_texts)

            # Chunk the text
            chunks = chunk_text(
                block_text,
                max_chunk_size=self.text_chunk_size,
                overlap=self.text_chunk_overlap,
            )
            self._text_chunks_per_block.append(chunks)

            for chunk in chunks:
                all_chunks.append(chunk)
                self._chunk_to_block_map.append(block_idx)

        if all_chunks:
            self._text_chunk_embeddings = embedding_model.embed(all_chunks)
        else:
            self._text_chunk_embeddings = None

        # Part 3: Build BM25 index
        # Use full text per block for BM25
        self._bm25_documents = []
        for agent in self.agent:
            original_texts = self._get_agent_texts(agent)
            full_text = "\n".join(original_texts)
            self._bm25_documents.append(full_text)

        if self._bm25_documents:
            # Create tokenizer with Chinese support if enabled
            tokenizer = create_multilingual_tokenizer(use_jieba=self.bm25_use_jieba)
            self._bm25_scorer = BM25Scorer(tokenizer=tokenizer)
            self._bm25_scorer.fit(self._bm25_documents)
        else:
            self._bm25_scorer = None

        self._embeddings_dirty = False
        logger.info(
            f"Embeddings rebuilt: {len(summaries)} summaries, "
            f"{len(all_chunks)} text chunks, {len(self._bm25_documents)} BM25 docs"
        )

    def _get_agent_texts(self, agent: Any) -> List[str]:
        """Get original texts from an agent (works for both KV and Text modes)."""
        if hasattr(agent, 'get_original_texts'):
            return agent.get_original_texts()
        elif hasattr(agent, 'original_texts'):
            return agent.original_texts if agent.original_texts else []
        elif hasattr(agent, 'current_block') and hasattr(agent.current_block, 'chunks'):
            # Text mode fallback
            return [chunk['text'] for chunk in agent.current_block.chunks]
        return []

    def _score_summary_similarity(self, query: str) -> np.ndarray:
        """
        Score blocks by summary embedding similarity.

        Args:
            query: Query string.

        Returns:
            Array of similarity scores per block.
        """
        if self._summary_embeddings is None or len(self._summary_embeddings) == 0:
            return np.zeros(len(self.agent))

        embedding_model = self._get_embedding_model()
        query_embedding = embedding_model.embed(query)

        similarities = cosine_similarity(query_embedding, self._summary_embeddings)
        return similarities.flatten()

    def _score_text_similarity(self, query: str) -> np.ndarray:
        """
        Score blocks by text chunk embedding similarity.

        This method:
        1. Computes query embedding
        2. Finds top-k most similar text chunks
        3. Maps chunks back to blocks and aggregates scores

        Args:
            query: Query string.

        Returns:
            Array of aggregated similarity scores per block.
        """
        if self._text_chunk_embeddings is None or len(self._text_chunk_embeddings) == 0:
            return np.zeros(len(self.agent))

        embedding_model = self._get_embedding_model()
        query_embedding = embedding_model.embed(query)

        # Get similarities for all chunks
        chunk_similarities = cosine_similarity(query_embedding, self._text_chunk_embeddings)
        chunk_similarities = chunk_similarities.flatten()

        # Get top-k chunks
        top_k = min(self.text_top_k, len(chunk_similarities))
        top_indices = np.argsort(chunk_similarities)[-top_k:][::-1]

        # Aggregate scores by block
        block_scores = np.zeros(len(self.agent))
        for chunk_idx in top_indices:
            block_idx = self._chunk_to_block_map[chunk_idx]
            block_scores[block_idx] += chunk_similarities[chunk_idx]

        # Normalize by number of contributions
        for block_idx in range(len(self.agent)):
            count = sum(1 for ci in top_indices if self._chunk_to_block_map[ci] == block_idx)
            if count > 0:
                block_scores[block_idx] /= count

        return block_scores

    def _score_bm25(self, query: str) -> np.ndarray:
        """
        Score blocks by BM25 keyword matching.

        Args:
            query: Query string.

        Returns:
            Array of BM25 scores per block.
        """
        if self._bm25_scorer is None:
            return np.zeros(len(self.agent))

        scores = self._bm25_scorer.score(query)
        return np.array(scores)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score > 1e-10:
            return (scores - min_score) / (max_score - min_score)
        return np.zeros_like(scores)

    def _map_blocks(self, user_query: str, max_blocks: Optional[int] = None) -> List[Any]:
        """
        Map the user query to relevant memory agents using hybrid scoring.

        Args:
            user_query: The user query.
            max_blocks: Maximum number of blocks to return.

        Returns:
            List of selected memory agents.
        """
        if max_blocks is None:
            max_blocks = self.max_blocks

        if not self.agent:
            logger.debug("No memory agents available for mapping")
            return []

        # If router is disabled, return ALL agents
        if not self.enable_router:
            logger.info(f"Router disabled, returning all {len(self.agent)} blocks")
            return self.agent.copy()

        # Rebuild embeddings if needed
        self._rebuild_embeddings()

        try:
            # Compute all three scores
            summary_scores = self._score_summary_similarity(user_query)
            text_scores = self._score_text_similarity(user_query)
            bm25_scores = self._score_bm25(user_query)

            # Normalize scores
            summary_scores_norm = self._normalize_scores(summary_scores)
            text_scores_norm = self._normalize_scores(text_scores)
            bm25_scores_norm = self._normalize_scores(bm25_scores)

            # Combine with weights
            combined_scores = (
                self.summary_weight * summary_scores_norm +
                self.text_weight * text_scores_norm +
                self.bm25_weight * bm25_scores_norm
            )

            # BM25 boost: auto-select blocks with high keyword match
            boosted_indices = set()
            if self.bm25_boost_threshold is not None and self.bm25_boost_threshold > 0:
                for idx, bm25_score in enumerate(bm25_scores_norm):
                    if bm25_score >= self.bm25_boost_threshold:
                        boosted_indices.add(idx)
                if boosted_indices:
                    logger.info(
                        f"BM25 boost: {len(boosted_indices)} blocks auto-selected "
                        f"(threshold={self.bm25_boost_threshold:.2f})"
                    )

            # Get top-k blocks by combined score
            top_indices = np.argsort(combined_scores)[-max_blocks:][::-1]

            # Merge boosted and top-k indices, prioritizing boosted ones
            selected_indices = []
            selected_set = set()

            # First add boosted blocks (sorted by BM25 score desc)
            boosted_sorted = sorted(boosted_indices, key=lambda i: bm25_scores_norm[i], reverse=True)
            for idx in boosted_sorted:
                if len(selected_indices) < max_blocks and idx not in selected_set:
                    selected_indices.append(idx)
                    selected_set.add(idx)

            # Then add top combined score blocks
            for idx in top_indices:
                if len(selected_indices) >= max_blocks:
                    break
                if idx not in selected_set and combined_scores[idx] > 0:
                    selected_indices.append(idx)
                    selected_set.add(idx)

            # Build selected agents list
            selected_agents = [self.agent[idx] for idx in selected_indices]

            # Ensure at least one block if agents exist
            if not selected_agents and self.agent:
                selected_agents = [self.agent[0]]

            # Logging
            if selected_indices:
                top_idx = selected_indices[0]
                boost_info = f", boosted={len(boosted_indices)}" if boosted_indices else ""
                logger.info(
                    f"Hybrid router selected {len(selected_agents)} blocks "
                    f"(scores: summary={summary_scores_norm[top_idx]:.3f}, "
                    f"text={text_scores_norm[top_idx]:.3f}, "
                    f"bm25={bm25_scores_norm[top_idx]:.3f}{boost_info})"
                )
            return selected_agents

        except Exception as e:
            logger.error(f"Hybrid scoring failed: {e}", exc_info=True)
            if self.use_llm_fallback and hasattr(self, 'llm'):
                logger.info("Falling back to LLM routing")
                return self._llm_map_blocks(user_query, max_blocks)
            # Return all blocks as fallback
            return self.agent[:max_blocks]

    def _llm_map_blocks(self, user_query: str, max_blocks: int) -> List[Any]:
        """Fallback LLM-based routing (original implementation)."""
        summary_blocks = "\n".join(
            map(
                lambda idx_agent: f"""
            <summary>
                <index>{idx_agent[0]}</index>
                <content>{idx_agent[1].summary}</content>
            </summary>
            """,
                enumerate(self.agent),
            )
        )

        user_prompt_formatted = f"""
            <query> {user_query} </query>
            <summary_list>
            {summary_blocks}
            </summary_list>
            Please provide the indices of the most relevant memory summaries to the query.
        """
        response = self.generate_response(user_prompt_formatted, max_tokens=4096)

        try:
            match = re.search(
                r"<summary_index>\s*(.*?)\s*</summary_index>",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                indices_str = match.group(1)
                indices = [int(num) for num in re.findall(r"\d+", indices_str)]
                seen = set()
                unique_indices = []
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        unique_indices.append(idx)
                valid_indices = [idx for idx in unique_indices if 0 <= idx < len(self.agent)]
                selected_indices = valid_indices[:max_blocks]
                return [self.agent[idx] for idx in selected_indices]
        except Exception as e:
            logger.error(f"LLM fallback parsing failed: {e}")

        return self.agent[:max_blocks]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Execute tool (not used by router)."""
        pass

    def map_reduce_blocks(self, user_query: str) -> List[str]:
        """
        Collect all responses from relevant memory agents.

        Args:
            user_query: The user query.

        Returns:
            List of responses from relevant memory blocks.
        """
        relevant_agents_list = self._map_blocks(user_query)
        if not relevant_agents_list:
            logger.debug("No relevant agents found for query")
            return []

        num_agents = len(relevant_agents_list)
        logger.info(f"Processing {num_agents} relevant memory blocks")

        # Pre-load all caches to CPU in parallel
        preload_workers = min(num_agents, self.max_parallel_cache_loads)
        logger.debug(f"Pre-loading {num_agents} caches with {preload_workers} workers")

        with ThreadPoolExecutor(max_workers=preload_workers) as executor:
            list(executor.map(lambda agent: agent.preload_cache(), relevant_agents_list))

        # Check if using shared model mode
        agents_share_model = (
            len(relevant_agents_list) > 0
            and hasattr(relevant_agents_list[0], '_owns_model')
            and not relevant_agents_list[0]._owns_model
        )

        results: List[str] = []

        if agents_share_model and self.query_batch_size > 1:
            logger.info(f"Using batch inference with batch_size={self.query_batch_size}")
            results = self._batch_query_agents(relevant_agents_list, user_query)
        elif agents_share_model:
            logger.debug("Shared model mode, batch_size=1, executing sequentially")
            results = self._sequential_query_agents(relevant_agents_list, user_query)
        else:
            logger.debug("Standalone model mode, executing queries in parallel")
            with ThreadPoolExecutor(max_workers=num_agents) as executor:
                results = list(
                    executor.map(lambda agent: agent.query(user_query), relevant_agents_list)
                )

        # Apply memory segment limit if configured
        if self.max_memory_segments is not None and self.max_memory_segments > 0:
            logger.debug(f"Applying memory segment limit: {self.max_memory_segments}")
            results = [
                limit_memory_segments(result, self.max_memory_segments)
                for result in results
            ]

        _empty_cuda_cache()
        logger.info(f"Collected {len(results)} results from memory blocks")
        return results

    def _sequential_query_agents(
        self, agents: List["MemoryAgent"], user_query: str
    ) -> List[str]:
        """Execute queries sequentially."""
        results: List[str] = []
        for i, agent in enumerate(agents):
            try:
                logger.debug(f"Querying agent {i+1}/{len(agents)}")
                result = agent.query(user_query)
                results.append(result)
                block_id = getattr(agent.current_block, 'block_id', f'agent_{i}')
                logger.info(f"Block {block_id} result: {result}")
            except Exception as e:
                logger.error(f"Error querying agent {i}: {e}", exc_info=True)
                results.append(f"[ERROR] Query failed: {e}")
            _empty_cuda_cache()
        return results

    def _batch_query_agents(
        self, agents: List["MemoryAgent"], user_query: str
    ) -> List[str]:
        """Execute queries using batch inference."""
        num_agents = len(agents)
        all_results: List[str] = [""] * num_agents

        for batch_start in range(0, num_agents, self.query_batch_size):
            batch_end = min(batch_start + self.query_batch_size, num_agents)
            batch_agents = agents[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            logger.debug(
                f"Processing batch {batch_start//self.query_batch_size + 1}: "
                f"agents {batch_start}-{batch_end-1}"
            )

            try:
                batch_results = self._execute_batch_query(batch_agents, user_query)
                for i, result in zip(batch_indices, batch_results):
                    all_results[i] = result
                    agent = agents[i]
                    block_id = getattr(agent.current_block, 'block_id', f'agent_{i}')
                    logger.info(f"Block {block_id} result: {result}")
            except Exception as e:
                logger.error(f"Batch query failed: {e}", exc_info=True)
                for i, agent in zip(batch_indices, batch_agents):
                    try:
                        result = agent.query(user_query)
                        all_results[i] = result
                    except Exception as inner_e:
                        all_results[i] = f"[ERROR] Query failed: {inner_e}"

            _empty_cuda_cache()

        return all_results

    def _execute_batch_query(
        self, agents: List["MemoryAgent"], user_query: str
    ) -> List[str]:
        """Execute a single batch of queries with true parallelism."""
        import torch
        from src.memory.memory_agent.agent import (
            BatchQueryInputs,
            batch_generate,
            pad_kv_cache_for_batch,
        )

        if not agents:
            return []

        batch_size = len(agents)

        model = agents[0].model
        tokenizer = agents[0].tokenizer
        layer_devices = agents[0].layer_devices
        primary_device = agents[0].primary_device

        caches: List[Any] = []
        global_offsets: List[int] = []
        query_inputs: List[Tuple[torch.Tensor, int]] = []
        valid_indices: List[int] = []

        for i, agent in enumerate(agents):
            cache_result = agent.get_cache_for_batch()
            if cache_result is None:
                logger.warning(f"Agent {i} has no cache available")
                continue

            cache, global_offset, _ = cache_result
            query_ids, query_len = agent.format_query_for_batch(user_query)

            caches.append(cache)
            global_offsets.append(global_offset)
            query_inputs.append((query_ids, query_len))
            valid_indices.append(i)

        if not caches:
            return ["No knowledge available."] * batch_size

        batched_cache, cache_lengths = pad_kv_cache_for_batch(
            caches, layer_devices, primary_device
        )
        max_cache_len = max(cache_lengths) if cache_lengths else 0

        max_query_len = max(q[1] for q in query_inputs)
        first_layer_device = layer_devices.get(0, primary_device)

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        batched_input_ids = torch.full(
            (len(valid_indices), max_query_len),
            pad_token_id,
            dtype=torch.long,
            device=first_layer_device
        )

        query_lengths: List[int] = []
        query_pad_sizes: List[int] = []
        for batch_idx, (query_ids, query_len) in enumerate(query_inputs):
            query_lengths.append(query_len)
            query_pad = max_query_len - query_len
            query_pad_sizes.append(query_pad)
            batched_input_ids[batch_idx, query_pad:] = query_ids.squeeze(0)

        batched_position_ids = torch.zeros(
            (len(valid_indices), max_query_len),
            dtype=torch.long,
            device=first_layer_device
        )

        total_len = max_cache_len + max_query_len
        batched_attention_mask = torch.zeros(
            (len(valid_indices), total_len),
            dtype=torch.long,
            device=first_layer_device
        )

        for batch_idx in range(len(valid_indices)):
            cache_len = cache_lengths[batch_idx]
            query_len = query_lengths[batch_idx]
            query_pad = query_pad_sizes[batch_idx]
            global_offset = global_offsets[batch_idx]

            for pos in range(query_len):
                batched_position_ids[batch_idx, query_pad + pos] = global_offset + pos

            cache_padding = max_cache_len - cache_len
            batched_attention_mask[batch_idx, cache_padding:max_cache_len] = 1
            batched_attention_mask[batch_idx, max_cache_len + query_pad:total_len] = 1

        batch_inputs = BatchQueryInputs(
            input_ids=batched_input_ids,
            position_ids=batched_position_ids,
            attention_mask=batched_attention_mask,
            past_key_values=batched_cache,
            cache_lengths=cache_lengths,
            query_lengths=query_lengths,
            global_offsets=global_offsets,
        )

        logger.debug(f"Executing batch generation with {len(valid_indices)} samples")
        batch_responses = batch_generate(
            model=model,
            tokenizer=tokenizer,
            batch_inputs=batch_inputs,
            max_new_tokens=8192,
        )

        cleaned_responses = []
        for response in batch_responses:
            cleaned = agents[0]._remove_thinking_content(response)
            cleaned_responses.append(cleaned)

        results = ["No knowledge available."] * batch_size
        for batch_idx, orig_idx in enumerate(valid_indices):
            results[orig_idx] = cleaned_responses[batch_idx]

        for cache in caches:
            del cache
        del batched_cache
        _empty_cuda_cache()

        return results

