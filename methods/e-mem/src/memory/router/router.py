import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

from src.agent.base import BaseAgent
from src.memory.memory_agent.agent import (
    BatchQueryInputs,
    batch_generate,
    pad_kv_cache_for_batch,
)
from src.utils.parser import limit_memory_segments
from src.utils.prompt import ROUTER_SYS_PROMPT

if TYPE_CHECKING:
    from src.memory.memory_agent.agent import MemoryAgent

logger = logging.getLogger(__name__)


class Router(BaseAgent):
    """
    Router that selects and queries relevant memory blocks.
    
    Supports multiple modes:
    1. Router enabled (default): Uses LLM to select relevant blocks
    2. Router disabled: Queries ALL inactive blocks (for evaluation/debugging)
    3. Batch inference mode: True GPU parallelism via batching
    4. Standalone mode: Parallel queries with separate models
    
    In shared model mode, queries are executed sequentially to avoid
    conflicts, but KV cache loading is still parallelized for I/O efficiency.
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
    ) -> None:
        """
        Initialize Router.
        
        Args:
            openai_config: OpenAI API configuration for routing decisions.
            system_prompt: System prompt for routing.
            max_memory_segments: Maximum memory segments per query result.
            max_blocks: Maximum blocks to query.
            max_parallel_cache_loads: Max parallel KV cache loads to GPU.
            query_batch_size: Queries to batch together (for future batch inference).
            enable_router: If False, skip LLM routing and query ALL blocks.
        """
        # Allow None openai_config when router is disabled
        if not openai_config and enable_router:
            raise NotImplementedError("Please provide openai_config for router.")
        
        # Initialize base class only if router is enabled
        if enable_router:
            super().__init__(openai_config, system_prompt)
        else:
            # Minimal init for disabled router
            self.openai_config = openai_config
            self.system_prompt = system_prompt
            
        self.name = "router"
        self.agent: List[Any] = []
        self.max_memory_segments = max_memory_segments
        self.max_blocks = max_blocks
        self.max_parallel_cache_loads = max_parallel_cache_loads
        self.query_batch_size = query_batch_size
        self.enable_router = enable_router
        
        router_status = "enabled" if enable_router else "DISABLED (querying all blocks)"
        logger.info(
            f"Router initialized (status={router_status}, max_memory_segments={max_memory_segments}, "
            f"max_blocks={max_blocks}, max_parallel_cache_loads={max_parallel_cache_loads})"
        )

    def add_blocks(self, memory_agent):
        """Add inactive memory agent to router for querying."""
        if memory_agent.is_active:
            logger.warning("Attempted to add active memory agent to router")
            return
        self.agent.append(memory_agent)
        logger.info(f"Added memory block to router, total blocks: {len(self.agent)}")

    def _map_blocks(self, user_query: str, max_blocks: Optional[int] = None) -> list:
        """
        Map the user query to relevant memory agents.

        Args:
            user_query (str): The user query.
            max_blocks (int): The maximum number of blocks to map. Uses self.max_blocks if None.

        Returns:
            list: A list of memory agents.
        """
        if max_blocks is None:
            max_blocks = self.max_blocks

        if not self.agent:
            logger.debug("No memory agents available for mapping")
            return []
        
        # If router is disabled, return ALL agents without LLM selection
        if not self.enable_router:
            logger.info(f"Router disabled, returning all {len(self.agent)} blocks for querying")
            return self.agent.copy()  # Return copy to avoid modification

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

        # Extract indices using robust regex
        try:
            # Match <summary_index>...</summary_index> - capture anything inside
            match = re.search(
                r"<summary_index>\s*(.*?)\s*</summary_index>",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                indices_str = match.group(1)
                # Extract all integers
                indices = [int(num) for num in re.findall(r"\d+", indices_str)]

                # Deduplicate while preserving order
                seen = set()
                unique_indices = []
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        unique_indices.append(idx)

                # Filter valid indices first, then limit to max_blocks
                valid_indices = [
                    idx for idx in unique_indices if 0 <= idx < len(self.agent)
                ]
                selected_indices = valid_indices[:max_blocks]
                selected_agents = [self.agent[idx] for idx in selected_indices]

                logger.info(
                    f"Mapped query to {len(selected_agents)} memory blocks (indices: {selected_indices})"
                )
                return selected_agents
            else:
                logger.warning(
                    f"No summary_index tag found, using all blocks. Response: {response[:200]}"
                )
                return self.agent[:max_blocks]
        except Exception as e:
            logger.error(
                f"Error parsing router response: {e}, using all blocks", exc_info=True
            )
            return self.agent[:max_blocks]
        
    
    def execute_tool(self, tool_name, arguments):
        pass


    def map_reduce_blocks(self, user_query: str) -> List[str]:
        """
        Collect all the responses from the relevant memory agents.
        
        Supports two modes:
        1. Batch inference mode (shared model): True GPU parallelism via batching
        2. Standalone mode: Parallel queries with separate models
        
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

        # Step 1: Pre-load all caches to CPU in parallel (I/O bound)
        preload_workers = min(num_agents, self.max_parallel_cache_loads)
        logger.debug(f"Pre-loading {num_agents} caches with {preload_workers} workers")
        
        with ThreadPoolExecutor(max_workers=preload_workers) as executor:
            list(executor.map(lambda agent: agent.preload_cache(), relevant_agents_list))
        
        # Step 2: Check if using shared model mode
        agents_share_model = (
            len(relevant_agents_list) > 0 
            and hasattr(relevant_agents_list[0], '_owns_model')
            and not relevant_agents_list[0]._owns_model
        )
        
        results: List[str] = []
        
        if agents_share_model and self.query_batch_size > 1:
            # True batch inference mode
            logger.info(f"Using batch inference with batch_size={self.query_batch_size}")
            results = self._batch_query_agents(relevant_agents_list, user_query)
        elif agents_share_model:
            # Shared model but batch_size=1: sequential with single model
            logger.debug("Shared model mode, batch_size=1, executing sequentially")
            results = self._sequential_query_agents(relevant_agents_list, user_query)
        else:
            # Standalone mode: parallel queries (each agent has own model)
            logger.debug("Standalone model mode, executing queries in parallel")
            with ThreadPoolExecutor(max_workers=num_agents) as executor:
                results = list(
                    executor.map(lambda agent: agent.query(user_query), relevant_agents_list)
                )

        # Step 3: Apply memory segment limit if configured
        if self.max_memory_segments is not None and self.max_memory_segments > 0:
            logger.debug(f"Applying memory segment limit: {self.max_memory_segments}")
            results = [
                limit_memory_segments(result, self.max_memory_segments)
                for result in results
            ]

        # Final cleanup
        torch.cuda.empty_cache()
        logger.info(f"Collected {len(results)} results from memory blocks")
        return results
    
    def _sequential_query_agents(
        self, agents: List["MemoryAgent"], user_query: str
    ) -> List[str]:
        """Execute queries sequentially (for shared model with batch_size=1)."""
        results: List[str] = []
        for i, agent in enumerate(agents):
            try:
                logger.debug(f"Querying agent {i+1}/{len(agents)}")
                result = agent.query(user_query)
                results.append(result)
                # Log each block's result
                block_id = getattr(agent.current_block, 'block_id', f'agent_{i}')
                logger.info(f"Block {block_id} result: {result}")
            except Exception as e:
                logger.error(f"Error querying agent {i}: {e}", exc_info=True)
                results.append(f"[ERROR] Query failed: {e}")
            torch.cuda.empty_cache()
        return results
    
    def _batch_query_agents(
        self, agents: List["MemoryAgent"], user_query: str
    ) -> List[str]:
        """
        Execute queries using true batch inference.
        
        This provides real GPU parallelism by batching multiple queries together.
        """
        num_agents = len(agents)
        all_results: List[str] = [""] * num_agents
        
        # Process in batches
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
                    # Log each block's result
                    agent = agents[i]
                    block_id = getattr(agent.current_block, 'block_id', f'agent_{i}')
                    logger.info(
                        f"Block {block_id} result: {result}" 
                    )
            except Exception as e:
                logger.error(f"Batch query failed: {e}", exc_info=True)
                # Fallback to sequential for this batch
                for i, agent in zip(batch_indices, batch_agents):
                    try:
                        result = agent.query(user_query)
                        all_results[i] = result
                        block_id = getattr(agent.current_block, 'block_id', f'agent_{i}')
                        logger.info(
                            f"Block {block_id} result (fallback): {result}"
                        )
                    except Exception as inner_e:
                        all_results[i] = f"[ERROR] Query failed: {inner_e}"
            
            torch.cuda.empty_cache()
        
        return all_results
    
    def _execute_batch_query(
        self, agents: List["MemoryAgent"], user_query: str
    ) -> List[str]:
        """
        Execute a single batch of queries with true parallelism.
        
        Args:
            agents: List of agents to query (batch)
            user_query: The query string
            
        Returns:
            List of responses
        """
        if not agents:
            return []
        
        batch_size = len(agents)
        
        # Get shared model resources from first agent
        model = agents[0].model
        tokenizer = agents[0].tokenizer
        layer_devices = agents[0].layer_devices
        primary_device = agents[0].primary_device
        
        # Step 1: Get all caches and query inputs
        caches: List[DynamicCache] = []
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
        
        # Step 2: Pad and batch KV caches (LEFT-PADDED)
        batched_cache, cache_lengths = pad_kv_cache_for_batch(
            caches, layer_devices, primary_device
        )
        max_cache_len = max(cache_lengths) if cache_lengths else 0
        
        # Step 3: Pad and batch query inputs (LEFT-PADDED to match KV cache)
        # CRITICAL: Both KV cache and input_ids must use LEFT-PADDING
        # This ensures [:, -1] always gets the last real token's logits
        max_query_len = max(q[1] for q in query_inputs)
        first_layer_device = layer_devices.get(0, primary_device)
        
        # LEFT-PAD input_ids with padding token
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
            # LEFT-PAD: put content at the END
            batched_input_ids[batch_idx, query_pad:] = query_ids.squeeze(0)
        
        # Step 4: Compute position_ids and attention_mask
        # Layout for each sample:
        # - Cache: [Pad, Pad, ..., CacheData]  (left-padded)
        # - Query: [Pad, Pad, ..., QueryData]  (left-padded)
        # - Position IDs: only for real query tokens, starting from global_offset
        
        # Position IDs for query tokens (LEFT-PADDED)
        batched_position_ids = torch.zeros(
            (len(valid_indices), max_query_len),
            dtype=torch.long,
            device=first_layer_device
        )
        
        # Attention mask: [batch, max_cache_len + max_query_len]
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
            
            # Position IDs for query tokens (LEFT-PADDED)
            # Real query tokens are at indices [query_pad:max_query_len]
            # Their positions start from global_offset
            for pos in range(query_len):
                batched_position_ids[batch_idx, query_pad + pos] = global_offset + pos
            
            # Attention mask:
            # - Cache part: mask out left-padding, allow real cache data
            cache_padding = max_cache_len - cache_len
            batched_attention_mask[batch_idx, cache_padding:max_cache_len] = 1
            # - Query part: mask out left-padding, allow real query tokens
            batched_attention_mask[batch_idx, max_cache_len + query_pad:total_len] = 1
        
        # Step 5: Create BatchQueryInputs
        batch_inputs = BatchQueryInputs(
            input_ids=batched_input_ids,
            position_ids=batched_position_ids,
            attention_mask=batched_attention_mask,
            past_key_values=batched_cache,
            cache_lengths=cache_lengths,
            query_lengths=query_lengths,
            global_offsets=global_offsets,
        )
        
        # Step 6: Execute batch generation
        logger.debug(f"Executing batch generation with {len(valid_indices)} samples")
        batch_responses = batch_generate(
            model=model,
            tokenizer=tokenizer,
            batch_inputs=batch_inputs,
            max_new_tokens=8192,
        )
        
        # Step 7: Clean thinking content from responses
        cleaned_responses = []
        for response in batch_responses:
            cleaned = agents[0]._remove_thinking_content(response)
            cleaned_responses.append(cleaned)
        
        # Step 8: Map back to original indices
        results = ["No knowledge available."] * batch_size
        for batch_idx, orig_idx in enumerate(valid_indices):
            results[orig_idx] = cleaned_responses[batch_idx]
        
        # Cleanup caches
        for cache in caches:
            del cache
        del batched_cache
        torch.cuda.empty_cache()
        
        return results