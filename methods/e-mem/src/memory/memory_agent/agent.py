import logging
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.memory.kv_block_manager.block import KVBlock
from src.utils.prompt import MEMORY_AGENT_SYS_PROMPT, SUMMARY_INSTRUCTION

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Load config
try:
    import config
    MAX_CONCURRENT_GPU_OPERATIONS = config.MAX_CONCURRENT_GPU_OPERATIONS
except (ImportError, AttributeError):
    MAX_CONCURRENT_GPU_OPERATIONS = 2

# Semaphore to limit concurrent GPU operations
_gpu_semaphore = threading.Semaphore(MAX_CONCURRENT_GPU_OPERATIONS)
# Lock for thread-safe model access when using shared model
_model_lock = threading.Lock()


@dataclass
class BatchQueryInputs:
    """Container for batch query inputs."""
    input_ids: torch.Tensor  # [batch, seq_len]
    position_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, total_len]
    past_key_values: "DynamicCache"  # Batched KV cache
    cache_lengths: List[int]  # Original cache length per sample
    query_lengths: List[int]  # Query length per sample
    global_offsets: List[int]  # Global offset per sample


def pad_kv_cache_for_batch(
    caches: List[DynamicCache],
    layer_devices: Dict[int, torch.device],
    primary_device: torch.device,
) -> Tuple[DynamicCache, List[int]]:
    """
    Pad and batch multiple KV caches into a single batched cache using LEFT-PADDING.
    
    NOTE: This uses left-padding strategy where shorter sequences are padded at the start.
    This requires the Attention Mask to have 0s for the padded prefix positions.
    The input_ids passed to batch_generate MUST also be left-padded to match.
    
    Args:
        caches: List of DynamicCache objects (one per agent)
        layer_devices: Device mapping for each layer
        primary_device: Fallback device
        
    Returns:
        Batched DynamicCache and list of original cache lengths
    """
    if not caches:
        return DynamicCache(), []
    
    # Get cache lengths
    cache_lengths = []
    for cache in caches:
        if len(cache) > 0 and len(cache[0]) > 0:
            cache_lengths.append(cache[0][0].shape[2])  # [batch, heads, seq_len, dim]
        else:
            cache_lengths.append(0)
    
    max_cache_len = max(cache_lengths) if cache_lengths else 0
    batch_size = len(caches)
    num_layers = len(caches[0]) if caches and len(caches[0]) > 0 else 0
    
    if max_cache_len == 0 or num_layers == 0:
        return DynamicCache(), cache_lengths
    
    # Pre-calculate padding sizes for efficiency
    pad_sizes = [max_cache_len - length for length in cache_lengths]
    
    # Create batched cache
    batched_cache = DynamicCache()
    
    for layer_idx in range(num_layers):
        # Get shape from first cache
        k_sample, v_sample = caches[0][layer_idx]
        num_heads = k_sample.shape[1]
        head_dim = k_sample.shape[3]
        target_device = layer_devices.get(layer_idx, primary_device)
        dtype = k_sample.dtype
        
        # Create padded tensors for this layer
        # Shape: [batch, num_heads, max_seq_len, head_dim]
        batched_k = torch.zeros(
            (batch_size, num_heads, max_cache_len, head_dim),
            dtype=dtype, device=target_device
        )
        batched_v = torch.zeros(
            (batch_size, num_heads, max_cache_len, head_dim),
            dtype=dtype, device=target_device
        )
        
        # Copy each cache into the batched tensor (left-padded)
        for batch_idx, cache in enumerate(caches):
            if cache_lengths[batch_idx] == 0:
                continue
                
            k, v = cache[layer_idx]
            # Move to target device with non_blocking for better performance
            k = k.to(target_device, non_blocking=True)
            v = v.to(target_device, non_blocking=True)
            
            # Left-pad: put content at the END of the tensor
            # Layout: [Pad, Pad, ..., Data, Data]
            start_pos = pad_sizes[batch_idx]
            batched_k[batch_idx, :, start_pos:, :] = k.squeeze(0)
            batched_v[batch_idx, :, start_pos:, :] = v.squeeze(0)
        
        batched_cache.update(batched_k, batched_v, layer_idx)
    
    # Set _seen_tokens to max length (important for transformers internals)
    batched_cache._seen_tokens = max_cache_len
    
    return batched_cache, cache_lengths


def batch_generate(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    batch_inputs: BatchQueryInputs,
    max_new_tokens: int = 8192,
    repetition_penalty: float = 1.1,
) -> List[str]:
    """
    Perform batch generation with shared model.
    
    IMPORTANT: This function expects LEFT-PADDED input_ids to match the left-padded
    KV cache. The last token position [:, -1] should contain valid logits.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        batch_inputs: Prepared batch inputs (input_ids must be LEFT-PADDED)
        max_new_tokens: Maximum tokens to generate per sample
        repetition_penalty: Penalty for repeated tokens
        
    Returns:
        List of generated responses
    """
    batch_size = batch_inputs.input_ids.shape[0]
    device = batch_inputs.input_ids.device
    
    eos_token_ids = [tokenizer.eos_token_id] if not isinstance(
        tokenizer.eos_token_id, (list, tuple)
    ) else list(tokenizer.eos_token_id)
    
    # Track generated tokens per sample
    generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    
    # Current inputs
    current_input_ids = batch_inputs.input_ids
    current_attention_mask = batch_inputs.attention_mask
    current_position_ids = batch_inputs.position_ids
    past_kv = batch_inputs.past_key_values
    
    # Current position for each sample (after first forward)
    # This is the semantic position, not the tensor index
    current_positions = [
        batch_inputs.global_offsets[i] + batch_inputs.query_lengths[i]
        for i in range(batch_size)
    ]
    
    with torch.no_grad():
        # First forward pass
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids,
            past_key_values=past_kv,
            use_cache=True,
        )
        
        # Get next tokens from the LAST position of each sequence
        # Since input_ids is LEFT-PADDED, [:, -1] is the last real token for all samples
        next_token_logits = outputs.logits[:, -1, :].clone()
        next_tokens = next_token_logits.argmax(dim=-1)  # [batch]
        
        for i in range(batch_size):
            token_id = next_tokens[i].item()
            generated_tokens[i].append(token_id)
            if token_id in eos_token_ids:
                finished[i] = True
        
        past_kv = outputs.past_key_values
        
        # Continue generation
        for step in range(max_new_tokens - 1):
            if all(finished):
                break
            
            # Prepare next input - single token per sample
            next_input_ids = next_tokens.unsqueeze(1)  # [batch, 1]
            
            # Update attention mask - append 1 for the new token
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((batch_size, 1), dtype=torch.long, device=device)
            ], dim=1)
            
            # Position IDs for next token (semantic positions)
            next_position_ids = torch.tensor(
                [[pos] for pos in current_positions],
                dtype=torch.long, device=device
            )
            
            outputs = model(
                input_ids=next_input_ids,
                attention_mask=current_attention_mask,
                position_ids=next_position_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            
            next_token_logits = outputs.logits[:, -1, :].clone()
            
            # Apply repetition penalty per sample
            for i in range(batch_size):
                if not finished[i]:
                    recent_tokens = set(generated_tokens[i][-50:])
                    for token_id in recent_tokens:
                        if next_token_logits[i, token_id] > 0:
                            next_token_logits[i, token_id] /= repetition_penalty
                        else:
                            next_token_logits[i, token_id] *= repetition_penalty
            
            next_tokens = next_token_logits.argmax(dim=-1)
            
            for i in range(batch_size):
                if not finished[i]:
                    token_id = next_tokens[i].item()
                    generated_tokens[i].append(token_id)
                    current_positions[i] += 1
                    if token_id in eos_token_ids:
                        finished[i] = True
            
            past_kv = outputs.past_key_values
    
    # Decode all responses
    responses = []
    for tokens in generated_tokens:
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        responses.append(response)
    
    return responses


class MemoryAgent:
    """
    Memory agent that manages KV cache for a block of text chunks.
    
    Supports two modes:
    1. Shared model mode (recommended): Pass shared_model and shared_tokenizer
    2. Standalone mode (legacy): Agent loads its own model
    
    Shared model mode is more memory efficient and allows true parallelism
    with batch inference.
    """
    
    def __init__(
        self,
        model_id: str,
        model_context_window: int = 32768,
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
        quantization_config: Optional[Dict] = None,
        max_memory: Optional[Dict] = None,
        offload_folder: Optional[str] = None,
        load_from_block_id: Optional[str] = None,
        load_timestamp: Optional[str] = None,
        block_size_ratio: float = 0.125,
        # New parameters for shared model support
        shared_model: Optional["PreTrainedModel"] = None,
        shared_tokenizer: Optional["PreTrainedTokenizer"] = None,
        shared_layer_devices: Optional[Dict[int, torch.device]] = None,
    ):
        """
        Initialize MemoryAgent.
        
        Args:
            model_id: HuggingFace model ID or local path
            model_context_window: Model's context window size
            attn_implementation: Attention implementation type
            device_map: Device mapping strategy
            quantization_config: Quantization configuration
            max_memory: Max memory per GPU device
            offload_folder: Folder for offloading weights
            load_from_block_id: Block ID to load from (for resuming)
            load_timestamp: Timestamp of block to load
            block_size_ratio: Block size as ratio of context window
            shared_model: Pre-loaded model to share (recommended for multi-agent)
            shared_tokenizer: Pre-loaded tokenizer to share
            shared_layer_devices: Pre-computed layer device mapping
        """
        self.model_id = model_id
        self.model_context_window = model_context_window
        self.block_size_ratio = block_size_ratio
        self.block_size = int(model_context_window * block_size_ratio)
        self.summary = None
        self._owns_model = shared_model is None  # Track if we own the model
        
        # Use shared model/tokenizer if provided, otherwise load our own
        if shared_model is not None and shared_tokenizer is not None:
            logger.debug(f"Using shared model for agent (model_id: {model_id})")
            self.model = shared_model
            self.tokenizer = shared_tokenizer
            self.layer_devices = shared_layer_devices or self._get_layer_devices()
            self.primary_device = self.layer_devices.get(0, self.model.device)
        else:
            # Legacy mode: load model ourselves
            logger.info(f"Loading model: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            logger.debug(f"Tokenizer loaded for {model_id}")
            
            model_kwargs = {
                "device_map": device_map,
                "attn_implementation": attn_implementation,
                "trust_remote_code": True
            }
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["dtype"] = torch.bfloat16
            
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory
            if offload_folder is not None:
                model_kwargs["offload_folder"] = offload_folder
            
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            logger.info(f"Model loaded successfully: {model_id}")
            
            # Get layer-to-device mapping
            self.layer_devices = self._get_layer_devices()
            self.primary_device = self.layer_devices.get(0, self.model.device)
            logger.info(f"Layer devices mapped: {len(self.layer_devices)} layers across devices")

        self.is_active = True  # False when block is full
        self.original_texts: List[str] = []  # Store original text chunks for hybrid routing
        
        self._extract_chat_tokens()
        logger.debug(f"Chat tokens extracted: role_start='{self.role_start}', role_end='{self.role_end}'")
        
        # Initialize or load block
        if load_from_block_id and load_timestamp:
            # Load existing block
            self.current_block = KVBlock(
                block_id=uuid.UUID(load_from_block_id),
                create_timestamp=load_timestamp,
                block_size=self.block_size
            )
            # Load state from disk
            cache_state = self.current_block.load_cache()
            
            # Validate model_id compatibility
            cached_model_id = cache_state.get("model_id")
            if cached_model_id and cached_model_id != model_id:
                raise ValueError(
                    f"Model mismatch: cache was created with '{cached_model_id}' but trying to load with '{model_id}'. "
                    f"KV cache dimensions are incompatible between different models. Please use clean_cache_first=True or use the same model."
                )
            
            self.global_offset = cache_state.get("global_offset", 0)
            self.saved_chunks = cache_state.get("saved_chunks", [])
            self.chunk_number = cache_state.get("chunk_number", 0)
            self.original_texts = cache_state.get("original_texts", [])
            
            # Load merged_cache if available (for active agent)
            if "merged_cache" in cache_state and cache_state["merged_cache"]:
                logger.info(f"Loading merged_cache for active agent (block {load_from_block_id})")
                self.merged_cache = DynamicCache()
                for layer_idx, (k, v) in enumerate(cache_state["merged_cache"]):
                    target_device = self.layer_devices.get(layer_idx, self.primary_device)
                    self.merged_cache.update(k.to(target_device), v.to(target_device), layer_idx)
            else:
                self.merged_cache = None  # Will be loaded on demand for inactive agents
            
            logger.info(f"Loaded existing block: {load_from_block_id} (chunks: {len(self.saved_chunks)}, offset: {self.global_offset})")
        else:
            # Create new block
            self.current_block = KVBlock(
                block_id=uuid.uuid4(),
                create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                block_size=self.block_size
            )
            self.global_offset = 0
            self.saved_chunks = []
            self.chunk_number = 0
            self.merged_cache = None
    
    def _get_layer_devices(self):
        """Get accurate device mapping for each layer using hf_device_map."""
        num_layers = len(self.model.model.layers)
        layer_devices = {}
        
        # Use hf_device_map (ground truth for device_map='auto')
        if hasattr(self.model, 'hf_device_map'):
            for key, device_index in self.model.hf_device_map.items():
                if 'model.layers.' in key:
                    try:
                        layer_idx = int(key.split('model.layers.')[1].split('.')[0])
                        if isinstance(device_index, int):
                            layer_devices[layer_idx] = torch.device(f'cuda:{device_index}')
                        else:
                            layer_devices[layer_idx] = torch.device(device_index)
                    except (IndexError, ValueError):
                        continue
        
        # Fallback for missing layers
        for i in range(num_layers):
            if i not in layer_devices:
                try:
                    dev = next(self.model.model.layers[i].parameters()).device
                    layer_devices[i] = torch.device('cuda:0') if dev.type == 'meta' else dev
                except Exception as e:
                    logger.error(f"Error mapping layer {i}: {e}")
                    layer_devices[i] = torch.device('cuda:0')
        
        return layer_devices
    
    def _extract_chat_tokens(self):
        """Extract chat format tokens."""
        test_user = [{"role": "user", "content": "TEST"}]
        test_output = self.tokenizer.apply_chat_template(test_user, tokenize=False, add_generation_prompt=False)
        lines = test_output.split('TEST')
        if len(lines) >= 2:
            before = lines[0].rstrip('\n').split('\n')[-1]
            after = lines[1].split('\n')[0]
            if 'user' in before:
                self.role_start = before.split('user')[0]
                self.role_end = after
            else:
                self.role_start = "<|im_start|>"
                self.role_end = "<|im_end|>"
        else:
            self.role_start = "<|im_start|>"
            self.role_end = "<|im_end|>"
    
    def _add_knowledge(self, text_chunks: List[str]) -> bool:
        """
        Add knowledge chunks incrementally.
        Returns True if block became full, False otherwise.
        Args:
            text_chunks (List[str]): List of text chunks to be added.   
        Returns:
            bool: True if block became full, False otherwise.
        """
        block_full = False
        
        for i, text_chunk in enumerate(text_chunks, 1):
            self.chunk_number += 1
            # Store original text for hybrid routing
            self.original_texts.append(text_chunk)
            
            # Format chunk
            if self.global_offset == 0:
                system_msg = [{"role": "system", "content": MEMORY_AGENT_SYS_PROMPT}]
                system_part = self.tokenizer.apply_chat_template(system_msg, tokenize=False, add_generation_prompt=False)
                formatted_chunk = system_part + f"{self.role_start}user\nHere is the context information:\n\n[Context {self.chunk_number}]\n{text_chunk}"
            else:
                formatted_chunk = f"\n\n[Context {self.chunk_number}]\n{text_chunk}"
            
            # Tokenize
            input_ids = self.tokenizer.encode(formatted_chunk, return_tensors="pt", add_special_tokens=False).to(self.primary_device)
            seq_len = input_ids.shape[1]
            
            # Position IDs
            position_ids = torch.arange(self.global_offset, self.global_offset + seq_len, dtype=torch.long, device=self.primary_device).unsqueeze(0)
            
            # Use cached merged KV (incremental update)
            past_kv = self.merged_cache
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=past_kv, use_cache=True)
            
            # Update merged cache incrementally (single copy)
            # Delete old cache first to free memory
            if past_kv is not None:
                del past_kv
            self.merged_cache = outputs.past_key_values
            
            # Store only metadata (no cache duplication)
            self.saved_chunks.append({"start": self.global_offset, "length": seq_len})
            self.global_offset += seq_len
            
            # Check if block is full
            block_full = (self.current_block.block_used + seq_len >= self.block_size)
            self.current_block.block_used += seq_len
            
            if block_full:
                logger.info(f"Block {self.current_block.block_id} is full ({self.current_block.block_used}/{self.block_size} tokens)")
                return True
        
        return block_full
    

    def add(self,text_chunks: List[str])->bool:
        """
        Check if the agent is active
        And if so, calling the _add_knowledge function
        Handle the return bool
        If the block is full, set the agent to inactive, and create summaries.
        """
        if not self.is_active:
            raise RuntimeError("The agent is inactive, since the block is already full. So no new knowledge can be added.")
        logger.debug(f"Adding {len(text_chunks)} text chunks to memory agent")
        block_full = self._add_knowledge(text_chunks)
        if block_full:
            logger.info("Memory agent became inactive, creating summaries")
            self.is_active = False
            try:
                self._create_summaries()
                logger.info("Summaries created successfully")
            except Exception as e:
                logger.error(f"Failed to create summaries: {e}", exc_info=True)
                # Keep is_active=False, summary will be None
                # Agent can still be queried but without summary for router

    def _create_summaries(self):
        """
        Create summaries of all the stored knowledge chunks.
        And this is only needed after the agent is no longer in a active state
        """
        logger.info("Creating summary for inactive agent")
        self.summary = self._agent_generate(instruction=SUMMARY_INSTRUCTION, max_new_tokens=8192)
        logger.info(f"Summary created (length: {len(self.summary)} chars)")
        
        # Save cache to disk BEFORE clearing
        if self.merged_cache is not None:
            logger.info(f"Saving cache to disk: {len(self.merged_cache)} layers")
            cache_state = {
                "global_offset": self.global_offset,
                "saved_chunks": self.saved_chunks,
                "chunk_number": self.chunk_number,
                "model_id": self.model_id,
                "merged_cache": [(k.cpu(), v.cpu()) for k, v in self.merged_cache],
                "original_texts": self.original_texts,  # Store original text for hybrid routing
            }
            self.current_block.save_cache(cache_state, 0)
            logger.info(f"Cache saved to {self.current_block.store_target}")
        else:
            logger.error("Cannot save cache: merged_cache is None")
        
        # Clear from GPU
        self.merged_cache = None
        torch.cuda.empty_cache()

    def _agent_generate(self,max_new_tokens: int=8192, instruction: str=None, question: str=None)-> str:
        """
        This is a base function for the agent to generate text
        It can be called by query and _create_summaries functions
        Args:
            max_new_tokens (int): Maximum number of tokens to generate.
            instruction (str): The instruction.
            question (str): The user question.
        Returns:
            str: The generated text.
        """
        if not self.saved_chunks:
            logger.warning(f"No knowledge available for generation (block_id={self.current_block.block_id}, is_active={self.is_active}, chunk_number={self.chunk_number})")
            return "No knowledge available."
        
        logger.debug(f"Generating response with max_new_tokens={max_new_tokens}")
        
        # Prepare base cache (use pre-loaded or load from disk)
        base_cache = self.merged_cache
        cache_loaded_from_disk = False
        if base_cache is None:
            if not self.is_active:
                # Check if cache was pre-loaded to CPU
                if hasattr(self, '_cpu_cache') and self._cpu_cache is not None:
                    logger.debug(f"Transferring pre-loaded cache to GPU (block {self.current_block.block_id})")
                    base_cache = DynamicCache()
                    for layer_idx, (k, v) in enumerate(self._cpu_cache):
                        target_device = self.layer_devices.get(layer_idx, self.primary_device)
                        base_cache.update(k.to(target_device), v.to(target_device), layer_idx)
                    cache_loaded_from_disk = True
                else:
                    # Fallback: load from disk (slower path)
                    cache_state = self.current_block.load_cache()
                    
                    # Validate model_id before loading cache
                    cached_model_id = cache_state.get("model_id")
                    if cached_model_id and cached_model_id != self.model_id:
                        raise ValueError(
                            f"Model mismatch: cache was created with '{cached_model_id}' but trying to load with '{self.model_id}'. "
                            f"KV cache dimensions are incompatible between different models."
                        )
                    
                    if "merged_cache" in cache_state and cache_state["merged_cache"]:
                        logger.info(f"Loading cache from disk for inactive agent (block {self.current_block.block_id})")
                        base_cache = DynamicCache()
                        for layer_idx, (k, v) in enumerate(cache_state["merged_cache"]):
                            target_device = self.layer_devices.get(layer_idx, self.primary_device)
                            base_cache.update(k.to(target_device), v.to(target_device), layer_idx)
                        cache_loaded_from_disk = True
                    else:
                        logger.error(f"No cache available for inactive agent (block {self.current_block.block_id})")
                        logger.error(f"Cache state keys: {cache_state.keys() if cache_state else 'None'}")
                        raise RuntimeError("No cache available for inactive agent. Block may not have been properly saved.")
            else:
                # Active agent with no cache - this shouldn't happen if chunks were added
                logger.warning("Active agent has no merged_cache but has saved_chunks")
                raise RuntimeError("Active agent cache inconsistency")
        
        # CRITICAL: Fork cache to prevent pollution (shallow copy for memory efficiency)
        generation_cache = DynamicCache()
        
        # Shallow copy - reuse tensors (they won't be modified during generation)
        for layer_idx in range(len(base_cache)):
            k, v = base_cache[layer_idx]
            # No .clone() to save memory - tensors are read-only during generation
            generation_cache.update(k, v, layer_idx)
        
        # Copy _seen_tokens metadata
        if hasattr(base_cache, '_seen_tokens'):
            generation_cache._seen_tokens = base_cache._seen_tokens
        elif len(base_cache) > 0:
            generation_cache._seen_tokens = base_cache[0][0].shape[-2]
        
        # Format query
        if question:
            # This means we are in the query mode
            formatted_query = f"\n\nBased on the context information provided above, please extract the original information that is relevant to the question(**REMEBER to give EXACT datetime** along with information, and the datetime format is 'YYYY-MM-DD HH:MM:SS'.):\n{question}{self.role_end}\n{self.role_start}assistant\n"
        elif instruction:
            formatted_query=f"\n\nBased on the context information provided above, please follow this instruction:\n{instruction}{self.role_end}\n{self.role_start}assistant\n"
        else:
            raise ValueError("Either question or instruction must be provided.")
        
        # Ensure all inputs on first layer device (model input device)
        first_layer_device = self.layer_devices.get(0, self.primary_device)
        input_ids = self.tokenizer.encode(formatted_query, return_tensors="pt", add_special_tokens=False).to(first_layer_device)
        query_len = input_ids.shape[1]
        
        # Position IDs
        query_position_ids = torch.arange(self.global_offset, self.global_offset + query_len, dtype=torch.long, device=first_layer_device).unsqueeze(0)
        
        # Attention mask
        cache_length = generation_cache.get_seq_length()
        if cache_length == 0 and len(generation_cache) > 0:
            # Double check: get from tensor shape
            cache_length = generation_cache[0][0].shape[-2]
        attention_mask = torch.ones((1, cache_length + query_len), dtype=torch.long, device=first_layer_device)
        
        # Manual generation with repetition penalty
        # CRITICAL: Use model.generate() or ensure cache is not mutated
        eos_token_ids = [self.tokenizer.eos_token_id] if not isinstance(self.tokenizer.eos_token_id, (list, tuple)) else self.tokenizer.eos_token_id
        generated_tokens = []
        repetition_penalty = 1.1
        
        # First forward pass with forked cache
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=query_position_ids,
                past_key_values=generation_cache,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :].clone()
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            generated_tokens.append(next_token.item())
            
            # Continue generation with new cache (separate from memory)
            past_kv = outputs.past_key_values
            current_position = self.global_offset + query_len + 1
            next_token_input = next_token
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=first_layer_device)], dim=1)
            
            for _ in range(max_new_tokens - 1):
                if generated_tokens[-1] in eos_token_ids:
                    break
                    
                next_position_ids = torch.tensor([[current_position]], dtype=torch.long, device=first_layer_device)
                outputs = self.model(
                    input_ids=next_token_input,
                    attention_mask=attention_mask,
                    position_ids=next_position_ids,
                    past_key_values=past_kv,
                    use_cache=True
                )
                next_token_logits = outputs.logits[:, -1, :].clone()
                
                # Repetition penalty
                for token_id in set(generated_tokens[-50:]):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty
                
                next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
                generated_tokens.append(next_token.item())
                next_token_input = next_token
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=first_layer_device)], dim=1)
                current_position += 1
                past_kv = outputs.past_key_values
        
        # Decode
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = self._remove_thinking_content(response)
        logger.debug(f"Generated response length: {len(response)} chars")
        
        # Clean up temporary cache immediately
        if 'generation_cache' in locals():
            del generation_cache
        if 'past_kv' in locals():
            del past_kv
        # For inactive agents, unload cache from GPU after query
        if cache_loaded_from_disk and base_cache is not None:
            logger.debug("Unloading cache from GPU for inactive agent")
            del base_cache
        # Clear CPU cache reference
        if hasattr(self, '_cpu_cache'):
            self._cpu_cache = None
        # Aggressive cleanup
        torch.cuda.empty_cache()
        
        return response
    
    def _remove_thinking_content(self, response: str) -> str:
        cleaned_response = response
        # Remove both <thinking> and <think> tags
        cleaned_response = re.sub(r'<thinking>.*?</thinking>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        
        cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)
        cleaned_response = cleaned_response.strip()  
        
        return cleaned_response
    
    def preload_cache(self):
        """Pre-load cache from disk to CPU memory (no GPU involved)."""
        if self.is_active or self.merged_cache is not None:
            return  # Already loaded or active
        
        cache_state = self.current_block.load_cache()
        
        # Validate model_id before preloading
        cached_model_id = cache_state.get("model_id")
        if cached_model_id and cached_model_id != self.model_id:
            logger.error(f"Model mismatch during preload: cache '{cached_model_id}' vs current '{self.model_id}'")
            self._cpu_cache = None
            return
        
        if "merged_cache" in cache_state and cache_state["merged_cache"]:
            logger.debug(f"Pre-loaded cache to CPU for block {self.current_block.block_id}")
            # Store in CPU memory, will transfer to GPU during generation
            self._cpu_cache = cache_state["merged_cache"]
        else:
            self._cpu_cache = None
    
    def query(self, question: str, max_new_tokens: int = 8192) -> str:
        """
        Query using cached knowledge.
        
        Args:
            question: The user question.
            max_new_tokens: Maximum number of tokens to generate.
            
        Returns:
            The generated response.
        """
        logger.debug(f"Querying memory agent: {question[:50]}...")
        # Use semaphore to limit concurrent GPU operations
        if not self.is_active:
            with _gpu_semaphore:
                # Use model lock for thread safety when sharing model
                if not self._owns_model:
                    with _model_lock:
                        result = self._agent_generate(max_new_tokens=max_new_tokens, question=question)
                else:
                    result = self._agent_generate(max_new_tokens=max_new_tokens, question=question)
            return result
        else:
            # Active agent - use lock if sharing model
            if not self._owns_model:
                with _model_lock:
                    return self._agent_generate(max_new_tokens=max_new_tokens, question=question)
            return self._agent_generate(max_new_tokens=max_new_tokens, question=question)
    
    def prepare_query_inputs(
        self, question: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, "DynamicCache", bool]]:
        """
        Prepare inputs for batch query (without running the model).
        
        This method prepares all inputs needed for generation, allowing
        multiple agents' queries to be batched together.
        
        Args:
            question: The user question.
            
        Returns:
            Tuple of (input_ids, position_ids, attention_mask, base_cache, cache_loaded_from_disk)
            or None if no knowledge available.
        """
        if not self.saved_chunks:
            logger.warning(f"No knowledge available (block_id={self.current_block.block_id})")
            return None
        
        # Prepare base cache
        base_cache = self.merged_cache
        cache_loaded_from_disk = False
        
        if base_cache is None:
            if not self.is_active:
                # Check if cache was pre-loaded to CPU
                if hasattr(self, '_cpu_cache') and self._cpu_cache is not None:
                    logger.debug(f"Transferring pre-loaded cache to GPU (block {self.current_block.block_id})")
                    base_cache = DynamicCache()
                    for layer_idx, (k, v) in enumerate(self._cpu_cache):
                        target_device = self.layer_devices.get(layer_idx, self.primary_device)
                        base_cache.update(k.to(target_device), v.to(target_device), layer_idx)
                    cache_loaded_from_disk = True
                else:
                    # Fallback: load from disk
                    cache_state = self.current_block.load_cache()
                    cached_model_id = cache_state.get("model_id")
                    if cached_model_id and cached_model_id != self.model_id:
                        raise ValueError(f"Model mismatch: cache '{cached_model_id}' vs '{self.model_id}'")
                    
                    if "merged_cache" in cache_state and cache_state["merged_cache"]:
                        logger.info(f"Loading cache from disk for inactive agent (block {self.current_block.block_id})")
                        base_cache = DynamicCache()
                        for layer_idx, (k, v) in enumerate(cache_state["merged_cache"]):
                            target_device = self.layer_devices.get(layer_idx, self.primary_device)
                            base_cache.update(k.to(target_device), v.to(target_device), layer_idx)
                        cache_loaded_from_disk = True
                    else:
                        raise RuntimeError("No cache available for inactive agent.")
            else:
                raise RuntimeError("Active agent cache inconsistency")
        
        # Format query
        formatted_query = f"\n\nBased on the context information provided above, please extract the original information that is relevant to the question(**REMEBER to give EXACT datetime** along with information, and the datetime format is 'YYYY-MM-DD HH:MM:SS'.):\n{question}{self.role_end}\n{self.role_start}assistant\n"
        
        first_layer_device = self.layer_devices.get(0, self.primary_device)
        input_ids = self.tokenizer.encode(formatted_query, return_tensors="pt", add_special_tokens=False).to(first_layer_device)
        query_len = input_ids.shape[1]
        
        # Position IDs
        position_ids = torch.arange(self.global_offset, self.global_offset + query_len, dtype=torch.long, device=first_layer_device).unsqueeze(0)
        
        # Attention mask
        cache_length = base_cache.get_seq_length()
        if cache_length == 0 and len(base_cache) > 0:
            cache_length = base_cache[0][0].shape[-2]
        attention_mask = torch.ones((1, cache_length + query_len), dtype=torch.long, device=first_layer_device)
        
        return input_ids, position_ids, attention_mask, base_cache, cache_loaded_from_disk
    
    def get_cache_for_batch(self) -> Optional[Tuple[DynamicCache, int, bool]]:
        """
        Get KV cache ready for batch processing.
        
        Returns:
            Tuple of (cache, global_offset, cache_loaded_from_disk) or None if not available.
        """
        if not self.saved_chunks:
            return None
        
        base_cache = self.merged_cache
        cache_loaded_from_disk = False
        
        if base_cache is None:
            if not self.is_active:
                if hasattr(self, '_cpu_cache') and self._cpu_cache is not None:
                    logger.debug(f"Transferring cache to GPU for batch (block {self.current_block.block_id})")
                    base_cache = DynamicCache()
                    for layer_idx, (k, v) in enumerate(self._cpu_cache):
                        target_device = self.layer_devices.get(layer_idx, self.primary_device)
                        base_cache.update(k.to(target_device), v.to(target_device), layer_idx)
                    cache_loaded_from_disk = True
                else:
                    cache_state = self.current_block.load_cache()
                    if "merged_cache" in cache_state and cache_state["merged_cache"]:
                        base_cache = DynamicCache()
                        for layer_idx, (k, v) in enumerate(cache_state["merged_cache"]):
                            target_device = self.layer_devices.get(layer_idx, self.primary_device)
                            base_cache.update(k.to(target_device), v.to(target_device), layer_idx)
                        cache_loaded_from_disk = True
                    else:
                        return None
            else:
                return None
        
        return base_cache, self.global_offset, cache_loaded_from_disk
    
    def format_query_for_batch(self, question: str) -> Tuple[torch.Tensor, int]:
        """
        Format query and return input_ids for batch processing.
        
        Args:
            question: The user question.
            
        Returns:
            Tuple of (input_ids tensor, query_length)
        """
        formatted_query = (
            f"\n\nBased on the context information provided above, please extract "
            f"the original information that is relevant to the question(**REMEBER "
            f"to give EXACT datetime** along with information, and the datetime "
            f"format is 'YYYY-MM-DD HH:MM:SS'.):\n{question}{self.role_end}\n"
            f"{self.role_start}assistant\n"
        )
        
        first_layer_device = self.layer_devices.get(0, self.primary_device)
        input_ids = self.tokenizer.encode(
            formatted_query, return_tensors="pt", add_special_tokens=False
        ).to(first_layer_device)
        
        return input_ids, input_ids.shape[1]
    
    def cleanup_after_query(self, base_cache: Optional["DynamicCache"], cache_loaded_from_disk: bool) -> None:
        """
        Cleanup resources after query completion.
        
        Args:
            base_cache: The cache used for query.
            cache_loaded_from_disk: Whether cache was loaded from disk.
        """
        # For inactive agents, unload cache from GPU after query
        if cache_loaded_from_disk and base_cache is not None:
            logger.debug("Unloading cache from GPU for inactive agent")
            del base_cache
        # Clear CPU cache reference
        if hasattr(self, '_cpu_cache'):
            self._cpu_cache = None
        # Aggressive cleanup
        torch.cuda.empty_cache()

    def get_original_texts(self) -> List[str]:
        """
        Get original text chunks for hybrid routing.
        
        Returns:
            List of original text chunks stored in this block.
        """
        # For inactive agents, may need to load from cache
        if not self.original_texts and not self.is_active:
            cache_state = self.current_block.load_cache()
            self.original_texts = cache_state.get("original_texts", [])
        return self.original_texts