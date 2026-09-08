import os
import numpy as np
import json
import time
import logging
import asyncio
import random
import re
import uuid
import hashlib
import shutil
import httpx
import subprocess
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
from datetime import datetime
from collections import defaultdict
from agent_system.environments.env_manager import *
from agent_system.environments.prompts.webshop import (
    WEBSHOP_TEMPLATE_REFLECTION,
    WEBSHOP_TEMPLATE_WITH_OPTIONAL_MEMORY,
    WEBSHOP_TEMPLATE_STRATEGY_GENERATION,
    WEBSHOP_TEMPLATE_ACTION_FROM_STRATEGY,
    WEBSHOP_TEMPLATE_STRATEGY_NO_RETRIEVAL,
    WEBSHOP_TEMPLATE_DIRECT_ACTION_FROM_RETRIEVAL,
    WEBSHOP_TEMPLATE_STRATEGY_REFRESH_DECISION,
    WEBSHOP_TEMPLATE_UPDATE_DECISION,
    WEBSHOP_TEMPLATE_UPDATE_DECISION_NO_RESET,
)
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError

# split = 'eval' or 'train'
split = os.environ.get("SPLIT", "eval")

# For eval env_num should equal 500 (number of eval goals) for deterministic eval
# For train env_num can be any number (random sample with replacement)
SPLIT2ENV_NUM = {'train': int(os.environ.get("TRAIN_SIZE", 10000)), 'eval': 500}

def _parse_env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        logging.warning(f"Invalid {name}={raw!r}; falling back to {default}.")
        value = default
    if value < minimum:
        logging.warning(f"{name}={value} is below minimum {minimum}; clamping to {minimum}.")
        value = minimum
    return value


def _parse_env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        logging.warning(f"Invalid {name}={raw!r}; falling back to {default}.")
        value = default
    if value < minimum:
        logging.warning(f"{name}={value} is below minimum {minimum}; clamping to {minimum}.")
        value = minimum
    return value


def truncate_middle_text(text: str, max_chars: int, marker: str = "\n... [truncated] ...\n") -> str:
    if text is None:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if len(marker) >= max_chars:
        return text[:max_chars]
    keep = max_chars - len(marker)
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


mem_type = os.environ.get("MEM_TYPE", None)
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
correct_only = os.environ.get("CORRECT_ONLY", "false").lower() == "true"
memory_mode = os.environ.get("MEMORY_MODE", "full")
topk = _parse_env_int("RETRIEVAL_TOPK", 1, minimum=1)
MODEL_CONTEXT_WINDOW_TOKENS = _parse_env_int("MODEL_CONTEXT_WINDOW_TOKENS", 30000, minimum=1024)
PROMPT_CHAR_PER_TOKEN = _parse_env_float("PROMPT_CHAR_PER_TOKEN", 4.0, minimum=1.0)
RETRIEVAL_OUTPUT_TOKEN_RESERVE = _parse_env_int("RETRIEVAL_OUTPUT_TOKEN_RESERVE", 4096, minimum=128)
RETRIEVAL_FIXED_OVERHEAD_CHARS = _parse_env_int("RETRIEVAL_FIXED_OVERHEAD_CHARS", 3000, minimum=0)
RETRIEVAL_PROMPT_MIN_CHARS = _parse_env_int("RETRIEVAL_PROMPT_MIN_CHARS", 512, minimum=64)

STEP_STRATEGY_MEM_TYPES = [
    "adamem-high",
    "adamem-max",
    "adamem-max-without-trajectory-memory",
    "adamem-max-without-strategy-memory",
    "adamem-low",
]

MEM_TYPES_WITH_RETRIEVAL_STATS = STEP_STRATEGY_MEM_TYPES

def parse_memory_request(response: str) -> tuple:
    """Parse LLM response to check if memory retrieval was requested.
    
    Returns:
        (has_request, reason) where has_request is True if <request_memory>yes</request_memory>
    """
    # Extract rationale first
    rationale_pattern = r'<memory_request_rationale>(.*?)</memory_request_rationale>'
    rationale_match = re.search(rationale_pattern, response, re.DOTALL | re.IGNORECASE)
    
    # Extract <request_memory> tag
    request_pattern = r'<request_memory>\s*(yes|no)\s*</request_memory>'
    request_match = re.search(request_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if request_match and request_match.group(1).strip().lower() == 'yes':
        if rationale_match:
            reason = rationale_match.group(1).strip()
        else:
            reason = "Memory requested without explicit rationale"
        
        return True, reason
    
    return False, ""

def parse_refresh_decision(response: str) -> tuple:
    """Parse LLM response to check if strategy refresh was requested.
    
    Returns:
        (should_refresh, reason) where should_refresh is True if <refresh_decision>yes</refresh_decision>
    """
    # Extract rationale first (optional)
    rationale_pattern = r'<think>(.*?)</think>'
    rationale_match = re.search(rationale_pattern, response, re.DOTALL | re.IGNORECASE)
    
    # Extract <refresh_decision> tag
    decision_pattern = r'<refresh_decision>\s*(yes|no)\s*</refresh_decision>'
    decision_match = re.search(decision_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if decision_match and decision_match.group(1).strip().lower() == 'yes':
        if rationale_match:
            reason = rationale_match.group(1).strip()
        else:
            reason = "Refresh requested without explicit rationale"
        
        return True, reason
    
    return False, ""

def parse_action_and_refresh(response: str) -> tuple:
    """Parse LLM response for action and refresh decision.
    
    Returns:
        (action, should_refresh, reason)
    """
    # Extract action
    action = extract_action_from_response(response)
    
    # Extract refresh decision
    should_refresh, reason = parse_refresh_decision(response)
    
    return action, should_refresh, reason

def extract_action_from_response(response: str) -> str:
    """Extract action from <action> tags."""
    pattern = r'<action>(.*?)</action>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()

def extract_strategy_from_response(response: str) -> str:
    """Extract strategy from <strategy> tags."""
    pattern = r'<strategy>(.*?)</strategy>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def build_env(env_name, env_num=1, batch_start_idx=0, reuse_workers=False):
    """
    Build environment manager.
    
    Args:
        env_name: Name of the environment
        env_num: Number of environments (workers) to create
        batch_start_idx: Starting index for goal assignment (used when reuse_workers=False)
        reuse_workers: If True, create workers without pre-assigned goals (for reuse across batches)
    """
    group_n = 1
    if env_name == "webshop":
        # Test WebShop Environment
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        
        # WebShop environment configuration
        env_kwargs = {
            'observation_mode': 'text',
            'num_products': None,  # Use all products
        }
        
        resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
        
        if split == "train":
            envs = build_webshop_envs(
                seed=1, 
                env_num=env_num, 
                group_n=group_n, 
                is_train=True, 
                env_kwargs=env_kwargs | {'split': 'train'},
                resources_per_worker=resources_per_worker,
                batch_start_idx=batch_start_idx
            )
        else:
            # For eval: always use is_train=False to get eval goals (0-499)
            # Note: Workers will initially be assigned goals [batch_start_idx, ..., batch_start_idx+env_num-1]
            # but when reuse_workers=True, we'll override these via reset_with_goals() in each batch.
            # The explicit goal_idx passed to worker.reset(goal_idx) takes precedence over assigned_goal_idx.
            envs = build_webshop_envs(
                seed=1, 
                env_num=env_num, 
                group_n=group_n, 
                is_train=False,  # This ensures self.goal_idxs = [0, 1, ..., 499]
                env_kwargs=env_kwargs | {'split': 'test'},
                resources_per_worker=resources_per_worker,
                batch_start_idx=batch_start_idx if not reuse_workers else 0
            )
        
        # Create a simple config object for WebshopEnvironmentManager
        from types import SimpleNamespace
        config = SimpleNamespace(
            env=SimpleNamespace(
                history_length=15,  # Number of recent steps to include in history
            )
        )
        
        env_manager = WebshopEnvironmentManager(envs, webshop_projection, config, mem_type=mem_type, topk=topk)
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
    
    return env_manager

class Agent:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.backend = os.environ.get("BACKEND", "vllm")  # 'vllm' or 'openai'
        
        if self.backend == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set for OpenAI backend")
            
            # Create single client for OpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.max_tokens = 4096  # Default for GPT-4o
            
            # Semaphore for rate limiting concurrent requests
            self.semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
            
            logging.info(f"Agent initialized with OpenAI backend, model {self.model_name}, max_tokens {self.max_tokens}")
        else:  # vllm backend
            ip_addrs_str = os.environ.get("OPENAI_BASE_IP_ADDR", "127.0.0.1")
            self.ip_addrs = [ip.strip() for ip in ip_addrs_str.split(",")]
            print("DEBUG: using vllm ip addresses", self.ip_addrs)
            self.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")  # vLLM doesn't check this
            
            # Create persistent clients for each IP
            self.clients = {
                ip: AsyncOpenAI(api_key=self.api_key, base_url=f"http://{ip}/v1")
                for ip in self.ip_addrs
            }

            self.max_tokens = 8192 if self.model_name == "Qwen/Qwen3-4B-Thinking-2507" else 2048
            logging.info(f"Agent initialized with vLLM backend, model {self.model_name}, max_tokens {self.max_tokens}")

        # Strategy model setup (separate from policy model when MEM_TYPE includes "step_strategy")
        self.strategy_model_name = os.environ.get("STRATEGY_MODEL_NAME", self.model_name)
        
        if self.backend == "openai":
            # For OpenAI, strategy uses same client but different model
            self.strategy_client = self.client
            self.strategy_max_tokens = 4096
            logging.info(f"Strategy model initialized with OpenAI backend, model {self.strategy_model_name}, max_tokens {self.strategy_max_tokens}")
        else:  # vllm
            strategy_ip_addrs_str = os.environ.get("OPENAI_BASE_IP_ADDR_STRATEGY", ip_addrs_str)
            self.strategy_ip_addrs = [ip.strip() for ip in strategy_ip_addrs_str.split(",")]
            print("DEBUG: using strategy vllm ip addresses", self.strategy_ip_addrs)
            
            # Create persistent clients for strategy model
            self.strategy_clients = {
                ip: AsyncOpenAI(api_key=self.api_key, base_url=f"http://{ip}/v1")
                for ip in self.strategy_ip_addrs
            }

            self.strategy_max_tokens = 8192 if self.strategy_model_name == "Qwen/Qwen3-4B-Thinking-2507" else 2048
            logging.info(f"Strategy model initialized with vLLM backend, model {self.strategy_model_name}, max_tokens {self.strategy_max_tokens}")

        # State for strategy reuse across steps
        self.active_strategies = {}  # env_idx -> current strategy string

        self.temperature = float(os.environ.get("TEMPERATURE", 0.7))

        # Cost tracking for API usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.input_token_cost_per_million = 0.05  # $0.05 per 1M input tokens
        self.output_token_cost_per_million = 0.40  # $0.40 per 1M output tokens

        # Retrieval truncation statistics
        self.truncation_stats = {
            "total": 0,
            "truncated": 0,
        }



    def _extract_httpx_error_details(self, resp: httpx.Response) -> str:
        try:
            return str(resp.json())
        except Exception:
            text = (resp.text or "").strip()
            return text if text else "<empty response body>"

    def _raise_httpx_for_status_with_details(
        self, resp: httpx.Response, context: str
    ) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_info = self._extract_httpx_error_details(resp)
            raise RuntimeError(
                f"{context} failed. Status {resp.status_code}. Details: {error_info}"
            ) from e

    def _format_vllm_exception(self, e: Exception, context: str) -> str:
        details = [f"{context} failed ({type(e).__name__})"]

        msg = str(e).strip()
        details.append(f"message={msg if msg else repr(e)}")

        status_code = getattr(e, "status_code", None)
        if status_code is not None:
            details.append(f"status={status_code}")

        body = getattr(e, "body", None)
        if body is not None:
            details.append(f"body={body}")

        response = getattr(e, "response", None)
        if response is not None:
            try:
                details.append(f"response_json={response.json()}")
            except Exception:
                try:
                    text = (response.text or "").strip()
                    if text:
                        details.append(f"response_text={text}")
                except Exception:
                    pass

        return "; ".join(details)

    def _record_truncation(self, original_text: str, truncated_text: str) -> None:
        stats = getattr(self, "truncation_stats", None)
        if stats is None:
            return
        stats["total"] += 1
        if truncated_text is not None and original_text is not None:
            if len(truncated_text) < len(original_text):
                stats["truncated"] += 1

    def _get_dynamic_retrieval_max_chars(
        self,
        base_prompt: str,
        output_token_reserve=None,
    ) -> int:
        reserve_tokens = output_token_reserve if output_token_reserve is not None else max(
            self.max_tokens,
            RETRIEVAL_OUTPUT_TOKEN_RESERVE,
        )
        usable_input_tokens = max(1, MODEL_CONTEXT_WINDOW_TOKENS - reserve_tokens)
        input_char_budget = int(usable_input_tokens * PROMPT_CHAR_PER_TOKEN)
        prompt_len = len(base_prompt or "")
        remaining_chars = input_char_budget - prompt_len - RETRIEVAL_FIXED_OVERHEAD_CHARS
        return max(RETRIEVAL_PROMPT_MIN_CHARS, remaining_chars)

    def _truncate_retrieval_for_prompt(
        self,
        base_prompt: str,
        retrieval_text: str,
        output_token_reserve=None,
    ) -> str:
        max_chars = self._get_dynamic_retrieval_max_chars(
            base_prompt=base_prompt,
            output_token_reserve=output_token_reserve,
        )
        truncated = truncate_middle_text(retrieval_text, max_chars)
        self._record_truncation(retrieval_text, truncated)
        return truncated

    def track_api_cost(self, response):
        """Track API costs based on token usage in the response."""
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0)
            output_tokens = getattr(response.usage, 'completion_tokens', 0)
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            # Calculate cost in dollars
            input_cost = (input_tokens / 1_000_000) * self.input_token_cost_per_million
            output_cost = (output_tokens / 1_000_000) * self.output_token_cost_per_million
            cost = input_cost + output_cost
            
            self.total_cost += cost
            
            logging.debug(f"API call: {input_tokens} input tokens (${input_cost:.6f}), {output_tokens} output tokens (${output_cost:.6f}), total cost: ${cost:.6f}")

    def get_cost_summary(self):
        """Get a summary of total API costs."""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_cost': self.total_cost,
            'input_cost': (self.total_input_tokens / 1_000_000) * self.input_token_cost_per_million,
            'output_cost': (self.total_output_tokens / 1_000_000) * self.output_token_cost_per_million
        }

    async def _retry_api_call(self, api_call_func, max_retries=3, base_delay=1.0):
        """Retry API call with exponential backoff."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await api_call_func()
            except RateLimitError as e:
                last_exception = e
                if attempt < max_retries:
                    # For rate limit errors, use the retry-after header if available
                    retry_after = getattr(e, 'retry_after', None)
                    if retry_after:
                        delay = float(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Rate limit exceeded, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                else:
                    logging.error(f"Rate limit error persisted after {max_retries + 1} attempts")
            except (APIConnectionError, APIError) as e:
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"API error ({type(e).__name__}), retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                else:
                    logging.error(f"API error persisted after {max_retries + 1} attempts: {e}")
            except Exception as e:
                # For unexpected errors, don't retry
                logging.error(f"Unexpected error: {e}")
                raise e
        
        # If we get here, all retries failed
        raise last_exception

    async def get_action_from_gpt(self, obs):
        if self.backend == "openai":
            async def _api_call():
                async with self.semaphore:
                    resp = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": obs}],
                        max_completion_tokens=self.max_tokens,
                    )
                self.track_api_cost(resp)
                ret = resp.choices[0].message.content.strip()
                if not ret:
                    raise RuntimeError(f"Empty response from OpenAI for prompt: {obs}")
                return ret
            
            return await self._retry_api_call(_api_call)
        else:  # vllm
            ip_addr = random.choice(self.ip_addrs)
            client = self.clients[ip_addr]  # Reuse existing client
            resp = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": obs}],
                temperature=self.temperature,
                extra_body={"min_tokens": 128}, 
                max_tokens=self.max_tokens,
            )
            # Note: vLLM may not provide usage info, so we skip cost tracking for vLLM
            ret = resp.choices[0].message.content.strip()
            if not ret:
                raise RuntimeError(f"Empty response {resp} from {ip_addr} for prompt: {obs}")
            return ret
    
    async def get_strategy_from_gpt(self, obs):
        if self.backend == "openai":
            async def _api_call():
                async with self.semaphore:
                    resp = await self.strategy_client.chat.completions.create(
                        model=self.strategy_model_name,
                        messages=[{"role": "user", "content": obs}],
                        max_completion_tokens=self.strategy_max_tokens,
                    )
                self.track_api_cost(resp)
                ret = resp.choices[0].message.content.strip()
                if not ret:
                    raise RuntimeError(f"Empty response from OpenAI strategy for prompt: {obs}")
                return ret
            
            return await self._retry_api_call(_api_call)
        else:  # vllm
            ip_addr = random.choice(self.strategy_ip_addrs)
            client = self.strategy_clients[ip_addr]  # Reuse existing client
            resp = await client.chat.completions.create(
                model=self.strategy_model_name,
                messages=[{"role": "user", "content": obs}],
                temperature=self.temperature,
                extra_body={"min_tokens": 128}, 
                max_tokens=self.strategy_max_tokens,
            )
            # Note: vLLM may not provide usage info, so we skip cost tracking for vLLM
            ret = resp.choices[0].message.content.strip()
            if not ret:
                raise RuntimeError(f"Empty response {resp} from strategy server {ip_addr} for prompt: {obs}")
            return ret
    
    async def get_action_with_adamem_high(
        self, 
        prompt: str, 
        env_manager, 
        env_idx: int,
        current_obs_text: str
    ):
        """
        Three-stage generation: initial response, strategy generation from memory, action from strategy.
        
        Args:
            prompt: The initial prompt for the agent
            env_manager: Environment manager (for memory retrieval)
            env_idx: Index of the environment (for debugging)
            current_obs_text: The current observation text for retrieval
            
        Returns:
            (final_action, retrieval_requested, retrieval_reason, retrieval_info)
            retrieval_info contains detailed info from all stages
        """
        # Stage 1: Get initial response
        initial_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_WITH_OPTIONAL_MEMORY
        initial_response = await self.get_action_from_gpt(initial_prompt)

        # Extract initial action
        initial_action = extract_action_from_response(initial_response)
        
        # Check if memory retrieval was requested
        has_request, reason = parse_memory_request(initial_response)
        
        if has_request and env_manager.memory_type == "adamem-high":
            try:
                # Import here to avoid circular dependency
                from utils import get_top_k_memories

                # Retrieve memories based on current observation
                top_k_memories = get_top_k_memories(current_obs_text, topk=topk)
                retrieved_exp = "\n\n".join([
                    f"Retrieved Item {idx}:\n{content}"
                    for idx, (content, relevance_score) in enumerate(top_k_memories)
                ])
                retrieved_exp = self._truncate_retrieval_for_prompt(
                    base_prompt=prompt,
                    retrieval_text=retrieved_exp,
                    output_token_reserve=self.max_tokens,
                )

                # Stage 2: Generate strategy from retrieved experiences
                strategy_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_STRATEGY_GENERATION.format(
                    k=topk,
                    retrieved_exp=retrieved_exp
                )
                strategy_response = await self.get_strategy_from_gpt(strategy_prompt)
                strategy = extract_strategy_from_response(strategy_response)

                if not strategy_response:
                    logging.warning(f"Env {env_idx}: Empty strategy_response after stage 2!")

                # Stage 3: Generate action based on strategy (no retrieved trajectories in context)
                action_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_ACTION_FROM_STRATEGY.format(
                    strategy=strategy if strategy else "[No strategy provided]"
                )
                final_response = await self.get_action_from_gpt(action_prompt)
                final_action = extract_action_from_response(final_response)
                
                # Debug: Check if final_response is empty
                if not final_response:
                    logging.warning(f"Env {env_idx}: Empty final_response after stage 3!")
                
                # Check if action changed from initial
                action_changed = (initial_action != final_action)
                
                # Package retrieval information
                retrieval_info = {
                    "initial_prompt": initial_prompt,
                    "initial_response": initial_response,
                    "initial_action": initial_action,
                    "strategy_prompt": strategy_prompt,
                    "strategy_response": strategy_response,
                    "strategy": strategy,
                    "action_prompt": action_prompt,
                    "final_response": final_response,
                    "final_action": final_action,
                    "action_changed": action_changed
                }
                
                return final_response, True, reason, retrieval_info
            except Exception as e:
                logging.warning(f"Three-stage memory retrieval failed for env {env_idx}: {e}. Using initial response.")
                retrieval_info = {
                    "initial_prompt": initial_prompt,
                    "initial_response": initial_response,
                    "initial_action": initial_action,
                    "strategy_prompt": None,
                    "strategy_response": None,
                    "strategy": None,
                    "action_prompt": None,
                    "final_response": None,
                    "final_action": initial_action,
                    "action_changed": False
                }
                return initial_response, False, "", retrieval_info
        else:
            # No retrieval - use initial action
            retrieval_info = {
                "initial_prompt": initial_prompt,
                "initial_response": initial_response,
                "initial_action": initial_action,
                "strategy_prompt": None,
                "strategy_response": None,
                "strategy": None,
                "action_prompt": None,
                "final_response": None,
                "final_action": initial_action,
                "action_changed": False
            }
            return initial_response, False, "", retrieval_info
    
    async def get_action_with_adamem_max(
        self, 
        prompt: str, 
        env_manager, 
        env_idx: int,
        current_obs_text: str
    ):
        """
        Three-stage generation (always retrieve): strategy generation from memory, action from strategy.
        
        Args:
            prompt: The initial prompt for the agent
            env_manager: Environment manager (for memory retrieval)
            env_idx: Index of the environment (for debugging)
            current_obs_text: The current observation text for retrieval
            
        Returns:
            (final_action, retrieval_requested, retrieval_reason, retrieval_info)
            retrieval_info contains detailed info from all stages
        """
        try:
            # Import here to avoid circular dependency
            from utils import get_top_k_memories
            
            # Always retrieve memories based on current observation
            top_k_memories = get_top_k_memories(current_obs_text, topk=topk)
            retrieved_exp = "\n\n".join([
                f"Retrieved Item {idx}:\n{content}"
                for idx, (content, relevance_score) in enumerate(top_k_memories)
            ])
            retrieved_exp = self._truncate_retrieval_for_prompt(
                base_prompt=prompt,
                retrieval_text=retrieved_exp,
                output_token_reserve=self.strategy_max_tokens,
            )

            # Stage 2: Generate strategy from retrieved experiences
            strategy_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_STRATEGY_GENERATION.format(
                k=topk,
                retrieved_exp=retrieved_exp
            )
            strategy_response = await self.get_strategy_from_gpt(strategy_prompt)
            strategy = extract_strategy_from_response(strategy_response)
            
            if not strategy_response:
                logging.warning(f"Env {env_idx}: Empty strategy_response after stage 2!")
            
            # Stage 3: Generate action based on strategy (no retrieved trajectories in context)
            action_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_ACTION_FROM_STRATEGY.format(
                strategy=strategy if strategy else "[No strategy provided]"
            )
            final_response = await self.get_action_from_gpt(action_prompt)
            final_action = extract_action_from_response(final_response)
            
            # Debug: Check if final_response is empty
            if not final_response:
                logging.warning(f"Env {env_idx}: Empty final_response after stage 3!")
            
            # Package retrieval information
            retrieval_info = {
                "strategy_prompt": strategy_prompt,
                "strategy_response": strategy_response,
                "strategy": strategy,
                "action_prompt": action_prompt,
                "final_response": final_response,
                "final_action": final_action,
            }
            
            return final_response, True, "always_retrieve", retrieval_info
        except Exception as e:
            logging.warning(f"Three-stage every step memory retrieval failed for env {env_idx}: {e}. Using direct action.")
            # Fallback to direct action
            direct_response = await self.get_action_from_gpt(prompt)
            direct_action = extract_action_from_response(direct_response)
            retrieval_info = {
                "strategy_prompt": None,
                "strategy_response": None,
                "strategy": None,
                "action_prompt": None,
                "final_response": direct_response,
                "final_action": direct_action,
            }
            return direct_response, False, "", retrieval_info
    
    async def get_action_with_adamem_max_without_trajectory(
        self, 
        prompt: str, 
        env_manager, 
        env_idx: int,
        current_obs_text: str
    ):
        """
        Three-stage generation without retrieval: strategy generation directly, then action from strategy.
        
        Args:
            prompt: The initial prompt for the agent
            env_manager: Environment manager (for consistency)
            env_idx: Index of the environment (for debugging)
            current_obs_text: The current observation text (for consistency)
            
        Returns:
            (final_action, retrieval_requested, retrieval_reason, retrieval_info)
            retrieval_info contains detailed info from stages
        """
        try:
            # Stage 1: Generate strategy directly without retrieval
            strategy_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_STRATEGY_NO_RETRIEVAL
            strategy_response = await self.get_strategy_from_gpt(strategy_prompt)
            strategy = extract_strategy_from_response(strategy_response)
            
            if not strategy_response:
                logging.warning(f"Env {env_idx}: Empty strategy_response after stage 1!")
            
            # Stage 2: Generate action based on strategy (no retrieved trajectories in context)
            action_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_ACTION_FROM_STRATEGY.format(
                strategy=strategy if strategy else "[No strategy provided]"
            )
            final_response = await self.get_action_from_gpt(action_prompt)
            final_action = extract_action_from_response(final_response)
            
            # Debug: Check if final_response is empty
            if not final_response:
                logging.warning(f"Env {env_idx}: Empty final_response after stage 2!")
            
            # Package retrieval information (no retrieval, so requested=False)
            retrieval_info = {
                "strategy_prompt": strategy_prompt,
                "strategy_response": strategy_response,
                "strategy": strategy,
                "action_prompt": action_prompt,
                "final_response": final_response,
                "final_action": final_action,
            }
            
            return final_response, False, "no_retrieval", retrieval_info
        except Exception as e:
            logging.warning(f"Three-stage no retrieval failed for env {env_idx}: {e}. Using direct action.")
            # Fallback to direct action
            direct_response = await self.get_action_from_gpt(prompt)
            direct_action = extract_action_from_response(direct_response)
            retrieval_info = {
                "strategy_prompt": None,
                "strategy_response": None,
                "strategy": None,
                "action_prompt": None,
                "final_response": direct_response,
                "final_action": direct_action,
            }
            return direct_response, False, "", retrieval_info
    
    async def get_action_with_adamem_max_without_strategy(
        self, 
        prompt: str, 
        env_manager, 
        env_idx: int,
        current_obs_text: str
    ):
        """
        Direct retrieval every step: retrieve memories and directly generate action based on them.
        
        Args:
            prompt: The initial prompt for the agent
            env_manager: Environment manager (for memory retrieval)
            env_idx: Index of the environment (for debugging)
            current_obs_text: The current observation text for retrieval
            
        Returns:
            (final_action, retrieval_requested, retrieval_reason, retrieval_info)
            retrieval_info contains detailed info from retrieval and action generation
        """
        try:
            # Import here to avoid circular dependency
            from utils import get_top_k_memories
            
            # Always retrieve memories based on current observation
            top_k_memories = get_top_k_memories(current_obs_text, topk=topk)
            retrieved_exp = "\n\n".join([
                f"Retrieved Item {idx}:\n{content}"
                for idx, (content, relevance_score) in enumerate(top_k_memories)
            ])
            retrieved_exp = self._truncate_retrieval_for_prompt(
                base_prompt=prompt,
                retrieval_text=retrieved_exp,
                output_token_reserve=self.max_tokens,
            )

            # Directly generate action based on retrieved experiences
            action_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_DIRECT_ACTION_FROM_RETRIEVAL.format(
                retrieved_exp=retrieved_exp
            )
            final_response = await self.get_action_from_gpt(action_prompt)
            final_action = extract_action_from_response(final_response)
            
            # Debug: Check if final_response is empty
            if not final_response:
                logging.warning(f"Env {env_idx}: Empty final_response after direct retrieval!")
            
            # Package retrieval information
            retrieval_info = {
                "action_prompt": action_prompt,
                "final_response": final_response,
                "final_action": final_action,
            }
            
            return final_response, True, "always_retrieve", retrieval_info
        except Exception as e:
            logging.warning(f"Direct retrieval every step failed for env {env_idx}: {e}. Using direct action.")
            # Fallback to direct action
            direct_response = await self.get_action_from_gpt(prompt)
            direct_action = extract_action_from_response(direct_response)
            retrieval_info = {
                "action_prompt": None,
                "final_response": direct_response,
                "final_action": direct_action,
            }
            return direct_response, False, "", retrieval_info

    async def get_action_with_adamem_low(
        self, 
        prompt: str, 
        env_manager, 
        env_idx: int,
        current_obs_text: str,
        recent_history: str = "",
    ):
        """
        Strategy reuse with conditional refresh: maintain strategy across steps, refresh when needed.
        Optimized to combine action generation with refresh decision in one call.
        
        Args:
            prompt: The initial prompt for the agent
            env_manager: Environment manager (for consistency)
            env_idx: Index of the environment (for debugging)
            current_obs_text: The current observation text for retrieval/refresh decision
            recent_history: Recent actions/observations to detect loops or deviations
            
        Returns:
            (final_action, retrieval_requested, retrieval_reason, retrieval_info)
            retrieval_info contains detailed info from stages
        """
        try:
            current_strategy = self.active_strategies.get(env_idx, "")
            
            if current_strategy is None:
                # No strategy yet - retrieve, generate strategy, generate action
                from utils import get_top_k_memories
                
                top_k_memories = get_top_k_memories(current_obs_text, topk=topk)
                retrieved_exp = "\n\n".join([
                    f"Retrieved Item {idx}:\n{content}"
                    for idx, (content, relevance_score) in enumerate(top_k_memories)
                ])
                retrieved_exp = self._truncate_retrieval_for_prompt(
                    base_prompt=prompt,
                    retrieval_text=retrieved_exp,
                    output_token_reserve=self.strategy_max_tokens,
                )

                strategy_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_STRATEGY_GENERATION.format(
                    k=topk,
                    retrieved_exp=retrieved_exp
                )
                strategy_response = await self.get_strategy_from_gpt(strategy_prompt)
                new_strategy = extract_strategy_from_response(strategy_response)
                
                if not new_strategy:
                    logging.warning(f"Env {env_idx}: Empty strategy_response after regeneration!")
                    new_strategy = "[No strategy generated]"
                
                # Store the new strategy
                self.active_strategies[env_idx] = new_strategy
                
                # Generate action from new strategy
                action_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_ACTION_FROM_STRATEGY.format(
                    strategy=new_strategy
                )
                final_response = await self.get_action_from_gpt(action_prompt)
                final_action = extract_action_from_response(final_response)
                
                if not final_response:
                    logging.warning(f"Env {env_idx}: Empty final_response after action generation!")
                
                # Package retrieval information
                retrieval_info = {
                    "refresh_decision": True,
                    "refresh_reason": "No existing strategy",
                    "refresh_response": None,
                    "refresh_prompt": None,
                    "strategy_prompt": strategy_prompt,
                    "strategy_response": strategy_response,
                    "strategy": new_strategy,
                    "action_prompt": action_prompt,
                    "final_response": final_response,
                    "final_action": final_action,
                }
                
                return final_response, True, "initial_strategy_generation", retrieval_info
            else:
                # Has strategy - combined action generation + refresh decision
                combined_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_STRATEGY_REFRESH_DECISION.format(
                    current_strategy=current_strategy
                )
                combined_response = await self.get_action_from_gpt(combined_prompt)
                
                initial_action, should_refresh, refresh_reason = parse_action_and_refresh(combined_response)
                
                if not should_refresh:
                    # Reuse strategy - return the initial action directly
                    retrieval_info = {
                        "refresh_decision": should_refresh,
                        "refresh_reason": refresh_reason,
                        "refresh_response": combined_response,
                        "refresh_prompt": combined_prompt,
                        "strategy_prompt": None,
                        "strategy_response": None,
                        "strategy": current_strategy,
                        "action_prompt": None,
                        "final_response": combined_response,
                        "final_action": initial_action,
                    }
                    
                    return combined_response, False, "strategy_reused", retrieval_info
                else:
                    # Refresh strategy - retrieve, generate new strategy, generate new action
                    from utils import get_top_k_memories
                    
                    top_k_memories = get_top_k_memories(current_obs_text, topk=topk)
                    retrieved_exp = "\n\n".join([
                        f"Retrieved Item {idx}:\n{content}"
                        for idx, (content, relevance_score) in enumerate(top_k_memories)
                    ])
                    retrieved_exp = self._truncate_retrieval_for_prompt(
                        base_prompt=prompt,
                        retrieval_text=retrieved_exp,
                        output_token_reserve=self.strategy_max_tokens,
                    )

                    strategy_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_STRATEGY_GENERATION.format(
                        k=topk,
                        retrieved_exp=retrieved_exp
                    )
                    strategy_response = await self.get_strategy_from_gpt(strategy_prompt)
                    new_strategy = extract_strategy_from_response(strategy_response)
                    
                    if not new_strategy:
                        logging.warning(f"Env {env_idx}: Empty strategy_response after regeneration!")
                        new_strategy = "[No strategy generated]"
                    
                    # Store the new strategy
                    self.active_strategies[env_idx] = new_strategy
                    
                    # Generate action from new strategy
                    action_prompt = prompt + "\n\n" + WEBSHOP_TEMPLATE_ACTION_FROM_STRATEGY.format(
                        strategy=new_strategy
                    )
                    final_response = await self.get_action_from_gpt(action_prompt)
                    final_action = extract_action_from_response(final_response)
                    
                    if not final_response:
                        logging.warning(f"Env {env_idx}: Empty final_response after action generation!")
                    
                    # Package retrieval information
                    retrieval_info = {
                        "refresh_decision": should_refresh,
                        "refresh_reason": refresh_reason,
                        "refresh_response": combined_response,
                        "refresh_prompt": combined_prompt,
                        "strategy_prompt": strategy_prompt,
                        "strategy_response": strategy_response,
                        "strategy": new_strategy,
                        "action_prompt": action_prompt,
                        "final_response": final_response,
                        "final_action": final_action,
                    }
                    
                    return final_response, True, "strategy_refreshed", retrieval_info
        except Exception as e:
            logging.warning(f"Step strategy reuse failed for env {env_idx}: {e}. Using direct action.")
            # Fallback to direct action
            direct_response = await self.get_action_from_gpt(prompt)
            direct_action = extract_action_from_response(direct_response)
            retrieval_info = {
                "refresh_decision": None,
                "refresh_reason": str(e),
                "refresh_response": None,
                "refresh_prompt": None,
                "strategy_prompt": None,
                "strategy_response": None,
                "strategy": None,
                "action_prompt": None,
                "final_response": direct_response,
                "final_action": direct_action,
            }
            return direct_response, False, "", retrieval_info

async def main():
    # -------- logging ----------
    log_dir = f"logs/webshop/{MODEL_NAME.replace('/', '_')}"
    os.makedirs(log_dir, exist_ok=True)

    def _add_file_suffix(path: str, suffix: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}_{suffix}{ext}"

    def _maybe_restart_vllm_between_batches(batch_idx: int, num_batches: int) -> None:
        """Optionally restart vLLM server between batches.

        Controlled by env var RESTART_VLLM_BETWEEN_BATCHES=true.
        """
        enabled = os.environ.get("RESTART_VLLM_BETWEEN_BATCHES", "false").lower() == "true"
        if not enabled:
            return

        script_path = os.environ.get(
            "RESTART_VLLM_SCRIPT",
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../scripts/restart_vllm_server.sh",
                )
            ),
        )
        openai_base_ip_addr = os.environ.get("OPENAI_BASE_IP_ADDR", "").strip()
        cuda_device = os.environ.get("CUDA_DEVICE", "").strip()
        gpu_util = os.environ.get("GPU_UTIL", "").strip()
        model_name = os.environ.get("MODEL_NAME", "").strip()

        missing = [
            name
            for name, value in [
                ("OPENAI_BASE_IP_ADDR", openai_base_ip_addr),
                ("CUDA_DEVICE", cuda_device),
                ("GPU_UTIL", gpu_util),
                ("MODEL_NAME", model_name),
            ]
            if not value
        ]
        if missing:
            logging.warning(
                "Skip vLLM restart between batches: missing env vars %s",
                missing,
            )
            return

        cmd = [
            script_path,
            openai_base_ip_addr,
            cuda_device,
            gpu_util,
            model_name,
        ]

        logging.info(
            f"Restarting vLLM between batches using script: {script_path}"
        )
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=True,
            )
            if result.stdout:
                logging.info(f"vLLM restart stdout:\n{result.stdout}")
            if result.stderr:
                logging.info(f"vLLM restart stderr:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(
                f"vLLM restart failed with return code {e.returncode}; "
                f"stdout={e.stdout}; stderr={e.stderr}"
            )
            raise

        wait_sec = float(os.environ.get("RESTART_VLLM_WAIT_SEC", "5"))
        if wait_sec > 0:
            logging.info(f"Waiting {wait_sec:.1f}s after vLLM restart")
            time.sleep(wait_sec)
    
    # Get strategy model name for filename distinction
    strategy_model_name = os.environ.get("STRATEGY_MODEL_NAME", MODEL_NAME)
    
    # Create stats file path
    if split == "train":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID for brevity
        traj_file = os.path.join(log_dir, f"traj_{split}_{timestamp}_{run_uuid}.json")
    else:
        traj_file = os.path.join(log_dir, f"traj_{split}.json")
    if mem_type:
        traj_file = traj_file.replace(".json", f"_{mem_type}.json")
        if correct_only:
            traj_file = traj_file.replace(".json", f"_correct_only.json")
    if memory_mode != "full":
        traj_file = traj_file.replace(".json", f"_{memory_mode}.json")
    # Add strategy model suffix if different from policy model
    if strategy_model_name != MODEL_NAME:
        strategy_suffix = strategy_model_name.replace('/', '_')
        traj_file = traj_file.replace(".json", f"_strategy-{strategy_suffix}.json")

    # Optional explicit run tag from launcher script, e.g. RUN_TAG=round1.
    run_tag = os.environ.get("RUN_TAG", "").strip()
    if run_tag:
        safe_run_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", run_tag)
        traj_file = _add_file_suffix(traj_file, safe_run_tag)

    # Final safeguard: never overwrite an existing trajectory file.
    if os.path.exists(traj_file):
        unique_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        traj_file = _add_file_suffix(traj_file, unique_tag)

    stats_file = traj_file.replace("traj_", "stats_").replace(".json", ".txt")
    
    # Setup logging to both console and stats file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(stats_file, encoding="utf-8"),  # Add stats file handler
            logging.StreamHandler()
        ],
    )
    
    logging.info(f"Saving trajectories to {traj_file}")
    logging.info(f"Saving statistics to {stats_file}")

    # -------- Parameters ----------
    max_steps = int(os.environ.get("MAX_STEPS", 15))
    env_num_total = SPLIT2ENV_NUM[split]  # Total number of test cases
    test_times = int(os.environ.get("TEST_TIMES", 1))  # Number of times to run the full dataset
    batch_size = int(os.environ.get("EVAL_BATCH_SIZE", env_num_total))  # Batch size for evaluation
    num_batches = (env_num_total + batch_size - 1) // batch_size  # Calculate number of batches
    env_name = "webshop"
    
    logging.info(f"Total test cases: {env_num_total}")
    logging.info(f"Test times: {test_times}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Number of batches: {num_batches}")

    # -------- Agent setup ----------
    agent = Agent()
    
    # -------- Create reusable environment workers (only once) ----------
    logging.info(f"Creating {batch_size} reusable workers for {'evaluation' if split == 'eval' else 'training'}...")
    env_manager = build_env(env_name, batch_size, batch_start_idx=0, reuse_workers=True)
    logging.info(f"Workers created and ready for reuse across batches")

    # Accumulated statistics across all runs
    all_run_success_rates = []  # Success rate for each full dataset run
    all_run_task_scores = []    # Average task score for each full dataset run
    all_trajectories = []

    # ======================= Main Loop: TEST_TIMES iterations =======================
    for run_idx in range(test_times):
        logging.info(f"\n{'='*80}")
        logging.info(f"========== Starting RUN {run_idx + 1}/{test_times} ==========")
        logging.info(f"{'='*80}\n")
        
        run_start_time = time.time()
        
        # Tracking for this run (aggregated across all batches)
        run_success_count = 0
        run_task_score_sum = 0.0
        run_total_envs = 0
        
        # Retrieval statistics for this run
        if mem_type in MEM_TYPES_WITH_RETRIEVAL_STATS:
            run_retrieval_stats = {
                'total_steps': 0,
                'total_requests': 0,
                'total_action_changes': 0,
                'steps_per_trajectory': [],
                'retrievals_per_trajectory': [],
                'action_changes_per_trajectory': []
            }
        
        # ======================= Batch Loop: Process dataset in batches =======================
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min(batch_start_idx + batch_size, env_num_total)
            current_batch_size = batch_end_idx - batch_start_idx

            _maybe_restart_vllm_between_batches(batch_idx, num_batches)
            
            logging.info(f"\n--- Batch {batch_idx + 1}/{num_batches}: {'Goals' if split == 'eval' else 'Sampled goals'} [{batch_start_idx}:{batch_end_idx}] ({current_batch_size} {'goals' if split == 'eval' else 'samples'}) ---")
            
            # Reuse workers for both training and evaluation
            if split == "eval":
                # Evaluation: reset with specific sequential goal indices
                goal_indices = list(range(batch_start_idx, batch_end_idx))
                # Pad with last goal if current_batch_size < batch_size (for last incomplete batch)
                if current_batch_size < batch_size:
                    goal_indices.extend([batch_end_idx - 1] * (batch_size - current_batch_size))
                    logging.info(f"Padding last batch: using {current_batch_size} unique goals, padding {batch_size - current_batch_size} workers with goal {batch_end_idx - 1}")
                
                obs, infos = env_manager.reset_with_goals(goal_indices)
                # Truncate results if we padded
                if current_batch_size < batch_size:
                    obs['text'] = obs['text'][:current_batch_size]
                    obs['anchor'] = obs['anchor'][:current_batch_size]
                    infos = infos[:current_batch_size]
            else:
                # Training: reset with randomly sampled goal indices
                # DON'T use seed for truly random sampling (non-deterministic)
                # Pad with additional samples if current_batch_size < batch_size (for last incomplete batch)
                if current_batch_size < batch_size:
                    obs, infos = env_manager.reset_with_random_goals(batch_size, seed=None)
                    # Truncate to actual batch size
                    obs['text'] = obs['text'][:current_batch_size]
                    obs['anchor'] = obs['anchor'][:current_batch_size]
                    infos = infos[:current_batch_size]
                    logging.info(f"Sampled {current_batch_size} random training goals (truncated from {batch_size})")
                else:
                    obs, infos = env_manager.reset_with_random_goals(current_batch_size, seed=None)
                    logging.info(f"Sampled {current_batch_size} random training goals")
            
            # Batch-level tracking
            batch_retrieval_stats = None
            if mem_type in MEM_TYPES_WITH_RETRIEVAL_STATS:
                batch_retrieval_stats = {
                    'total_steps': 0,
                    'total_requests': 0,
                    'total_action_changes': 0
                }
                # Per-trajectory tracking (will aggregate into run stats)
                trajectory_steps = [0] * current_batch_size
                trajectory_retrievals = [0] * current_batch_size
                trajectory_action_changes = [0] * current_batch_size
            
            env_dones = [False] * current_batch_size
            
            # Batch statistics
            batch_success = np.zeros(current_batch_size, dtype=bool)
            batch_task_scores = np.zeros(current_batch_size, dtype=float)
            
            trajs = [{"won": None, "task_score": None, "steps": [], "timing_per_step": []} for i in range(current_batch_size)]
            
            # Reset agent strategies for new trajectories
            agent.active_strategies = {}
            if hasattr(agent, "ttsft_env_adapters"):
                agent.ttsft_env_adapters = {}
            if hasattr(agent, "ttsft_env_icl_trajectories"):
                agent.ttsft_env_icl_trajectories = {}
            if hasattr(agent, "ttsft_env_last_sft_signature"):
                agent.ttsft_env_last_sft_signature = {}
            if hasattr(agent, "ttsft_env_adapter_history"):
                agent.ttsft_env_adapter_history = {}
            if hasattr(agent, "ttsft_env_adapter_counter"):
                agent.ttsft_env_adapter_counter = {}
            
            # ======================= Step Loop: Run environments =======================
            for step_idx in range(max_steps):
                logging.info(f"Step {step_idx}; Dones ({np.array(env_dones).sum().item()}/{current_batch_size}); SR {batch_success.mean().item():.3f}")

                # --- Assemble actions with optional memory retrieval ---
                actions = ["None"] * current_batch_size
                retrieval_requests = [False] * current_batch_size
                retrieval_reasons = [""] * current_batch_size
                retrieval_infos = [None] * current_batch_size  # Store detailed retrieval info
                step_timings = [{
                    "retrieval_time": 0.0,
                    "strategy_time": 0.0,
                    "action_time": 0.0,
                    "retrieval_latency": 0.0,
                    "augmentation_latency": 0.0,
                    "training_latency": 0.0,
                    "lora_io_latency": 0.0,
                    "inference_latency": 0.0,
                } for _ in range(current_batch_size)]
                
                active_indices = [i for i in range(current_batch_size) if not env_dones[i]]
                if active_indices:
                    prompts = [obs["text"][i] for i in active_indices]
                    
                    # Async gather all action generations (with optional memory)
                    if mem_type == "adamem-high":
                        current_obs_texts = [obs["text"][i] for i in active_indices]
                        tasks = [
                            agent.get_action_with_adamem_high(
                                prompt, 
                                env_manager, 
                                idx,
                                current_obs_text
                            ) 
                            for prompt, idx, current_obs_text 
                            in zip(prompts, active_indices, current_obs_texts)
                        ]
                        results = await asyncio.gather(*tasks)
                        
                        for idx, (action, requested, reason, retrieval_info) in zip(active_indices, results):
                            actions[idx] = action
                            retrieval_requests[idx] = requested
                            retrieval_reasons[idx] = reason
                            retrieval_infos[idx] = retrieval_info
                        
                        # Track retrieval statistics
                        num_requested = sum(retrieval_requests)
                        # round_requests += num_requested  # Removed: tracking at batch level now
                        batch_retrieval_stats['total_steps'] += len(active_indices)
                        batch_retrieval_stats['total_requests'] += num_requested
                        
                        # Track action changes
                        num_action_changes = sum(1 for idx in active_indices 
                                                if retrieval_infos[idx] and retrieval_infos[idx].get('action_changed', False))
                        # round_action_changes += num_action_changes  # Removed: tracking at batch level now
                        batch_retrieval_stats['total_action_changes'] += num_action_changes
                        
                        # Update per-trajectory counters
                        for idx in active_indices:
                            trajectory_steps[idx] += 1
                            if retrieval_requests[idx]:
                                trajectory_retrievals[idx] += 1
                            if retrieval_infos[idx] and retrieval_infos[idx].get('action_changed', False):
                                trajectory_action_changes[idx] += 1
                        
                        logging.info(f"  Memory retrieval (three-stage): {num_requested}/{len(active_indices)} requested, {num_action_changes} action changes")
                    
                    elif mem_type == "adamem-max":
                        current_obs_texts = [obs["text"][i] for i in active_indices]
                        tasks = [
                            agent.get_action_with_adamem_max(
                                prompt, 
                                env_manager, 
                                idx,
                                current_obs_text
                            ) 
                            for prompt, idx, current_obs_text 
                            in zip(prompts, active_indices, current_obs_texts)
                        ]
                        results = await asyncio.gather(*tasks)
                        
                        for idx, (action, requested, reason, retrieval_info) in zip(active_indices, results):
                            actions[idx] = action
                            retrieval_requests[idx] = requested
                            retrieval_reasons[idx] = reason
                            retrieval_infos[idx] = retrieval_info
                        
                        # Track retrieval statistics
                        num_requested = sum(retrieval_requests)
                        # round_requests += num_requested  # Removed: tracking at batch level now
                        batch_retrieval_stats['total_steps'] += len(active_indices)
                        batch_retrieval_stats['total_requests'] += num_requested
                        
                        # For every_step, always retrieve, no action_changes since no initial
                        # round_action_changes += 0  # Removed: tracking at batch level now
                        batch_retrieval_stats['total_action_changes'] += 0
                        
                        # Update per-trajectory counters
                        for idx in active_indices:
                            trajectory_steps[idx] += 1
                            if retrieval_requests[idx]:
                                trajectory_retrievals[idx] += 1
                            # No action_changes for every_step
                        
                        logging.info(f"  Memory retrieval (three-stage every step): {num_requested}/{len(active_indices)} always retrieved")
                    
                    elif mem_type == "adamem-max-without-trajectory-memory":
                        current_obs_texts = [obs["text"][i] for i in active_indices]
                        tasks = [
                            agent.get_action_with_adamem_max_without_trajectory(
                                prompt, 
                                env_manager, 
                                idx,
                                current_obs_text
                            ) 
                            for prompt, idx, current_obs_text 
                            in zip(prompts, active_indices, current_obs_texts)
                        ]
                        results = await asyncio.gather(*tasks)
                        
                        for idx, (action, requested, reason, retrieval_info) in zip(active_indices, results):
                            actions[idx] = action
                            retrieval_requests[idx] = requested
                            retrieval_reasons[idx] = reason
                            retrieval_infos[idx] = retrieval_info
                        
                        # Track retrieval statistics (no retrieval, so all zero)
                        num_requested = 0
                        # round_requests += num_requested  # Removed: tracking at batch level now
                        batch_retrieval_stats['total_steps'] += len(active_indices)
                        batch_retrieval_stats['total_requests'] += num_requested
                        
                        # No action changes since no initial action
                        # round_action_changes += 0  # Removed: tracking at batch level now
                        batch_retrieval_stats['total_action_changes'] += 0
                        
                        # Update per-trajectory counters
                        for idx in active_indices:
                            trajectory_steps[idx] += 1
                            # No retrievals or action changes
                        
                        logging.info(f"  Memory retrieval (three-stage no retrieval): {num_requested}/{len(active_indices)} always no retrieval")
                    
                    elif mem_type == "adamem-max-without-strategy-memory":
                        current_obs_texts = [obs["text"][i] for i in active_indices]
                        tasks = [
                            agent.get_action_with_adamem_max_without_strategy(
                                prompt, 
                                env_manager, 
                                idx,
                                current_obs_text
                            ) 
                            for prompt, idx, current_obs_text 
                            in zip(prompts, active_indices, current_obs_texts)
                        ]
                        results = await asyncio.gather(*tasks)
                        
                        for idx, (action, requested, reason, retrieval_info) in zip(active_indices, results):
                            actions[idx] = action
                            retrieval_requests[idx] = requested
                            retrieval_reasons[idx] = reason
                            retrieval_infos[idx] = retrieval_info
                        
                        # Track retrieval statistics
                        num_requested = sum(retrieval_requests)
                        # round_requests += num_requested  # Removed: tracking at batch level now
                        batch_retrieval_stats['total_steps'] += len(active_indices)
                        batch_retrieval_stats['total_requests'] += num_requested
                        
                        # For direct retrieval every step, always retrieve, no action_changes
                        # round_action_changes += 0  # Removed: tracking at batch level now
                        batch_retrieval_stats['total_action_changes'] += 0
                        
                        # Update per-trajectory counters
                        for idx in active_indices:
                            trajectory_steps[idx] += 1
                            if retrieval_requests[idx]:
                                trajectory_retrievals[idx] += 1
                            # No action_changes for direct retrieval every step
                        
                        logging.info(f"  Memory retrieval (direct every step): {num_requested}/{len(active_indices)} always retrieved")
                    
                    elif mem_type == "adamem-low":
                        current_obs_texts = [obs["text"][i] for i in active_indices]
                        # Build recent history for each env (last 3 steps)
                        recent_histories = []
                        for idx in active_indices:
                            steps = trajs[idx]["steps"]
                            if len(steps) >= 3:
                                recent = steps[-3:]
                            else:
                                recent = steps
                            history_str = "\n".join([
                                f"Obs: {step.get('observation', 'N/A')}, Action: {step.get('action', 'N/A')}"
                                for step in recent
                            ])
                            recent_histories.append(history_str)
                        
                        tasks = [
                            agent.get_action_with_adamem_low(
                                prompt, 
                                env_manager, 
                                idx,
                                current_obs_text,
                                recent_history,
                            )
                            for prompt, idx, current_obs_text, recent_history 
                            in zip(prompts, active_indices, current_obs_texts, recent_histories)
                        ]
                        results = await asyncio.gather(*tasks)
                        
                        for idx, (action, requested, reason, retrieval_info) in zip(active_indices, results):
                            actions[idx] = action
                            retrieval_requests[idx] = requested
                            retrieval_reasons[idx] = reason
                            retrieval_infos[idx] = retrieval_info
                        
                        # Track retrieval statistics
                        num_requested = sum(retrieval_requests)
                        # round_requests += num_requested  # Removed: tracking at batch level now
                        batch_retrieval_stats['total_steps'] += len(active_indices)
                        batch_retrieval_stats['total_requests'] += num_requested
                        
                        # For reuse, refreshes are retrievals, no action_changes
                        # round_action_changes += 0  # Removed: tracking at batch level now
                        batch_retrieval_stats['total_action_changes'] += 0
                        
                        # Update per-trajectory counters
                        for idx in active_indices:
                            trajectory_steps[idx] += 1
                            if retrieval_requests[idx]:
                                trajectory_retrievals[idx] += 1
                            # No action_changes for reuse
                        
                        logging.info(f"  Memory retrieval (strategy reuse): {num_requested}/{len(active_indices)} refreshed")
                        curr_action = actions[i]
                        step_item = {
                            "step_idx": step_idx,
                            "env_num": batch_start_idx + i,
                            "curr_prompt": curr_prompt,
                            "curr_action": curr_action,
                        }
                        
                        # Add retrieval information for memory-enabled variants
                        if mem_type in MEM_TYPES_WITH_RETRIEVAL_STATS and retrieval_infos[i] is not None:
                            step_item["retrieval_requested"] = retrieval_requests[i]
                            step_item["retrieval_reason"] = retrieval_reasons[i]
                            if mem_type == "adamem-high":
                                # adamem-high has both initial decision and strategy
                                step_item["initial_prompt"] = retrieval_infos[i]["initial_prompt"]
                                step_item["initial_response"] = retrieval_infos[i]["initial_response"]
                                step_item["initial_action"] = retrieval_infos[i].get("initial_action")
                                step_item["strategy_prompt"] = retrieval_infos[i]["strategy_prompt"]
                                step_item["strategy_response"] = retrieval_infos[i]["strategy_response"]
                                step_item["strategy"] = retrieval_infos[i]["strategy"]
                                step_item["action_prompt"] = retrieval_infos[i]["action_prompt"]
                                step_item["final_response"] = retrieval_infos[i]["final_response"]
                                step_item["final_action"] = retrieval_infos[i].get("final_action")
                                step_item["action_changed"] = retrieval_infos[i].get("action_changed", False)
                            elif mem_type in ["adamem-max", "adamem-max-without-trajectory-memory"]:
                                step_item["strategy_prompt"] = retrieval_infos[i]["strategy_prompt"]
                                step_item["strategy_response"] = retrieval_infos[i]["strategy_response"]
                                step_item["strategy"] = retrieval_infos[i]["strategy"]
                                step_item["action_prompt"] = retrieval_infos[i]["action_prompt"]
                                step_item["final_response"] = retrieval_infos[i]["final_response"]
                                step_item["final_action"] = retrieval_infos[i].get("final_action")
                            elif mem_type == "adamem-max-without-strategy-memory":
                                step_item["action_prompt"] = retrieval_infos[i]["action_prompt"]
                                step_item["final_response"] = retrieval_infos[i]["final_response"]
                                step_item["final_action"] = retrieval_infos[i].get("final_action")
                            elif mem_type == "adamem-low":
                                step_item["refresh_decision"] = retrieval_infos[i]["refresh_decision"]
                                step_item["refresh_reason"] = retrieval_infos[i]["refresh_reason"]
                                step_item["refresh_response"] = retrieval_infos[i]["refresh_response"]
                                step_item["refresh_prompt"] = retrieval_infos[i]["refresh_prompt"]
                                step_item["strategy_prompt"] = retrieval_infos[i]["strategy_prompt"]
                                step_item["strategy_response"] = retrieval_infos[i]["strategy_response"]
                                step_item["strategy"] = retrieval_infos[i]["strategy"]
                                step_item["action_prompt"] = retrieval_infos[i]["action_prompt"]
                                step_item["final_response"] = retrieval_infos[i]["final_response"]
                                step_item["final_action"] = retrieval_infos[i].get("final_action")
                    
                    trajs[i]["steps"].append(step_item)
                    trajs[i]["timing_per_step"].append(step_timings[i])

                # --- Environment stepping ---
                obs, rewards, dones, infos = env_manager.step(actions)

                # --- Determine endings and successes ---
                for i in range(current_batch_size):
                    if env_dones[i]:
                        continue

                    if dones[i]:
                        env_dones[i] = True
                        won = bool(infos[i].get("won", False))
                        task_score = float(infos[i].get("task_score", 0.0))
                        
                        batch_success[i] = won
                        batch_task_scores[i] = task_score
                        
                        trajs[i]["won"] = won
                        trajs[i]["task_score"] = task_score
                        
                        # Save per-trajectory statistics to run stats
                        if mem_type in MEM_TYPES_WITH_RETRIEVAL_STATS:
                            run_retrieval_stats['steps_per_trajectory'].append(trajectory_steps[i])
                            run_retrieval_stats['retrievals_per_trajectory'].append(trajectory_retrievals[i])
                            run_retrieval_stats['action_changes_per_trajectory'].append(trajectory_action_changes[i])

                if all(env_dones):
                    logging.info("All batch environments finished early!")
                    break
        
            # Aggregate batch retrieval stats into run stats
            if mem_type in MEM_TYPES_WITH_RETRIEVAL_STATS:
                run_retrieval_stats['total_steps'] += batch_retrieval_stats['total_steps']
                run_retrieval_stats['total_requests'] += batch_retrieval_stats['total_requests']
                run_retrieval_stats['total_action_changes'] += batch_retrieval_stats['total_action_changes']
        
            # Aggregate batch results into run results
            all_trajectories.extend(trajs)
            with open(traj_file, 'w') as writer:
                json.dump(all_trajectories, writer, indent=2)

            run_success_count += batch_success.sum()
            run_task_score_sum += batch_task_scores.sum()
            run_total_envs += current_batch_size
            
            logging.info(f"Batch {batch_idx + 1}/{num_batches} completed: SR={batch_success.mean():.4f}, Avg Score={batch_task_scores.mean():.4f}")
    
        # ======================= Run completion: Calculate and save run statistics =======================
        run_success_rate = run_success_count / run_total_envs if run_total_envs > 0 else 0.0
        run_avg_task_score = run_task_score_sum / run_total_envs if run_total_envs > 0 else 0.0
        
        all_run_success_rates.append(run_success_rate)
        all_run_task_scores.append(run_avg_task_score)
        
        
        logging.info(f"\nRun {run_idx + 1}/{test_times} completed:")
        logging.info(f"  Success rate: {run_success_rate:.4f} ({run_success_count}/{run_total_envs})")
        logging.info(f"  Avg task score: {run_avg_task_score:.4f}")
        logging.info(f"  Time elapsed: {time.time() - run_start_time:.2f}s\n")

    # ======================= Final Summary Across All Runs =======================
    logging.info("=" * 80)
    logging.info("=============== Final Summary ===============")
    logging.info("=" * 80)
    logging.info(
        f"Total runs: {test_times} | Test cases per run: {env_num_total} | Total evaluations: {env_num_total * test_times}"
    )

    if test_times > 1:
        # Show statistics with std when TEST_TIMES > 1
        logging.info(
            f"Overall success rate: "
            f"{100*np.mean(all_run_success_rates):.1f} ± {100*np.std(all_run_success_rates, ddof=1):.1f}%"
        )
        logging.info(
            f"Overall task score: "
            f"{np.mean(all_run_task_scores):.4f} ± {np.std(all_run_task_scores, ddof=1):.4f}"
        )
    else:
        # Simple statistics when TEST_TIMES == 1
        logging.info(f"Overall success rate: {100*all_run_success_rates[0]:.1f}%")
        logging.info(f"Overall task score: {all_run_task_scores[0]:.4f}")

    # Log truncation statistics
    trunc_stats = getattr(agent, "truncation_stats", None)
    if trunc_stats and trunc_stats.get("total", 0) > 0:
        trunc_total = trunc_stats["total"]
        trunc_count = trunc_stats.get("truncated", 0)
        trunc_rate = trunc_count / max(trunc_total, 1)
        logging.info(
            f"Prompt truncation rate: {trunc_rate:.4%} "
            f"({trunc_count}/{trunc_total})"
        )

    # Log API cost summary if using OpenAI backend
    if agent.backend == "openai":
        cost_summary = agent.get_cost_summary()
        logging.info("=============== API Cost Summary ===============")
        logging.info(
            f"Total input tokens: {cost_summary['total_input_tokens']:,} "
            f"(${cost_summary['input_cost']:.4f})"
        )
        logging.info(
            f"Total output tokens: {cost_summary['total_output_tokens']:,} "
            f"(${cost_summary['output_cost']:.4f})"
        )
        logging.info(
            f"Total tokens: {cost_summary['total_tokens']:,} "
            f"(Total cost: ${cost_summary['total_cost']:.4f})"
        )
        logging.info(
            f"Average cost per run: ${cost_summary['total_cost'] / max(test_times, 1):.4f}"
        )

if __name__ == "__main__":
    asyncio.run(main())
