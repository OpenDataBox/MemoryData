import os
import uuid
import math
import subprocess
import numpy as np
import json
import time
import hashlib
import logging
import asyncio
import random
import re
import shutil
import httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
from datetime import datetime
from collections import defaultdict
from agent_system.environments.env_manager import *
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# split = "eval_in_distribution" # 'train' or 'eval_in_distribution' or 'eval_out_of_distribution'
# split = 'train'
# split = 'eval_out_of_distribution'
split = os.environ["SPLIT"]
# TODO: we can control that total_tasks = test_times (total_steps) * env_num (batch_size)
SPLIT2ENV_NUM = {'train' : int(os.environ.get("EVAL_BATCH_SIZE", 150)), 'eval_in_distribution' : int(os.environ.get("EVAL_BATCH_SIZE", 140)), 'eval_out_of_distribution' : int(os.environ.get("EVAL_BATCH_SIZE", 134))} # for eval env_num should equal game cnt for deterministic eval; for train no need (random sample with replacement)
# SPLIT2ENV_NUM = {'train' : 5000, 'eval_in_distribution' : 10, 'eval_out_of_distribution' : 10} # for eval env_num should equal game cnt for deterministic eval; for train no need (random sample with replacement)
mem_type = os.environ.get("MEM_TYPE", None)
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
_correct_only_val = os.environ.get("CORRECT_ONLY", "false").lower()
correct_only = _correct_only_val == "true"
mix_mode = _correct_only_val == "mix"


def _parse_env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        logging.warning(
            f"Invalid {name}={raw!r}; falling back to {default}."
        )
        value = default
    if value < minimum:
        logging.warning(
            f"{name}={value} is below minimum {minimum}; clamping to {minimum}."
        )
        value = minimum
    return value


def _parse_env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        logging.warning(
            f"Invalid {name}={raw!r}; falling back to {default}."
        )
        value = default
    if value < minimum:
        logging.warning(
            f"{name}={value} is below minimum {minimum}; clamping to {minimum}."
        )
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


topk = _parse_env_int("RETRIEVAL_TOPK", 1, minimum=1)
MODEL_CONTEXT_WINDOW_TOKENS = _parse_env_int("MODEL_CONTEXT_WINDOW_TOKENS", 131072, minimum=1024)
PROMPT_CHAR_PER_TOKEN = _parse_env_float("PROMPT_CHAR_PER_TOKEN", 4.0, minimum=1.0)
RETRIEVAL_OUTPUT_TOKEN_RESERVE = _parse_env_int("RETRIEVAL_OUTPUT_TOKEN_RESERVE", 4096, minimum=128)
RETRIEVAL_FIXED_OVERHEAD_CHARS = _parse_env_int("RETRIEVAL_FIXED_OVERHEAD_CHARS", 3000, minimum=0)
RETRIEVAL_PROMPT_MIN_CHARS = _parse_env_int("RETRIEVAL_PROMPT_MIN_CHARS", 512, minimum=64)
CONCURRENT_ENV_BATCH_SIZE = _parse_env_int("CONCURRENT_ENV_BATCH_SIZE", 0, minimum=0)  # 0 = no batching


async def _batched_gather(coros, batch_size: int):
    """Run coroutines in sequential batches to limit vLLM concurrency.

    If batch_size <= 0, falls back to a single asyncio.gather (no batching).
    """
    if batch_size <= 0 or len(coros) <= batch_size:
        return list(await asyncio.gather(*coros))
    results = []
    for i in range(0, len(coros), batch_size):
        results.extend(await asyncio.gather(*coros[i : i + batch_size]))
    return results


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


async def timed_retrieval(query: str, topk: int = 1):
    """Wrapper for get_top_k_memories that tracks timing."""
    from utils import get_top_k_memories
    start_time = time.time()
    result = get_top_k_memories(query, topk=topk)
    elapsed = time.time() - start_time
    return result, elapsed

def build_env(env_name, env_num=1, start_idx=0):
    group_n = 1
    if env_name == "alfworld":
        from agent_system.environments.env_package.alfworld import alfworld_projection
        from agent_system.environments.env_package.alfworld import build_alfworld_envs
        alf_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
        env_kwargs = {
            'eval_dataset': split,
        }
        resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
        if split == "train":
            envs = build_alfworld_envs(alf_config_path, seed=1, env_num=env_num, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        else:
            envs = build_alfworld_envs(alf_config_path, seed=1, env_num=env_num, group_n=group_n, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker, start_idx=start_idx)
        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, 'alfworld/AlfredThorEnv', mem_type=mem_type, topk=topk)
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
        # Tracks how often retrieval/ICL prompts are truncated.
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

    def _parse_ttsft_reasoningbank_memories(self, query_state: str, topk_value: int):
        from utils import get_top_k_memories

        top_k_memories = get_top_k_memories(
            query_state,
            topk=topk_value,
            memory_type="reasoningbank",
        )
        retrieved_exp = "\n\n".join([
            f"Retrieved Strategy {idx}:\n{content}"
            for idx, (content, relevance_score) in enumerate(top_k_memories)
        ])
        return retrieved_exp

    def _build_reasoningbank_prompt(self, prompt: str, query_state: str, topk_value: int):
        retrieved_exp = self._parse_ttsft_reasoningbank_memories(
            query_state,
            topk_value,
        )
        retrieved_exp = self._truncate_retrieval_for_prompt(
            base_prompt=prompt,
            retrieval_text=retrieved_exp,
            output_token_reserve=self.max_tokens,
        )
        return (
            f"{prompt}\n\n"
            "Below are some strategy items that are accumulated from past interactions from the environment that may be helpful to solve the task. "
            "You can use it when you feel it's relevant. In each step, please first explicitly discuss if you want to use each strategy item or not, and then take action.\n\n"
            f"{retrieved_exp}"
        )

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
            
            logging.info(f"API call: {input_tokens} input tokens (${input_cost:.6f}), {output_tokens} output tokens (${output_cost:.6f}), total cost: ${cost:.6f}")

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
        start_time = time.time()
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
            
            result = await self._retry_api_call(_api_call)
            elapsed = time.time() - start_time
            return result, elapsed
        else:  # vllm
            ip_addr = random.choice(self.ip_addrs)
            client = self.clients[ip_addr]  # Reuse existing client
            try:
                resp = await client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": obs}],
                    temperature=self.temperature,
                    extra_body={"min_tokens": 128}, 
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                raise RuntimeError(
                    self._format_vllm_exception(
                        e,
                        f"vLLM action completion on {ip_addr}",
                    )
                ) from e
            # Note: vLLM may not provide usage info, so we skip cost tracking for vLLM
            ret = resp.choices[0].message.content.strip()
            if not ret:
                raise RuntimeError(f"Empty response {resp} from {ip_addr} for prompt: {obs}")
            elapsed = time.time() - start_time
            return ret, elapsed
    
    async def get_strategy_from_gpt(self, obs):
        start_time = time.time()
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
            
            result = await self._retry_api_call(_api_call)
            elapsed = time.time() - start_time
            return result, elapsed
        else:  # vllm
            ip_addr = random.choice(self.strategy_ip_addrs)
            client = self.strategy_clients[ip_addr]  # Reuse existing client
            try:
                resp = await client.chat.completions.create(
                    model=self.strategy_model_name,
                    messages=[{"role": "user", "content": obs}],
                    temperature=self.temperature,
                    extra_body={"min_tokens": 128}, 
                    max_tokens=self.strategy_max_tokens,
                )
            except Exception as e:
                raise RuntimeError(
                    self._format_vllm_exception(
                        e,
                        f"vLLM strategy completion on {ip_addr}",
                    )
                ) from e
            # Note: vLLM may not provide usage info, so we skip cost tracking for vLLM
            ret = resp.choices[0].message.content.strip()
            if not ret:
                raise RuntimeError(f"Empty response {resp} from strategy server {ip_addr} for prompt: {obs}")
            elapsed = time.time() - start_time
            return ret, elapsed
    
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
        initial_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_WITH_OPTIONAL_MEMORY
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
                
                # Stage 2: Generate strategy from retrieved experiences
                strategy_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_STRATEGY_GENERATION.format(
                    k=topk,
                    retrieved_exp=retrieved_exp
                )
                strategy_response = await self.get_strategy_from_gpt(strategy_prompt)
                strategy = extract_strategy_from_response(strategy_response)
                
                if not strategy_response:
                    logging.warning(f"Env {env_idx}: Empty strategy_response after stage 2!")
                
                # Stage 3: Generate action based on strategy (no retrieved trajectories in context)
                action_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_ACTION_FROM_STRATEGY.format(
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
            
            # Stage 2: Generate strategy from retrieved experiences
            strategy_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_STRATEGY_GENERATION.format(
                k=topk,
                retrieved_exp=retrieved_exp
            )
            strategy_response = await self.get_strategy_from_gpt(strategy_prompt)
            strategy = extract_strategy_from_response(strategy_response)
            
            if not strategy_response:
                logging.warning(f"Env {env_idx}: Empty strategy_response after stage 2!")
            
            # Stage 3: Generate action based on strategy (no retrieved trajectories in context)
            action_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_ACTION_FROM_STRATEGY.format(
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
            strategy_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_STRATEGY_NO_RETRIEVAL
            strategy_response = await self.get_strategy_from_gpt(strategy_prompt)
            strategy = extract_strategy_from_response(strategy_response)
            
            if not strategy_response:
                logging.warning(f"Env {env_idx}: Empty strategy_response after stage 1!")
            
            # Stage 2: Generate action based on strategy (no retrieved trajectories in context)
            action_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_ACTION_FROM_STRATEGY.format(
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
            
            # Directly generate action based on retrieved experiences
            action_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_DIRECT_ACTION_FROM_RETRIEVAL.format(
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
            direct_response, direct_time = await self.get_action_from_gpt(prompt)
            direct_action = extract_action_from_response(direct_response)
            retrieval_info = {
                "action_prompt": None,
                "final_response": direct_response,
                "final_action": direct_action,
            }
            timing_info = {
                "retrieval_time": 0.0,
                "strategy_time": 0.0,
                "action_time": direct_time
            }
            return direct_response, False, "", retrieval_info, timing_info
    
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
            (final_action, retrieval_requested, retrieval_reason, retrieval_info, timing_info)
            retrieval_info contains detailed info from stages
            timing_info contains {retrieval_time, strategy_time, action_time}
        """
        # Generate initial action without strategy guidance
        direct_prompt = prompt + "\n\n" + ALFWORLD_ACTION_INSTR
        direct_response, direct_time = await self.get_action_from_gpt(direct_prompt)
        initial_action = extract_action_from_response(direct_response)
        
        try:
            current_strategy = self.active_strategies.get(env_idx, None)
            
            if current_strategy is None:
                # No strategy yet - retrieve, generate strategy, generate action
                from utils import get_top_k_memories, get_top_k_memories_mix

                retrieval_start = time.time()
                if mix_mode:
                    success_mems, failure_mems = get_top_k_memories_mix(current_obs_text, topk=topk)
                    top_k_memories = sorted(success_mems + failure_mems, key=lambda x: x[1], reverse=True)
                else:
                    top_k_memories = get_top_k_memories(current_obs_text, topk=topk)
                retrieval_time = time.time() - retrieval_start
                
                retrieved_exp = "\n\n".join([
                    f"Retrieved Item {idx}:\n{content}" 
                    for idx, (content, relevance_score) in enumerate(top_k_memories)
                ])
                retrieved_exp = self._truncate_retrieval_for_prompt(
                    base_prompt=prompt,
                    retrieval_text=retrieved_exp,
                    output_token_reserve=self.strategy_max_tokens,
                )
                
                strategy_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_STRATEGY_GENERATION.format(
                    k=topk,
                    retrieved_exp=retrieved_exp
                )
                strategy_response, strategy_gen_time = await self.get_strategy_from_gpt(strategy_prompt)
                new_strategy = extract_strategy_from_response(strategy_response)
                
                # Strategy time includes both initial action (counts as thinking) and strategy generation
                strategy_time = direct_time + strategy_gen_time
                
                if not new_strategy:
                    logging.warning(f"Env {env_idx}: Empty strategy_response after regeneration!")
                    new_strategy = "[No strategy generated]"
                
                # Store the new strategy
                self.active_strategies[env_idx] = new_strategy
                
                # Generate action from new strategy
                action_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_ACTION_FROM_STRATEGY.format(
                    strategy=new_strategy
                )
                final_response, action_time = await self.get_action_from_gpt(action_prompt)
                final_action = extract_action_from_response(final_response)
                
                if not final_response:
                    logging.warning(f"Env {env_idx}: Empty final_response after action generation!")
                
                # Package retrieval information
                retrieval_info = {
                    "direct_prompt": direct_prompt,
                    "initial_response": direct_response,
                    "initial_action": initial_action,
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
                
                timing_info = {
                    "retrieval_time": retrieval_time,
                    "strategy_time": strategy_time,
                    "action_time": action_time
                }
                
                return final_response, True, "initial_strategy_generation", retrieval_info, timing_info
            else:
                # Has strategy - combined action generation + refresh decision
                combined_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_STRATEGY_REFRESH_DECISION.format(
                    current_strategy=current_strategy
                )
                combined_response, combined_time = await self.get_action_from_gpt(combined_prompt)
                
                initial_action_from_combined, should_refresh, refresh_reason = parse_action_and_refresh(combined_response)
                
                if not should_refresh:
                    # Reuse strategy - initial action time counts as action generation
                    retrieval_info = {
                        "direct_prompt": direct_prompt,
                        "initial_response": direct_response,
                        "initial_action": initial_action,
                        "refresh_decision": should_refresh,
                        "refresh_reason": refresh_reason,
                        "refresh_response": combined_response,
                        "refresh_prompt": combined_prompt,
                        "strategy_prompt": None,
                        "strategy_response": None,
                        "strategy": current_strategy,
                        "action_prompt": None,
                        "final_response": combined_response,
                        "final_action": initial_action_from_combined,
                    }
                    
                    timing_info = {
                        "retrieval_time": 0.0,
                        "strategy_time": 0.0,
                        "action_time": direct_time  # Only initial action time
                    }
                    
                    return combined_response, False, "strategy_reused", retrieval_info, timing_info
                else:
                    # Refresh strategy - retrieve, generate new strategy, generate new action
                    from utils import get_top_k_memories, get_top_k_memories_mix

                    retrieval_start = time.time()
                    if mix_mode:
                        success_mems, failure_mems = get_top_k_memories_mix(current_obs_text, topk=topk)
                        top_k_memories = success_mems + failure_mems
                    else:
                        top_k_memories = get_top_k_memories(current_obs_text, topk=topk)
                    retrieval_time = time.time() - retrieval_start
                    
                    retrieved_exp = "\n\n".join([
                        f"Retrieved Item {idx}:\n{content}" 
                        for idx, (content, relevance_score) in enumerate(top_k_memories)
                    ])
                    retrieved_exp = self._truncate_retrieval_for_prompt(
                        base_prompt=prompt,
                        retrieval_text=retrieved_exp,
                        output_token_reserve=self.strategy_max_tokens,
                    )
                    
                    strategy_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_STRATEGY_GENERATION.format(
                        k=topk,
                        retrieved_exp=retrieved_exp
                    )
                    strategy_response, strategy_gen_time = await self.get_strategy_from_gpt(strategy_prompt)
                    new_strategy = extract_strategy_from_response(strategy_response)
                    
                    # Strategy time includes initial action + strategy generation
                    strategy_time = direct_time + strategy_gen_time
                    
                    if not new_strategy:
                        logging.warning(f"Env {env_idx}: Empty strategy_response after regeneration!")
                        new_strategy = "[No strategy generated]"
                    
                    # Store the new strategy
                    self.active_strategies[env_idx] = new_strategy
                    
                    # Generate action from new strategy
                    action_prompt = prompt + "\n\n" + ALFWORLD_TEMPLATE_ACTION_FROM_STRATEGY.format(
                        strategy=new_strategy
                    )
                    final_response, action_time = await self.get_action_from_gpt(action_prompt)
                    final_action = extract_action_from_response(final_response)
                    
                    if not final_response:
                        logging.warning(f"Env {env_idx}: Empty final_response after action generation!")
                    
                    # Package retrieval information
                    retrieval_info = {
                        "direct_prompt": direct_prompt,
                        "initial_response": direct_response,
                        "initial_action": initial_action,
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
                    
                    timing_info = {
                        "retrieval_time": retrieval_time,
                        "strategy_time": strategy_time,
                        "action_time": action_time
                    }
                    
                    return final_response, True, "strategy_refreshed", retrieval_info, timing_info
        except Exception as e:
            logging.warning(f"Step strategy reuse failed for env {env_idx}: {e}. Using direct action.")
            retrieval_info = {
                "direct_prompt": direct_prompt,
                "initial_response": direct_response,
                "initial_action": initial_action,
                "error": str(e)
            }
            timing_info = {
                "retrieval_time": 0.0,
                "strategy_time": 0.0,
                "action_time": direct_time
            }
            return direct_response, False, "", retrieval_info, timing_info

async def main():
    # -------- logging ----------
    log_dir = f"logs/alfworld/{MODEL_NAME.replace('/', '_')}"
    os.makedirs(log_dir, exist_ok=True)

    def _add_file_suffix(path: str, suffix: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}_{suffix}{ext}"
    
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
        elif mix_mode:
            traj_file = traj_file.replace(".json", f"_mix.json")
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
    max_steps = int(os.environ.get("MAX_STEPS", 50))
    env_num = SPLIT2ENV_NUM[split] # 200
    test_times = 1000 if split == "train" else int(os.environ.get("TEST_TIMES", 3))
    env_name = "alfworld" 

    # Keywords for 6 subtasks
    TASKS = [
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ]

    # Episode-level batching: run ENV_BATCH_SIZE envs to completion before next batch.
    # Defaults to env_num (no batching) when unset.
    ENV_BATCH_SIZE = int(os.environ.get("ENV_BATCH_SIZE", env_num))
    if ENV_BATCH_SIZE <= 0:
        ENV_BATCH_SIZE = env_num
    num_env_batches = math.ceil(env_num / ENV_BATCH_SIZE)

    def _maybe_restart_vllm_between_batches(batch_idx: int, num_batches: int) -> None:
        """Optionally restart the vLLM server between episode batches.

        Controlled by RESTART_VLLM_BETWEEN_BATCHES=true.
        Only runs between batches (not before the first or after the last).
        """
        if batch_idx == 0 or not (os.environ.get("RESTART_VLLM_BETWEEN_BATCHES", "false").lower() == "true"):
            return
        script_path = os.environ.get(
            "RESTART_VLLM_SCRIPT",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts/restart_vllm_server.sh")),
        )
        openai_base_ip_addr = os.environ.get("OPENAI_BASE_IP_ADDR", "").strip()
        cuda_device = os.environ.get("CUDA_DEVICE", "").strip()
        gpu_util = os.environ.get("GPU_UTIL", "").strip()
        model_name_env = os.environ.get("MODEL_NAME", "").strip()
        missing = [n for n, v in [("OPENAI_BASE_IP_ADDR", openai_base_ip_addr), ("CUDA_DEVICE", cuda_device), ("GPU_UTIL", gpu_util), ("MODEL_NAME", model_name_env)] if not v]
        if missing:
            logging.warning("Skip vLLM restart between batches: missing env vars %s", missing)
            return
        logging.info(f"Restarting vLLM between batches (batch {batch_idx}/{num_batches - 1})")
        try:
            result = subprocess.run(
                [script_path, openai_base_ip_addr, cuda_device, gpu_util, model_name_env],
                check=True, text=True, capture_output=True,
            )
            if result.stdout:
                logging.info(f"vLLM restart stdout:\n{result.stdout}")
            if result.stderr:
                logging.info(f"vLLM restart stderr:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"vLLM restart failed: rc={e.returncode} stdout={e.stdout} stderr={e.stderr}")
            raise
        wait_sec = float(os.environ.get("RESTART_VLLM_WAIT_SEC", "5"))
        if wait_sec > 0:
            logging.info(f"Waiting {wait_sec:.1f}s after vLLM restart")
            time.sleep(wait_sec)

    # -------- Agent setup ----------
    agent = Agent()

    # Accumulated statistics
    overall_success_rates = []         # Overall success per round
    task_success_history = defaultdict(list)  # Subtask success per round

    # Retrieval statistics tracking
    if mem_type in MEM_TYPES_WITH_RETRIEVAL_STATS:
        retrieval_stats = {
            'total_steps': 0,
            'total_requests': 0,
            'total_action_changes': 0,  # Track how many times action changed after retrieval
            'requests_per_round': [],
            'action_changes_per_round': [],
            'steps_per_trajectory': [],      # Track steps taken in each trajectory
            'retrievals_per_trajectory': [],  # Track retrievals in each trajectory
            'action_changes_per_trajectory': [],  # Track action changes in each trajectory
        }

    traj_items = []
    if mem_type:
        logging.info(f"Saving to {traj_file}")
    else:
        logging.info(f"Saving to {traj_file}")

    # ======================= Main Loop =======================
    for test_idx in range(test_times):
        logging.info(f"\n========== Start test {test_idx} ==========")
        start_time = time.time()

        # Round-level accumulators (aggregated across all batches)
        overall_success_this_round = np.zeros(env_num, dtype=bool)
        task_success_cnt = defaultdict(int)
        task_total_cnt = defaultdict(int)
        round_requests = 0
        round_action_changes = 0
        round_trajs = []  # accumulates batch_trajs from every batch

        # ======================= Batch Loop =======================
        for batch_idx in range(num_env_batches):
            batch_start = batch_idx * ENV_BATCH_SIZE
            batch_end = min(batch_start + ENV_BATCH_SIZE, env_num)
            current_batch_size = batch_end - batch_start

            if num_env_batches > 1:
                logging.info(
                    f"\n--- Batch {batch_idx + 1}/{num_env_batches}: "
                    f"envs [{batch_start}:{batch_end}] ({current_batch_size} envs) ---"
                )
                _maybe_restart_vllm_between_batches(batch_idx, num_env_batches)

            env_manager = build_env(
                env_name, current_batch_size,
                start_idx=(batch_start if split != "train" else 0),
            )
            kwargs = {}
            obs, infos = env_manager.reset(kwargs)
            env_dones = [False] * current_batch_size

            # Reset agent state for this batch
            agent.active_strategies = {}

            if mem_type in MEM_TYPES_WITH_RETRIEVAL_STATS:
                trajectory_steps = [0] * current_batch_size
                trajectory_retrievals = [0] * current_batch_size
                trajectory_action_changes = [0] * current_batch_size

            batch_trajs = [{"won": None, "steps": [], "timing_per_step": [], "round_idx": test_idx} for i in range(current_batch_size)]

            for step_idx in range(max_steps):
                logging.info(f"Step {step_idx}; Dones ({np.array(env_dones).sum().item()}/{current_batch_size}); SR {overall_success_this_round.mean().item()}")

                # --- Assemble actions with optional memory retrieval ---
                actions = ["None"] * current_batch_size
                retrieval_requests = [False] * current_batch_size
                retrieval_reasons = [""] * current_batch_size
                retrieval_infos = [None] * current_batch_size
    
                # Track timing for each environment
                step_timings = [{
                    "retrieval_time": 0.0,
                    "strategy_time": 0.0,
                    "action_time": 0.0
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
                                batch_start + idx,
                                current_obs_text
                            )
                            for prompt, idx, current_obs_text
                            in zip(prompts, active_indices, current_obs_texts)
                        ]
                        results = await _batched_gather(tasks, CONCURRENT_ENV_BATCH_SIZE)
                        
                        for idx, (action, requested, reason, retrieval_info) in zip(active_indices, results):
                            actions[idx] = action
                            retrieval_requests[idx] = requested
                            retrieval_reasons[idx] = reason
                            retrieval_infos[idx] = retrieval_info
                        
                        # Track retrieval statistics
                        num_requested = sum(retrieval_requests)
                        round_requests += num_requested
                        retrieval_stats['total_steps'] += len(active_indices)
                        retrieval_stats['total_requests'] += num_requested
                        
                        # Track action changes
                        num_action_changes = sum(1 for idx in active_indices 
                                                if retrieval_infos[idx] and retrieval_infos[idx].get('action_changed', False))
                        round_action_changes += num_action_changes
                        retrieval_stats['total_action_changes'] += num_action_changes
                        
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
                                batch_start + idx,
                                current_obs_text
                            )
                            for prompt, idx, current_obs_text
                            in zip(prompts, active_indices, current_obs_texts)
                        ]
                        results = await _batched_gather(tasks, CONCURRENT_ENV_BATCH_SIZE)
                        
                        for idx, (action, requested, reason, retrieval_info) in zip(active_indices, results):
                            actions[idx] = action
                            retrieval_requests[idx] = requested
                            retrieval_reasons[idx] = reason
                            retrieval_infos[idx] = retrieval_info
                        
                        # Track retrieval statistics
                        num_requested = sum(retrieval_requests)
                        round_requests += num_requested
                        retrieval_stats['total_steps'] += len(active_indices)
                        retrieval_stats['total_requests'] += num_requested
                        
                        # For every_step, always retrieve, no action_changes since no initial
                        round_action_changes += 0
                        retrieval_stats['total_action_changes'] += 0
                        
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
                                batch_start + idx,
                                current_obs_text
                            )
                            for prompt, idx, current_obs_text
                            in zip(prompts, active_indices, current_obs_texts)
                        ]
                        results = await _batched_gather(tasks, CONCURRENT_ENV_BATCH_SIZE)
                        
                        for idx, (action, requested, reason, retrieval_info) in zip(active_indices, results):
                            actions[idx] = action
                            retrieval_requests[idx] = requested
                            retrieval_reasons[idx] = reason
                            retrieval_infos[idx] = retrieval_info
                        
                        # Track retrieval statistics (no retrieval, so all zero)
                        num_requested = 0
                        round_requests += num_requested
                        retrieval_stats['total_steps'] += len(active_indices)
                        retrieval_stats['total_requests'] += num_requested
                        
                        # No action changes since no initial action
                        round_action_changes += 0
                        retrieval_stats['total_action_changes'] += 0
                        
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
                                batch_start + idx,
                                current_obs_text
                            )
                            for prompt, idx, current_obs_text
                            in zip(prompts, active_indices, current_obs_texts)
                        ]
                        results = await _batched_gather(tasks, CONCURRENT_ENV_BATCH_SIZE)
                        
                        for idx, (action, requested, reason, retrieval_info) in zip(active_indices, results):
                            actions[idx] = action
                            retrieval_requests[idx] = requested
                            retrieval_reasons[idx] = reason
                            retrieval_infos[idx] = retrieval_info
                        
                        # Track retrieval statistics
                        num_requested = sum(retrieval_requests)
                        round_requests += num_requested
                        retrieval_stats['total_steps'] += len(active_indices)
                        retrieval_stats['total_requests'] += num_requested
                        
                        # For direct retrieval every step, always retrieve, no action_changes
                        round_action_changes += 0
                        retrieval_stats['total_action_changes'] += 0
                        
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
                                batch_start + idx,
                                current_obs_text,
                                recent_history,
                            )
                            for prompt, idx, current_obs_text, recent_history
                            in zip(prompts, active_indices, current_obs_texts, recent_histories)
                        ]
                        results = await _batched_gather(tasks, CONCURRENT_ENV_BATCH_SIZE)
                        
                        for active_env_idx, (action, requested, reason, retrieval_info, timing_info) in enumerate(results):
                            idx = active_indices[active_env_idx]
                            actions[idx] = action
                            retrieval_requests[idx] = requested
                            retrieval_reasons[idx] = reason
                            retrieval_infos[idx] = retrieval_info
                            step_timings[idx] = timing_info
                        
                        # Track retrieval statistics
                        num_requested = sum(retrieval_requests)
                        round_requests += num_requested
                        retrieval_stats['total_steps'] += len(active_indices)
                        retrieval_stats['total_requests'] += num_requested
                        
                        # For reuse, refreshes are retrievals, no action_changes
                        round_action_changes += 0
                        retrieval_stats['total_action_changes'] += 0
                        
                        # Update per-trajectory counters
                        for idx in active_indices:
                            trajectory_steps[idx] += 1
                            if retrieval_requests[idx]:
                                trajectory_retrievals[idx] += 1
                            # No action_changes for reuse
                        
                        logging.info(f"  Memory retrieval (strategy reuse): {num_requested}/{len(active_indices)} refreshed")
                    
                    else:
                        # Original logic for other memory types (no memory or simple retrieval)
                        if mem_type in ["reasoningbank", "synapse"]:
                            # These types have retrieval built into the prompt via env_manager
                            # The retrieval time is negligible as it's done during prompt construction
                            tasks = [agent.get_action_from_gpt(prompt) for prompt in prompts]
                            results = await _batched_gather(tasks, CONCURRENT_ENV_BATCH_SIZE)
                            for active_env_idx, (action, action_time) in enumerate(results):
                                idx = active_indices[active_env_idx]
                                actions[idx] = action
                                step_timings[idx]["action_time"] = action_time
                                # Retrieval is embedded in prompt, so we don't have separate timing
                                # For these methods, retrieval happens in env_manager.build_text_obs
                        else:
                            # No memory at all
                            tasks = [agent.get_action_from_gpt(prompt) for prompt in prompts]
                            results = await _batched_gather(tasks, CONCURRENT_ENV_BATCH_SIZE)
                            for active_env_idx, (action, action_time) in enumerate(results):
                                idx = active_indices[active_env_idx]
                                actions[idx] = action
                                step_timings[idx]["action_time"] = action_time
    
                for i in range(current_batch_size):
                    if env_dones[i]:
                        curr_prompt = "None"
                        curr_action = "None"
                        step_item = {
                            "step_idx": step_idx,
                            "env_num": i,
                            "curr_prompt": curr_prompt,
                            "curr_action": curr_action,
                        }
                    else:
                        curr_prompt = obs["text"][i]
                        curr_action = actions[i]
                        step_item = {
                            "step_idx": step_idx,
                            "env_num": i,
                            "curr_prompt": curr_prompt,
                            "curr_action": curr_action,
                        }
                        
                        # Add retrieval information for AdaMEM variants
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
                                step_item["direct_prompt"] = retrieval_infos[i]["direct_prompt"]
                                step_item["initial_response"] = retrieval_infos[i]["initial_response"]
                                step_item["initial_action"] = retrieval_infos[i]["initial_action"]
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
                    
                    batch_trajs[i]["steps"].append(step_item)
    
                    # Add timing information for this step
                    batch_trajs[i]["timing_per_step"].append(step_timings[i])
    
                # --- Environment stepping ---
                obs, rewards, dones, infos = env_manager.step(actions)
    
                # --- Determine endings and successes ---
                for i in range(current_batch_size):
                    if env_dones[i]:
                        continue
    
                    if dones[i]:
                        env_dones[i] = True
                        won = bool(infos[i].get("won", False))
                        overall_success_this_round[batch_start + i] = won
                        batch_trajs[i]["won"] = won
    
                        
                        # Save per-trajectory statistics
                        if mem_type in MEM_TYPES_WITH_RETRIEVAL_STATS:
                            retrieval_stats['steps_per_trajectory'].append(trajectory_steps[i])
                            retrieval_stats['retrievals_per_trajectory'].append(trajectory_retrievals[i])
                            retrieval_stats['action_changes_per_trajectory'].append(trajectory_action_changes[i])
    
                        # Parse task type
                        gamefile = infos[i].get("extra.gamefile", "")
                        matched = False
                        for task in TASKS:
                            if task in gamefile:
                                task_total_cnt[task] += 1
                                if won:
                                    task_success_cnt[task] += 1
                                matched = True
                                break
                        if not matched:
                            # Unrecognized tasks are also counted in total
                            task_total_cnt["other"] += 1
                            if won:
                                task_success_cnt["other"] += 1
    
                if all(env_dones):
                    logging.info("All environments finished early!")
                    break

            # Accumulate this batch's trajectories into the round collection.
            round_trajs.extend(batch_trajs)

            # Shut down this batch's Ray workers before starting the next batch.
            env_manager.envs.close()
            del env_manager

        # -------- Single round results --------
        round_success_rate = overall_success_this_round.mean()
        overall_success_rates.append(round_success_rate)

        # Track retrieval requests for this round
        if mem_type in MEM_TYPES_WITH_RETRIEVAL_STATS:
            retrieval_stats['requests_per_round'].append(round_requests)
            retrieval_stats['action_changes_per_round'].append(round_action_changes)

        traj_items.extend(round_trajs)

        with open(traj_file, 'w') as writer:
            json.dump(traj_items, writer, indent=2)
        logging.info(f"Trajectories saved to {traj_file} after test {test_idx}")

        logging.info(f"Test {test_idx} overall success: {round_success_rate:.4f}")

        for task in TASKS + ["other"]:
            if task_total_cnt.get(task, 0) > 0:
                rate = task_success_cnt[task] / task_total_cnt[task]
                task_success_history[task].append(rate)
                logging.info(
                    f"    {task:<35s}: {rate:.4f} "
                    f"({task_success_cnt[task]}/{task_total_cnt[task]})"
                )

        logging.info(
            f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n"
        )

    # ======================= Final Summary =======================
    logging.info("=============== Final Summary ===============")
    logging.info(
        f"Total tests: {test_times} | Envs / test: {env_num} | Total envs: {env_num * test_times}"
    )
    logging.info(
        f"Overall success avg ± std: "
        f"{100*np.mean(overall_success_rates):.1f} ± {100*np.std(overall_success_rates, ddof=1):.1f}"
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
            f"Average cost per test: ${cost_summary['total_cost'] / max(test_times, 1):.4f}"
        )

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

    # Log retrieval statistics
    if mem_type in MEM_TYPES_WITH_RETRIEVAL_STATS:
        retrieval_rate = retrieval_stats['total_requests'] / max(retrieval_stats['total_steps'], 1)
        action_change_rate = retrieval_stats['total_action_changes'] / max(retrieval_stats['total_requests'], 1)

        if mem_type == "adamem-low":
            reuse_rate = 1 - retrieval_rate
            logging.info(
                f"Strategy refresh rate: {retrieval_rate:.4f} "
                f"({retrieval_stats['total_requests']}/{retrieval_stats['total_steps']})"
            )
            logging.info(
                f"Strategy reuse rate: {reuse_rate:.4f} "
                f"({retrieval_stats['total_steps'] - retrieval_stats['total_requests']}/{retrieval_stats['total_steps']})"
            )
        else:
            logging.info(
                f"Memory retrieval rate: {retrieval_rate:.4f} "
                f"({retrieval_stats['total_requests']}/{retrieval_stats['total_steps']})"
            )
        
        if mem_type != "adamem-low":
            logging.info(
                f"Action change rate (when memory retrieved): {action_change_rate:.4f} "
                f"({retrieval_stats['total_action_changes']}/{retrieval_stats['total_requests']})"
            )
        
        if retrieval_stats['requests_per_round']:
            logging.info(
                f"Requests per round: avg={np.mean(retrieval_stats['requests_per_round']):.2f} ± "
                f"{np.std(retrieval_stats['requests_per_round'], ddof=1):.2f}"
            )
        
        if retrieval_stats['action_changes_per_round']:
            logging.info(
                f"Action changes per round: avg={np.mean(retrieval_stats['action_changes_per_round']):.2f} ± "
                f"{np.std(retrieval_stats['action_changes_per_round'], ddof=1):.2f}"
            )
        
        # Log trajectory-level statistics
        if retrieval_stats['steps_per_trajectory']:
            avg_steps = np.mean(retrieval_stats['steps_per_trajectory'])
            std_steps = np.std(retrieval_stats['steps_per_trajectory'], ddof=1)
            logging.info(
                f"Steps per trajectory: avg={avg_steps:.2f} ± {std_steps:.2f}"
            )
        
        if retrieval_stats['retrievals_per_trajectory']:
            avg_retrievals = np.mean(retrieval_stats['retrievals_per_trajectory'])
            std_retrievals = np.std(retrieval_stats['retrievals_per_trajectory'], ddof=1)
            retrieval_rate_per_traj = avg_retrievals / max(avg_steps, 1)
            logging.info(
                f"Retrievals per trajectory: avg={avg_retrievals:.2f} ± {std_retrievals:.2f} "
                f"(rate={retrieval_rate_per_traj:.4f})"
            )
        
        if retrieval_stats['action_changes_per_trajectory']:
            avg_action_changes = np.mean(retrieval_stats['action_changes_per_trajectory'])
            std_action_changes = np.std(retrieval_stats['action_changes_per_trajectory'], ddof=1)
            action_change_rate_per_traj = avg_action_changes / max(avg_retrievals, 1)
            logging.info(
                f"Action changes per trajectory: avg={avg_action_changes:.1f} ± {std_action_changes:.1f} "
                f"(rate when retrieved={action_change_rate_per_traj:.4f})"
            )

    # ======================= Timing Statistics =======================
    logging.info("=============== Timing Statistics ===============")
    
    # Calculate per-instance timing statistics
    timing_stats = {
        "per_instance": [],  # List of {total, retrieval, strategy, action} for each instance
        "per_instance_step_count": [],  # Number of steps for each instance
        "per_round": []  # Per-round averages
    }
    
    for traj in traj_items:
        if traj.get("timing_per_step"):
            instance_total_time = 0.0
            instance_retrieval_time = 0.0
            instance_strategy_time = 0.0
            instance_action_time = 0.0
            instance_steps = len(traj["timing_per_step"])
            
            for step_timing in traj["timing_per_step"]:
                instance_retrieval_time += step_timing.get("retrieval_time", 0.0)
                instance_strategy_time += step_timing.get("strategy_time", 0.0)
                instance_action_time += step_timing.get("action_time", 0.0)
            
            instance_total_time = instance_retrieval_time + instance_strategy_time + instance_action_time
            
            timing_stats["per_instance"].append({
                "total": instance_total_time,
                "retrieval": instance_retrieval_time,
                "strategy": instance_strategy_time,
                "action": instance_action_time,
                "steps": instance_steps,
                "round_idx": traj.get("round_idx", 0)
            })
            timing_stats["per_instance_step_count"].append(instance_steps)
    
    # Calculate per-round statistics
    if timing_stats["per_instance"]:
        # Group instances by round
        round_groups = {}
        for inst in timing_stats["per_instance"]:
            round_idx = inst["round_idx"]
            if round_idx not in round_groups:
                round_groups[round_idx] = []
            round_groups[round_idx].append(inst)
        
        # Calculate averages for each round
        for round_idx in sorted(round_groups.keys()):
            round_instances = round_groups[round_idx]
            round_avg = {
                "round_idx": round_idx,
                "total": np.mean([inst["total"] for inst in round_instances]),
                "retrieval": np.mean([inst["retrieval"] for inst in round_instances]),
                "strategy": np.mean([inst["strategy"] for inst in round_instances]),
                "action": np.mean([inst["action"] for inst in round_instances]),
                "steps": np.mean([inst["steps"] for inst in round_instances]),
                "num_instances": len(round_instances)
            }
            timing_stats["per_round"].append(round_avg)
    
    # Calculate aggregate statistics
    if timing_stats["per_instance"]:
        total_times = [inst["total"] for inst in timing_stats["per_instance"]]
        retrieval_times = [inst["retrieval"] for inst in timing_stats["per_instance"]]
        strategy_times = [inst["strategy"] for inst in timing_stats["per_instance"]]
        action_times = [inst["action"] for inst in timing_stats["per_instance"]]
        
        # Overall statistics (all instances)
        logging.info(f"Total instances: {len(timing_stats['per_instance'])}")
        logging.info(
            f"Average total time per instance (all): {np.mean(total_times):.2f}s ± {np.std(total_times, ddof=1):.2f}s"
        )
        logging.info(
            f"Average retrieval time per instance (all): {np.mean(retrieval_times):.2f}s ± {np.std(retrieval_times, ddof=1):.2f}s "
            f"({np.mean(retrieval_times) / max(np.mean(total_times), 0.001) * 100:.1f}% of total)"
        )
        logging.info(
            f"Average strategy time per instance (all): {np.mean(strategy_times):.2f}s ± {np.std(strategy_times, ddof=1):.2f}s "
            f"({np.mean(strategy_times) / max(np.mean(total_times), 0.001) * 100:.1f}% of total)"
        )
        logging.info(
            f"Average action time per instance (all): {np.mean(action_times):.2f}s ± {np.std(action_times, ddof=1):.2f}s "
            f"({np.mean(action_times) / max(np.mean(total_times), 0.001) * 100:.1f}% of total)"
        )
        
        # Per-round statistics (average of round averages)
        if timing_stats["per_round"] and len(timing_stats["per_round"]) > 1:
            logging.info("\n--- Per-Round Averages ---")
            for round_stat in timing_stats["per_round"]:
                logging.info(
                    f"Round {round_stat['round_idx']}: "
                    f"total={round_stat['total']:.2f}s, "
                    f"retrieval={round_stat['retrieval']:.2f}s, "
                    f"strategy={round_stat['strategy']:.2f}s, "
                    f"action={round_stat['action']:.2f}s "
                    f"({round_stat['num_instances']} instances)"
                )
            
            # Average of round averages
            round_total_avg = np.mean([r["total"] for r in timing_stats["per_round"]])
            round_total_std = np.std([r["total"] for r in timing_stats["per_round"]], ddof=1) if len(timing_stats["per_round"]) > 1 else 0.0
            round_retrieval_avg = np.mean([r["retrieval"] for r in timing_stats["per_round"]])
            round_retrieval_std = np.std([r["retrieval"] for r in timing_stats["per_round"]], ddof=1) if len(timing_stats["per_round"]) > 1 else 0.0
            round_strategy_avg = np.mean([r["strategy"] for r in timing_stats["per_round"]])
            round_strategy_std = np.std([r["strategy"] for r in timing_stats["per_round"]], ddof=1) if len(timing_stats["per_round"]) > 1 else 0.0
            round_action_avg = np.mean([r["action"] for r in timing_stats["per_round"]])
            round_action_std = np.std([r["action"] for r in timing_stats["per_round"]], ddof=1) if len(timing_stats["per_round"]) > 1 else 0.0
            
            logging.info("\n--- Average of Round Averages ---")
            logging.info(
                f"Average total time per instance (round avg): {round_total_avg:.2f}s ± {round_total_std:.2f}s"
            )
            logging.info(
                f"Average retrieval time per instance (round avg): {round_retrieval_avg:.2f}s ± {round_retrieval_std:.2f}s "
                f"({round_retrieval_avg / max(round_total_avg, 0.001) * 100:.1f}% of total)"
            )
            logging.info(
                f"Average strategy time per instance (round avg): {round_strategy_avg:.2f}s ± {round_strategy_std:.2f}s "
                f"({round_strategy_avg / max(round_total_avg, 0.001) * 100:.1f}% of total)"
            )
            logging.info(
                f"Average action time per instance (round avg): {round_action_avg:.2f}s ± {round_action_std:.2f}s "
                f"({round_action_avg / max(round_total_avg, 0.001) * 100:.1f}% of total)"
            )
        
        # Per-step timing
        avg_steps = np.mean(timing_stats["per_instance_step_count"])
        logging.info(
            f"\nAverage time per step: {np.mean(total_times) / max(avg_steps, 1):.3f}s"
        )
        
        # Save detailed timing stats to a separate JSON file
        timing_file = traj_file.replace(".json", "_timing.json")
        with open(timing_file, 'w') as f:
            json.dump(timing_stats, f, indent=2)
        logging.info(f"Detailed timing statistics saved to {timing_file}")

    for task in TASKS + ["other"]:
        if task_success_history.get(task):
            logging.info(
                f"{task:<35s}: "
                f"{np.mean(task_success_history[task]):.4f} ± "
                f"{np.std(task_success_history[task], ddof=1):.4f}"
            )

if __name__ == "__main__":
    asyncio.run(main())
