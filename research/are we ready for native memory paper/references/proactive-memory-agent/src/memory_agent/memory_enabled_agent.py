"""Memory-enabled Terminus2 agent — V3 Multi-Turn Tool Use Architecture.

Design:
1. Two-phase design (Phase 1 bank + Phase 2 context decision)
2. with_strategy prompts (proven effective)
3. Designed for RL: both phases are separate forward passes in same conversation
4. Sliding window: memory agent sees task description + last N steps
5. System prompt: <memory_context> injected only when Phase 2 decides intervention
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2, Command
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext

from .memory import (
    UniversalMemory,
    MemoryAgent,
    MemoryAgentResult,
    MemoryAgentTrigger,
    TriggerConfig,
)

logger = logging.getLogger(__name__)


class MemoryEnabledTerminus2(Terminus2):
    """Terminus2 with V3 single-phase memory system.

    The memory agent:
    - Maintains a lean knowledge bank (every turn)
    - Decides whether to inject context via Phase 2 text generation (default: no-op)
    - When injecting, wraps context in <memory_context> system prompt
    - All done in a SINGLE LLM call per step
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str,
        memory_model_name: str = "openrouter/anthropic/claude-opus-4.6",
        memory_api_base: str | None = None,
        memory_api_key: str | None = None,
        memory_temperature: float = 0.3,
        memory_max_thinking_tokens: int | None = None,
        memory_keep_thinking: bool = False,
        memory_use_structured_tools: bool = True,
        trigger_interval: int = 1,
        trigger_on_first_step: bool = True,
        sliding_window_size: int = 8,
        reset_status_on_episode: bool = True,
        reset_knowledge_on_episode: bool = False,
        reset_procedural_on_episode: bool = False,
        injection_method: str = "system_prompt",
        truncation_pairs: int = 0,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, truncation_pairs=truncation_pairs, **kwargs)

        # Memory components
        self._memory = UniversalMemory()
        self._memory_llm = LiteLLM(
            model_name=memory_model_name,
            api_base=memory_api_base,
            api_key=memory_api_key,
            temperature=memory_temperature,
            max_thinking_tokens=memory_max_thinking_tokens,
        )
        self._memory_agent = MemoryAgent(llm=self._memory_llm, memory=self._memory, keep_thinking=memory_keep_thinking, use_structured_tools=memory_use_structured_tools)
        self._injection_method = injection_method

        # Snapshot support (set by runner if enroot.save_snapshots=true)
        self._environment = None
        self._snapshot_dir: Path | None = None

        # Reset config
        self._reset_status_on_episode = reset_status_on_episode
        self._reset_knowledge_on_episode = reset_knowledge_on_episode
        self._reset_procedural_on_episode = reset_procedural_on_episode

        # Trigger — every turn by default
        self._trigger = MemoryAgentTrigger(
            config=TriggerConfig(
                interval=trigger_interval,
                trigger_on_first_step=trigger_on_first_step,
            )
        )

        # Sliding window size for memory agent context
        self._sliding_window_size = sliding_window_size

        # Task description (for sliding window grounding)
        self._task_description: str = ""

        # Phase 2: Pending context injection — None means no intervention
        self._pending_injection: str | None = None
        self._injection_used: bool = False

        # Accumulated trajectory steps (NEVER cleared — sliding window)
        # Each entry: {step, observation, analysis, plan, commands}
        self._accumulated_observations: list[dict] = []

        # Tracking
        self._memory_trigger_count: int = 0
        self._memory_operation_count: int = 0
        self._injection_count: int = 0
        self._noop_count: int = 0
        self._memory_step_in_episode: int = 0

        # Memory trajectory log
        self._memory_trajectory_steps: list[dict] = []

    @property
    def memory(self) -> UniversalMemory:
        return self._memory

    # -----------------------------------------------------------------
    # Trajectory accumulation & sliding window
    # -----------------------------------------------------------------

    def _accumulate_step(
        self,
        step: int,
        observation: str,
        analysis: str | None = None,
        plan: str | None = None,
        commands: list | None = None,
    ) -> None:
        """Accumulate a full step: what the agent thought + what happened."""
        self._accumulated_observations.append({
            "step": step,
            "observation": observation,
            "analysis": analysis,
            "plan": plan,
            "commands": [str(c) for c in commands] if commands else None,
        })

    def _get_memory_agent_context(self) -> str:
        """Build context for the memory agent: task description + sliding window.

        The memory agent acts as selective attention — it always sees:
        1. The original task description (for grounding in requirements)
        2. A sliding window of the last N steps (recent behavior)
        The memory BANK provides long-term context beyond the window.
        """
        parts = []

        # Always include task description
        if self._task_description:
            parts.append(f"[Task Description]\n{self._task_description}")

        # Sliding window of recent steps (skip first entry which is task description,
        # already included above via [Task Description] section)
        window = self._accumulated_observations[1:][-self._sliding_window_size:]
        if window:
            parts.append("[Recent Trajectory (last {} steps)]".format(len(window)))
            for entry in window:
                parts.append(self._format_step_entry(entry))

        return "\n\n".join(parts)

    @staticmethod
    def _format_step_entry(entry: dict) -> str:
        step = entry["step"]
        sections = [f"[Step {step}]"]

        if entry.get("analysis"):
            sections.append(f"Agent Analysis: {entry['analysis']}")

        if entry.get("plan"):
            sections.append(f"Agent Plan: {entry['plan']}")

        if entry.get("commands"):
            cmds = entry["commands"][:5]
            cmd_str = "; ".join(cmds)
            if len(entry["commands"]) > 5:
                cmd_str += f" (+{len(entry['commands']) - 5} more)"
            sections.append(f"Commands Executed: {cmd_str}")

        obs = entry.get("observation", "")
        # observation is already truncated to ~10KB by Terminus2._limit_output_length
        # Pass it through unchanged so memory agent sees exactly what action agent saw
        sections.append(f"Terminal Output: {obs}")

        return "\n".join(sections)

    # -----------------------------------------------------------------
    # Memory agent trigger
    # -----------------------------------------------------------------

    async def _trigger_memory_agent(self, step_count: int) -> None:
        """Run the memory agent (single phase) with sliding window context."""
        self._memory_trigger_count += 1
        trigger_step = self._memory_step_in_episode

        # Build context: task description + sliding window
        combined_trajectory = self._get_memory_agent_context()

        try:
            result: MemoryAgentResult = await self._memory_agent.process(
                observation=combined_trajectory,
                step_count=step_count,
            )

            self._memory_operation_count += len(result.operations)

            # Handle context decision
            if result.should_inject and result.context_for_action:
                self._pending_injection = result.context_for_action
                self._injection_used = False
                self._injection_count += 1
                logger.info(
                    f"[Step {step_count}] INJECT: "
                    f"{result.context_for_action[:100]}..."
                )
            else:
                self._noop_count += 1
                logger.debug(
                    f"[Step {step_count}] NO-OP"
                )

            # Record in trajectory
            self._record_memory_step(
                trigger_step=trigger_step,
                step_count=step_count,
                observation=combined_trajectory,
                result=result,
            )

            # Save container snapshot if enabled
            if self._snapshot_dir and self._environment:
                await self._save_container_snapshot(step_count)

        except Exception as e:
            logger.error(f"Memory agent error at step {step_count}: {e}", exc_info=True)
            self._record_memory_step(
                trigger_step=trigger_step,
                step_count=step_count,
                observation=combined_trajectory,
                result=None,
                error=str(e),
            )

    # -----------------------------------------------------------------
    # Injection mechanism
    # -----------------------------------------------------------------

    def _get_memory_system_prompt(self) -> str | None:
        """Return memory system prompt when Phase 2 decided to inject.

        Returns None when no intervention is pending.
        After context is used once, it is cleared (one-shot).
        """
        if self._pending_injection and not self._injection_used:
            self._injection_used = True
            prompt = (
                "<memory_context>\n"
                "The following context from memory may be relevant to your current work "
                "(these are observations, not directives \u2014 verify before acting):\n"
                f"{self._pending_injection}\n"
                "</memory_context>"
            )
            # Clear after use
            self._pending_injection = None
            self._injection_used = False
            return prompt
        return None

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # Override agent loop (truncation inherited from Terminus2)
    # -----------------------------------------------------------------

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat,
        logging_dir: Path | None = None,
        original_instruction: str = "",
    ) -> int:
        """Override agent loop to integrate V3 memory system."""
        if self._context is None:
            raise RuntimeError("Agent context is not set.")
        if self._session is None:
            raise RuntimeError("Session is not set.")

        # Patch chat.chat to inject memory context (when available)
        _original_chat_fn = chat.chat

        async def _memory_aware_chat(prompt, logging_path=None, **kwargs):
            memory_context = self._get_memory_system_prompt()
            if memory_context:
                if self._injection_method == "user_turn":
                    # Append memory context to the user prompt
                    prompt = prompt + "\n\n" + memory_context
                else:
                    # Inject as system prompt (default)
                    if "system" not in kwargs:
                        kwargs["system"] = memory_context
            return await _original_chat_fn(prompt, logging_path=logging_path, **kwargs)

        chat.chat = _memory_aware_chat

        # Reset memory state for new episode
        self._memory_step_in_episode = 0
        self._trigger.reset()
        self._accumulated_observations.clear()
        self._pending_injection = None
        self._injection_used = False
        self._task_description = ""

        # Optionally reset memory based on config
        if self._reset_status_on_episode:
            self._memory.update_status("")
        if self._reset_knowledge_on_episode:
            self._memory.knowledge.clear()
        if self._reset_procedural_on_episode:
            self._memory.procedural.clear()

        prompt = initial_prompt

        # Store task description for sliding window (always available)
        self._task_description = prompt

        # Accumulate step 1 and trigger memory agent (bank management)
        self._accumulate_step(step=1, observation=prompt)
        should_trigger, reason = self._trigger.should_trigger(step_count=1)
        if should_trigger:
            await self._trigger_memory_agent(step_count=1)

        for episode in range(self._max_episodes):
            self._n_episodes = episode + 1
            self._memory_step_in_episode = episode + 1

            if not await self._session.is_session_alive():
                logger.debug("Session has ended")
                return episode + 1

            # Check proactive summarization
            if original_instruction and self._enable_summarize:
                proactive_summary_result = await self._check_proactive_summarization(
                    chat, original_instruction, self._session
                )
                if proactive_summary_result:
                    handoff_prompt, subagent_refs = proactive_summary_result
                    self._pending_subagent_refs = subagent_refs
                    self._pending_handoff_prompt = handoff_prompt
                    prompt = handoff_prompt

            logging_paths = self._setup_episode_logging(logging_dir, episode)

            # Capture injection BEFORE chat call (chat consumes & clears it)
            step_injection = self._pending_injection

            # Track tokens
            tokens_before_input = chat.total_input_tokens
            tokens_before_output = chat.total_output_tokens
            tokens_before_cache = chat.total_cache_tokens
            cost_before = chat.total_cost

            # Handle LLM interaction
            try:
                (
                    commands,
                    is_task_complete,
                    feedback,
                    analysis,
                    plan,
                    llm_response,
                ) = await self._handle_llm_interaction(
                    chat, prompt, logging_paths, original_instruction, self._session
                )
            except ContextLengthExceededError:
                # Truncation already attempted by Terminus2._query_llm if configured
                logger.warning(f"[Step {episode + 1}] Context length exceeded, terminating task.")
                from harbor.models.trajectories import Step as TrajStep
                self._trajectory_steps.append(
                    TrajStep(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="CONTEXT_LENGTH_EXCEEDED: Task terminated because the action agent's context window was full.",
                        extra={"context_length_exceeded": True},
                    )
                )
                self._dump_trajectory()
                raise

            # Handle pending subagent refs (summarization)
            if self._pending_subagent_refs:
                from harbor.models.trajectories import Step, Observation, ObservationResult
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="Performed context summarization.",
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    subagent_trajectory_ref=self._pending_subagent_refs
                                )
                            ]
                        ),
                    )
                )
                self._pending_subagent_refs = None

            # Handle handoff prompt
            if self._pending_handoff_prompt:
                from harbor.models.trajectories import Step
                if self._linear_history:
                    self._split_trajectory_on_summarization(self._pending_handoff_prompt)
                else:
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="user",
                            message=self._pending_handoff_prompt,
                        )
                    )
                self._pending_handoff_prompt = None

            # Build trajectory message
            if self._save_raw_content_in_trajectory:
                message_content = llm_response.content
            else:
                message_parts = []
                if analysis:
                    message_parts.append(f"Analysis: {analysis}")
                if plan:
                    message_parts.append(f"Plan: {plan}")
                message_content = "\n".join(message_parts) if message_parts else ""

            # Update context
            self._context.n_input_tokens = chat.total_input_tokens
            self._context.n_output_tokens = chat.total_output_tokens
            self._context.n_cache_tokens = chat.total_cache_tokens
            self._context.cost_usd = chat.total_cost if chat.total_cost > 0 else None

            self._record_asciinema_marker(f"Episode {episode}: {len(commands)} commands")

            # Handle parsing errors
            if feedback and "ERROR:" in feedback:
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    f"Please fix these issues and provide a proper "
                    f"{self._get_error_response_type()}."
                )
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before
                self._record_step(
                    episode, llm_response, message_content, prompt,
                    tokens_before_input, tokens_before_output,
                    cache_tokens_used, step_cost, chat, is_error=True
                )
                continue

            # Execute commands
            timeout_occurred, terminal_output = await self._execute_commands(
                commands, self._session
            )

            was_pending_completion = self._pending_completion

            # Build observation
            if is_task_complete:
                if self._pending_completion:
                    observation = terminal_output
                else:
                    self._pending_completion = True
                    observation = self._get_completion_confirmation_message(terminal_output)
            else:
                self._pending_completion = False
                if feedback and "WARNINGS:" in feedback:
                    observation = (
                        f"Previous response had warnings:\n{feedback}\n\n"
                        f"{self._limit_output_length(terminal_output)}"
                    )
                else:
                    observation = self._limit_output_length(terminal_output)

            # Record step
            cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
            step_cost = chat.total_cost - cost_before
            self._record_step_full(
                episode, commands, is_task_complete, llm_response,
                message_content, observation, tokens_before_input,
                tokens_before_output, cache_tokens_used, step_cost, chat,
                injection=step_injection,
            )

            # Dump trajectory
            self._dump_trajectory()

            # Check task completion
            if is_task_complete:
                if was_pending_completion:
                    logger.info(
                        f"Task complete. Memory: {len(self._memory)} entries, "
                        f"{self._memory_trigger_count} triggers, "
                        f"{self._injection_count} injections, "
                        f"{self._noop_count} no-ops"
                    )
                    return episode + 1
                else:
                    prompt = observation
                    continue

            # === MEMORY: accumulate full step and check trigger ===
            self._accumulate_step(
                step=episode + 2,
                observation=observation,
                analysis=analysis,
                plan=plan,
                commands=commands,
            )
            should_trigger, reason = self._trigger.should_trigger(step_count=episode + 2)
            if should_trigger:
                logger.debug(f"Memory trigger at step {episode + 2}: {reason}")
                await self._trigger_memory_agent(step_count=episode + 2)

            prompt = observation

        return self._n_episodes

    # -----------------------------------------------------------------
    # Step recording
    # -----------------------------------------------------------------

    def _record_step(
        self, episode, llm_response, message_content, observation,
        tokens_before_input, tokens_before_output, cache_tokens_used,
        step_cost, chat, is_error=False
    ):
        """Record an error step in the trajectory."""
        from harbor.models.trajectories import Step, Observation, ObservationResult, Metrics

        self._trajectory_steps.append(
            Step(
                step_id=len(self._trajectory_steps) + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=self._model_name,
                message=llm_response.content,
                reasoning_content=llm_response.reasoning_content,
                observation=Observation(
                    results=[ObservationResult(content=observation)]
                ),
                metrics=Metrics(
                    prompt_tokens=chat.total_input_tokens - tokens_before_input,
                    completion_tokens=chat.total_output_tokens - tokens_before_output,
                    cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                    cost_usd=step_cost if step_cost > 0 else None,
                    prompt_token_ids=llm_response.prompt_token_ids,
                    completion_token_ids=llm_response.completion_token_ids,
                    logprobs=llm_response.logprobs,
                ),
            )
        )

    def _record_step_full(
        self, episode, commands, is_task_complete, llm_response,
        message_content, observation, tokens_before_input,
        tokens_before_output, cache_tokens_used, step_cost, chat,
        injection: str | None = None,
    ):
        """Record a full step in the trajectory."""
        from harbor.models.trajectories import (
            Step, Observation, ObservationResult, Metrics, ToolCall
        )

        tool_calls = None
        observation_results = []

        if not self._save_raw_content_in_trajectory:
            tool_calls_list = []
            if commands:
                for i, cmd in enumerate(commands):
                    tool_call_id = f"call_{episode}_{i + 1}"
                    tool_calls_list.append(
                        ToolCall(
                            tool_call_id=tool_call_id,
                            function_name="bash_command",
                            arguments={
                                "keystrokes": cmd.keystrokes,
                                "duration": cmd.duration_sec,
                            },
                        )
                    )
            if tool_calls_list:
                tool_calls = tool_calls_list

            if observation:
                observation_results.append(ObservationResult(content=observation))
        else:
            if observation:
                observation_results.append(ObservationResult(content=observation))

        # Build metadata with injection info
        extra = {}
        if injection:
            extra["memory_context"] = injection
        if is_task_complete:
            extra["task_complete"] = True

        self._trajectory_steps.append(
            Step(
                step_id=len(self._trajectory_steps) + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=self._model_name,
                message=message_content if not self._save_raw_content_in_trajectory else llm_response.content,
                reasoning_content=llm_response.reasoning_content,
                tool_calls=tool_calls,
                observation=Observation(results=observation_results) if observation_results else None,
                extra=extra if extra else None,
                metrics=Metrics(
                    prompt_tokens=chat.total_input_tokens - tokens_before_input,
                    completion_tokens=chat.total_output_tokens - tokens_before_output,
                    cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                    cost_usd=step_cost if step_cost > 0 else None,
                    prompt_token_ids=llm_response.prompt_token_ids,
                    completion_token_ids=llm_response.completion_token_ids,
                    logprobs=llm_response.logprobs,
                ),
            )
        )

    # -----------------------------------------------------------------
    # Container snapshot
    # -----------------------------------------------------------------

    async def _save_container_snapshot(self, step_count: int) -> None:
        """Export current container state as sqsh file for RL rollout acceleration."""
        try:
            container_name = getattr(self._environment, '_container_name', None)
            if not container_name:
                return

            sqsh_path = self._snapshot_dir / f"snapshot_step_{step_count:04d}.sqsh"
            logger.info(f"[Step {step_count}] Saving container snapshot to {sqsh_path}...")

            import asyncio
            data_dir = getattr(self._environment, '_data_dir', '')
            env = {**__import__('os').environ, "ENROOT_DATA_PATH": str(data_dir)}
            proc = await asyncio.create_subprocess_exec(
                "enroot", "export", "-f", "-o", str(sqsh_path), container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

            if proc.returncode == 0:
                size_gb = sqsh_path.stat().st_size / 1e9
                logger.info(f"[Step {step_count}] Snapshot saved ({size_gb:.1f} GB)")
            else:
                logger.warning(f"[Step {step_count}] Snapshot failed: {stderr.decode()[:200]}")
        except Exception as e:
            logger.warning(f"[Step {step_count}] Snapshot error: {e}")

    # -----------------------------------------------------------------
    # Memory trajectory logging
    # -----------------------------------------------------------------

    def _record_memory_step(
        self,
        trigger_step: int,
        step_count: int,
        observation: str,
        result: MemoryAgentResult | None,
        error: str | None = None,
    ) -> None:
        """Record a memory agent invocation in the trajectory."""
        entry = {
            "step_id": len(self._memory_trajectory_steps) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger_step": trigger_step,
            "step_count": step_count,
            "observation_len": len(observation),
        }

        if result:
            entry["operations"] = [
                {
                    "action": op.action,
                    "memory_id": op.memory_id,
                    "content": op.content,
                    "success": op.success,
                    "error": op.error,
                }
                for op in result.operations
            ]
            entry["should_inject"] = result.should_inject
            entry["context_for_action"] = result.context_for_action or ""
            entry["raw_response_phase1"] = result.raw_response_phase1 or ""
            entry["raw_response_phase2"] = result.raw_response_phase2 or ""
            entry["reasoning_phase1"] = result.reasoning_phase1 or ""
            entry["reasoning_phase2"] = result.reasoning_phase2 or ""
            entry["user_prompt_phase1"] = result.user_prompt_phase1 or ""
            entry["user_prompt_phase2"] = result.user_prompt_phase2 or ""
            entry["observation"] = observation or ""
            entry["memory_snapshot"] = {
                "status": self._memory.status or "",
                "knowledge_count": len(self._memory.knowledge),
                "procedural_count": len(self._memory.procedural),
            }
        else:
            entry["error"] = error

        self._memory_trajectory_steps.append(entry)

    def save_memory(self, path: str) -> None:
        """Save memory bank and trajectory to files."""
        from pathlib import Path as P
        p = P(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save memory bank
        self._memory.save_to_file(p)

        # Save memory trajectory
        traj_path = p.parent / "trajectory_memory.json"
        traj_data = {
            "schema_version": "memory-trajectory-v3",
            "memory_model_name": self._memory_llm._model_name if self._memory_llm else "unknown",
            "trigger_config": {
                "interval": self._trigger.config.interval,
                "trigger_on_first_step": self._trigger.config.trigger_on_first_step,
            },
            "summary": {
                "total_triggers": self._memory_trigger_count,
                "total_operations": self._memory_operation_count,
                "total_injections": self._injection_count,
                "total_noops": self._noop_count,
                "final_memory_size": len(self._memory),
            },
            "steps": self._memory_trajectory_steps,
        }
        traj_path.write_text(json.dumps(traj_data, indent=2, default=str))

    def get_memory_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "memory_size": len(self._memory),
            "knowledge_count": len(self._memory.knowledge),
            "procedural_count": len(self._memory.procedural),
            "trigger_count": self._memory_trigger_count,
            "operation_count": self._memory_operation_count,
            "injection_count": self._injection_count,
            "noop_count": self._noop_count,
        }
