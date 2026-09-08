"""Memory Agent V3 — Multi-Turn Tool Use Architecture.

Key design:
- Phase 1: LLM call with tools → bank management (save/delete/update)
- Tool results returned to model
- Phase 2: LLM generates text response → context decision (inject or no_intervention)
- Both phases in ONE conversation (multi-turn), sharing context
- For RL: two forward passes per step, but in same conversation with shared KV cache
"""

import json
import logging
import re
from dataclasses import dataclass

from .universal_memory import UniversalMemory, MemoryEntry

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Definitions — Phase 1 only (bank management)
# =============================================================================

BANK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_save_knowledge",
            "description": (
                "Save an important fact to the knowledge bank. Use for: "
                "task requirements, environment facts, file paths, API details, "
                "key constraints from the task description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact to save. Be precise and specific.",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save_procedural",
            "description": (
                "Record a debugging experience, failed approach, or solution. "
                "Use for: error patterns, things that didn't work and why, "
                "successful fixes, performance observations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The experience to record. Include what was tried and what happened.",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_update_status",
            "description": "Update your internal tracking of task progress. This is for YOUR reference only, not shown to the action agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Current task status/progress summary.",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete an outdated or incorrect memory entry by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to delete.",
                    }
                },
                "required": ["memory_id"],
            },
        },
    },
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MemoryOperation:
    """Record of a memory operation performed."""
    action: str
    memory_id: str | None = None
    content: str | None = None
    success: bool = True
    error: str | None = None


@dataclass
class MemoryAgentResult:
    """Result from a Memory Agent processing step."""
    operations: list[MemoryOperation]
    should_inject: bool
    context_for_action: str | None = None
    raw_response_phase1: str | None = None
    raw_response_phase2: str | None = None
    reasoning_phase1: str | None = None
    reasoning_phase2: str | None = None
    user_prompt_phase1: str | None = None
    user_prompt_phase2: str | None = None


# =============================================================================
# System Prompts — with_strategy variant
# =============================================================================

PHASE1_SYSTEM = '''You are a Memory Bank Manager. Your job is to maintain a lean, accurate knowledge bank.

## Your Tools
- memory_save_knowledge: Save important FACTS (task requirements, env details, API constraints, file paths)
- memory_save_procedural: Record EXPERIENCES (failed approaches, error patterns, successful fixes)
- memory_update_status: Track progress internally (not shown to action agent)
- memory_delete: Remove outdated/incorrect entries

## Guidelines
1. **Be selective**: Only save information that would be LOST if the action agent's context window scrolls past it
2. **Prioritize task requirements**: API signatures, output formats, specific constraints from the task description
3. **Record failures**: What was tried, why it failed, what the error was — so it's not repeated
4. **Keep it lean**: Delete entries that are superseded by newer information
5. **Don't save obvious things**: Terminal output the agent just saw, general knowledge about tools

## What to save at Step 1 (task description):
- Specific API requirements (function signatures, argument orders, input types)
- Required output files and formats
- Constraints that are easy to forget mid-task
- Key domain-specific details

## What to save at later steps:
- Environment discoveries (package versions, tool availability)
- Error patterns and fixes
- Approach changes and why
- Performance or score tracking
'''

PHASE2_SYSTEM = '''You are a Selective Attention module for an Action Agent working on a coding task.

Your role is like selective attention — when the agent's working memory (context window) has scrolled past important information, you can restore that context. But you also know when things are fine and the agent doesn't need help.

## PROCESS:
1. Review the memory bank contents
2. Review the agent's recent trajectory (what it's doing now)
3. DECIDE: Does the agent need a context reminder right now?

## WHEN TO INTERVENE (output <context_for_action>):
- The agent has forgotten a key requirement or constraint stored in memory
- The agent is about to repeat a recorded failure pattern
- The agent's current approach contradicts a specific fact in the memory bank
- A format requirement, API detail, or constraint from the task seems forgotten

## WHEN NOT TO INTERVENE (output <no_intervention/>):
- The agent's actions are consistent with memory bank contents
- You're not confident that a specific fact is being forgotten
- The information you'd provide is already visible in the recent trajectory
- You'd only be restating what the agent already knows

## OUTPUT FORMAT:
If intervention needed, write:
<context_for_action>
Your synthesized context note here. You can:
- Highlight requirements that seem forgotten
- Note relevant past experiences from memory
- Connect dots between memory entries and current behavior
- Flag patterns that match recorded failures
Write as helpful observations, not commands.
</context_for_action>

If no intervention needed, write:
<no_intervention/>

## GUIDELINES:
- Your DEFAULT is <no_intervention/> — only intervene when you see a real gap
- When you do intervene, be thoughtful: synthesize, don't just dump memory entries
- Frame as observations: "The task requires X" not "You MUST do X"
- Don't include information already visible in the recent trajectory
- Be calibrated: don't pass things you're not sure about
- You CAN connect dots and highlight patterns — not limited to verbatim copies from bank
- The action agent is a strong model — help it remember, don't try to control it
'''


# =============================================================================
# Memory Agent
# =============================================================================

class MemoryAgent:
    """Memory Agent V3 — Multi-turn tool use architecture.

    Each step is a multi-turn conversation:
      Turn 1 (Phase 1): System=PHASE1_SYSTEM, tools=BANK_TOOLS → model returns tool calls
      Turn 2 (Phase 2): Tool results + System=PHASE2_SYSTEM → model returns text (inject/noop)

    Both turns share the same conversation context.
    For RL training, each turn is a separate forward pass with its own prompt_ids/logprobs.
    """

    def __init__(
        self,
        llm=None,
        memory: UniversalMemory | None = None,
        max_memory_in_prompt: int = 50,
        use_bm25_prefilter: bool = True,
        keep_thinking: bool = False,
        use_structured_tools: bool = True,
    ):
        self.llm = llm
        self.memory = memory if memory is not None else UniversalMemory()
        self.max_memory_in_prompt = max_memory_in_prompt
        self.use_bm25_prefilter = use_bm25_prefilter
        self.keep_thinking = keep_thinking
        self.use_structured_tools = use_structured_tools

    # -----------------------------------------------------------------
    # LLM calls (overridable for training)
    # -----------------------------------------------------------------

    async def _call_llm_phase1(self, user_prompt: str):
        """Phase 1: bank management with tools."""
        kwargs = dict(prompt=user_prompt, system=PHASE1_SYSTEM)
        if self.use_structured_tools:
            kwargs["tools"] = BANK_TOOLS
        return await self.llm.call(**kwargs)

    async def _call_llm_phase2(self, user_prompt: str):
        """Phase 2: context decision, text only (no tools)."""
        return await self.llm.call(
            prompt=user_prompt,
            system=PHASE2_SYSTEM,
        )

    # -----------------------------------------------------------------
    # Memory formatting
    # -----------------------------------------------------------------

    def _format_memory_list(self, entries: list[MemoryEntry]) -> str:
        if not entries:
            return "(empty)"
        return "\n".join([f"[{e.id}] {e.content}" for e in entries])

    def _format_memory_bank(self, observation: str = "") -> str:
        """Format the memory bank for the LLM prompt."""
        k_count = len(self.memory.knowledge)
        p_count = len(self.memory.procedural)
        total = k_count + p_count

        if total <= self.max_memory_in_prompt or not self.use_bm25_prefilter:
            knowledge_str = self._format_memory_list(self.memory.knowledge)
            procedural_str = self._format_memory_list(self.memory.procedural)
        else:
            k_filtered = self.memory.search_knowledge(observation, top_k=20)
            p_filtered = self.memory.search_procedural(observation, top_k=20)
            knowledge_str = self._format_memory_list(k_filtered) + f"\n(showing top 20 of {k_count})"
            procedural_str = self._format_memory_list(p_filtered) + f"\n(showing top 20 of {p_count})"

        return f"""<memory_bank>
<status>{self.memory.status or "(no status)"}</status>
<knowledge>
{knowledge_str}
</knowledge>
<procedural>
{procedural_str}
</procedural>
</memory_bank>"""

    # -----------------------------------------------------------------
    # Prompt building
    # -----------------------------------------------------------------

    def _build_phase1_prompt(self, observation: str, step_count: int) -> str:
        memory_content = self._format_memory_bank(observation)
        return f"""## Step {step_count}

## Current Memory Bank
{memory_content}

## Context (Task Description + Recent Trajectory)
{observation}

## Your Task
Analyze the trajectory and update the memory bank. Save important facts, record experiences, delete outdated entries. Be selective — only save what would be lost as the agent's context scrolls.
"""

    def _build_phase2_prompt(self, observation: str, step_count: int) -> str:
        """Build Phase 2 prompt with UPDATED memory bank (after Phase 1 ops)."""
        memory_content = self._format_memory_bank(observation)
        return f"""## Step {step_count}

## Memory Bank (just updated)
{memory_content}

## Context (Task Description + Recent Trajectory)
{observation}

## Your Task
Review the memory bank and the agent's recent trajectory. Decide:
- Is the agent missing or forgetting something important from the memory bank?
- Is the agent heading toward a pattern that matches a recorded failure?

If yes, write a <context_for_action> note with the relevant context.
If no, write <no_intervention/>.
"""

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    async def process(
        self,
        observation: str,
        step_count: int = 0,
    ) -> MemoryAgentResult:
        """Process observation through two-phase multi-turn conversation.

        Phase 1: Bank management (tool calls)
        Phase 2: Context decision (text generation with updated memory bank)

        Args:
            observation: Memory agent context (task description + sliding window).
            step_count: Current step number.

        Returns:
            MemoryAgentResult with operations, injection decision, and metadata.
        """
        # Phase 1: Bank management
        phase1_prompt = self._build_phase1_prompt(observation, step_count)
        operations = []
        raw1 = None

        reasoning1 = None
        try:
            response1 = await self._call_llm_phase1(phase1_prompt)
            raw1 = getattr(response1, 'content', None)
            reasoning1 = getattr(response1, 'reasoning_content', None)
            operations = self._execute_tool_calls(response1)
        except Exception as e:
            logger.error(f"Phase 1 error: {e}", exc_info=True)

        logger.info(
            f"Phase 1: {len(operations)} ops, "
            f"bank={len(self.memory.knowledge)}K/{len(self.memory.procedural)}P"
        )

        # Phase 2: Context decision (with updated memory bank)
        phase2_prompt = self._build_phase2_prompt(observation, step_count)
        should_inject = False
        context_for_action = None
        raw2 = None
        reasoning2 = None

        try:
            response2 = await self._call_llm_phase2(phase2_prompt)
            raw2 = getattr(response2, 'content', None) or ""
            reasoning2 = getattr(response2, 'reasoning_content', None)
            context_for_action = self._parse_phase2_response(raw2)
            should_inject = context_for_action is not None
        except Exception as e:
            logger.error(f"Phase 2 error: {e}", exc_info=True)

        logger.info(f"Phase 2: {'INJECT' if should_inject else 'NO-OP'}")

        return MemoryAgentResult(
            operations=operations,
            should_inject=should_inject,
            context_for_action=context_for_action,
            raw_response_phase1=raw1,
            raw_response_phase2=raw2,
            reasoning_phase1=reasoning1,
            reasoning_phase2=reasoning2,
            user_prompt_phase1=phase1_prompt,
            user_prompt_phase2=phase2_prompt,
        )

    # -----------------------------------------------------------------
    # Tool execution (Phase 1)
    # -----------------------------------------------------------------

    def _parse_tool_calls_from_text(self, text: str) -> list[tuple[str, dict]]:
        """Fallback: parse tool calls from raw XML text when structured tool_calls are empty.

        Handles the XML format output by SFT-trained models:
            <tool_call>
            <function=memory_save_knowledge>
            <parameter=content>...</parameter>
            </function>
            </tool_call>
        """
        parsed = []
        for block in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
            block_text = block.group(1)
            # Extract function name: <function=memory_xxx> or {"function=memory_xxx>
            func_match = re.search(r'[<{]"?function=(\w+)', block_text)
            if not func_match:
                continue
            name = func_match.group(1)

            # Extract parameters
            args = {}
            for param in re.finditer(
                r"<parameter=(\w+)>(.*?)(?:</parameter>|$)", block_text, re.DOTALL
            ):
                args[param.group(1)] = param.group(2).strip()

            # Fallback: if no <parameter=content> found, extract text between
            # <function=name>\n and </parameter> or </function>
            if not args.get("content") and name != "delete":
                func_end = func_match.end()
                remaining = block_text[func_end:].lstrip(">\n")
                # Strip closing tags
                remaining = re.sub(r"</parameter>\s*</function>.*", "", remaining, flags=re.DOTALL)
                remaining = re.sub(r"</function>.*", "", remaining, flags=re.DOTALL)
                remaining = remaining.strip()
                if remaining:
                    # Unwrap JSON format: {"content": "..."} or {"content": "...", "memory_id": "..."}
                    if remaining.startswith("{") and '"content"' in remaining[:20]:
                        try:
                            parsed_json = json.loads(remaining)
                            args.update(parsed_json)
                        except json.JSONDecodeError:
                            args["content"] = remaining
                    else:
                        args["content"] = remaining

            parsed.append((name, args))
        return parsed

    def _execute_tool_calls(self, response) -> list[MemoryOperation]:
        """Execute bank management tool calls from Phase 1 response."""
        operations = []
        tool_calls = getattr(response, "tool_calls", None) or []

        # Fallback: if no structured tool_calls, or if structured calls have empty content
        has_empty_content = any(
            not (self._extract_tool_call(tc)[1].get("content"))
            for tc in tool_calls
        ) if tool_calls else False

        if not tool_calls or has_empty_content:
            raw = getattr(response, "content", None) or ""
            raw = self._strip_thinking(raw)
            if "<tool_call>" in raw:
                # Use fallback parser instead of broken structured calls
                tool_calls = []
                parsed = self._parse_tool_calls_from_text(raw)
                if parsed:
                    logger.debug(f"Parsed {len(parsed)} tool calls from raw text (fallback)")
                    for name, args in parsed:
                        tool_calls.append(type("TC", (), {
                            "function": type("F", (), {"name": name, "arguments": args})()
                        })())

        for tool_call in tool_calls:
            name, args = self._extract_tool_call(tool_call)
            if not name:
                continue

            try:
                if name == "memory_save_knowledge":
                    content = args.get("content", "")
                    mem_id = self.memory.save_knowledge(content)
                    operations.append(MemoryOperation(action="save_knowledge", memory_id=mem_id, content=content))

                elif name == "memory_save_procedural":
                    content = args.get("content", "")
                    mem_id = self.memory.save_procedural(content)
                    operations.append(MemoryOperation(action="save_procedural", memory_id=mem_id, content=content))

                elif name == "memory_update_status":
                    content = args.get("content", "")
                    self.memory.update_status(content)
                    operations.append(MemoryOperation(action="update_status", content=content))

                elif name == "memory_delete":
                    mem_id = args.get("memory_id", "")
                    success = self.memory.delete(mem_id)
                    operations.append(MemoryOperation(
                        action="delete", memory_id=mem_id, success=success,
                        error=None if success else "ID not found",
                    ))

            except Exception as e:
                logger.warning(f"Failed to execute {name}: {e}")
                operations.append(MemoryOperation(action=name, success=False, error=str(e)))

        return operations

    # -----------------------------------------------------------------
    # Phase 2 response parsing
    # -----------------------------------------------------------------

    def _strip_thinking(self, text: str) -> str:
        """Strip <think>...</think> blocks from text unless keep_thinking is True."""
        if self.keep_thinking:
            return text
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    def _parse_phase2_response(self, raw: str) -> str | None:
        """Parse Phase 2 text response.

        Returns context_for_action string if intervention, None if no-op.
        """
        # Strip thinking tokens if present in raw content
        raw = self._strip_thinking(raw)
        match = re.search(
            r"<context_for_action>(.*?)</context_for_action>",
            raw,
            re.DOTALL,
        )
        if match:
            context = match.group(1).strip()
            if context:
                return context

        if "<no_intervention" in raw:
            return None

        # Fallback: no XML tags found → default no-op
        logger.warning("Phase 2: No XML tags found, defaulting to no intervention")
        return None

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _extract_tool_call(self, tool_call) -> tuple[str | None, dict]:
        """Extract name and args from a tool call object."""
        if hasattr(tool_call, "function"):
            name = tool_call.function.name
            args = tool_call.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
        else:
            name = getattr(tool_call, "name", None)
            args = getattr(tool_call, "args", {}) or getattr(tool_call, "arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
        return name, args

    def get_tools_schema(self) -> list[dict]:
        """Get the bank management tool schema."""
        return BANK_TOOLS
