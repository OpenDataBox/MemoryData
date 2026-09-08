# pyright: reportMissingImports=false
"""
MCP tool wrappers for Claude/agent integration with SimpleMem-Cross.

Provides MCP-compatible tool definitions and a dispatch registry so that
agents (e.g. Claude Desktop, Cursor, or any MCP client) can interact with
cross-session memory through a standard tool-call interface.

Usage::

    from cross.api_mcp import create_mcp_tools

    registry = create_mcp_tools(orchestrator)
    definitions = registry.get_tool_definitions()   # for tools/list
    result = await registry.call_tool("cross_session_start", {...})
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCPToolRegistry
# ---------------------------------------------------------------------------


class MCPToolRegistry:
    """Registry of MCP tool definitions backed by a cross-session orchestrator.

    Each tool is exposed as an async method that delegates to the
    orchestrator (duck-typed).  The registry also provides
    :meth:`get_tool_definitions` for the ``tools/list`` MCP response and
    :meth:`call_tool` for the ``tools/call`` dispatcher.

    Parameters
    ----------
    orchestrator:
        Duck-typed object that exposes the cross-session lifecycle and
        query methods.  Typically an instance of the project's
        ``CrossSessionOrchestrator``, but any object with the right
        method signatures will work.
    """

    def __init__(self, orchestrator: Any) -> None:
        self._orchestrator = orchestrator
        self._tool_map: Dict[str, Any] = {
            "cross_session_start": self.cross_session_start,
            "cross_session_message": self.cross_session_message,
            "cross_session_tool_use": self.cross_session_tool_use,
            "cross_session_stop": self.cross_session_stop,
            "cross_session_end": self.cross_session_end,
            "cross_session_search": self.cross_session_search,
            "cross_session_context": self.cross_session_context,
            "cross_session_stats": self.cross_session_stats,
        }

    # ------------------------------------------------------------------
    # MCP schema helpers
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return MCP-compatible tool schema definitions.

        Each entry contains ``name``, ``description``, and ``inputSchema``
        (a JSON Schema object) as required by the MCP ``tools/list``
        response format.
        """
        return [
            # --- cross_session_start -----------------------------------
            {
                "name": "cross_session_start",
                "description": (
                    "Start a new cross-session memory session. "
                    "Creates a session record in the memory system and "
                    "returns a memory_session_id along with any relevant "
                    "context from previous sessions. Call this at the "
                    "beginning of every new agent conversation."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {
                            "type": "string",
                            "description": (
                                "Tenant identifier for multi-tenant isolation."
                            ),
                        },
                        "content_session_id": {
                            "type": "string",
                            "description": (
                                "Unique session ID from the agent framework "
                                "(e.g. the conversation ID)."
                            ),
                        },
                        "project": {
                            "type": "string",
                            "description": (
                                "Project name or path used to scope memories."
                            ),
                        },
                        "user_prompt": {
                            "type": "string",
                            "description": (
                                "Optional initial user prompt that started "
                                "the session. Used for semantic retrieval of "
                                "relevant past context."
                            ),
                        },
                    },
                    "required": ["tenant_id", "content_session_id", "project"],
                },
            },
            # --- cross_session_message ---------------------------------
            {
                "name": "cross_session_message",
                "description": (
                    "Record a chat message event in the current memory "
                    "session. Call this for each user or assistant message "
                    "that should be remembered across sessions."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_session_id": {
                            "type": "string",
                            "description": (
                                "Internal memory session identifier returned "
                                "by cross_session_start."
                            ),
                        },
                        "content": {
                            "type": "string",
                            "description": "The message text content.",
                        },
                        "role": {
                            "type": "string",
                            "description": (
                                "Speaker role: 'user', 'assistant', or "
                                "'system'. Defaults to 'user'."
                            ),
                        },
                    },
                    "required": ["memory_session_id", "content"],
                },
            },
            # --- cross_session_tool_use --------------------------------
            {
                "name": "cross_session_tool_use",
                "description": (
                    "Record a tool invocation event in the current memory "
                    "session. Call this after each tool use to capture what "
                    "tools were called and their results for future context."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_session_id": {
                            "type": "string",
                            "description": (
                                "Internal memory session identifier returned "
                                "by cross_session_start."
                            ),
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool that was called.",
                        },
                        "tool_input": {
                            "type": "string",
                            "description": ("Serialised input passed to the tool."),
                        },
                        "tool_output": {
                            "type": "string",
                            "description": ("Serialised output returned by the tool."),
                        },
                    },
                    "required": [
                        "memory_session_id",
                        "tool_name",
                        "tool_input",
                        "tool_output",
                    ],
                },
            },
            # --- cross_session_stop ------------------------------------
            {
                "name": "cross_session_stop",
                "description": (
                    "Stop and finalize the current memory session. Triggers "
                    "observation extraction, summary generation, and memory "
                    "persistence. Call this when the agent conversation is "
                    "wrapping up, before cross_session_end."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_session_id": {
                            "type": "string",
                            "description": (
                                "Internal memory session identifier returned "
                                "by cross_session_start."
                            ),
                        },
                    },
                    "required": ["memory_session_id"],
                },
            },
            # --- cross_session_end -------------------------------------
            {
                "name": "cross_session_end",
                "description": (
                    "End and clean up the memory session. Releases any "
                    "transient resources associated with the session. "
                    "Call this as the final step after cross_session_stop."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_session_id": {
                            "type": "string",
                            "description": (
                                "Internal memory session identifier returned "
                                "by cross_session_start."
                            ),
                        },
                    },
                    "required": ["memory_session_id"],
                },
            },
            # --- cross_session_search ----------------------------------
            {
                "name": "cross_session_search",
                "description": (
                    "Search cross-session memories using semantic and "
                    "keyword matching. Returns relevant memory entries "
                    "from past sessions that match the query."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural language search query for finding "
                                "relevant cross-session memories."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": (
                                "Maximum number of results to return. Defaults to 10."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
            # --- cross_session_context ---------------------------------
            {
                "name": "cross_session_context",
                "description": (
                    "Get cross-session context suitable for injection into "
                    "a system prompt. Assembles a token-budgeted bundle of "
                    "session summaries, observations, and semantic matches. "
                    "Use at session start to prime the agent with relevant "
                    "past knowledge."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_prompt": {
                            "type": "string",
                            "description": (
                                "Optional user prompt to drive semantic "
                                "retrieval of relevant past memories."
                            ),
                        },
                    },
                    "required": [],
                },
            },
            # --- cross_session_stats -----------------------------------
            {
                "name": "cross_session_stats",
                "description": (
                    "Get statistics about the cross-session memory system "
                    "including session counts, event counts, observation "
                    "counts, and memory entry totals."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch an MCP tool call to the appropriate handler.

        Parameters
        ----------
        name:
            The tool name as declared in :meth:`get_tool_definitions`.
        arguments:
            The parsed ``arguments`` dict from the MCP ``tools/call`` request.

        Returns
        -------
        A dict suitable for serialisation as the MCP tool-call result.
        On failure the dict contains an ``"error"`` key with a message.
        """
        handler = self._tool_map.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return await handler(**arguments)
        except Exception as exc:
            logger.exception("Tool call %s failed", name)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def cross_session_start(
        self,
        tenant_id: str,
        content_session_id: str,
        project: str,
        user_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a new cross-session, return memory_session_id + context.

        Delegates to ``orchestrator.session_start(...)`` which is expected
        to create the session record and optionally build a context bundle.
        """
        try:
            result = await _await_if_coro(
                self._orchestrator.session_start(
                    tenant_id=tenant_id,
                    content_session_id=content_session_id,
                    project=project,
                    user_prompt=user_prompt,
                )
            )
            return _normalise_result(result, fallback_key="session")
        except Exception as exc:
            logger.exception("cross_session_start failed")
            return {"error": str(exc)}

    async def cross_session_message(
        self,
        memory_session_id: str,
        content: str,
        role: str = "user",
    ) -> Dict[str, Any]:
        """Record a message event in the active session.

        Delegates to ``orchestrator.session_message(...)`` or
        ``orchestrator.record_message(...)``.
        """
        try:
            fn = _resolve_method(
                self._orchestrator,
                "session_message",
                "record_message",
            )
            result = await _await_if_coro(
                fn(
                    memory_session_id=memory_session_id,
                    content=content,
                    role=role,
                )
            )
            return _normalise_result(result, fallback_key="event")
        except Exception as exc:
            logger.exception("cross_session_message failed")
            return {"error": str(exc)}

    async def cross_session_tool_use(
        self,
        memory_session_id: str,
        tool_name: str,
        tool_input: str,
        tool_output: str,
    ) -> Dict[str, Any]:
        """Record a tool use event in the active session.

        Delegates to ``orchestrator.session_tool_use(...)`` or
        ``orchestrator.record_tool_use(...)``.
        """
        try:
            fn = _resolve_method(
                self._orchestrator,
                "session_tool_use",
                "record_tool_use",
            )
            result = await _await_if_coro(
                fn(
                    memory_session_id=memory_session_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=tool_output,
                )
            )
            return _normalise_result(result, fallback_key="event")
        except Exception as exc:
            logger.exception("cross_session_tool_use failed")
            return {"error": str(exc)}

    async def cross_session_stop(
        self,
        memory_session_id: str,
    ) -> Dict[str, Any]:
        """Stop and finalize the session (extract observations, summarize).

        Delegates to ``orchestrator.session_stop(...)`` or
        ``orchestrator.finalize_session(...)``.
        """
        try:
            fn = _resolve_method(
                self._orchestrator,
                "session_stop",
                "finalize_session",
            )
            result = await _await_if_coro(fn(memory_session_id=memory_session_id))
            return _normalise_result(result, fallback_key="finalization")
        except Exception as exc:
            logger.exception("cross_session_stop failed")
            return {"error": str(exc)}

    async def cross_session_end(
        self,
        memory_session_id: str,
    ) -> Dict[str, Any]:
        """End session and release resources.

        Delegates to ``orchestrator.session_end(...)`` or
        ``orchestrator.end_session(...)``.
        """
        try:
            fn = _resolve_method(
                self._orchestrator,
                "session_end",
                "end_session",
            )
            result = await _await_if_coro(fn(memory_session_id=memory_session_id))
            return _normalise_result(result, fallback_key="status")
        except Exception as exc:
            logger.exception("cross_session_end failed")
            return {"error": str(exc)}

    async def cross_session_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Search cross-session memories.

        Delegates to ``orchestrator.search(...)`` or
        ``orchestrator.session_search(...)``.
        """
        try:
            fn = _resolve_method(
                self._orchestrator,
                "search",
                "session_search",
            )
            result = await _await_if_coro(fn(query=query, top_k=top_k))
            return _normalise_result(result, fallback_key="results")
        except Exception as exc:
            logger.exception("cross_session_search failed")
            return {"error": str(exc)}

    async def cross_session_context(
        self,
        user_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get context for system prompt injection.

        Delegates to ``orchestrator.get_context(...)`` or
        ``orchestrator.session_context(...)``.
        """
        try:
            fn = _resolve_method(
                self._orchestrator,
                "get_context",
                "session_context",
            )
            result = await _await_if_coro(fn(user_prompt=user_prompt))
            return _normalise_result(result, fallback_key="context")
        except Exception as exc:
            logger.exception("cross_session_context failed")
            return {"error": str(exc)}

    async def cross_session_stats(self) -> Dict[str, Any]:
        """Get memory system statistics.

        Delegates to ``orchestrator.get_stats(...)`` or
        ``orchestrator.session_stats(...)``.
        """
        try:
            fn = _resolve_method(
                self._orchestrator,
                "get_stats",
                "session_stats",
            )
            result = await _await_if_coro(fn())
            return _normalise_result(result, fallback_key="stats")
        except Exception as exc:
            logger.exception("cross_session_stats failed")
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_mcp_tools(orchestrator: Any) -> MCPToolRegistry:
    """Convenience factory for creating an :class:`MCPToolRegistry`.

    Parameters
    ----------
    orchestrator:
        The cross-session orchestrator instance (duck-typed).

    Returns
    -------
    A fully-initialised :class:`MCPToolRegistry` ready for use.

    Example::

        registry = create_mcp_tools(orchestrator)
        tool_defs = registry.get_tool_definitions()
        result = await registry.call_tool("cross_session_stats", {})
    """
    return MCPToolRegistry(orchestrator)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _await_if_coro(value: Any) -> Any:
    """Await *value* if it is a coroutine, otherwise return it directly.

    This allows tools to work transparently with both sync and async
    orchestrator implementations.
    """
    import asyncio

    if asyncio.iscoroutine(value) or asyncio.isfuture(value):
        return await value
    return value


def _resolve_method(obj: Any, *names: str) -> Any:
    """Resolve the first available method name on *obj*.

    Parameters
    ----------
    obj:
        The target object (typically the orchestrator).
    *names:
        Method names to try in order of preference.

    Returns
    -------
    The bound method.

    Raises
    ------
    AttributeError
        If none of the given names exist on *obj*.
    """
    for name in names:
        method = getattr(obj, name, None)
        if method is not None and callable(method):
            return method
    tried = ", ".join(names)
    raise AttributeError(f"Orchestrator {type(obj).__name__!r} has none of: {tried}")


def _normalise_result(result: Any, *, fallback_key: str) -> Dict[str, Any]:
    """Coerce an orchestrator return value into a plain dict.

    If the result is already a dict it is returned as-is.  Dataclass
    instances are converted via their ``__dict__``.  Pydantic models
    are converted via ``.model_dump()`` or ``.dict()``.  Everything else
    is wrapped as ``{fallback_key: result}``.
    """
    if isinstance(result, dict):
        return result

    # Pydantic v2+
    model_dump = getattr(result, "model_dump", None)
    if model_dump is not None and callable(model_dump):
        return cast(Dict[str, Any], model_dump())

    # Pydantic v1
    dict_method = getattr(result, "dict", None)
    if dict_method is not None and callable(dict_method):
        try:
            return cast(Dict[str, Any], dict_method())
        except Exception:
            pass

    # Dataclass
    if hasattr(result, "__dataclass_fields__"):
        return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}

    # Fallback
    return {fallback_key: result}
