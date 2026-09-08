# pyright: reportMissingImports=false
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from cross.types import ContextBundle, EventKind, FinalizationReport, SessionEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class HookResult(BaseModel):
    """Result returned by every hook invocation.

    Attributes:
        context_bundle: Cross-session context assembled at session start.
        finalization_report: Report produced when a session is finalized.
        events_recorded: Running count of events persisted by the hook call.
    """

    context_bundle: Optional[ContextBundle] = None
    finalization_report: Optional[FinalizationReport] = None
    events_recorded: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class SessionHooks(ABC):
    """Abstract interface for agent lifecycle hooks.

    Agent frameworks implement (or use a provided subclass of) this ABC
    to integrate with SimpleMem-Cross.  The lifecycle follows:

        SessionStart -> (UserMessage | ToolUse)* -> SessionStop -> SessionEnd
    """

    @abstractmethod
    async def on_session_start(
        self,
        tenant_id: str,
        content_session_id: str,
        project: str,
        user_prompt: Optional[str] = None,
    ) -> HookResult:
        """Called when a new agent session begins.

        Should create a session record and, optionally, build a
        :class:`ContextBundle` with relevant cross-session memories.

        Args:
            tenant_id: Tenant identifier for multi-tenant isolation.
            content_session_id: Unique session ID from the agent framework.
            project: Project name or path for scoping memories.
            user_prompt: The initial user prompt that started the session.

        Returns:
            HookResult with ``context_bundle`` populated.
        """

    @abstractmethod
    async def on_user_message(
        self,
        memory_session_id: str,
        content: str,
    ) -> HookResult:
        """Called on each user message during an active session.

        Args:
            memory_session_id: Internal memory session identifier.
            content: Raw text content of the user message.

        Returns:
            HookResult (typically with ``events_recorded`` incremented).
        """

    @abstractmethod
    async def on_tool_use(
        self,
        memory_session_id: str,
        tool_name: str,
        tool_input: str,
        tool_output: str,
    ) -> HookResult:
        """Called after each tool invocation by the agent.

        Args:
            memory_session_id: Internal memory session identifier.
            tool_name: Name of the tool that was invoked.
            tool_input: Serialised input passed to the tool.
            tool_output: Serialised output returned by the tool.

        Returns:
            HookResult (typically with ``events_recorded`` incremented).
        """

    @abstractmethod
    async def on_session_stop(
        self,
        memory_session_id: str,
    ) -> HookResult:
        """Called when the session is stopping — triggers memory finalization.

        This is where observations are extracted, summaries generated, and
        cross-session memories persisted.

        Args:
            memory_session_id: Internal memory session identifier.

        Returns:
            HookResult with ``finalization_report`` populated.
        """

    @abstractmethod
    async def on_session_end(
        self,
        memory_session_id: str,
    ) -> HookResult:
        """Called for final cleanup after finalization.

        Any transient resources tied to the session should be released here.

        Args:
            memory_session_id: Internal memory session identifier.

        Returns:
            HookResult (empty on success).
        """


# ---------------------------------------------------------------------------
# Default implementation
# ---------------------------------------------------------------------------


class DefaultHooks(SessionHooks):
    """Standard hook implementation that delegates to a session manager.

    This is the primary implementation shipped with SimpleMem-Cross.  It
    requires a *session_manager* (duck-typed) and optionally a
    *context_injector* for building the startup context bundle.
    """

    def __init__(
        self,
        session_manager: Any,
        context_injector: Any = None,
    ) -> None:
        """Initialise with the required collaborators.

        Args:
            session_manager: Object exposing ``start_session``,
                ``record_event``, ``finalize_session``, and
                ``end_session`` methods (all may be sync or async).
            context_injector: Optional object with a ``build_context``
                method used to assemble the :class:`ContextBundle`.
        """
        self._session_manager = session_manager
        self._context_injector = context_injector

    # -- lifecycle hooks ----------------------------------------------------

    async def on_session_start(
        self,
        tenant_id: str,
        content_session_id: str,
        project: str,
        user_prompt: Optional[str] = None,
    ) -> HookResult:
        """Create a session record and build the context bundle."""
        logger.info(
            "on_session_start: tenant=%s session=%s project=%s",
            tenant_id,
            content_session_id,
            project,
        )
        try:
            memory_session_id = await self._await_if_coro(
                self._session_manager.start_session(
                    tenant_id=tenant_id,
                    content_session_id=content_session_id,
                    project=project,
                    user_prompt=user_prompt,
                )
            )

            context_bundle: Optional[ContextBundle] = None
            if self._context_injector is not None:
                context_bundle = await self._await_if_coro(
                    self._context_injector.build_context(
                        tenant_id=tenant_id,
                        project=project,
                        user_prompt=user_prompt,
                    )
                )

            logger.debug(
                "on_session_start completed: memory_session_id=%s has_context=%s",
                memory_session_id,
                context_bundle is not None,
            )
            return HookResult(context_bundle=context_bundle)

        except Exception:
            logger.exception(
                "on_session_start failed for session=%s", content_session_id
            )
            return HookResult()

    async def on_user_message(
        self,
        memory_session_id: str,
        content: str,
    ) -> HookResult:
        """Record a user message event."""
        logger.info("on_user_message: session=%s", memory_session_id)
        try:
            event = SessionEvent(
                memory_session_id=memory_session_id,
                timestamp=datetime.now(timezone.utc),
                kind=EventKind.message,
                title="user_message",
                payload_json=json.dumps({"content": content}),
            )
            await self._await_if_coro(
                self._session_manager.record_event(
                    memory_session_id=memory_session_id,
                    event=event,
                )
            )
            return HookResult(events_recorded=1)

        except Exception:
            logger.exception("on_user_message failed for session=%s", memory_session_id)
            return HookResult()

    async def on_tool_use(
        self,
        memory_session_id: str,
        tool_name: str,
        tool_input: str,
        tool_output: str,
    ) -> HookResult:
        """Record a tool-use event."""
        logger.info("on_tool_use: session=%s tool=%s", memory_session_id, tool_name)
        try:
            payload = {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output,
            }
            event = SessionEvent(
                memory_session_id=memory_session_id,
                timestamp=datetime.now(timezone.utc),
                kind=EventKind.tool_use,
                title=tool_name,
                payload_json=json.dumps(payload),
            )
            await self._await_if_coro(
                self._session_manager.record_event(
                    memory_session_id=memory_session_id,
                    event=event,
                )
            )
            return HookResult(events_recorded=1)

        except Exception:
            logger.exception(
                "on_tool_use failed for session=%s tool=%s",
                memory_session_id,
                tool_name,
            )
            return HookResult()

    async def on_session_stop(
        self,
        memory_session_id: str,
    ) -> HookResult:
        """Finalize memory: extract observations, generate summary, persist."""
        logger.info("on_session_stop: session=%s", memory_session_id)
        try:
            report = await self._await_if_coro(
                self._session_manager.finalize_session(
                    memory_session_id=memory_session_id,
                )
            )
            finalization: Optional[FinalizationReport] = None
            if isinstance(report, FinalizationReport):
                finalization = report
            logger.info(
                "on_session_stop completed: session=%s report=%s",
                memory_session_id,
                finalization is not None,
            )
            return HookResult(finalization_report=finalization)

        except Exception:
            logger.exception("on_session_stop failed for session=%s", memory_session_id)
            return HookResult()

    async def on_session_end(
        self,
        memory_session_id: str,
    ) -> HookResult:
        """Perform final cleanup for the session."""
        logger.info("on_session_end: session=%s", memory_session_id)
        try:
            await self._await_if_coro(
                self._session_manager.end_session(
                    memory_session_id=memory_session_id,
                )
            )
            logger.debug("on_session_end completed: session=%s", memory_session_id)
            return HookResult()

        except Exception:
            logger.exception("on_session_end failed for session=%s", memory_session_id)
            return HookResult()

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    async def _await_if_coro(value: Any) -> Any:
        """Await *value* if it is a coroutine, otherwise return it directly.

        This allows the hooks to work with both sync and async session
        managers without requiring the caller to know which flavour is in
        use.
        """
        import asyncio

        if asyncio.iscoroutine(value) or asyncio.isfuture(value):
            return await value
        return value


# ---------------------------------------------------------------------------
# No-op implementation (for testing / disabled mode)
# ---------------------------------------------------------------------------


class NoOpHooks(SessionHooks):
    """No-op hook implementation — every method returns an empty HookResult.

    Useful for testing or when cross-session memory is intentionally disabled.
    """

    async def on_session_start(
        self,
        tenant_id: str,
        content_session_id: str,
        project: str,
        user_prompt: Optional[str] = None,
    ) -> HookResult:
        """Return empty result without side effects."""
        return HookResult()

    async def on_user_message(
        self,
        memory_session_id: str,
        content: str,
    ) -> HookResult:
        """Return empty result without side effects."""
        return HookResult()

    async def on_tool_use(
        self,
        memory_session_id: str,
        tool_name: str,
        tool_input: str,
        tool_output: str,
    ) -> HookResult:
        """Return empty result without side effects."""
        return HookResult()

    async def on_session_stop(
        self,
        memory_session_id: str,
    ) -> HookResult:
        """Return empty result without side effects."""
        return HookResult()

    async def on_session_end(
        self,
        memory_session_id: str,
    ) -> HookResult:
        """Return empty result without side effects."""
        return HookResult()
