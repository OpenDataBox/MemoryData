"""Simplified trigger for Memory Agent V3.

Design:
- Trigger every turn by default (interval=1) for continuous bank management
- Whether to surface facts is still decided by the memory agent itself (Phase 2)
"""

from dataclasses import dataclass


@dataclass
class TriggerConfig:
    """Configuration for Memory Agent triggering.

    Attributes:
        interval: Run memory agent every N steps (default: 1 = every turn)
        trigger_on_first_step: Always trigger on step 1 (to capture task description)
    """
    interval: int = 1
    trigger_on_first_step: bool = True


class MemoryAgentTrigger:
    """Determines when to invoke the Memory Agent.

    Simple interval-based: every N steps + first step.
    The memory agent itself decides whether to inject (Phase 2).
    """

    def __init__(self, config: TriggerConfig | None = None):
        self.config = config or TriggerConfig()
        self._last_triggered_step: int = 0

    def should_trigger(self, step_count: int, force: bool = False) -> tuple[bool, str]:
        """Check if memory agent should run at this step.

        Args:
            step_count: Current step number (1-indexed).
            force: Force trigger regardless.

        Returns:
            (should_trigger, reason)
        """
        if force:
            self._last_triggered_step = step_count
            return True, "forced"

        if step_count == 1 and self.config.trigger_on_first_step:
            self._last_triggered_step = step_count
            return True, "first_step"

        steps_since_last = step_count - self._last_triggered_step
        if steps_since_last >= self.config.interval:
            self._last_triggered_step = step_count
            return True, f"periodic:{self.config.interval}"

        return False, "no_trigger"

    def reset(self) -> None:
        """Reset trigger state for new episode."""
        self._last_triggered_step = 0

    @property
    def last_triggered_step(self) -> int:
        return self._last_triggered_step
