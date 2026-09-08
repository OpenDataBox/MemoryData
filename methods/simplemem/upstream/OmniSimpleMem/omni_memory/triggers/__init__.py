"""
Modal Entropy Triggers for Omni-Memory.

These triggers implement the entropy-driven filtering that determines
whether incoming multimodal data is worth storing.
"""

from omni_memory.triggers.visual_trigger import VisualEntropyTrigger
from omni_memory.triggers.audio_trigger import AudioEntropyTrigger
from omni_memory.triggers.base import BaseTrigger, TriggerResult

__all__ = [
    "VisualEntropyTrigger",
    "AudioEntropyTrigger",
    "BaseTrigger",
    "TriggerResult",
]
