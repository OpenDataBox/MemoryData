"""
Omni-Memory: High-Efficiency Unified Multimodal Memory for Agents

A compression-first, cost-aware, lazy-expansion memory system that preserves
entropy rather than raw signals for multimodal agent interactions.
"""

__version__ = "0.1.0"
__author__ = "Omni-Memory Team"

from omni_memory.core.mau import MultimodalAtomicUnit, ModalityType
from omni_memory.core.event import EventNode, EventLevel
from omni_memory.core.config import OmniMemoryConfig
from omni_memory.orchestrator import OmniMemoryOrchestrator

__all__ = [
    "MultimodalAtomicUnit",
    "ModalityType",
    "EventNode",
    "EventLevel",
    "OmniMemoryConfig",
    "OmniMemoryOrchestrator",
]
