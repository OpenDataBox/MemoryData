"""
Parametric Memory module for Omni-Memory.

Provides memory distillation from episodic memories into a lightweight
neural model for fast, differentiable recall.
"""

from omni_memory.parametric.memory_distiller import MemoryDistiller
from omni_memory.parametric.parametric_store import ParametricMemoryStore
from omni_memory.parametric.consolidator import MemoryConsolidator

__all__ = [
    "MemoryDistiller",
    "ParametricMemoryStore", 
    "MemoryConsolidator",
]
