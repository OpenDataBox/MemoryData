"""
Memory routing module for benchmark-safe self-evolution.

Provides intelligent routing between Episodic (original MAU) and Semantic
(derived/distilled) memory stores.
"""

from omni_memory.routing.router import MemoryRouter, RouteResult
from omni_memory.routing.policy import (
    BenchmarkSafeGuard,
    RetentionPolicy,
    CircuitBreaker,
)
from omni_memory.routing.features import compute_route_features

__all__ = [
    "MemoryRouter",
    "RouteResult",
    "BenchmarkSafeGuard",
    "RetentionPolicy",
    "CircuitBreaker",
    "compute_route_features",
]
