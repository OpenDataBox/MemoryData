"""
Self-Evolution Module for Omni-Memory (Meta-Memory Evolution Framework).

Components:
- MetaController: Adaptive parameter optimization via feedback-driven updates
- ExperienceEngine: Experience accumulation and self-reflection
- StrategyOptimizer: Thompson Sampling-based retrieval strategy selection
"""

from omni_memory.evolution.evolution_config import (
    EvolutionConfig,
    MetaControllerConfig,
    ExperienceEngineConfig,
    StrategyOptimizerConfig,
)
from omni_memory.evolution.meta_controller import (
    MetaController,
    AdaptiveParameters,
    EvolutionMetrics,
)
from omni_memory.evolution.experience_engine import (
    ExperienceEngine,
    MetaExperience,
    QueryFeatures,
)
from omni_memory.evolution.strategy_optimizer import (
    StrategyOptimizer,
    BetaPosterior,
)

__all__ = [
    "EvolutionConfig",
    "MetaControllerConfig",
    "ExperienceEngineConfig",
    "StrategyOptimizerConfig",
    "MetaController",
    "AdaptiveParameters",
    "EvolutionMetrics",
    "ExperienceEngine",
    "MetaExperience",
    "QueryFeatures",
    "StrategyOptimizer",
    "BetaPosterior",
]
