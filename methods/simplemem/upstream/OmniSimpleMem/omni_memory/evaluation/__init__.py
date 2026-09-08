"""
Evaluation module for Omni-Memory.

Provides comprehensive evaluation on standard benchmarks:
- ScienceQA (multimodal reasoning)
- LoCoMo (long-term conversation memory)
- MSR-VTT (video-text retrieval)
- Custom visual memory benchmarks
"""

from omni_memory.evaluation.evaluator import OmniMemoryEvaluator, EvaluationConfig
from omni_memory.evaluation.metrics import (
    compute_accuracy,
    compute_recall_at_k,
    compute_token_efficiency,
    compute_latency_metrics,
)
from omni_memory.evaluation.benchmarks import (
    ScienceQABenchmark,
    LoCoMoBenchmark,
    MSRVTTBenchmark,
    VisualMemoryBenchmark,
    DocBenchBenchmark,
    MMLongBenchDocBenchmark,
)

__all__ = [
    "OmniMemoryEvaluator",
    "EvaluationConfig",
    "compute_accuracy",
    "compute_recall_at_k",
    "compute_token_efficiency",
    "compute_latency_metrics",
    "ScienceQABenchmark",
    "LoCoMoBenchmark",
    "MSRVTTBenchmark",
    "VisualMemoryBenchmark",
    "DocBenchBenchmark",
    "MMLongBenchDocBenchmark",
]
