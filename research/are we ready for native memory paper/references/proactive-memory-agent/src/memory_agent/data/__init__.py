"""Task data loading adapters.

This module provides adapters for loading tasks from different data sources:
- ParquetAdapter: Loads from terminal-bench-rl-data parquet files
- HarborAdapter: Loads from harbor's cached task format (~/.cache/harbor/tasks/)

Usage:
    from memory_agent.data import create_adapter, Task

    # Parquet adapter
    adapter = create_adapter("parquet", data_path="/path/to/data", split="val")
    tasks = adapter.load_tasks(task_names=["task1"], sqsh_dir="/path/to/sqsh")

    # Harbor adapter
    adapter = create_adapter("harbor", tasks_dir="~/.cache/harbor/tasks")
    tasks = adapter.load_tasks(task_names=["chess-best-move"], sqsh_dir="/path/to/sqsh")
"""

from pathlib import Path
from typing import Any

from .base import AdapterType, Task, TaskAdapter, get_sqsh_path
from .harbor_adapter import HarborAdapter
from .parquet_adapter import ParquetAdapter

__all__ = [
    "Task",
    "TaskAdapter",
    "AdapterType",
    "ParquetAdapter",
    "HarborAdapter",
    "create_adapter",
    "load_tasks",
    "get_sqsh_path",
]


def create_adapter(
    adapter_type: AdapterType,
    **kwargs: Any,
) -> TaskAdapter:
    """Factory function to create a task adapter.

    Args:
        adapter_type: Type of adapter ("parquet" or "harbor").
        **kwargs: Adapter-specific arguments.
            For "parquet":
                - data_path: Path to parquet files directory
                - split: Dataset split ("train" or "val")
            For "harbor":
                - tasks_dir: Path to harbor tasks cache directory
                - sqsh_pattern: Pattern for sqsh file names
                - sqsh_prefix: Optional sqsh filename prefix filter

    Returns:
        TaskAdapter instance.

    Raises:
        ValueError: If adapter_type is not recognized.
    """
    if adapter_type == "parquet":
        return ParquetAdapter(
            data_path=kwargs.get("data_path", "."),
            split=kwargs.get("split", "val"),
        )
    elif adapter_type == "harbor":
        return HarborAdapter(
            tasks_dir=kwargs.get("tasks_dir"),
            sqsh_pattern=kwargs.get(
                "sqsh_pattern",
                "ghcr.io+laude-institute+terminal-bench+{task_name}+2.0.sqsh",
            ),
            sqsh_prefix=kwargs.get("sqsh_prefix"),
        )
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


def load_tasks(
    adapter_type: AdapterType = "parquet",
    task_names: list[str] | None = None,
    sqsh_dir: str | Path | None = None,
    **kwargs: Any,
) -> list[Task]:
    """Convenience function to load tasks using the specified adapter.

    This is a simpler interface that creates the adapter and loads tasks
    in one call, for backward compatibility with existing code.

    Args:
        adapter_type: Type of adapter ("parquet" or "harbor").
        task_names: Optional list of task names to filter by.
        sqsh_dir: Optional path to sqsh directory for filtering.
        **kwargs: Additional adapter-specific arguments.

    Returns:
        List of Task objects.
    """
    adapter = create_adapter(adapter_type, **kwargs)
    return adapter.load_tasks(task_names=task_names, sqsh_dir=sqsh_dir)
