"""Base classes for task data loading."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Task(BaseModel):
    """A terminal-bench task to evaluate."""

    task_name: str
    instruction: str
    test_weights: dict[str, float]
    dockerfile_contents: str = ""
    py_test_file_contents: str = ""  # For parquet format
    max_test_timeout_sec: int = 300
    category: str | None = None
    difficulty: str | None = None
    # Harbor-specific fields
    docker_image: str = ""  # ghcr.io/laude-institute/terminal-bench/<task>:<version>
    test_script: str = ""  # Contents of tests/test.sh
    test_outputs_py: str = ""  # Contents of tests/test_outputs.py


class TaskAdapter(ABC):
    """Abstract base class for task data adapters."""

    @abstractmethod
    def load_tasks(
        self,
        task_names: list[str] | None = None,
        sqsh_dir: str | Path | None = None,
    ) -> list[Task]:
        """Load tasks from the data source.

        Args:
            task_names: Optional list of task names to filter by.
            sqsh_dir: Optional path to sqsh directory for filtering.

        Returns:
            List of Task objects.
        """
        pass


AdapterType = Literal["parquet", "harbor"]


def get_sqsh_path(sqsh_dir: str | Path, task_name: str) -> Path:
    """Get the expected sqsh image path for a task.

    For parquet format:
        task_name with / and : replaced by +, plus .sqsh extension.

    For harbor format (ghcr.io/laude-institute/terminal-bench/<task>:<version>):
        e.g., ghcr.io+laude-institute+terminal-bench+chess-best-move+2.0.sqsh
    """
    safe_name = task_name.replace("/", "+").replace(":", "+")
    return Path(sqsh_dir) / f"{safe_name}.sqsh"
