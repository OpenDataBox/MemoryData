"""Parquet task data adapter.

Loads tasks from parquet files in terminal-bench-rl-data format.
"""

import logging
from pathlib import Path

import pyarrow.parquet as pq

from .base import Task, TaskAdapter, get_sqsh_path

logger = logging.getLogger(__name__)


def filter_parquet_by_sqsh(parquet_path: str | Path, sqsh_dir: str | Path) -> str:
    """Return a path to a parquet filtered to rows whose sqsh image exists.

    Reads the top-level ``task_name`` column and drops rows whose
    corresponding ``{sqsh_dir}/{task_name}.sqsh`` file is absent, using the
    same :func:`get_sqsh_path` logic as :class:`ParquetAdapter`.  If nothing
    needs to be filtered the original path is returned unchanged; otherwise a
    temp file is written and its path returned.
    """
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)

    if "task_name" not in table.schema.names:
        logger.warning("[sqsh-filter] No 'task_name' column in %s — skipping filter", parquet_path)
        return str(parquet_path)

    task_names = table.column("task_name").to_pylist()
    keep = [get_sqsh_path(sqsh_dir, name).exists() if name else False for name in task_names]

    n_orig = len(table)
    n_keep = sum(keep)
    n_skip = n_orig - n_keep

    if n_skip == 0:
        logger.info("[sqsh-filter] %s: all %d rows have sqsh images", parquet_path, n_orig)
        return str(parquet_path)

    filtered = table.filter(pa.array(keep, type=pa.bool_()))
    tmp_path = tempfile.mktemp(suffix=".parquet", prefix="sqsh_filtered_")
    pq.write_table(filtered, tmp_path)
    logger.info("[sqsh-filter] %s: %d → %d rows (%d skipped, missing sqsh)", parquet_path, n_orig, n_keep, n_skip)
    return tmp_path


class ParquetAdapter(TaskAdapter):
    """Loads tasks from parquet files."""

    def __init__(self, data_path: str | Path, split: str = "val"):
        """Initialize the parquet adapter.

        Args:
            data_path: Path to directory containing parquet files.
            split: Which split to load ("train" or "val").
        """
        self.data_path = Path(data_path)
        self.split = split

    def load_tasks(
        self,
        task_names: list[str] | None = None,
        sqsh_dir: str | Path | None = None,
    ) -> list[Task]:
        """Load tasks from a parquet file.

        Args:
            task_names: Optional list of task names to filter by.
            sqsh_dir: Optional path to sqsh directory for filtering.

        Returns:
            List of Task objects.
        """
        parquet_path = self.data_path / f"{self.split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        table = pq.read_table(parquet_path)
        tasks = []

        for i in range(len(table)):
            # Extract fields from the extra_info struct
            extra_info = table.column("extra_info")[i].as_py()
            if extra_info is None:
                continue

            task_name = extra_info.get("task_name")
            if not task_name:
                continue

            # Filter by task_names if specified
            if task_names and task_name not in task_names:
                continue

            # Filter by sqsh availability if sqsh_dir specified
            if sqsh_dir:
                sqsh_path = get_sqsh_path(sqsh_dir, task_name)
                if not sqsh_path.exists():
                    logger.debug(f"Skipping {task_name}: no sqsh image at {sqsh_path}")
                    continue

            # Extract test_weights, filtering out None values
            raw_weights = extra_info.get("test_weights", {})
            test_weights = {k: v for k, v in raw_weights.items() if v is not None}

            tasks.append(
                Task(
                    task_name=task_name,
                    instruction=extra_info.get("instruction", ""),
                    test_weights=test_weights,
                    dockerfile_contents=extra_info.get("dockerfile_contents", ""),
                    py_test_file_contents=extra_info.get("py_test_file_contents", ""),
                    max_test_timeout_sec=extra_info.get("max_test_timeout_sec", 300),
                    category=extra_info.get("category"),
                    difficulty=extra_info.get("difficulty"),
                )
            )

        logger.info(f"Loaded {len(tasks)} tasks from {parquet_path}")
        return tasks
