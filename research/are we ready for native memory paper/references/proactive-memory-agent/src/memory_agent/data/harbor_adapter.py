"""Harbor task data adapter.

Loads tasks from harbor's cached task format at ~/.cache/harbor/tasks/.
"""

import logging
import tomllib
from pathlib import Path

from .base import Task, TaskAdapter

logger = logging.getLogger(__name__)

# Default harbor task cache directory
DEFAULT_HARBOR_TASKS_DIR = Path.home() / ".cache" / "harbor" / "tasks"


class HarborAdapter(TaskAdapter):
    """Loads tasks from harbor's cached task format.

    Harbor tasks are stored in ~/.cache/harbor/tasks/<hash-id>/<task-name>/
    with the following structure:
        - instruction.md: Task instruction
        - task.toml: Task metadata (docker_image, timeout, etc.)
        - tests/test.sh: Test runner script
        - tests/test_outputs.py: Actual test code
        - environment/: Environment files
        - solution/: Solution files
    """

    def __init__(
        self,
        tasks_dir: str | Path | None = None,
        sqsh_pattern: str = "ghcr.io+laude-institute+terminal-bench+{task_name}+2.0.sqsh",
        sqsh_prefix: str | None = None,
    ):
        """Initialize the harbor adapter.

        Args:
            tasks_dir: Path to harbor tasks cache directory.
                       Defaults to ~/.cache/harbor/tasks/
            sqsh_pattern: Pattern for sqsh file names.
                         Use {task_name} as placeholder for the task name.
            sqsh_prefix: Optional filename prefix to restrict sqsh matches
                        (e.g. "alexgshaw+").
        """
        self.tasks_dir = Path(tasks_dir) if tasks_dir else DEFAULT_HARBOR_TASKS_DIR
        self.sqsh_pattern = sqsh_pattern
        self.sqsh_prefix = sqsh_prefix

    def load_tasks(
        self,
        task_names: list[str] | None = None,
        sqsh_dir: str | Path | None = None,
    ) -> list[Task]:
        """Load tasks from harbor's cached task format.

        Args:
            task_names: Optional list of task names to filter by.
            sqsh_dir: Optional path to sqsh directory for filtering.

        Returns:
            List of Task objects (deduplicated by task_name).
        """
        if not self.tasks_dir.exists():
            raise FileNotFoundError(f"Harbor tasks directory not found: {self.tasks_dir}")

        # Use dict to deduplicate by task_name
        task_dict: dict[str, Task] = {}

        # Iterate through all hash-id directories
        for hash_dir in self.tasks_dir.iterdir():
            if not hash_dir.is_dir():
                continue

            # Each hash_dir contains task name subdirectories
            for task_dir in hash_dir.iterdir():
                if not task_dir.is_dir():
                    continue

                task_name = task_dir.name

                # Filter by task_names if specified
                if task_names and task_name not in task_names:
                    continue

                try:
                    task = self._load_task(task_dir, task_name)
                    if task:
                        # Filter by sqsh availability if sqsh_dir specified
                        if sqsh_dir:
                            sqsh_path = self._find_sqsh_path(
                                sqsh_dir=sqsh_dir,
                                task_name=task_name,
                                docker_image=task.docker_image,
                            )
                            if not sqsh_path:
                                logger.debug(
                                    f"Skipping {task_name}: no matching sqsh image "
                                    f"(docker_image={task.docker_image!r}, prefix={self.sqsh_prefix!r})"
                                )
                                continue

                        # Skip if already loaded (deduplicate by task_name)
                        if task_name in task_dict:
                            continue

                        task_dict[task_name] = task
                except Exception as e:
                    logger.warning(f"Failed to load task {task_name}: {e}")
                    continue

        tasks = list(task_dict.values())
        logger.info(f"Loaded {len(tasks)} tasks from {self.tasks_dir}")
        return tasks

    def _load_task(self, task_dir: Path, task_name: str) -> Task | None:
        """Load a single task from its directory.

        Args:
            task_dir: Path to the task directory.
            task_name: Name of the task.

        Returns:
            Task object or None if essential files are missing.
        """
        # Read instruction.md
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.exists():
            logger.warning(f"Missing instruction.md for {task_name}")
            return None

        instruction = instruction_path.read_text()

        # Read task.toml for metadata
        task_toml_path = task_dir / "task.toml"
        metadata = {}
        docker_image = ""
        timeout_sec = 300

        if task_toml_path.exists():
            try:
                with open(task_toml_path, "rb") as f:
                    toml_data = tomllib.load(f)

                metadata = toml_data.get("metadata", {})
                docker_image = toml_data.get("environment", {}).get("docker_image", "")
                timeout_sec = toml_data.get("verifier", {}).get("timeout_sec", 300)
            except Exception as e:
                logger.warning(f"Failed to parse task.toml for {task_name}: {e}")

        # Read test files
        tests_dir = task_dir / "tests"
        test_script = ""
        test_outputs_py = ""

        if tests_dir.exists():
            test_sh_path = tests_dir / "test.sh"
            if test_sh_path.exists():
                test_script = test_sh_path.read_text()

            test_outputs_path = tests_dir / "test_outputs.py"
            if test_outputs_path.exists():
                test_outputs_py = test_outputs_path.read_text()

        # For harbor format, test_weights are typically uniform (1.0 per test)
        # We'll extract test function names from test_outputs.py
        test_weights = self._extract_test_weights(test_outputs_py)

        return Task(
            task_name=task_name,
            instruction=instruction,
            test_weights=test_weights,
            dockerfile_contents="",  # Harbor uses docker_image directly
            py_test_file_contents=test_outputs_py,
            max_test_timeout_sec=int(timeout_sec),
            category=metadata.get("category"),
            difficulty=metadata.get("difficulty"),
            docker_image=docker_image,
            test_script=test_script,
            test_outputs_py=test_outputs_py,
        )

    def _extract_test_weights(self, test_outputs_py: str) -> dict[str, float]:
        """Extract test function names and assign equal weights.

        Args:
            test_outputs_py: Contents of test_outputs.py file.

        Returns:
            Dictionary mapping test names to weights (all 1.0).
        """
        import re

        weights = {}
        # Match "def test_xxx(" pattern
        pattern = r"def\s+(test_\w+)\s*\("
        for match in re.finditer(pattern, test_outputs_py):
            test_name = match.group(1)
            weights[test_name] = 1.0

        return weights

    def _find_sqsh_path(
        self,
        sqsh_dir: str | Path,
        task_name: str,
        docker_image: str = "",
    ) -> Path | None:
        """Find a matching sqsh image path for a harbor task.

        Args:
            sqsh_dir: Directory containing sqsh files.
            task_name: Name of the task.
            docker_image: Docker image name from task.toml.

        Returns:
            Path to a matching sqsh file, or None if not found.
        """
        for sqsh_path in self._get_sqsh_candidates(
            sqsh_dir=sqsh_dir,
            task_name=task_name,
            docker_image=docker_image,
        ):
            if self.sqsh_prefix and not sqsh_path.name.startswith(self.sqsh_prefix):
                continue
            if sqsh_path.exists():
                return sqsh_path
        return None

    def _get_sqsh_candidates(
        self,
        sqsh_dir: str | Path,
        task_name: str,
        docker_image: str = "",
    ) -> list[Path]:
        """Get candidate sqsh image paths for a harbor task.

        Candidate order:
        1) (Opt-in) Derived from docker_image when sqsh_prefix is set, e.g.
           alexgshaw/task:20251031 -> alexgshaw+task+20251031.sqsh
        2) Legacy task-name pattern (sqsh_pattern), e.g.
           ghcr.io+laude-institute+terminal-bench+{task_name}+2.0.sqsh
        """
        candidates: list[Path] = []
        sqsh_dir_path = Path(sqsh_dir)

        # Keep legacy behavior by default: only consider docker-image-derived
        # sqsh names when the caller explicitly sets a prefix filter.
        if self.sqsh_prefix and docker_image:
            docker_sqsh_name = f"{docker_image.replace('/', '+').replace(':', '+')}.sqsh"
            candidates.append(sqsh_dir_path / docker_sqsh_name)

        pattern_sqsh_name = self.sqsh_pattern.format(task_name=task_name)
        pattern_sqsh_path = sqsh_dir_path / pattern_sqsh_name
        if pattern_sqsh_path not in candidates:
            candidates.append(pattern_sqsh_path)

        return candidates
