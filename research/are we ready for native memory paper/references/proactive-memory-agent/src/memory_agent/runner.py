"""Main evaluation runner for MemoryAgent V3.

Orchestrates: load task -> create enroot container -> run memory-enabled agent -> compute reward.
Always runs memory-enabled mode (this is V3, not a baseline runner).
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from harbor.environments.enroot.enroot import EnrootEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from .config import MemoryAgentV3Config
from .data import Task, create_adapter
from .utils import visualize_trajectory

logger = logging.getLogger(__name__)


class TaskResult(BaseModel):
    """Result of running a single task."""

    task_name: str
    reward: float
    n_input_tokens: int | None = None
    n_output_tokens: int | None = None
    cost_usd: float | None = None
    duration_sec: float = 0.0
    error: str | None = None
    trajectory_path: str | None = None
    memory_path: str | None = None
    memory_stats: dict | None = None
    test_output: str | None = None


def create_agent(config: MemoryAgentV3Config, logs_dir: Path):
    """Create agent — baseline (Terminus2) or memory-enabled (MemoryEnabledTerminus2)."""
    common_kwargs = dict(
        logs_dir=logs_dir,
        model_name=config.model.model_name,
        api_base=config.model.api_base,
        api_key=config.model.api_key,
        parser_name=config.model.parser_name,
        max_turns=config.model.max_turns,
        temperature=config.model.temperature,
        enable_summarize=config.model.enable_summarize,
        record_terminal_session=config.model.record_terminal_session,
        truncation_pairs=getattr(config.model, "truncation_pairs", 0),
    )

    if not config.memory.enabled:
        from harbor.agents.terminus_2.terminus_2 import Terminus2
        logger.info("Creating baseline agent (no memory)")
        return Terminus2(**common_kwargs)

    from .memory_enabled_agent import MemoryEnabledTerminus2
    logger.info("Creating memory-enabled agent")
    return MemoryEnabledTerminus2(
        **common_kwargs,
        memory_model_name=config.memory.model_name,
        memory_api_base=config.memory.api_base,
        memory_api_key=config.memory.api_key,
        memory_temperature=config.memory.temperature,
        memory_max_thinking_tokens=getattr(config.memory, "max_thinking_tokens", None),
        memory_keep_thinking=getattr(config.memory, "keep_thinking", False),
        trigger_interval=config.memory.trigger_interval,
        trigger_on_first_step=config.memory.trigger_on_first_step,
        sliding_window_size=config.memory.sliding_window_size,
        reset_status_on_episode=config.memory.reset_status_on_episode,
        reset_knowledge_on_episode=config.memory.reset_knowledge_on_episode,
        reset_procedural_on_episode=config.memory.reset_procedural_on_episode,
        injection_method=getattr(config.memory, "injection_method", "system_prompt"),
        memory_use_structured_tools=getattr(config.memory, "use_structured_tools", True),
    )


async def run_task(task: Task, config: MemoryAgentV3Config, run_output_dir: Path) -> TaskResult:
    """Run a single task through the full pipeline."""
    start_time = time.time()
    task_output_dir = run_output_dir / task.task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    trial_paths = TrialPaths(trial_dir=task_output_dir)
    trial_paths.mkdir()

    docker_image = task.docker_image if task.docker_image else task.task_name
    task_env_config = EnvironmentConfig(docker_image=docker_image)

    environment = EnrootEnvironment(
        environment_dir=task_output_dir,
        environment_name=task.task_name,
        session_id=f"{task.task_name}",
        trial_paths=trial_paths,
        task_env_config=task_env_config,
        sqsh_dir=config.enroot.sqsh_dir,
        data_dir=str(Path(config.enroot.data_dir).expanduser()),
    )

    # Keep container alive so we can export before deletion
    environment._keep_containers = True

    context = AgentContext()
    memory_path = None
    memory_stats = None

    max_time_s = getattr(config.run, 'max_time_s', 2400)  # default 40 min

    try:
        logger.info(f"[{task.task_name}] Starting enroot environment...")
        await environment.start(force_build=False)

        agent = create_agent(config, trial_paths.agent_dir)

        # Enable container snapshots if configured
        if getattr(config.enroot, 'save_snapshots', False) and hasattr(agent, '_snapshot_dir'):
            snapshot_dir = task_output_dir / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            agent._snapshot_dir = snapshot_dir
            agent._environment = environment

        logger.info(f"[{task.task_name}] Setting up agent...")
        await agent.setup(environment)

        logger.info(f"[{task.task_name}] Running V3 memory-enabled agent (timeout={max_time_s}s)...")
        import asyncio as _asyncio_timeout
        try:
            await _asyncio_timeout.wait_for(
                agent.run(task.instruction, environment, context),
                timeout=max_time_s,
            )
        except _asyncio_timeout.TimeoutError:
            logger.warning(f"[{task.task_name}] Task timed out after {max_time_s}s")

        # Save memory bank and trajectory
        if config.memory.save_memory_per_task and hasattr(agent, "save_memory"):
            memory_path = str(task_output_dir / "memory.json")
            agent.save_memory(memory_path)
            try:
                from .utils import visualize_memory, visualize_memory_trajectory

                visualize_memory(Path(memory_path))
                memory_traj_path = task_output_dir / "trajectory_memory.json"
                if memory_traj_path.exists():
                    visualize_memory_trajectory(memory_traj_path)
            except Exception as e:
                logger.warning(f"[{task.task_name}] Failed to visualize memory: {e}")

        if hasattr(agent, "get_memory_stats"):
            memory_stats = agent.get_memory_stats()

        # Read reward from container — harbor's test.sh writes it to /logs/verifier/reward.txt
        logger.info(f"[{task.task_name}] Reading reward from container...")
        reward_result = await environment.exec(
            "cat /logs/verifier/reward.txt 2>/dev/null || echo '0'",
            timeout_sec=60,
        )
        try:
            reward = float((reward_result.stdout or "0").strip())
        except ValueError:
            reward = 0.0

        test_output_result = await environment.exec(
            "cat /logs/verifier/test_output.log 2>/dev/null || echo ''",
            timeout_sec=60,
        )
        test_output = test_output_result.stdout or ""

        verifier_dir = task_output_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        (verifier_dir / "test_output.txt").write_text(test_output)
        (verifier_dir / "reward.txt").write_text(str(reward))
        logger.info(f"[{task.task_name}] Reward: {reward:.4f}")

        duration = time.time() - start_time

        # Find and visualize trajectory
        traj_path = None
        traj_files = list(trial_paths.agent_dir.glob("trajectory*.json"))
        if traj_files:
            traj_path = str(traj_files[0])
            try:
                visualize_trajectory(Path(traj_path))
            except Exception as e:
                logger.warning(f"[{task.task_name}] Failed to visualize trajectory: {e}")

        # Move artifacts to task_output_dir for easier access
        import shutil

        for pattern in [
            "trajectory*.json",
            "trajectory*.html",
            "*_memory*.json",
            "*_memory*.html",
        ]:
            for f in trial_paths.agent_dir.glob(pattern):
                dest = task_output_dir / f.name
                try:
                    shutil.move(str(f), str(dest))
                    if f.name.startswith("trajectory") and f.suffix == ".json":
                        traj_path = str(dest)
                except Exception as e:
                    logger.warning(f"[{task.task_name}] Failed to move {f.name}: {e}")

        memory_info = ""
        if memory_stats:
            memory_info = (
                f", memory={memory_stats.get('memory_size', 0)} entries"
                f", injections={memory_stats.get('injection_count', 0)}"
                f", noops={memory_stats.get('noop_count', 0)}"
            )
        logger.info(
            f"[{task.task_name}] Done: reward={reward:.4f}, "
            f"duration={duration:.1f}s, cost=${context.cost_usd or 0:.4f}{memory_info}"
        )

        return TaskResult(
            task_name=task.task_name,
            reward=reward,
            n_input_tokens=context.n_input_tokens,
            n_output_tokens=context.n_output_tokens,
            cost_usd=context.cost_usd,
            duration_sec=duration,
            trajectory_path=traj_path,
            memory_path=memory_path,
            memory_stats=memory_stats,
            test_output=test_output,
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"[{task.task_name}] Error: {e}", exc_info=True)
        return TaskResult(
            task_name=task.task_name,
            reward=0.0,
            duration_sec=duration,
            error=str(e),
        )
    finally:
        # Export container rootfs as squashfs for potential retesting
        try:
            container_name = environment._container_name
            if container_name:
                sqsh_path = task_output_dir / "container.sqsh"
                logger.info(f"[{task.task_name}] Exporting container to {sqsh_path}...")
                import asyncio as _asyncio
                proc = await _asyncio.create_subprocess_exec(
                    "enroot", "export", "-f", "-o", str(sqsh_path), container_name,
                    stdout=_asyncio.subprocess.PIPE,
                    stderr=_asyncio.subprocess.PIPE,
                    env={**__import__('os').environ, "ENROOT_DATA_PATH": str(environment._data_dir)},
                )
                _, stderr = await _asyncio.wait_for(proc.communicate(), timeout=300)
                if proc.returncode == 0:
                    logger.info(f"[{task.task_name}] Container exported ({sqsh_path.stat().st_size / 1e9:.1f} GB)")
                else:
                    logger.warning(f"[{task.task_name}] Container export failed: {stderr.decode()[:200]}")
        except Exception as e:
            logger.warning(f"[{task.task_name}] Container export error: {e}")

        try:
            environment._keep_containers = False  # Allow deletion now
            await environment.stop(delete=True)
        except Exception as e:
            logger.warning(f"[{task.task_name}] Error stopping environment: {e}")


async def run_evaluation(
    config: MemoryAgentV3Config, run_id: str | None = None
) -> list[TaskResult]:
    """Run the full evaluation pipeline."""
    adapter = create_adapter(
        adapter_type=config.run.adapter_type,
        data_path=config.run.data_path,
        split=config.run.split,
        tasks_dir=config.run.tasks_dir,
        sqsh_prefix=config.run.harbor_sqsh_prefix,
    )
    tasks = adapter.load_tasks(
        task_names=config.run.task_names,
        sqsh_dir=config.enroot.sqsh_dir,
    )

    if not tasks:
        logger.warning("No tasks found to evaluate")
        return []

    logger.info(f"Running V3 memory evaluation on {len(tasks)} tasks")
    logger.info(f"Action Model: {config.model.model_name}")
    logger.info(f"Memory Model: {config.memory.model_name}")
    logger.info(f"Trigger interval: {config.memory.trigger_interval}")

    # Resume mode
    if config.run.resume_run_dir:
        run_output_dir = Path(config.run.resume_run_dir)
        completed_tasks = {
            p.parent.parent.name for p in run_output_dir.glob("*/verifier/reward.txt")
        }
        original_count = len(tasks)
        tasks = [t for t in tasks if t.task_name not in completed_tasks]
        logger.info(
            f"Resuming {run_output_dir.name}: skipping {len(completed_tasks)} completed, "
            f"{len(tasks)}/{original_count} remaining"
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = Path(config.run.output_dir) / f"run_{run_id}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output dir: {run_output_dir}")

    n_parallel = getattr(config.run, "n_parallel", 1)
    semaphore = asyncio.Semaphore(n_parallel)
    completed = 0

    async def run_with_semaphore(i: int, task) -> TaskResult:
        nonlocal completed
        async with semaphore:
            logger.info(f"--- Task {i+1}/{len(tasks)}: {task.task_name} ---")
            result = await run_task(task, config, run_output_dir)
            completed += 1
            logger.info(f"Progress: {completed}/{len(tasks)} done (reward={result.reward:.2f})")
            return result

    results = await asyncio.gather(
        *(run_with_semaphore(i, task) for i, task in enumerate(tasks))
    )

    # Save aggregate results
    results_path = run_output_dir / "results.json"
    memory_stats_list = [r.memory_stats for r in results if r.memory_stats]

    summary = {
        "mode": "memory-v3-reflective",
        "n_tasks": len(results),
        "avg_reward": sum(r.reward for r in results) / len(results) if results else 0,
        "total_cost_usd": sum(r.cost_usd or 0 for r in results),
        "total_duration_sec": sum(r.duration_sec for r in results),
        "n_errors": sum(1 for r in results if r.error),
    }
    if memory_stats_list:
        summary["memory"] = {
            "total_memories": sum(s.get("memory_size", 0) for s in memory_stats_list),
            "total_triggers": sum(s.get("trigger_count", 0) for s in memory_stats_list),
            "total_operations": sum(s.get("operation_count", 0) for s in memory_stats_list),
            "total_injections": sum(s.get("injection_count", 0) for s in memory_stats_list),
            "total_noops": sum(s.get("noop_count", 0) for s in memory_stats_list),
        }

    results_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(),
        "summary": summary,
        "tasks": [r.model_dump() for r in results],
    }
    results_path.write_text(json.dumps(results_data, indent=2, default=str))
    logger.info(f"Results saved to {results_path}")

    avg_reward = summary["avg_reward"]
    total_cost = summary["total_cost_usd"]
    n_errors = summary["n_errors"]
    logger.info(
        f"Summary: avg_reward={avg_reward:.4f}, "
        f"total_cost=${total_cost:.4f}, errors={n_errors}/{len(results)}"
    )

    return results
