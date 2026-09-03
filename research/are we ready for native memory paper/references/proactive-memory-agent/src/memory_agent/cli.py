"""CLI entrypoint for MemoryAgent V3."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import memory_agent

from .config import load_config
from .data import create_adapter
from .runner import run_evaluation

app = typer.Typer(name="memory-agent-v3", help="MemoryAgent V3 — Reflective Memory for Terminal-Bench.")
console = Console()


def _setup_logging(verbose: bool = False, log_dir: Path | None = None, run_id: str | None = None):
    """Setup logging to both console and file."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler()]
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"run_{run_id}.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    if log_dir:
        logging.info(f"Logging to: {log_dir / f'run_{run_id}.log'}")


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to YAML config file"),
    task_names: Optional[list[str]] = typer.Option(
        None, "--task", "-t", help="Specific task names to run (can be repeated)"
    ),
    resume_run_dir: Optional[Path] = typer.Option(
        None, "--resume-run-dir", "-r", help="Resume an interrupted run"
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Override run ID (e.g. SLURM_JOB_ID)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    log_dir: Optional[Path] = typer.Option(
        Path("logs"), "--log-dir", "-l", help="Directory to save logs"
    ),
):
    """Run V3 memory-enabled evaluation on terminal-bench tasks."""
    from datetime import datetime

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    _setup_logging(verbose, log_dir=log_dir, run_id=run_id)

    config = load_config(config_path)

    if task_names:
        config.run.task_names = task_names
    if resume_run_dir:
        config.run.resume_run_dir = str(resume_run_dir)

    console.print("[bold]Running MemoryAgent V3 (reflective) evaluation[/bold]")
    console.print(f"  Config: {config_path}")
    console.print(f"  Action Model: {config.model.model_name}")
    console.print(f"  Memory Model: {config.memory.model_name}")
    console.print(f"  Trigger interval: {config.memory.trigger_interval}")
    console.print(f"  Split: {config.run.split}")

    results = asyncio.run(run_evaluation(config, run_id=run_id))

    if not results:
        console.print("[yellow]No results[/yellow]")
        return

    # Print results table
    table = Table(title="MemoryAgent V3 Results")
    table.add_column("Task", style="cyan")
    table.add_column("Reward", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Mem", justify="right")
    table.add_column("Inj/Nop", justify="right")
    table.add_column("Status")

    for r in results:
        status = "[red]ERROR[/red]" if r.error else "[green]OK[/green]"
        mem_count = r.memory_stats.get("memory_size", 0) if r.memory_stats else 0
        inj_count = r.memory_stats.get("injection_count", 0) if r.memory_stats else 0
        nop_count = r.memory_stats.get("noop_count", 0) if r.memory_stats else 0
        table.add_row(
            r.task_name,
            f"{r.reward:.4f}",
            f"{r.duration_sec:.1f}s",
            f"${r.cost_usd or 0:.4f}",
            f"{mem_count}",
            f"{inj_count}/{nop_count}",
            status,
        )

    console.print(table)

    avg_reward = sum(r.reward for r in results) / len(results)
    total_inj = sum((r.memory_stats or {}).get("injection_count", 0) for r in results)
    total_nop = sum((r.memory_stats or {}).get("noop_count", 0) for r in results)
    console.print(
        f"\n[bold]Average reward: {avg_reward:.4f}  |  "
        f"Total injections: {total_inj}  |  Total no-ops: {total_nop}[/bold]"
    )


@app.command("list-tasks")
def list_tasks_cmd(
    adapter_type: str = typer.Option(
        "parquet", "--adapter-type", "-a", help="Data adapter type"
    ),
    data_path: Optional[Path] = typer.Option(
        None, "--data-path", "-d", help="Path to parquet files"
    ),
    tasks_dir: Optional[Path] = typer.Option(
        None, "--tasks-dir", help="Path to harbor tasks directory"
    ),
    split: str = typer.Option("val", "--split", "-s", help="Dataset split"),
    sqsh_dir: Optional[Path] = typer.Option(
        None, "--sqsh-dir", help="Filter to tasks with pre-built sqsh images"
    ),
    sqsh_prefix: Optional[str] = typer.Option(
        None, "--sqsh-prefix", help="sqsh filename prefix filter"
    ),
):
    """List available tasks in the dataset."""
    _setup_logging()

    adapter = create_adapter(
        adapter_type=adapter_type,
        data_path=str(data_path) if data_path else ".",
        split=split,
        tasks_dir=str(tasks_dir) if tasks_dir else None,
        sqsh_prefix=sqsh_prefix,
    )
    tasks = adapter.load_tasks(sqsh_dir=str(sqsh_dir) if sqsh_dir else None)

    table = Table(title=f"Tasks ({split} split)")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Task Name", style="cyan")
    table.add_column("Category")
    table.add_column("Difficulty")

    for i, task in enumerate(tasks, 1):
        table.add_row(str(i), task.task_name, task.category or "—", task.difficulty or "—")

    console.print(table)
    console.print(f"\nTotal: {len(tasks)} tasks")


def main():
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
