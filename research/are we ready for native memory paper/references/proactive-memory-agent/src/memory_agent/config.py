"""Configuration models for MemoryAgent V3."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """LLM configuration for the action agent."""

    model_name: str = "openrouter/anthropic/claude-sonnet-4"
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_turns: int = 50
    parser_name: str = "xml"
    enable_summarize: bool = True
    record_terminal_session: bool = True
    truncation_pairs: int = Field(
        default=0,
        description="When context overflows, drop N oldest (assistant,user) pairs after the first turn. 0 = disabled (use summarization fallback).",
    )


class MemoryConfig(BaseModel):
    """V3 Memory Agent configuration.

    Design notes:
    - No max_knowledge_in_context / max_procedural_in_context (no fallback context)
    - trigger_interval=1 (every turn), sliding_window_size=8
    - Phase 2 surfaces facts only, Layer 1 requirements guard always-on
    - enabled=false creates a baseline agent without memory
    """

    enabled: bool = Field(
        default=True,
        description="Enable memory agent. Set to false for baseline runs.",
    )

    model_name: str = Field(
        default="openrouter/anthropic/claude-opus-4.6",
        description="Model for the Memory Agent",
    )
    api_base: str | None = Field(
        default=None,
        description="API base URL for the Memory Agent model",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for the Memory Agent model",
    )
    temperature: float = Field(
        default=0.3,
        description="Temperature for Memory Agent (lower = more consistent)",
    )
    max_thinking_tokens: int | None = Field(
        default=None,
        description="Enable thinking/reasoning for Anthropic models. Set to e.g. 4096 to enable.",
    )
    keep_thinking: bool = Field(
        default=False,
        description="Keep <think> tokens in raw responses. Set True for RL training (policy needs full output). "
                    "When False, thinking is stripped before tool call parsing and phase2 response parsing.",
    )

    # Trigger settings
    trigger_interval: int = Field(
        default=1,
        description="Run memory agent every N steps (V3 default: 1 = every turn)",
    )
    trigger_on_first_step: bool = Field(
        default=True,
        description="Always trigger on first step to capture task requirements",
    )

    # Sliding window for memory agent context
    sliding_window_size: int = Field(
        default=8,
        description="Memory agent sees task description + last N steps (selective attention window)",
    )

    # Memory persistence
    reset_status_on_episode: bool = Field(
        default=True,
        description="Reset status when starting new episode",
    )
    reset_knowledge_on_episode: bool = Field(
        default=False,
        description="Reset knowledge memories when starting new episode",
    )
    reset_procedural_on_episode: bool = Field(
        default=False,
        description="Reset procedural memories when starting new episode",
    )

    # Injection method
    injection_method: str = Field(
        default="system_prompt",
        description="Where to inject memory context: 'system_prompt' (append to system message) or 'user_turn' (append to last user message)",
    )

    use_structured_tools: bool = Field(
        default=True,
        description="Send tools schema to LLM for structured tool_calls. Set False to skip tools param and rely on fallback XML text parser.",
    )

    # Output
    save_memory_per_task: bool = Field(
        default=True,
        description="Save memory.json and trajectory_memory.json after each task",
    )


class EnrootConfig(BaseModel):
    """Enroot container runtime configuration."""

    sqsh_dir: str = Field(description="Path to directory containing pre-built .sqsh images")
    data_dir: str = Field(
        default="~/.local/share/enroot",
        description="Enroot data path for containers",
    )
    save_snapshots: bool = Field(
        default=False,
        description="Save container sqsh snapshots at each memory agent trigger step",
    )
    snapshot_dir: str | None = Field(
        default=None,
        description="Directory for container snapshots. Required if save_snapshots=true.",
    )


class RunConfig(BaseModel):
    """Evaluation run configuration."""

    adapter_type: Literal["parquet", "harbor"] = Field(
        default="parquet",
        description="Data adapter type",
    )
    data_path: str = Field(
        default="",
        description="Path to directory containing parquet files (for parquet adapter)",
    )
    tasks_dir: str | None = Field(
        default=None,
        description="Path to harbor tasks cache directory",
    )
    split: str = "val"
    task_names: list[str] | None = None
    harbor_sqsh_prefix: str | None = Field(
        default=None,
        description="Optional sqsh filename prefix filter for harbor adapter",
    )
    output_dir: str = "./outputs"
    n_parallel: int = Field(default=1, description="Number of tasks to run concurrently")
    resume_run_dir: str | None = Field(
        default=None,
        description="Resume an interrupted run: skip tasks that already have verifier/reward.txt",
    )


class MemoryAgentV3Config(BaseModel):
    """Top-level configuration for MemoryAgent V3."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    enroot: EnrootConfig
    run: RunConfig


def load_config(config_path: Path) -> MemoryAgentV3Config:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return MemoryAgentV3Config.model_validate(data)
