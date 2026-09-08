"""Configuration schema validation using Pydantic."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(..., description="OpenAI API key")
    base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API base URL"
    )
    model: str = Field(default="gpt-4o-mini", description="Model name to use")


class TokenizerConfig(BaseModel):
    """Tokenizer configuration."""

    model_id: str = Field(..., description="Tokenizer model ID or local path")


class APIModelRoleConfig(BaseModel):
    """OpenAI-compatible model configuration for a specific runtime role."""

    openai_config: OpenAIConfig = Field(..., description="OpenAI API configuration")


class MemoryAgentModelConfig(BaseModel):
    """Memory-agent configuration for both KV and text modes."""

    model_id: Optional[str] = Field(
        default=None,
        description="HuggingFace model ID or local path. Required for KV mode and "
        "optional for text mode when tokenizer.model_id is configured.",
    )
    openai_config: Optional[OpenAIConfig] = Field(
        default=None,
        description="OpenAI API configuration. Required for text-mode memory agents.",
    )
    model_context_window: int = Field(
        default=32768, ge=1024, le=131072, description="Model context window size"
    )
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = Field(
        default="sdpa", description="Attention implementation"
    )
    device_map: str = Field(default="auto", description="Device mapping strategy")
    quantization_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Quantization configuration"
    )


class ModelConfig(BaseModel):
    """Model configuration with a required memory model and general LLM defaults."""

    memory_agent_model: MemoryAgentModelConfig = Field(
        ..., description="Model configuration for memory agents"
    )
    general_model: APIModelRoleConfig = Field(
        ..., description="Default OpenAI-compatible model configuration for non-memory roles"
    )
    manager_model: Optional[APIModelRoleConfig] = Field(
        default=None,
        description="Optional override for the top-level manager/tool-calling model",
    )
    aggregator_model: Optional[APIModelRoleConfig] = Field(
        default=None,
        description="Optional override for the aggregation/summarization model",
    )
    question_answer_model: Optional[APIModelRoleConfig] = Field(
        default=None,
        description="Optional override for final question-answer generation",
    )
    router_fallback_model: Optional[APIModelRoleConfig] = Field(
        default=None,
        description="Optional override for the router or router-fallback model",
    )

    def get_general_openai_config(self) -> OpenAIConfig:
        """Return the default non-memory OpenAI configuration."""
        return self.general_model.openai_config

    def _get_role_override_or_general(
        self, role_config: Optional[APIModelRoleConfig]
    ) -> OpenAIConfig:
        """Return a role override when present, otherwise the general config."""
        if role_config is not None:
            return role_config.openai_config
        return self.get_general_openai_config()

    def get_memory_agent_config(self) -> MemoryAgentModelConfig:
        """Return the effective memory-agent configuration."""
        return self.memory_agent_model

    def get_manager_openai_config(self) -> OpenAIConfig:
        """Return the effective manager/chat model configuration."""
        return self._get_role_override_or_general(self.manager_model)

    def get_aggregator_openai_config(self) -> OpenAIConfig:
        """Return the effective aggregation model configuration."""
        return self._get_role_override_or_general(self.aggregator_model)

    def get_router_openai_config(self) -> OpenAIConfig:
        """Return the effective router model configuration."""
        return self._get_role_override_or_general(self.router_fallback_model)

    def get_question_answer_openai_config(self) -> OpenAIConfig:
        """Return the effective evaluation question-answer model configuration."""
        return self._get_role_override_or_general(self.question_answer_model)


class HybridRouterConfig(BaseModel):
    """Hybrid router configuration."""

    embedding_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface",
        description="Embedding model provider. 'huggingface' uses sentence-transformers, "
        "'openai' uses OpenAI-compatible embedding APIs.",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model name. Default: 'sentence-transformers/all-MiniLM-L6-v2' "
        "for HuggingFace, 'text-embedding-3-small' for OpenAI.",
    )
    embedding_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional embedding configuration (e.g., api_key, base_url for OpenAI).",
    )
    summary_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for summary embedding similarity score.",
    )
    text_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for text embedding similarity score.",
    )
    bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 keyword matching score.",
    )
    summary_top_k: int = Field(
        default=10, ge=1, description="Top-k summaries to consider for scoring."
    )
    text_top_k: int = Field(
        default=20, ge=1, description="Top-k text chunks to consider for scoring."
    )
    bm25_top_k: int = Field(
        default=10, ge=1, description="Top-k BM25 results to consider."
    )
    text_chunk_size: int = Field(
        default=512,
        ge=100,
        description="Maximum chunk size for text embedding (characters).",
    )
    text_chunk_overlap: int = Field(
        default=50, ge=0, description="Overlap between text chunks (characters)."
    )
    use_llm_fallback: bool = Field(
        default=False,
        description="Use LLM-based routing as fallback when embedding fails.",
    )
    bm25_use_jieba: bool = Field(
        default=True,
        description="Use jieba tokenizer for Chinese text support in BM25. "
        "Set to False if your corpus is English-only for slightly better performance.",
    )
    bm25_boost_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="If a block's normalized BM25 score exceeds this threshold (0-1), "
        "it will be auto-selected regardless of combined score. "
        "Useful for exact keyword matches. Set to None to disable. "
        "Recommended: 0.7-0.9 for strict keyword matching.",
    )


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    storage_mode: Literal["kv_cache", "text"] = Field(
        default="kv_cache", description="Storage backend mode"
    )
    clean_cache_first: bool = Field(
        default=True, description="Whether to clean cache on initialization"
    )
    router_system_prompt: Optional[str] = Field(
        default=None, description="Custom router system prompt"
    )
    overlap_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Overlap ratio between KV blocks (0.0-1.0)",
    )
    overlap_mode: Literal["chunk", "token"] = Field(
        default="chunk", description="Overlap handling strategy"
    )
    block_size_ratio: float = Field(
        default=0.125,
        gt=0.0,
        le=1.0,
        description="Block size ratio relative to model context window (0.0-1.0)",
    )
    max_concurrent_gpu_operations: int = Field(
        default=2, ge=1, le=8, description="Number of concurrent GPU operations"
    )
    max_memory_segments: int = Field(
        default=5, ge=1, description="Maximum number of memory segments to return"
    )
    max_blocks: int = Field(
        default=5, ge=1, description="Maximum number of memory blocks for router"
    )
    query_batch_size: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of queries to batch together for inference. "
        "Higher values improve throughput but require more GPU memory. "
        "Set to 2 for single GPU with limited memory, 4-8 for multi-GPU setups.",
    )
    max_parallel_cache_loads: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum number of KV caches to load to GPU in parallel. "
        "Adjust based on available GPU memory.",
    )
    kv_cache_on_gpu: bool = Field(
        default=False,
        description="Keep inactive KV caches on GPU for faster queries. "
        "Only enable if you have sufficient GPU memory (e.g., multiple A100s).",
    )
    enable_router: bool = Field(
        default=True,
        description="Enable router for selecting relevant blocks. "
        "Set to False to query ALL blocks directly (useful for evaluation/debugging).",
    )
    router_type: Literal["llm", "hybrid"] = Field(
        default="hybrid",
        description="Router type: 'llm' for LLM-based routing (legacy), "
        "'hybrid' for embedding + BM25 hybrid routing (recommended).",
    )
    hybrid_router: HybridRouterConfig = Field(
        default_factory=HybridRouterConfig,
        description="Configuration for hybrid router (only used when router_type='hybrid').",
    )

    @field_validator("overlap_ratio")
    @classmethod
    def validate_overlap_ratio(cls, v: float) -> float:
        """Validate overlap ratio is reasonable."""
        if v > 0.5:
            raise ValueError(
                "overlap_ratio should not exceed 0.5 to maintain block efficiency"
            )
        return v


class LocomoEvalConfig(BaseModel):
    """LoComo evaluation configuration."""

    dataset_path: str = Field(..., description="Path to dataset JSON file")
    output_dir: str = Field(
        default="evaluation/locomo/results", description="Output directory for results"
    )
    ratio: float = Field(
        default=1.0, gt=0.0, le=1.0, description="Portion of dataset to evaluate"
    )
    specific_questions_path: Optional[str] = Field(
        default=None, description="Path to specific questions JSON file"
    )
    conversation_auto_save: bool = Field(
        default=True, description="Enable auto-save for conversation turns"
    )
    categories: List[int] = Field(
        default=[1, 2, 3, 4, 5], description="Categories to evaluate"
    )

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: List[int]) -> List[int]:
        """Validate categories are in valid range."""
        for cat in v:
            if cat < 1 or cat > 5:
                raise ValueError(f"Category {cat} out of valid range [1-5]")
        return v


class HotpotQAEvalConfig(BaseModel):
    """HotpotQA evaluation configuration."""

    dataset_path: str = Field(..., description="Path to dataset JSON file")
    output_dir: str = Field(
        default="evaluation/hotpotqa/results",
        description="Output directory for results",
    )
    ratio: float = Field(
        default=1.0, gt=0.0, le=1.0, description="Portion of dataset to evaluate"
    )
    max_tokens_per_chunk: int = Field(
        default=2048, ge=256, description="Maximum tokens per context chunk"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_dir: str = Field(
        default="evaluation/locomo/logs", description="Directory for log files"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )


class MaxMemoryConfig(BaseModel):
    """GPU memory configuration per device."""

    model_config = {"extra": "allow"}


class AppConfig(BaseModel):
    """Main application configuration."""

    tokenizer: Optional[TokenizerConfig] = Field(
        default=None, description="Tokenizer configuration"
    )
    model: ModelConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    max_memory: Optional[Dict[str, str]] = Field(
        default=None, description="Max memory per GPU device"
    )
    locomo_eval: Optional[LocomoEvalConfig] = Field(
        default=None, description="LoComo evaluation config"
    )
    hotpotqa_eval: Optional[HotpotQAEvalConfig] = Field(
        default=None, description="HotpotQA evaluation config"
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def get_tokenizer_model_id(self) -> Optional[str]:
        """Return the effective tokenizer model ID."""
        if self.tokenizer is not None:
            return self.tokenizer.model_id
        return self.model.get_memory_agent_config().model_id

    def get_model_label(self) -> str:
        """Return a human-readable model label for logging/results."""
        model_id = self.get_tokenizer_model_id()
        if model_id:
            return model_id
        qa_config = self.model.get_question_answer_openai_config()
        if qa_config is not None:
            return qa_config.model
        raise ValueError("No effective model label could be resolved from the config.")

    def get_runtime_model_summary(self) -> Dict[str, Optional[str]]:
        """Return a structured summary of all effective model roles."""
        memory_agent = self.model.get_memory_agent_config()
        manager_config = self.model.get_manager_openai_config()
        aggregator_config = self.model.get_aggregator_openai_config()
        router_config = self.model.get_router_openai_config()
        question_answer_config = self.model.get_question_answer_openai_config()

        return {
            "storage_mode": self.memory.storage_mode,
            "tokenizer_model": self.get_tokenizer_model_id(),
            "memory_agent_model": (
                memory_agent.model_id
                or (
                    memory_agent.openai_config.model
                    if memory_agent.openai_config is not None
                    else None
                )
            ),
            "general_model": self.model.get_general_openai_config().model,
            "manager_model": (
                manager_config.model if manager_config is not None else None
            ),
            "aggregator_model": (
                aggregator_config.model if aggregator_config is not None else None
            ),
            "router_fallback_model": (
                router_config.model if router_config is not None else None
            ),
            "question_answer_model": (
                question_answer_config.model
                if question_answer_config is not None
                else None
            ),
        }

    def to_chat_manager_kwargs(self) -> Dict[str, Any]:
        """Build runtime kwargs for create_chat_manager()."""
        memory_agent = self.model.get_memory_agent_config()
        manager_config = self.model.get_manager_openai_config()
        aggregator_config = self.model.get_aggregator_openai_config()
        router_config = self.model.get_router_openai_config()

        kwargs: Dict[str, Any] = {
            "storage_mode": self.memory.storage_mode,
            "model_id": self.get_tokenizer_model_id(),
            "chat_openai_config": manager_config.model_dump(),
            "aggregator_openai_config": aggregator_config.model_dump(),
            "router_openai_config": router_config.model_dump(),
            "clean_cache_first": self.memory.clean_cache_first,
            "model_context_window": memory_agent.model_context_window,
            "router_system_prompt": self.memory.router_system_prompt,
            "overlap_mode": self.memory.overlap_mode,
            "overlap_ratio": self.memory.overlap_ratio,
            "block_size_ratio": self.memory.block_size_ratio,
            "max_memory": self.max_memory,
            "max_memory_segments": self.memory.max_memory_segments,
            "max_blocks": self.memory.max_blocks,
            "query_batch_size": self.memory.query_batch_size,
            "max_parallel_cache_loads": self.memory.max_parallel_cache_loads,
            "enable_router": self.memory.enable_router,
            "router_type": self.memory.router_type,
            "hybrid_router_config": self.memory.hybrid_router.model_dump(),
        }

        if self.memory.storage_mode == "text":
            kwargs["memory_agent_openai_config"] = (
                memory_agent.openai_config.model_dump()
                if memory_agent.openai_config is not None
                else None
            )
        else:
            kwargs["attn_implementation"] = memory_agent.attn_implementation
            kwargs["device_map"] = memory_agent.device_map
            kwargs["quantization_config"] = memory_agent.quantization_config

        return kwargs

    @model_validator(mode="after")
    def validate_config_consistency(self) -> "AppConfig":
        """Validate configuration consistency across sections."""
        total_ratio = self.memory.block_size_ratio + self.memory.overlap_ratio
        if total_ratio > 0.9:
            raise ValueError(
                f"block_size_ratio ({self.memory.block_size_ratio}) + "
                f"overlap_ratio ({self.memory.overlap_ratio}) = {total_ratio} "
                "exceeds 0.9, which may cause memory issues"
            )

        tokenizer_model_id = self.get_tokenizer_model_id()
        memory_agent = self.model.get_memory_agent_config()

        if self.memory.storage_mode == "kv_cache":
            if memory_agent.model_id is None:
                raise ValueError(
                    "KV cache mode requires model.memory_agent_model.model_id."
                )
        else:
            if tokenizer_model_id is None:
                raise ValueError(
                    "Text mode requires tokenizer.model_id."
                )
            if memory_agent.openai_config is None:
                raise ValueError(
                    "Text mode requires model.memory_agent_model.openai_config."
                )

        return self


def load_and_validate_config(config_dict: Dict[str, Any]) -> AppConfig:
    """
    Load and validate configuration from a dictionary.

    Args:
        config_dict: Configuration dictionary (typically from YAML)

    Returns:
        Validated AppConfig instance

    Raises:
        ValidationError: If configuration is invalid
    """
    return AppConfig.model_validate(config_dict)


def validate_openai_config(config_dict: Dict[str, Any]) -> OpenAIConfig:
    """Validate OpenAI configuration standalone."""
    return OpenAIConfig.model_validate(config_dict)


def validate_memory_config(config_dict: Dict[str, Any]) -> MemoryConfig:
    """Validate memory configuration standalone."""
    return MemoryConfig.model_validate(config_dict)
