import uuid

from datetime import datetime
from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from memos.configs.base import BaseConfig
from memos.configs.llm import LLMConfigFactory


class BaseMemChatConfig(BaseConfig):
    """Base configuration class for MemChat."""

    user_id: str = Field(..., description="User ID for the MemChat")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Session ID for the MemChat"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp for the MemChat",
    )
    config_filename: str = Field(
        default="config.json",
        description="Filename for storing the MemChat configuration",
    )


class SimpleMemChatConfig(BaseMemChatConfig):
    """Simple MemChat configuration class."""

    chat_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM configuration for the MemChat",
    )
    max_turns_window: int = Field(
        default=15,
        description="Maximum number of turns to keep in the conversation history",
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of memories to retrieve for each query",
    )
    enable_textual_memory: bool = Field(
        default=False,
        description="Enable textual memory for the MemChat",
    )
    enable_activation_memory: bool = Field(
        default=False,
        description="Enable activation memory for the MemChat",
    )
    enable_parametric_memory: bool = Field(
        default=False,
        description="Enable parametric memory for the MemChat",
    )


class MemChatConfigFactory(BaseConfig):
    """Factory class for creating MemChat configurations."""

    backend: str = Field(..., description="Backend for MemChat")
    config: dict[str, Any] = Field(..., description="Configuration for the MemChat backend")

    backend_to_class: ClassVar[dict[str, Any]] = {
        "simple": SimpleMemChatConfig,
    }

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, backend: str) -> str:
        """Validate the backend field."""
        if backend not in cls.backend_to_class:
            raise ValueError(f"Invalid backend: {backend}")
        return backend

    @model_validator(mode="after")
    def create_config(self) -> "MemChatConfigFactory":
        config_class = self.backend_to_class[self.backend]
        self.config = config_class(**self.config)
        return self
