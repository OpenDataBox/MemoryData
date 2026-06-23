from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from memos.configs.base import BaseConfig


class BaseParserConfig(BaseConfig):
    """Base configuration class for parser models."""


class MarkItDownParserConfig(BaseParserConfig):
    pass


class ParserConfigFactory(BaseConfig):
    """Factory class for creating Parser configurations."""

    backend: str = Field(..., description="Backend for parser")
    config: dict[str, Any] = Field(..., description="Configuration for the parser backend")

    backend_to_class: ClassVar[dict[str, Any]] = {
        "markitdown": MarkItDownParserConfig,
    }

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, backend: str) -> str:
        """Validate the backend field."""
        if backend not in cls.backend_to_class:
            raise ValueError(f"Invalid backend: {backend}")
        return backend

    @model_validator(mode="after")
    def create_config(self) -> "ParserConfigFactory":
        config_class = self.backend_to_class[self.backend]
        self.config = config_class(**self.config)
        return self
