from abc import abstractmethod
from typing import Any

from memos.configs.memory import BaseActMemoryConfig
from memos.memories.base import BaseMemory


class BaseActMemory(BaseMemory):
    @abstractmethod
    def __init__(self, config: BaseActMemoryConfig) -> None:
        """Initialize the activation memory with a configuration."""

    @abstractmethod
    def extract(self, text: str) -> Any:
        """Extract memory based on the texts."""

    @abstractmethod
    def add(self, memories: list) -> None:
        """Add memories."""

    @abstractmethod
    def get(self, memory_id: str) -> Any | None:
        """Get a memory by its ID."""

    @abstractmethod
    def get_by_ids(self, memory_ids: list[str]) -> list[Any | None]:
        """Get memories by their IDs."""

    @abstractmethod
    def get_all(self) -> list[Any]:
        """Get all memories."""

    @abstractmethod
    def delete(self, memory_ids: list[str]) -> None:
        """Delete memories.
        Args:
            memory_ids (list[str]): List of memory IDs to delete.
        """

    @abstractmethod
    def delete_all(self) -> None:
        """Delete all memories."""
