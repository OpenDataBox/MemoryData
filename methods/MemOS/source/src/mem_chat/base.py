from abc import ABC, abstractmethod

from memos.configs.mem_chat import BaseMemChatConfig
from memos.mem_cube.base import BaseMemCube


class BaseMemChat(ABC):
    """Base class for all MemChat."""

    @abstractmethod
    def __init__(self, config: BaseMemChatConfig):
        """Initialize the MemChat with the given configuration."""

    @property
    @abstractmethod
    def mem_cube(self) -> BaseMemCube:
        """The memory cube associated with this MemChat."""

    @mem_cube.setter
    @abstractmethod
    def mem_cube(self, value: BaseMemCube) -> None:
        """The memory cube associated with this MemChat."""

    @abstractmethod
    def run(self) -> None:
        """Run the MemChat.

        This `run` method can represent the core logic of a MemChat.
        It could be an iterative chat process.
        """
