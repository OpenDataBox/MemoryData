from abc import ABC, abstractmethod

from memos.configs.parser import BaseParserConfig


class BaseParser(ABC):
    """Base class for all parsers."""

    @abstractmethod
    def __init__(self, config: BaseParserConfig):
        """Initialize the parser with the given configuration."""

    @abstractmethod
    def parse(self, file_path: str) -> str:
        """Parse the file at the given path and return its content as a string."""
