from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Base class for all memory implementations."""

    @abstractmethod
    def load(self, dir: str) -> None:
        """Load memories from os.path.join(dir, self.config.memory_filename)
        Args:
            dir (str): The directory containing the memory files.
        """

    @abstractmethod
    def dump(self, dir: str) -> None:
        """Dump memories to os.path.join(dir, self.config.memory_filename)
        Args:
            dir (str): The directory where the memory files will be saved.
        """
