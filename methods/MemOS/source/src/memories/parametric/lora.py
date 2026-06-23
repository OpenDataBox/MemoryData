################################################################
# TODO:
# This file currently serves as a placeholder.
# The actual implementation will be added here in the future.
# Please do not use this as a functional module yet.
################################################################

import os

from memos.configs.memory import LoRAMemoryConfig
from memos.memories.parametric.base import BaseParaMemory


class LoRAMemory(BaseParaMemory):
    """
    LoRA Memory for parametric memories.
    This memory type is designed to store and retrieve low-rank adaptation (LoRA) parameters.
    """

    def __init__(self, config: LoRAMemoryConfig) -> None:
        """Initialize the LoRA Memory with a configuration."""
        self.config = config

    def load(self, dir: str) -> None:
        """Load memories from os.path.join(dir, self.config.memory_filename)

        Args:
            dir (str): The directory containing the memory files.
        """

    def dump(self, dir: str) -> None:
        """Dump memories to os.path.join(dir, self.config.memory_filename)

        Args:
            dir (str): The directory where the memory files will be saved.
        """
        path = os.path.join(dir, self.config.memory_filename)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"Placeholder")
