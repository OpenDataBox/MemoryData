from typing import Any, ClassVar

from memos.configs.mem_chat import MemChatConfigFactory
from memos.mem_chat.base import BaseMemChat
from memos.mem_chat.simple import SimpleMemChat


class MemChatFactory(BaseMemChat):
    """Factory class for creating MemChat instances."""

    backend_to_class: ClassVar[dict[str, Any]] = {
        "simple": SimpleMemChat,
    }

    @classmethod
    def from_config(cls, config_factory: MemChatConfigFactory) -> BaseMemChat:
        backend = config_factory.backend
        if backend not in cls.backend_to_class:
            raise ValueError(f"Invalid backend: {backend}")
        mem_chat_class = cls.backend_to_class[backend]
        return mem_chat_class(config_factory.config)
