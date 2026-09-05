"""Comprehensive verification of Text Storage implementation."""
import os
import uuid
from datetime import datetime

import pytest

from src.conversation_manager.chat_handler import ChatManager, TextStorageChatManager
from src.conversation_manager.factory import create_chat_manager
from src.memory.core.text_loop_handler import (
    TextAddHandler,
    TextMemoryHandler,
    TextQueryHandler,
)
from src.memory.kv_block_manager.text_block import TextBlock, clear_text_cache
from src.memory.memory_agent.text_agent import TextMemoryAgent


def test_all_imports():
    """Test that all modules can be imported."""
    assert TextBlock is not None
    assert TextMemoryAgent is not None
    assert TextMemoryHandler is not None
    assert TextAddHandler is not None
    assert TextQueryHandler is not None
    assert create_chat_manager is not None
    assert ChatManager is not None
    assert TextStorageChatManager is not None


def test_text_block_comprehensive():
    """Test TextBlock comprehensive functionality."""
    block = TextBlock(
        block_id=uuid.uuid4(),
        create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        block_size=1000
    )
    
    # Add chunks
    block.add_chunk("First memory", 100)
    block.add_chunk("Second memory", 200)
    
    assert block.chunk_num == 2
    assert block.block_used == 300
    assert "First memory" in block.get_all_text()
    assert "Second memory" in block.get_all_text()
    
    # Test full detection
    block.add_chunk("Large memory", 800)
    assert block.is_full()
    
    clear_text_cache()


def test_factory_validation():
    """Test factory function parameter validation."""
    with pytest.raises(ValueError, match="Invalid storage_mode"):
        create_chat_manager(
            storage_mode="invalid",
            chat_openai_config={"api_key": "test"},
            aggregator_openai_config={"api_key": "test"},
            memory_agent_openai_config={"api_key": "test"},
            router_openai_config={"api_key": "test"},
        )


def test_file_structure():
    """Test that all required files exist."""
    required_files = [
        "src/memory/kv_block_manager/text_block.py",
        "src/memory/memory_agent/text_agent.py",
        "src/memory/core/text_loop_handler.py",
        "src/conversation_manager/factory.py",
        "docs/TEXT_STORAGE_README.md",
        "docs/QUICKSTART_TEXT_STORAGE.md",
        "examples/example_text_storage.py",
        "tests/test_text_storage.py"
    ]
    
    for file in required_files:
        assert os.path.exists(file), f"Missing file: {file}"


def test_storage_directories():
    """Test storage directories creation."""
    assert os.path.exists("text_data")
    clear_text_cache()
