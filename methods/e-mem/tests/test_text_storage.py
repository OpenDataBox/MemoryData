"""Test text storage implementation."""
import uuid
from datetime import datetime

import pytest

from src.conversation_manager.chat_handler import TextStorageChatManager
from src.conversation_manager.factory import create_chat_manager
from src.memory.core.text_loop_handler import TextMemoryHandler
from src.memory.kv_block_manager.text_block import TextBlock, clear_text_cache
from src.memory.memory_agent.text_agent import TextMemoryAgent


def test_imports():
    """Test that all modules can be imported."""
    assert TextBlock is not None
    assert TextMemoryAgent is not None
    assert TextMemoryHandler is not None
    assert create_chat_manager is not None
    assert TextStorageChatManager is not None


def test_text_block(temp_text_dir):
    """Test TextBlock functionality."""
    block = TextBlock(
        block_id=uuid.uuid4(),
        create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        block_size=1000
    )
    
    # Add chunk
    is_full = block.add_chunk("Test memory chunk", 100)
    assert block.chunk_num == 1
    assert block.block_used == 100
    assert not is_full
    
    # Get text
    text = block.get_all_text()
    assert "Test memory chunk" in text
    
    # Test full detection
    block.add_chunk("Large memory", 900)
    assert block.is_full()


def test_factory_function():
    """Test factory function validation."""
    with pytest.raises(ValueError, match="Invalid storage_mode"):
        create_chat_manager(
            storage_mode="invalid",
            chat_openai_config={"api_key": "test"},
            aggregator_openai_config={"api_key": "test"},
            memory_agent_openai_config={"api_key": "test"},
            router_openai_config={"api_key": "test"},
        )


def test_cleanup(temp_text_dir):
    """Test cache cleanup."""
    clear_text_cache()
    # Should not raise any errors


def test_text_handler_overlap_mode_chunk(temp_text_dir):
    """Test TextMemoryHandler with chunk overlap mode."""
    handler = TextMemoryHandler(
        model_id="test-model",
        openai_config={"api_key": "test"},
        overlap_ratio=0.1,
        overlap_mode="chunk"
    )
    assert handler.add_handler.overlap_mode == "chunk"


def test_text_handler_overlap_mode_token(temp_text_dir):
    """Test TextMemoryHandler with token overlap mode."""
    handler = TextMemoryHandler(
        model_id="test-model",
        openai_config={"api_key": "test"},
        overlap_ratio=0.1,
        overlap_mode="token"
    )
    assert handler.add_handler.overlap_mode == "token"


def test_text_handler_overlap_mode_default(temp_text_dir):
    """Test TextMemoryHandler default overlap mode."""
    handler = TextMemoryHandler(
        model_id="test-model",
        openai_config={"api_key": "test"}
    )
    assert handler.add_handler.overlap_mode == "chunk"  # Default should be chunk


def test_text_handler_with_memory_segment_params(temp_text_dir):
    """Test TextMemoryHandler with max_memory_segments and max_blocks."""
    handler = TextMemoryHandler(
        model_id="test-model",
        openai_config={"api_key": "test"},
        max_memory_segments=3,
        max_blocks=10,
    )
    assert handler.query_handler.router.max_memory_segments == 3
    assert handler.query_handler.router.max_blocks == 10


def test_text_block_save_and_load(temp_text_dir):
    """Test TextBlock save and load functionality."""
    block_id = uuid.uuid4()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    block = TextBlock(
        block_id=block_id,
        create_timestamp=timestamp,
        block_size=1000,
    )

    # Add some chunks - add_chunk auto-saves
    block.add_chunk("First memory chunk", 50)
    block.add_chunk("Second memory chunk", 60)

    # Create new block with same ID and load
    loaded_block = TextBlock(
        block_id=block_id,
        create_timestamp=timestamp,
        block_size=1000,
    )
    loaded_block.load()

    # Verify data
    text = loaded_block.get_all_text()
    assert "First memory chunk" in text
    assert "Second memory chunk" in text
    assert loaded_block.chunk_num == 2


def test_text_block_is_full(temp_text_dir):
    """Test TextBlock is_full detection."""
    block = TextBlock(
        block_id=uuid.uuid4(),
        create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        block_size=100,  # Small block
    )

    # Not full initially
    assert not block.is_full()

    # Add chunk that fills it
    block.add_chunk("Large content", 100)
    assert block.is_full()


def test_text_block_get_all_text_empty(temp_text_dir):
    """Test TextBlock get_all_text with no chunks."""
    block = TextBlock(
        block_id=uuid.uuid4(),
        create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        block_size=1000,
    )

    text = block.get_all_text()
    assert text == ""


def test_text_handler_add_memory(temp_text_dir):
    """Test TextMemoryHandler add_memory method."""
    from unittest.mock import Mock, patch

    # Mock the tokenizer to avoid network calls
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = list(range(50))  # Simulate 50 tokens

    with patch(
        "src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    ):
        handler = TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            clean_cache_first=True,
        )

        # Add memory
        handler.add_memory("Test memory content")

        # Verify memory was added
        assert handler.add_handler.active_memory_agent is not None
        assert len(handler.add_handler.active_memory_agent.current_block.chunks) > 0
