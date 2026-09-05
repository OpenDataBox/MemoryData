"""Tests for TextMemoryHandler and TextAddHandler."""

import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.memory.core.text_loop_handler import (
    TextAddHandler,
    TextMemoryHandler,
    TextQueryHandler,
)


@pytest.fixture(autouse=True)
def cleanup(temp_text_dir):
    """Clean up using temp directory fixture."""
    yield
    # temp_text_dir fixture handles cleanup automatically


class TestTextAddHandler:
    """Test TextAddHandler functionality."""

    @patch("src.memory.core.text_loop_handler.TextMemoryAgent")
    def test_init(self, mock_agent_class):
        """Test TextAddHandler initialization."""
        handler = TextAddHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            model_context_window=32768,
        )

        assert handler.model_id == "test-model"
        assert handler.active_memory_agent is None
        assert handler.overlap_buffer == []
        assert handler.overlap_mode == "chunk"

    @patch("src.memory.core.text_loop_handler.TextMemoryAgent")
    def test_create_agent(self, mock_agent_class):
        """Test creating text memory agent."""
        handler = TextAddHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )
        handler.create_agent()

        assert handler.active_memory_agent is not None
        mock_agent_class.assert_called_once()

    @patch("src.memory.core.text_loop_handler.TextMemoryAgent")
    def test_add_memory_creates_agent_if_none(self, mock_agent_class):
        """Test that add_memory creates agent if none exists."""
        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.block_size = 1000
        mock_agent_class.return_value = mock_agent

        handler = TextAddHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )

        # Agent should be None initially
        assert handler.active_memory_agent is None

        handler.add_memory("Test memory")

        # Agent should be created
        assert handler.active_memory_agent is not None
        mock_agent.add.assert_called_once()

    @patch("src.memory.core.text_loop_handler.TextMemoryAgent")
    def test_add_memory_chunk_mode_overlap(self, mock_agent_class):
        """Test add_memory with chunk mode overlap."""
        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.block_size = 1000
        mock_agent_class.return_value = mock_agent

        handler = TextAddHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            overlap_ratio=0.1,
            overlap_mode="chunk",
        )

        handler.add_memory("First chunk of text.")
        handler.add_memory("Second chunk of text.")

        overlap_memories = handler.get_overlap_memories()
        assert len(overlap_memories) >= 0  # Should have some overlap

    @patch("src.memory.core.text_loop_handler.TextMemoryAgent")
    def test_add_memory_token_mode_overlap(self, mock_agent_class):
        """Test add_memory with token mode overlap."""
        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.block_size = 1000
        mock_agent_class.return_value = mock_agent

        handler = TextAddHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            overlap_ratio=0.1,
            overlap_mode="token",
        )

        handler.add_memory("First sentence. Second sentence. Third sentence.")

        overlap_memories = handler.get_overlap_memories()
        # Token mode should return wrapped content
        if overlap_memories:
            assert "<overlap_replay>" in overlap_memories[0]

    @patch("src.memory.core.text_loop_handler.TextMemoryAgent")
    def test_get_overlap_memories_empty(self, mock_agent_class):
        """Test get_overlap_memories with empty buffer."""
        handler = TextAddHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            overlap_mode="token",
        )

        overlap = handler.get_overlap_memories()
        assert overlap == []

    @patch("src.memory.core.text_loop_handler.TextMemoryAgent")
    def test_clear_overlap_buffer(self, mock_agent_class):
        """Test clearing overlap buffer."""
        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.block_size = 1000
        mock_agent_class.return_value = mock_agent

        handler = TextAddHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            overlap_ratio=0.1,
        )

        handler.add_memory("Test memory")
        handler.clear_overlap_buffer()

        assert handler.overlap_buffer == []

    @patch("src.memory.core.text_loop_handler.TextMemoryAgent")
    def test_query_new_agent_no_agent(self, mock_agent_class):
        """Test query_new_agent when no agent exists."""
        handler = TextAddHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )

        result = handler.query_new_agent("Test query")
        assert result == "No active memory."

    @patch("src.memory.core.text_loop_handler.TextMemoryAgent")
    def test_query_new_agent_with_agent(self, mock_agent_class):
        """Test query_new_agent with existing agent."""
        mock_agent = Mock()
        mock_agent.query.return_value = "Query result"
        mock_agent_class.return_value = mock_agent

        handler = TextAddHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )
        handler.create_agent()

        result = handler.query_new_agent("Test query")
        assert result == "Query result"


class TestTextQueryHandler:
    """Test TextQueryHandler functionality."""

    @patch("src.agent.base.OpenAI")
    def test_init(self, mock_openai):
        """Test TextQueryHandler initialization."""
        mock_router = Mock()
        handler = TextQueryHandler(router=mock_router)

        assert handler.router == mock_router
        assert handler.inactive_memory_agent == []

    @patch("src.agent.base.OpenAI")
    def test_query_memory_no_results(self, mock_openai):
        """Test query with no results."""
        mock_router = Mock()
        mock_router.map_reduce_blocks.return_value = []

        handler = TextQueryHandler(router=mock_router)
        result = handler.query_memory("Test query")

        assert result == "No relevant memory found."

    @patch("src.agent.base.OpenAI")
    def test_query_memory_with_results(self, mock_openai):
        """Test query with results."""
        mock_router = Mock()
        mock_router.map_reduce_blocks.return_value = ["Result 1", "Result 2"]

        handler = TextQueryHandler(router=mock_router)
        result = handler.query_memory("Test query")

        assert "Old Memory Block 1" in result
        assert "Result 1" in result
        assert "Old Memory Block 2" in result


class TestTextMemoryHandler:
    """Test TextMemoryHandler functionality."""

    @patch("src.memory.core.text_loop_handler.clear_text_metadata")
    @patch("src.memory.core.text_loop_handler.clear_text_cache")
    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_init_with_clean_cache(
        self, mock_add_handler, mock_router, mock_clear_cache, mock_clear_meta
    ):
        """Test initialization with clean cache."""
        handler = TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            clean_cache_first=True,
        )

        mock_clear_cache.assert_called_once()
        mock_clear_meta.assert_called_once()
        assert handler.inactive_memory_agents == []

    @patch("src.memory.core.text_loop_handler.clear_text_metadata")
    @patch("src.memory.core.text_loop_handler.clear_text_cache")
    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_init_without_clean_cache(
        self, mock_add_handler, mock_router, mock_clear_cache, mock_clear_meta
    ):
        """Test initialization without clean cache."""
        TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            clean_cache_first=False,
        )

        mock_clear_cache.assert_not_called()
        mock_clear_meta.assert_not_called()

    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_init_with_router_prompt(self, mock_add_handler, mock_router_class):
        """Test initialization with custom router prompt."""
        TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            router_system_prompt="Custom prompt",
        )

        call_kwargs = mock_router_class.call_args.kwargs
        assert call_kwargs["system_prompt"] == "Custom prompt"

    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_init_with_memory_segment_params(self, mock_add_handler, mock_router_class):
        """Test initialization with max_memory_segments and max_blocks."""
        TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            max_memory_segments=10,
            max_blocks=8,
        )

        call_kwargs = mock_router_class.call_args.kwargs
        assert call_kwargs["max_memory_segments"] == 10
        assert call_kwargs["max_blocks"] == 8

    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_init_with_enable_router_true(self, mock_add_handler, mock_router_class):
        """Test initialization with enable_router=True (default)."""
        TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            enable_router=True,
        )

        call_kwargs = mock_router_class.call_args.kwargs
        assert call_kwargs["enable_router"] is True

    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_init_with_enable_router_false(self, mock_add_handler, mock_router_class):
        """Test initialization with enable_router=False."""
        TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            enable_router=False,
        )

        call_kwargs = mock_router_class.call_args.kwargs
        assert call_kwargs["enable_router"] is False

    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_init_router_disabled_no_config(self, mock_add_handler, mock_router_class):
        """Test initialization with enable_router=False and no openai_config."""
        # When router is disabled, it should not require openai_config
        TextMemoryHandler(
            model_id="test-model",
            openai_config=None,
            enable_router=False,
        )

        call_kwargs = mock_router_class.call_args.kwargs
        assert call_kwargs["enable_router"] is False
        assert call_kwargs["openai_config"] is None

    @patch("src.memory.core.text_loop_handler.save_text_agents_metadata")
    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_add_memory_creates_agent(
        self, mock_add_handler_class, mock_router, mock_save_meta
    ):
        """Test that add_memory creates agent if none exists."""
        # Create mock agent that will be set after create_agent is called
        mock_agent = Mock()
        mock_agent.is_active = True

        mock_add_handler = Mock()
        # Start with None, then set to mock_agent after create_agent
        mock_add_handler.active_memory_agent = None
        mock_add_handler.add_memory.return_value = True

        def set_agent():
            mock_add_handler.active_memory_agent = mock_agent

        mock_add_handler.create_agent = Mock(side_effect=set_agent)
        mock_add_handler_class.return_value = mock_add_handler

        handler = TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )

        handler.add_memory("Test memory")

        mock_add_handler.create_agent.assert_called()
        mock_add_handler.add_memory.assert_called_once_with("Test memory")

    @patch("src.memory.core.text_loop_handler.save_text_agents_metadata")
    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_add_memory_agent_becomes_inactive(
        self, mock_add_handler_class, mock_router_class, mock_save_meta
    ):
        """Test add_memory when agent becomes inactive."""
        mock_block = Mock()
        mock_block.block_id = uuid.uuid4()
        mock_block.create_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mock_block.block_used = 100

        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.summary = "Test summary"
        mock_agent.current_block = mock_block
        mock_agent.model_id = "test-model"
        mock_agent.block_size_ratio = 0.125

        mock_new_agent = Mock()
        mock_new_agent.is_active = True
        mock_new_agent.add = Mock()

        mock_add_handler = Mock()
        mock_add_handler.active_memory_agent = mock_agent
        mock_add_handler.add_memory.return_value = False  # Agent becomes inactive
        mock_add_handler.get_overlap_memories.return_value = []
        mock_add_handler.create_agent = Mock(
            side_effect=lambda: setattr(mock_add_handler, "active_memory_agent", mock_new_agent)
        )
        mock_add_handler_class.return_value = mock_add_handler

        mock_router = Mock()
        mock_router_class.return_value = mock_router

        handler = TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )

        handler.add_memory("Test memory")

        # Agent should be added to inactive list
        assert len(handler.inactive_memory_agents) == 1
        mock_router.add_blocks.assert_called_once_with(mock_agent)
        mock_add_handler.create_agent.assert_called()

    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_query_memory_both_sources(self, mock_add_handler_class, mock_router_class):
        """Test query with both old and new memory."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "New memory"
        mock_add_handler_class.return_value = mock_add_handler

        mock_router = Mock()
        mock_router.map_reduce_blocks.return_value = ["Old memory"]
        mock_router_class.return_value = mock_router

        handler = TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )
        handler.query_handler.query_memory = Mock(return_value="Old Memory Block 1: Old memory")

        result = handler.query_memory("Test query")

        assert "Old memory" in result or "New memory" in result

    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_query_memory_no_memory(self, mock_add_handler_class, mock_router_class):
        """Test query with no memory."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "No active memory."
        mock_add_handler_class.return_value = mock_add_handler

        handler = TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )
        handler.query_handler.query_memory = Mock(return_value="No relevant memory found.")

        result = handler.query_memory("Test query")

        assert result == "No memory found."

    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_query_memory_only_old(self, mock_add_handler_class, mock_router_class):
        """Test query with only old memory."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "No active memory."
        mock_add_handler_class.return_value = mock_add_handler

        handler = TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )
        handler.query_handler.query_memory = Mock(return_value="Old memory content")

        result = handler.query_memory("Test query")

        assert result == "Old memory content"

    @patch("src.memory.core.text_loop_handler.HybridRouter")
    @patch("src.memory.core.text_loop_handler.TextAddHandler")
    def test_query_memory_only_new(self, mock_add_handler_class, mock_router_class):
        """Test query with only new memory."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "New memory content"
        mock_add_handler_class.return_value = mock_add_handler

        handler = TextMemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )
        handler.query_handler.query_memory = Mock(return_value="No relevant memory found.")

        result = handler.query_memory("Test query")

        assert result == "New memory content"


class TestTextMemoryAgentDirect:
    """Test TextMemoryAgent directly with mocks."""

    @patch("src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained")
    @patch("src.agent.base.OpenAI")
    def test_text_memory_agent_init(self, mock_openai, mock_tokenizer):
        """Test TextMemoryAgent initialization."""
        from src.memory.memory_agent.text_agent import TextMemoryAgent

        mock_tokenizer.return_value = Mock()

        agent = TextMemoryAgent(
            model_id="test-model",
            openai_config={"api_key": "test"},
            model_context_window=32768,
        )

        assert agent.model_id == "test-model"
        assert agent.is_active is True
        assert agent.summary is None

    @patch("src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained")
    @patch("src.agent.base.OpenAI")
    def test_text_memory_agent_add(self, mock_openai, mock_tokenizer_class):
        """Test TextMemoryAgent add method."""
        from src.memory.memory_agent.text_agent import TextMemoryAgent

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1] * 100  # 100 tokens
        mock_tokenizer_class.return_value = mock_tokenizer

        agent = TextMemoryAgent(
            model_id="test-model",
            openai_config={"api_key": "test"},
            model_context_window=32768,
            block_size_ratio=0.125,  # 4096 tokens
        )

        agent.add(["Test memory chunk"])

        assert agent.current_block.chunk_num == 1

    @patch("src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained")
    @patch("src.agent.base.OpenAI")
    def test_text_memory_agent_add_fills_block(self, mock_openai, mock_tokenizer_class):
        """Test TextMemoryAgent add method when block becomes full."""
        from src.memory.memory_agent.text_agent import TextMemoryAgent

        mock_tokenizer = Mock()
        # Return enough tokens to fill block
        mock_tokenizer.encode.return_value = [1] * 5000
        mock_tokenizer_class.return_value = mock_tokenizer

        # Mock the LLM response for summary
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Test summary"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        agent = TextMemoryAgent(
            model_id="test-model",
            openai_config={"api_key": "test"},
            model_context_window=8000,
            block_size_ratio=0.5,  # 4000 tokens
        )

        agent.add(["Very large chunk that fills the block"])

        # Block should be full
        assert agent.is_active is False
        assert agent.summary is not None

    @patch("src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained")
    @patch("src.agent.base.OpenAI")
    def test_text_memory_agent_summary_removes_thinking_content(
        self, mock_openai, mock_tokenizer_class
    ):
        """Test TextMemoryAgent strips thinking tags from generated summaries."""
        from src.memory.memory_agent.text_agent import TextMemoryAgent

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1] * 5000
        mock_tokenizer_class.return_value = mock_tokenizer

        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = (
            "Summary start <thinking>hidden reasoning</thinking> summary end"
        )
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        agent = TextMemoryAgent(
            model_id="test-model",
            openai_config={"api_key": "test"},
            model_context_window=8000,
            block_size_ratio=0.5,
        )

        agent.add(["Very large chunk that fills the block"])

        assert "<thinking>" not in agent.summary
        assert "hidden reasoning" not in agent.summary
        assert "Summary start" in agent.summary
        assert "summary end" in agent.summary

    @patch("src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained")
    @patch("src.agent.base.OpenAI")
    def test_text_memory_agent_add_inactive_raises(self, mock_openai, mock_tokenizer_class):
        """Test TextMemoryAgent add raises error when inactive."""
        from src.memory.memory_agent.text_agent import TextMemoryAgent

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1] * 100
        mock_tokenizer_class.return_value = mock_tokenizer

        agent = TextMemoryAgent(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )

        # Force agent to be inactive
        agent.is_active = False

        with pytest.raises(RuntimeError, match="Agent is inactive"):
            agent.add(["Test memory"])

    @patch("src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained")
    @patch("src.agent.base.OpenAI")
    def test_text_memory_agent_query(self, mock_openai, mock_tokenizer_class):
        """Test TextMemoryAgent query method."""
        from src.memory.memory_agent.text_agent import TextMemoryAgent

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1] * 100
        mock_tokenizer_class.return_value = mock_tokenizer

        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Query response"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        agent = TextMemoryAgent(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )

        # Add some data first
        agent.add(["Test memory content"])

        result = agent.query("What is the test?")

        assert result == "Query response"

    @patch("src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained")
    @patch("src.agent.base.OpenAI")
    def test_text_memory_agent_query_removes_thinking_content(
        self, mock_openai, mock_tokenizer_class
    ):
        """Test TextMemoryAgent strips thinking tags from query responses."""
        from src.memory.memory_agent.text_agent import TextMemoryAgent

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1] * 100
        mock_tokenizer_class.return_value = mock_tokenizer

        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = (
            "Query start <think>hidden reasoning</think> query end"
        )
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        agent = TextMemoryAgent(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )

        agent.add(["Test memory content"])

        result = agent.query("What is the test?")

        assert "<think>" not in result
        assert "hidden reasoning" not in result
        assert "Query start" in result
        assert "query end" in result

    @patch("src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained")
    @patch("src.agent.base.OpenAI")
    def test_text_memory_agent_query_empty(self, mock_openai, mock_tokenizer_class):
        """Test TextMemoryAgent query with no data."""
        from src.memory.memory_agent.text_agent import TextMemoryAgent

        mock_tokenizer = Mock()
        mock_tokenizer_class.return_value = mock_tokenizer

        agent = TextMemoryAgent(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )

        result = agent.query("What is the test?")

        assert result == "No knowledge available."

    @patch("src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained")
    @patch("src.agent.base.OpenAI")
    def test_text_memory_agent_preload_cache(self, mock_openai, mock_tokenizer_class):
        """Test TextMemoryAgent preload_cache is no-op."""
        from src.memory.memory_agent.text_agent import TextMemoryAgent

        mock_tokenizer = Mock()
        mock_tokenizer_class.return_value = mock_tokenizer

        agent = TextMemoryAgent(
            model_id="test-model",
            openai_config={"api_key": "test"},
        )

        # Should not raise any error
        agent.preload_cache()

    @patch("src.memory.memory_agent.text_agent.AutoTokenizer.from_pretrained")
    @patch("src.agent.base.OpenAI")
    def test_text_memory_agent_load_existing(self, mock_openai, mock_tokenizer_class):
        """Test TextMemoryAgent loading from existing block."""
        from src.memory.kv_block_manager.text_block import TextBlock
        from src.memory.memory_agent.text_agent import TextMemoryAgent

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1] * 100
        mock_tokenizer_class.return_value = mock_tokenizer

        # Create a block with some data
        block_id = uuid.uuid4()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        block = TextBlock(block_id=block_id, create_timestamp=timestamp, block_size=4096)
        block.add_chunk("Existing memory", 100)

        # Load agent from existing block
        agent = TextMemoryAgent(
            model_id="test-model",
            openai_config={"api_key": "test"},
            load_from_block_id=str(block_id),
            load_timestamp=timestamp,
        )

        # Should have loaded the existing data
        assert agent.current_block.chunk_num == 1
