"""Tests for MemoryHandler."""
from unittest.mock import Mock, patch

import torch

from src.memory.core.loop_handler import AddHandler, MemoryHandler, QueryHandler


def create_mock_model():
    """Helper to create properly configured mock model."""
    mock_model = Mock()
    mock_model.device = torch.device("cpu")
    mock_layers = [Mock() for _ in range(4)]
    for layer in mock_layers:
        layer.parameters.return_value = iter([torch.tensor([1.0])])
    mock_model.model.layers = mock_layers
    # Remove hf_device_map to simulate non-auto device mapping
    if hasattr(mock_model, 'hf_device_map'):
        del mock_model.hf_device_map
    return mock_model


def create_mock_tokenizer():
    """Helper to create properly configured mock tokenizer."""
    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = (
        "<|im_start|>system\nTest<|im_end|>\n<|im_start|>user\nTEST<|im_end|>\n"
    )
    mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_tokenizer.decode.return_value = "Test response"
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.pad_token_id = 0
    return mock_tokenizer


class TestAddHandler:
    """Test AddHandler functionality."""
    
    @patch("src.memory.core.loop_handler.AutoModelForCausalLM")
    @patch("src.memory.core.loop_handler.AutoTokenizer")
    def test_init(self, mock_tokenizer_class, mock_model_class):
        """Test AddHandler initialization."""
        mock_tokenizer_class.from_pretrained.return_value = create_mock_tokenizer()
        mock_model_class.from_pretrained.return_value = create_mock_model()
        
        handler = AddHandler(
            model_id="test-model",
            model_context_window=32768,
            attn_implementation="sdpa",
            device_map="cpu"
        )
        
        assert handler.model_id == "test-model"
        assert handler.active_memory_agent is None
        assert handler._shared_model is not None
        assert handler._shared_tokenizer is not None
    
    @patch("src.memory.core.loop_handler.AutoModelForCausalLM")
    @patch("src.memory.core.loop_handler.AutoTokenizer")
    @patch("src.memory.core.loop_handler.MemoryAgent")
    def test_create_agent(self, mock_agent_class, mock_tokenizer_class, mock_model_class):
        """Test creating agent."""
        mock_tokenizer_class.from_pretrained.return_value = create_mock_tokenizer()
        mock_model_class.from_pretrained.return_value = create_mock_model()
        
        handler = AddHandler(model_id="test-model")
        handler.create_agent()
        
        assert handler.active_memory_agent is not None
        mock_agent_class.assert_called_once()
        # Verify shared model is passed
        call_kwargs = mock_agent_class.call_args.kwargs
        assert "shared_model" in call_kwargs
        assert "shared_tokenizer" in call_kwargs
    
    @patch("src.memory.core.loop_handler.AutoModelForCausalLM")
    @patch("src.memory.core.loop_handler.AutoTokenizer")
    @patch("src.memory.core.loop_handler.MemoryAgent")
    def test_add_memory(self, mock_agent_class, mock_tokenizer_class, mock_model_class):
        """Test adding memory."""
        mock_tokenizer_class.from_pretrained.return_value = create_mock_tokenizer()
        mock_model_class.from_pretrained.return_value = create_mock_model()
        
        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.block_size = 1000
        mock_agent_class.return_value = mock_agent
        
        handler = AddHandler(model_id="test-model")
        is_active = handler.add_memory("Test memory")
        
        assert is_active
        mock_agent.add.assert_called_once_with(["Test memory"])
    
    @patch("src.memory.core.loop_handler.AutoModelForCausalLM")
    @patch("src.memory.core.loop_handler.AutoTokenizer")
    @patch("src.memory.core.loop_handler.MemoryAgent")
    def test_query_new_agent(self, mock_agent_class, mock_tokenizer_class, mock_model_class):
        """Test querying new agent."""
        mock_tokenizer_class.from_pretrained.return_value = create_mock_tokenizer()
        mock_model_class.from_pretrained.return_value = create_mock_model()
        
        mock_agent = Mock()
        mock_agent.query.return_value = "Test response"
        mock_agent.saved_chunks = [{"start": 0, "length": 10}]
        mock_agent_class.return_value = mock_agent
        
        handler = AddHandler(model_id="test-model")
        handler.create_agent()
        result = handler.query_new_agent("Test query")
        
        assert result == "Test response"
        mock_agent.query.assert_called_once_with("Test query")
    
    @patch("src.memory.core.loop_handler.AutoModelForCausalLM")
    @patch("src.memory.core.loop_handler.AutoTokenizer")
    def test_query_no_agent(self, mock_tokenizer_class, mock_model_class):
        """Test querying when no agent exists."""
        mock_tokenizer_class.from_pretrained.return_value = create_mock_tokenizer()
        mock_model_class.from_pretrained.return_value = create_mock_model()
        
        handler = AddHandler(model_id="test-model")
        result = handler.query_new_agent("Test query")
        
        assert result == "No active memory."
    
    @patch("src.memory.core.loop_handler.AutoModelForCausalLM")
    @patch("src.memory.core.loop_handler.AutoTokenizer")
    @patch("src.memory.core.loop_handler.MemoryAgent")
    def test_overlap_mode_chunk(self, mock_agent_class, mock_tokenizer_class, mock_model_class):
        """Test chunk mode overlap handling."""
        mock_tokenizer_class.from_pretrained.return_value = create_mock_tokenizer()
        mock_model_class.from_pretrained.return_value = create_mock_model()
        
        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.block_size = 1000
        mock_agent_class.return_value = mock_agent
        
        handler = AddHandler(
            model_id="test-model",
            overlap_ratio=0.1,
            overlap_mode="chunk"
        )
        
        # Add multiple chunks
        handler.add_memory("First chunk of text.")
        handler.add_memory("Second chunk of text.")
        handler.add_memory("Third chunk of text.")
        
        # In chunk mode, overlap_buffer should contain whole chunks
        overlap_memories = handler.get_overlap_memories()
        assert len(overlap_memories) > 0
        # All items should be complete chunks
        assert all(isinstance(mem, str) for mem in overlap_memories)
    
    @patch("src.memory.core.loop_handler.AutoModelForCausalLM")
    @patch("src.memory.core.loop_handler.AutoTokenizer")
    @patch("src.memory.core.loop_handler.MemoryAgent")
    def test_overlap_mode_token(self, mock_agent_class, mock_tokenizer_class, mock_model_class):
        """Test token mode overlap handling."""
        mock_tokenizer_class.from_pretrained.return_value = create_mock_tokenizer()
        mock_model_class.from_pretrained.return_value = create_mock_model()
        
        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.block_size = 1000
        mock_agent_class.return_value = mock_agent
        
        handler = AddHandler(
            model_id="test-model",
            overlap_ratio=0.1,
            overlap_mode="token"
        )
        
        # Add text with multiple sentences
        text_with_sentences = "First sentence. Second sentence. Third sentence. Fourth sentence."
        handler.add_memory(text_with_sentences)
        
        # In token mode, overlap_buffer should contain sentences
        overlap_memories = handler.get_overlap_memories()
        assert len(overlap_memories) > 0
        # Items should be sentences (may be split)
        assert all(isinstance(mem, str) for mem in overlap_memories)
    
    @patch("src.memory.core.loop_handler.AutoModelForCausalLM")
    @patch("src.memory.core.loop_handler.AutoTokenizer")
    def test_overlap_mode_initialization(self, mock_tokenizer_class, mock_model_class):
        """Test that overlap_mode is correctly initialized."""
        mock_tokenizer_class.from_pretrained.return_value = create_mock_tokenizer()
        mock_model_class.from_pretrained.return_value = create_mock_model()
        
        handler_chunk = AddHandler(
            model_id="test-model",
            overlap_mode="chunk"
        )
        assert handler_chunk.overlap_mode == "chunk"
        
        handler_token = AddHandler(
            model_id="test-model",
            overlap_mode="token"
        )
        assert handler_token.overlap_mode == "token"
        
        handler_default = AddHandler(model_id="test-model")
        assert handler_default.overlap_mode == "chunk"  # Default should be chunk
    
    @patch("src.memory.core.loop_handler.AutoModelForCausalLM")
    @patch("src.memory.core.loop_handler.AutoTokenizer")
    @patch("src.memory.core.loop_handler.MemoryAgent")
    def test_get_overlap_memories_returns_copy(self, mock_agent_class, mock_tokenizer_class, mock_model_class):
        """Test that get_overlap_memories returns a copy, not a reference."""
        mock_tokenizer_class.from_pretrained.return_value = create_mock_tokenizer()
        mock_model_class.from_pretrained.return_value = create_mock_model()
        
        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.block_size = 1000
        mock_agent_class.return_value = mock_agent
        
        handler = AddHandler(
            model_id="test-model",
            overlap_ratio=0.1,
            overlap_mode="chunk"
        )
        
        handler.add_memory("Test memory")
        overlap1 = handler.get_overlap_memories()
        overlap2 = handler.get_overlap_memories()
        
        # Should be different objects (copy)
        assert overlap1 is not overlap2
        # But should have same content
        assert overlap1 == overlap2
    
    @patch("src.memory.core.loop_handler.AutoModelForCausalLM")
    @patch("src.memory.core.loop_handler.AutoTokenizer")
    @patch("src.memory.core.loop_handler.MemoryAgent")
    def test_clear_overlap_buffer(self, mock_agent_class, mock_tokenizer_class, mock_model_class):
        """Test clearing overlap buffer."""
        mock_tokenizer_class.from_pretrained.return_value = create_mock_tokenizer()
        mock_model_class.from_pretrained.return_value = create_mock_model()
        
        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.block_size = 1000
        mock_agent_class.return_value = mock_agent
        
        handler = AddHandler(
            model_id="test-model",
            overlap_ratio=0.1,
            overlap_mode="chunk"
        )
        
        handler.add_memory("Test memory")
        assert len(handler.get_overlap_memories()) > 0
        
        handler.clear_overlap_buffer()
        assert len(handler.get_overlap_memories()) == 0


class TestQueryHandler:
    """Test QueryHandler functionality."""
    
    def test_init(self, mock_openai_config):
        """Test QueryHandler initialization."""
        with patch("src.memory.core.loop_handler.HybridRouter"):
            mock_router = Mock()
            handler = QueryHandler(router=mock_router)
            
            assert handler.router == mock_router
            assert handler.inactive_memory_agent == []
    
    def test_query_memory_no_results(self):
        """Test querying with no results."""
        mock_router = Mock()
        mock_router.map_reduce_blocks.return_value = []
        
        handler = QueryHandler(router=mock_router)
        result = handler.query_memory("Test query")
        
        assert result == "No relevant memory found."
    
    def test_query_memory_with_results(self):
        """Test querying with results."""
        mock_router = Mock()
        mock_router.map_reduce_blocks.return_value = ["Result 1", "Result 2"]
        
        handler = QueryHandler(router=mock_router)
        result = handler.query_memory("Test query")
        
        assert result == "Old Memory Block 1: Result 1\nOld Memory Block 2: Result 2"


class TestMemoryHandler:
    """Test MemoryHandler functionality."""
    
    @patch("src.memory.core.loop_handler.clear_metadata")
    @patch("src.memory.core.loop_handler.clear_kv_cache")
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_init(self, mock_add_handler, mock_router, mock_clear_kv, mock_clear_meta, mock_openai_config):
        """Test MemoryHandler initialization."""
        handler = MemoryHandler(
            model_id="test-model",
            openai_config=mock_openai_config,
            clean_cache_first=True
        )
        
        assert handler.add_handler is not None
        assert handler.inactive_memory_agents == []
        mock_clear_kv.assert_called_once()
        mock_clear_meta.assert_called_once()
    
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_add_memory_active(self, mock_add_handler_class, mock_router):
        """Test adding memory when agent stays active."""
        mock_add_handler = Mock()
        mock_add_handler.add_memory.return_value = True
        mock_add_handler_class.return_value = mock_add_handler
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.add_memory("Test memory")
        
        mock_add_handler.add_memory.assert_called_once_with("Test memory")
        assert len(handler.inactive_memory_agents) == 0
    
    @patch("src.memory.core.loop_handler.save_agents_metadata")
    @patch("src.memory.core.loop_handler.load_agents_metadata")
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_add_memory_becomes_inactive(self, mock_add_handler_class, mock_router_class, mock_load_meta, mock_save_meta):
        """Test adding memory when agent becomes inactive."""
        import os
        os.environ['EVAL_SESSION_ID'] = 'test_session'
        mock_load_meta.return_value = []
        
        mock_block = Mock()
        mock_block.block_id = "test-id"
        mock_block.create_timestamp = "20231201_120000"
        mock_block.block_used = 100
        mock_block.chunk_num = 5
        
        # First agent (will become inactive)
        mock_agent = Mock()
        mock_agent.is_active = True  # Start as active
        mock_agent.summary = "Test summary"
        mock_agent.current_block = mock_block
        mock_agent.chunk_number = 5
        mock_agent.model_id = "test-model"
        mock_agent.saved_chunks = []
        
        # New agent (will be created)
        mock_new_agent = Mock()
        mock_new_agent.is_active = True
        mock_new_agent.add = Mock()
        
        mock_add_handler = Mock()
        # First call returns False (agent became inactive)
        mock_add_handler.add_memory.return_value = False
        mock_add_handler.active_memory_agent = mock_agent
        mock_add_handler.get_overlap_memories.return_value = []
        mock_add_handler.create_agent = Mock(side_effect=lambda: setattr(mock_add_handler, 'active_memory_agent', mock_new_agent))
        mock_add_handler_class.return_value = mock_add_handler
        
        mock_router = Mock()
        mock_router_class.return_value = mock_router
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.add_memory("Test memory")
        
        assert len(handler.inactive_memory_agents) == 1
        mock_add_handler.create_agent.assert_called_once()
        mock_router.add_blocks.assert_called_once_with(mock_agent)
        
        if 'EVAL_SESSION_ID' in os.environ:
            del os.environ['EVAL_SESSION_ID']
    
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_query_memory_both_sources(self, mock_add_handler_class, mock_router_class):
        """Test querying memory from both old and new sources."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "New memory"
        mock_add_handler_class.return_value = mock_add_handler
        
        mock_router = Mock()
        mock_router.map_reduce_blocks.return_value = ["Old memory"]
        mock_router_class.return_value = mock_router
        
        mock_query_handler = Mock()
        mock_query_handler.query_memory.return_value = "Old memory"
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.query_handler.query_memory = mock_query_handler.query_memory
        
        result = handler.query_memory("Test query")
        
        assert "Old memory" in result
        assert "New memory" in result
    
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_query_memory_no_memory(self, mock_add_handler_class, mock_router_class):
        """Test querying when no memory exists."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "No active memory."
        mock_add_handler_class.return_value = mock_add_handler
        
        mock_router = Mock()
        mock_router_class.return_value = mock_router
        
        mock_query_handler = Mock()
        mock_query_handler.query_memory.return_value = "No relevant memory found."
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.query_handler.query_memory = mock_query_handler.query_memory
        
        result = handler.query_memory("Test query")
        
        assert result == "No memory found."
    
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_query_memory_only_old(self, mock_add_handler_class, mock_router_class):
        """Test querying with only old memory."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "No active memory."
        mock_add_handler_class.return_value = mock_add_handler
        
        mock_router = Mock()
        mock_router_class.return_value = mock_router
        
        mock_query_handler = Mock()
        mock_query_handler.query_memory.return_value = "Old memory content"
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.query_handler.query_memory = mock_query_handler.query_memory
        
        result = handler.query_memory("Test query")
        
        assert result == "Old memory content"
    
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_query_memory_only_new(self, mock_add_handler_class, mock_router_class):
        """Test querying with only new memory."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "New memory content"
        mock_add_handler_class.return_value = mock_add_handler
        
        mock_router = Mock()
        mock_router_class.return_value = mock_router
        
        mock_query_handler = Mock()
        mock_query_handler.query_memory.return_value = "No relevant memory found."
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.query_handler.query_memory = mock_query_handler.query_memory
        
        result = handler.query_memory("Test query")
        
        assert result == "New memory content"
    
    @patch("src.memory.core.loop_handler.clear_metadata")
    @patch("src.memory.core.loop_handler.clear_kv_cache")
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_init_no_clean_cache(self, mock_add_handler, mock_router, mock_clear_kv, mock_clear_meta):
        """Test initialization without cleaning cache."""
        _ = MemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            clean_cache_first=False
        )
        
        mock_clear_kv.assert_not_called()
        mock_clear_meta.assert_not_called()
    
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_init_with_custom_router_prompt(self, mock_add_handler, mock_router_class):
        """Test initialization with custom router prompt."""
        _ = MemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            router_system_prompt="Custom router prompt"
        )
        
        # Verify router was called with custom prompt
        call_args = mock_router_class.call_args
        assert call_args.kwargs["system_prompt"] == "Custom router prompt"
    
    @patch("src.memory.core.loop_handler.clear_metadata")
    @patch("src.memory.core.loop_handler.clear_kv_cache")
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_init_with_overlap_mode(self, mock_add_handler_class, mock_router_class, mock_clear_kv, mock_clear_meta):
        """Test initialization with overlap_mode parameter."""
        mock_add_handler = Mock()
        mock_add_handler_class.return_value = mock_add_handler
        
        # Create handler to trigger AddHandler initialization
        _ = MemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            overlap_mode="token"
        )
        
        # Verify AddHandler was called with overlap_mode
        assert mock_add_handler_class.called, "AddHandler should have been called"
        call_args = mock_add_handler_class.call_args
        assert call_args is not None, "AddHandler should have been called with arguments"
        # AddHandler is now called with keyword arguments
        kwargs = call_args.kwargs
        assert "overlap_mode" in kwargs, f"overlap_mode should be in kwargs: {kwargs}"
        assert kwargs["overlap_mode"] == "token", f"overlap_mode should be 'token', got {kwargs.get('overlap_mode')}"

    @patch("src.memory.core.loop_handler.save_agents_metadata")
    @patch("src.memory.core.loop_handler.load_agents_metadata")
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_add_memory_with_overlap_token_mode(
        self, mock_add_handler_class, mock_router_class, mock_load_meta, mock_save_meta
    ):
        """Test add_memory with overlap memories in token mode."""
        import os
        os.environ['EVAL_SESSION_ID'] = 'test_session'
        mock_load_meta.return_value = []

        mock_block = Mock()
        mock_block.block_id = "test-id"
        mock_block.create_timestamp = "20231201_120000"
        mock_block.block_used = 100
        mock_block.chunk_num = 5

        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.summary = "Test summary"
        mock_agent.current_block = mock_block
        mock_agent.chunk_number = 5
        mock_agent.model_id = "test-model"
        mock_agent.saved_chunks = []

        mock_new_agent = Mock()
        mock_new_agent.is_active = True
        mock_new_agent.add = Mock()

        mock_add_handler = Mock()
        mock_add_handler.add_memory.return_value = False  # Agent becomes inactive
        mock_add_handler.active_memory_agent = mock_agent
        mock_add_handler.overlap_mode = "token"
        mock_add_handler.get_overlap_memories.return_value = ["overlap content"]
        mock_add_handler.create_agent = Mock(
            side_effect=lambda: setattr(mock_add_handler, 'active_memory_agent', mock_new_agent)
        )
        mock_add_handler_class.return_value = mock_add_handler

        mock_router = Mock()
        mock_router_class.return_value = mock_router

        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.add_memory("Test memory")

        # Overlap memories should be added to new agent in token mode
        mock_new_agent.add.assert_called_once_with(["overlap content"])

        if 'EVAL_SESSION_ID' in os.environ:
            del os.environ['EVAL_SESSION_ID']

    @patch("src.memory.core.loop_handler.save_agents_metadata")
    @patch("src.memory.core.loop_handler.load_agents_metadata")
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_add_memory_with_overlap_chunk_mode(
        self, mock_add_handler_class, mock_router_class, mock_load_meta, mock_save_meta
    ):
        """Test add_memory with overlap memories in chunk mode."""
        import os
        os.environ['EVAL_SESSION_ID'] = 'test_session'
        mock_load_meta.return_value = []

        mock_block = Mock()
        mock_block.block_id = "test-id"
        mock_block.create_timestamp = "20231201_120000"
        mock_block.block_used = 100
        mock_block.chunk_num = 5

        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.summary = "Test summary"
        mock_agent.current_block = mock_block
        mock_agent.chunk_number = 5
        mock_agent.model_id = "test-model"
        mock_agent.saved_chunks = []

        mock_new_agent = Mock()
        mock_new_agent.is_active = True
        mock_new_agent.add = Mock()

        mock_add_handler = Mock()
        mock_add_handler.add_memory.return_value = False  # Agent becomes inactive
        mock_add_handler.active_memory_agent = mock_agent
        mock_add_handler.overlap_mode = "chunk"
        mock_add_handler.get_overlap_memories.return_value = ["chunk1", "chunk2"]
        mock_add_handler.create_agent = Mock(
            side_effect=lambda: setattr(mock_add_handler, 'active_memory_agent', mock_new_agent)
        )
        mock_add_handler_class.return_value = mock_add_handler

        mock_router = Mock()
        mock_router_class.return_value = mock_router

        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.add_memory("Test memory")

        # Overlap memories should be added individually in chunk mode
        assert mock_new_agent.add.call_count == 2
        mock_new_agent.add.assert_any_call(["chunk1"])
        mock_new_agent.add.assert_any_call(["chunk2"])

        if 'EVAL_SESSION_ID' in os.environ:
            del os.environ['EVAL_SESSION_ID']

    @patch("src.memory.core.loop_handler.save_agents_metadata")
    @patch("src.memory.core.loop_handler.load_agents_metadata")
    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_add_memory_already_inactive_agent(
        self, mock_add_handler_class, mock_router_class, mock_load_meta, mock_save_meta
    ):
        """Test add_memory when agent is already inactive before add."""
        import os
        os.environ['EVAL_SESSION_ID'] = 'test_session'
        mock_load_meta.return_value = []

        mock_block = Mock()
        mock_block.block_id = "test-id"
        mock_block.create_timestamp = "20231201_120000"
        mock_block.block_used = 100
        mock_block.chunk_num = 5

        # Agent that is already inactive
        mock_agent = Mock()
        mock_agent.is_active = False  # Already inactive
        mock_agent.summary = "Test summary"
        mock_agent.current_block = mock_block
        mock_agent.chunk_number = 5
        mock_agent.model_id = "test-model"
        mock_agent.saved_chunks = []

        mock_new_agent = Mock()
        mock_new_agent.is_active = True
        mock_new_agent.add = Mock()

        mock_add_handler = Mock()
        mock_add_handler.add_memory.return_value = True  # New agent stays active
        mock_add_handler.active_memory_agent = mock_agent
        mock_add_handler.overlap_mode = "chunk"
        mock_add_handler.get_overlap_memories.return_value = []
        mock_add_handler.create_agent = Mock(
            side_effect=lambda: setattr(mock_add_handler, 'active_memory_agent', mock_new_agent)
        )
        mock_add_handler_class.return_value = mock_add_handler

        mock_router = Mock()
        mock_router_class.return_value = mock_router

        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.add_memory("Test memory")

        # Should have transitioned and created new agent
        mock_add_handler.create_agent.assert_called()
        mock_router.add_blocks.assert_called_with(mock_agent)

        if 'EVAL_SESSION_ID' in os.environ:
            del os.environ['EVAL_SESSION_ID']

    @patch("src.memory.core.loop_handler.HybridRouter")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_init_with_memory_segment_params(self, mock_add_handler, mock_router_class):
        """Test initialization with max_memory_segments and max_blocks."""
        _ = MemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            max_memory_segments=10,
            max_blocks=8
        )

        # Verify Router was called with correct params
        call_kwargs = mock_router_class.call_args.kwargs
        assert call_kwargs["max_memory_segments"] == 10
        assert call_kwargs["max_blocks"] == 8
