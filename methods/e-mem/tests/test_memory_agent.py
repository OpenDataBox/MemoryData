"""Tests for MemoryAgent."""
from unittest.mock import Mock, patch

import torch

from src.memory.memory_agent.agent import MemoryAgent


def create_mock_model():
    """Helper to create properly configured mock model."""
    mock_model = Mock()
    mock_model.device = torch.device("cpu")
    mock_layers = [Mock() for _ in range(4)]
    for layer in mock_layers:
        layer.parameters.return_value = iter([torch.tensor([1.0])])
    mock_model.model.layers = mock_layers
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
    return mock_tokenizer


class TestMemoryAgent:
    """Test MemoryAgent functionality."""
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_init(self, mock_block, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test MemoryAgent initialization."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n<|im_start|>user\nTEST<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test with default block_size_ratio (0.125)
        agent = MemoryAgent(model_id="test-model", model_context_window=1000)
        
        assert agent.model_id == "test-model"
        assert agent.block_size == 125  # 1000 * 0.125 = 125
        assert agent.block_size_ratio == 0.125
        assert agent.is_active
        assert agent.global_offset == 0
        assert agent.chunk_number == 0
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_extract_chat_tokens(self, mock_block, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test chat token extraction."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>user\nTEST<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        
        assert agent.role_start == "<|im_start|>"
        assert agent.role_end == "<|im_end|>"
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_add_knowledge_first_chunk(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test adding first knowledge chunk."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_output = Mock()
        mock_output.past_key_values = [(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)) for _ in range(4)]
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_block = Mock()
        mock_block.block_used = 0
        mock_block.save_cache.return_value = False
        mock_block_class.return_value = mock_block
        
        agent = MemoryAgent(model_id="test-model", model_context_window=1000)
        result = agent._add_knowledge(["Test chunk"])
        
        assert not result
        assert agent.chunk_number == 1
        assert agent.global_offset == 5
        assert len(agent.saved_chunks) == 1
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_remove_thinking_content(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test removing thinking tags from response."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        
        text_with_thinking = "Answer <thinking>internal thoughts</thinking> more text"
        cleaned = agent._remove_thinking_content(text_with_thinking)
        
        assert "<thinking>" not in cleaned
        assert "internal thoughts" not in cleaned
        assert "Answer" in cleaned
        assert "more text" in cleaned
