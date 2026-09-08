"""Extended tests for MemoryAgent to improve coverage."""
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import DynamicCache

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


class TestMemoryAgentExtended:
    """Extended tests for MemoryAgent."""
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_add_inactive_agent_raises_error(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test that adding to inactive agent raises error."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        agent.is_active = False
        
        with pytest.raises(RuntimeError, match="inactive"):
            agent.add(["Test chunk"])
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_add_knowledge_multiple_chunks(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test adding multiple knowledge chunks."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_output = Mock()
        mock_output.past_key_values = [(torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64)) for _ in range(4)]
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_block = Mock()
        mock_block.block_used = 0
        mock_block_class.return_value = mock_block
        
        agent = MemoryAgent(model_id="test-model")
        agent._add_knowledge(["Chunk 1", "Chunk 2", "Chunk 3"])
        
        assert agent.chunk_number == 3
        assert len(agent.saved_chunks) == 3
        assert agent.global_offset == 9  # 3 tokens * 3 chunks
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_query_with_merged_cache(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test querying with merged cache in memory."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Answer"
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_output = Mock()
        mock_output.past_key_values = DynamicCache()
        for i in range(4):
            mock_output.past_key_values.update(torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64), i)
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{"start": 0, "length": 5}]
        agent.global_offset = 5
        
        # Set merged_cache
        agent.merged_cache = DynamicCache()
        for i in range(4):
            agent.merged_cache.update(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64), i)
        
        result = agent.query("Test question")
        assert result == "Answer"
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_query_loads_from_disk(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test querying loads cache from disk when merged_cache is None."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Answer"
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_output = Mock()
        mock_output.past_key_values = DynamicCache()
        for i in range(4):
            mock_output.past_key_values.update(torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64), i)
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_block = Mock()
        mock_block.load_cache.return_value = {
            "merged_cache": [(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)) for _ in range(4)]
        }
        mock_block_class.return_value = mock_block
        
        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{"start": 0, "length": 5}]
        agent.global_offset = 5
        agent.merged_cache = None
        agent.is_active = False  # Set to inactive to trigger disk load
        agent.current_block = mock_block
        
        result = agent.query("Test question")
        assert result == "Answer"
        mock_block.load_cache.assert_called_once()
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_create_summaries_saves_cache(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test that _create_summaries saves cache to disk."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Summary"
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_output = Mock()
        mock_output.past_key_values = DynamicCache()
        for i in range(4):
            mock_output.past_key_values.update(torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64), i)
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_block = Mock()
        mock_block_class.return_value = mock_block
        
        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{"start": 0, "length": 5}]
        agent.global_offset = 5
        agent.merged_cache = DynamicCache()
        for i in range(4):
            agent.merged_cache.update(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64), i)
        
        agent._create_summaries()
        
        assert agent.summary == "Summary"
        assert agent.merged_cache is None
        mock_block.save_cache.assert_called_once()
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_get_layer_devices_with_hf_device_map(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test _get_layer_devices with hf_device_map."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cuda:0")
        mock_layers = [Mock() for _ in range(4)]
        for layer in mock_layers:
            layer.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.model.layers = mock_layers
        mock_model.hf_device_map = {
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.layers.2": "cuda:0",
            "model.layers.3": torch.device("cuda:1")
        }
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        
        assert 0 in agent.layer_devices
        assert 1 in agent.layer_devices
        assert agent.layer_devices[0].type == "cuda"
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_agent_generate_with_instruction(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test _agent_generate with instruction instead of question."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Summary"
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_output = Mock()
        mock_output.past_key_values = DynamicCache()
        for i in range(4):
            mock_output.past_key_values.update(torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64), i)
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{"start": 0, "length": 5}]
        agent.global_offset = 5
        agent.merged_cache = DynamicCache()
        for i in range(4):
            agent.merged_cache.update(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64), i)
        
        result = agent._agent_generate(instruction="Summarize", max_new_tokens=10)
        assert result == "Summary"
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_remove_think_tags(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test removing <think> tags."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        
        text = "Answer <think>thoughts</think> more"
        cleaned = agent._remove_thinking_content(text)
        
        assert "<think>" not in cleaned
        assert "thoughts" not in cleaned
        assert "Answer" in cleaned
