"""Additional tests for MemoryAgent to cover batch operations and edge cases."""
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import DynamicCache

from src.memory.memory_agent.agent import (
    BatchQueryInputs,
    MemoryAgent,
    batch_generate,
    pad_kv_cache_for_batch,
)


def create_mock_model():
    """Helper to create properly configured mock model."""
    mock_model = Mock()
    mock_model.device = torch.device("cpu")
    mock_layers = [Mock() for _ in range(4)]
    for layer in mock_layers:
        layer.parameters.return_value = iter([torch.tensor([1.0])])
    mock_model.model.layers = mock_layers
    if hasattr(mock_model, "hf_device_map"):
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


class TestPadKvCacheForBatch:
    """Test pad_kv_cache_for_batch function."""

    def test_empty_caches(self):
        """Test with empty cache list."""
        layer_devices = {0: torch.device("cpu")}
        primary_device = torch.device("cpu")

        batched_cache, cache_lengths = pad_kv_cache_for_batch(
            [], layer_devices, primary_device
        )

        assert isinstance(batched_cache, DynamicCache)
        assert cache_lengths == []

    def test_single_cache(self):
        """Test with single cache."""
        cache = DynamicCache()
        for i in range(2):
            cache.update(
                torch.randn(1, 4, 10, 64),
                torch.randn(1, 4, 10, 64),
                i
            )

        layer_devices = {0: torch.device("cpu"), 1: torch.device("cpu")}
        primary_device = torch.device("cpu")

        batched_cache, cache_lengths = pad_kv_cache_for_batch(
            [cache], layer_devices, primary_device
        )

        assert cache_lengths == [10]
        assert len(batched_cache) == 2

    def test_multiple_caches_with_padding(self):
        """Test with multiple caches of different lengths."""
        caches = []
        for seq_len in [5, 10, 3]:
            cache = DynamicCache()
            for i in range(2):
                cache.update(
                    torch.randn(1, 4, seq_len, 64),
                    torch.randn(1, 4, seq_len, 64),
                    i
                )
            caches.append(cache)

        layer_devices = {0: torch.device("cpu"), 1: torch.device("cpu")}
        primary_device = torch.device("cpu")

        batched_cache, cache_lengths = pad_kv_cache_for_batch(
            caches, layer_devices, primary_device
        )

        assert cache_lengths == [5, 10, 3]
        assert len(batched_cache) == 2
        # Check that batch dimension is correct
        k, v = batched_cache[0]
        assert k.shape[0] == 3  # batch size
        assert k.shape[2] == 10  # max cache length

    def test_empty_cache_in_list(self):
        """Test with empty cache in list."""
        cache = DynamicCache()
        for i in range(2):
            cache.update(
                torch.randn(1, 4, 5, 64),
                torch.randn(1, 4, 5, 64),
                i
            )

        empty_cache = DynamicCache()

        layer_devices = {0: torch.device("cpu"), 1: torch.device("cpu")}
        primary_device = torch.device("cpu")

        batched_cache, cache_lengths = pad_kv_cache_for_batch(
            [cache, empty_cache], layer_devices, primary_device
        )

        assert cache_lengths == [5, 0]


class TestBatchQueryInputs:
    """Test BatchQueryInputs dataclass."""

    def test_batch_query_inputs_creation(self):
        """Test creating BatchQueryInputs."""
        inputs = BatchQueryInputs(
            input_ids=torch.tensor([[1, 2, 3]]),
            position_ids=torch.tensor([[0, 1, 2]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
            past_key_values=DynamicCache(),
            cache_lengths=[10],
            query_lengths=[3],
            global_offsets=[10],
        )

        assert inputs.input_ids.shape == (1, 3)
        assert inputs.cache_lengths == [10]


class TestBatchGenerate:
    """Test batch_generate function."""

    def test_batch_generate_basic(self):
        """Test basic batch generation."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode.return_value = "Generated text"

        # Create mock outputs
        mock_output = Mock()
        mock_output.logits = torch.zeros(2, 1, 100)
        mock_output.logits[:, :, 0] = 10.0  # Make token 0 most likely (EOS)
        mock_output.past_key_values = DynamicCache()
        mock_model.return_value = mock_output

        # Create batch inputs
        batch_inputs = BatchQueryInputs(
            input_ids=torch.tensor([[1, 2], [3, 4]]),
            position_ids=torch.tensor([[10, 11], [20, 21]]),
            attention_mask=torch.ones(2, 12, dtype=torch.long),
            past_key_values=DynamicCache(),
            cache_lengths=[10, 10],
            query_lengths=[2, 2],
            global_offsets=[10, 20],
        )

        results = batch_generate(
            mock_model,
            mock_tokenizer,
            batch_inputs,
            max_new_tokens=5,
        )

        assert len(results) == 2
        assert results[0] == "Generated text"

    def test_batch_generate_with_eos_list(self):
        """Test batch generation with eos_token_id as list."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = [0, 1]  # List of EOS tokens
        mock_tokenizer.decode.return_value = "Text"

        mock_output = Mock()
        mock_output.logits = torch.zeros(2, 1, 100)
        mock_output.logits[:, :, 0] = 10.0
        mock_output.past_key_values = DynamicCache()
        mock_model.return_value = mock_output

        batch_inputs = BatchQueryInputs(
            input_ids=torch.tensor([[1, 2], [3, 4]]),
            position_ids=torch.tensor([[10, 11], [20, 21]]),
            attention_mask=torch.ones(2, 12, dtype=torch.long),
            past_key_values=DynamicCache(),
            cache_lengths=[10, 10],
            query_lengths=[2, 2],
            global_offsets=[10, 20],
        )

        results = batch_generate(
            mock_model,
            mock_tokenizer,
            batch_inputs,
            max_new_tokens=5,
        )

        assert len(results) == 2


class TestMemoryAgentSharedModel:
    """Test MemoryAgent with shared model."""

    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_init_with_shared_model(self, mock_block_class, temp_kv_dir):
        """Test initialization with shared model."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()
        layer_devices = {0: torch.device("cpu")}

        agent = MemoryAgent(
            model_id="test-model",
            shared_model=mock_model,
            shared_tokenizer=mock_tokenizer,
            shared_layer_devices=layer_devices,
        )

        assert agent.model is mock_model
        assert agent.tokenizer is mock_tokenizer
        assert agent.layer_devices == layer_devices
        assert not agent._owns_model

    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_init_with_shared_model_no_layer_devices(self, mock_block_class, temp_kv_dir):
        """Test initialization with shared model but no layer devices."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        agent = MemoryAgent(
            model_id="test-model",
            shared_model=mock_model,
            shared_tokenizer=mock_tokenizer,
        )

        assert agent.model is mock_model
        assert agent.layer_devices is not None


class TestMemoryAgentPrepareQueryInputs:
    """Test MemoryAgent.prepare_query_inputs method."""

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_prepare_query_inputs_no_knowledge(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test prepare_query_inputs with no knowledge."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = []

        result = agent.prepare_query_inputs("test question")
        assert result is None

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_prepare_query_inputs_with_merged_cache(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test prepare_query_inputs with merged cache."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{"start": 0, "length": 5}]
        agent.global_offset = 5
        agent.merged_cache = DynamicCache()
        for i in range(4):
            agent.merged_cache.update(
                torch.randn(1, 8, 5, 64),
                torch.randn(1, 8, 5, 64),
                i
            )

        result = agent.prepare_query_inputs("test question")

        assert result is not None
        input_ids, position_ids, attention_mask, base_cache, loaded = result
        assert input_ids.shape[1] == 3
        assert not loaded

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_prepare_query_inputs_loads_from_cpu_cache(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test prepare_query_inputs loads from CPU cache."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{"start": 0, "length": 5}]
        agent.global_offset = 5
        agent.merged_cache = None
        agent.is_active = False
        agent._cpu_cache = [
            (torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64))
            for _ in range(4)
        ]

        result = agent.prepare_query_inputs("test question")

        assert result is not None
        input_ids, position_ids, attention_mask, base_cache, loaded = result
        assert loaded


class TestMemoryAgentGetCacheForBatch:
    """Test MemoryAgent.get_cache_for_batch method."""

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_get_cache_for_batch_no_knowledge(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test get_cache_for_batch with no knowledge."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = []

        result = agent.get_cache_for_batch()
        assert result is None

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_get_cache_for_batch_with_merged_cache(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test get_cache_for_batch with merged cache."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{"start": 0, "length": 5}]
        agent.global_offset = 10
        agent.merged_cache = DynamicCache()
        for i in range(4):
            agent.merged_cache.update(
                torch.randn(1, 8, 5, 64),
                torch.randn(1, 8, 5, 64),
                i
            )

        result = agent.get_cache_for_batch()

        assert result is not None
        cache, global_offset, loaded = result
        assert global_offset == 10
        assert not loaded

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_get_cache_for_batch_active_agent_no_cache(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test get_cache_for_batch with active agent but no cache."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{"start": 0, "length": 5}]
        agent.merged_cache = None
        agent.is_active = True

        result = agent.get_cache_for_batch()
        assert result is None


class TestMemoryAgentFormatQueryForBatch:
    """Test MemoryAgent.format_query_for_batch method."""

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_format_query_for_batch(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test format_query_for_batch."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")

        input_ids, query_len = agent.format_query_for_batch("test question")

        assert input_ids.shape[1] == 5
        assert query_len == 5


class TestMemoryAgentCleanupAfterQuery:
    """Test MemoryAgent.cleanup_after_query method."""

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_cleanup_after_query(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test cleanup_after_query."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent._cpu_cache = [(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64))]

        cache = DynamicCache()
        for i in range(4):
            cache.update(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64), i)

        agent.cleanup_after_query(cache, cache_loaded_from_disk=True)

        assert agent._cpu_cache is None


class TestMemoryAgentGetOriginalTexts:
    """Test MemoryAgent.get_original_texts method."""

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_get_original_texts_active_agent(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test get_original_texts with active agent."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.original_texts = ["text1", "text2"]

        texts = agent.get_original_texts()
        assert texts == ["text1", "text2"]

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_get_original_texts_inactive_agent_loads_from_cache(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test get_original_texts loads from cache for inactive agent."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_block = Mock()
        mock_block.load_cache.return_value = {"original_texts": ["loaded1", "loaded2"]}
        mock_block_class.return_value = mock_block

        agent = MemoryAgent(model_id="test-model")
        agent.original_texts = []
        agent.is_active = False

        texts = agent.get_original_texts()
        assert texts == ["loaded1", "loaded2"]


class TestMemoryAgentPreloadCache:
    """Test MemoryAgent.preload_cache method."""

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_preload_cache_active_agent(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test preload_cache with active agent (should skip)."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.is_active = True

        agent.preload_cache()
        # Should not load anything for active agent

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_preload_cache_already_loaded(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test preload_cache when cache already loaded."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.merged_cache = DynamicCache()

        agent.preload_cache()
        # Should skip when merged_cache exists

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_preload_cache_model_mismatch(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test preload_cache with model mismatch."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_block = Mock()
        mock_block.load_cache.return_value = {
            "model_id": "different-model",
            "merged_cache": [(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64))]
        }
        mock_block_class.return_value = mock_block

        agent = MemoryAgent(model_id="test-model")
        agent.is_active = False
        agent.merged_cache = None

        agent.preload_cache()
        assert agent._cpu_cache is None


class TestMemoryAgentLoadFromBlock:
    """Test MemoryAgent loading from existing block."""

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_load_from_block_model_mismatch(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test loading from block with model mismatch raises error."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_block = Mock()
        mock_block.load_cache.return_value = {
            "model_id": "different-model",
            "global_offset": 10,
            "saved_chunks": [],
            "chunk_number": 1,
            "original_texts": [],
        }
        mock_block_class.return_value = mock_block

        with pytest.raises(ValueError, match="Model mismatch"):
            MemoryAgent(
                model_id="test-model",
                load_from_block_id="12345678-1234-5678-1234-567812345678",
                load_timestamp="20240101_120000",
            )

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_load_from_block_with_merged_cache(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test loading from block with merged cache."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_block = Mock()
        mock_block.load_cache.return_value = {
            "model_id": "test-model",
            "global_offset": 10,
            "saved_chunks": [{"start": 0, "length": 10}],
            "chunk_number": 1,
            "original_texts": ["text1"],
            "merged_cache": [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)) for _ in range(4)],
        }
        mock_block_class.return_value = mock_block

        agent = MemoryAgent(
            model_id="test-model",
            load_from_block_id="12345678-1234-5678-1234-567812345678",
            load_timestamp="20240101_120000",
        )

        assert agent.global_offset == 10
        assert agent.merged_cache is not None


class TestMemoryAgentAgentGenerate:
    """Test MemoryAgent._agent_generate edge cases."""

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_agent_generate_no_knowledge(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test _agent_generate with no knowledge."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = []

        result = agent._agent_generate(question="test")
        assert result == "No knowledge available."

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_agent_generate_neither_question_nor_instruction(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test _agent_generate without question or instruction raises error."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model

        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{"start": 0, "length": 5}]
        agent.merged_cache = DynamicCache()
        for i in range(4):
            agent.merged_cache.update(
                torch.randn(1, 8, 5, 64),
                torch.randn(1, 8, 5, 64),
                i
            )

        with pytest.raises(ValueError, match="Either question or instruction"):
            agent._agent_generate()


class TestMemoryAgentAddKnowledge:
    """Test MemoryAgent._add_knowledge edge cases."""

    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_add_knowledge_block_becomes_full(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test _add_knowledge when block becomes full."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = create_mock_model()
        mock_output = Mock()
        mock_output.past_key_values = [(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)) for _ in range(4)]
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        mock_block = Mock()
        mock_block.block_used = 95  # Near the limit
        mock_block.block_id = "test-block-id"
        mock_block_class.return_value = mock_block

        agent = MemoryAgent(model_id="test-model", model_context_window=100)
        agent.block_size = 100

        result = agent._add_knowledge(["Test chunk"])

        # Block should be full after adding 5 tokens to 95 used
        assert result is True
