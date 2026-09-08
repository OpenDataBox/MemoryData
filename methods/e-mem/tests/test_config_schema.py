"""Tests for configuration schema validation."""

import pytest
from pydantic import ValidationError

from src.config.schema import (
    AppConfig,
    HotpotQAEvalConfig,
    LocomoEvalConfig,
    LoggingConfig,
    MemoryConfig,
    ModelConfig,
    OpenAIConfig,
    load_and_validate_config,
)


class TestOpenAIConfig:
    """Test OpenAIConfig validation."""

    def test_valid_config(self):
        """Test valid OpenAI configuration."""
        config = OpenAIConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-4o-mini"

    def test_custom_values(self):
        """Test OpenAI config with custom values."""
        config = OpenAIConfig(
            api_key="custom-key",
            base_url="https://custom.api.com/v1",
            model="gpt-4",
        )
        assert config.api_key == "custom-key"
        assert config.base_url == "https://custom.api.com/v1"
        assert config.model == "gpt-4"

    def test_missing_api_key_fails(self):
        """Test that missing api_key raises error."""
        with pytest.raises(ValidationError):
            OpenAIConfig()


class TestModelConfig:
    """Test ModelConfig validation."""

    def test_valid_config(self):
        """Test valid model configuration."""
        config = ModelConfig(
            memory_agent_model={
                "model_id": "Qwen/Qwen3-4B",
                "model_context_window": 32768,
            },
            general_model={"openai_config": {"api_key": "test-key"}},
        )
        assert config.memory_agent_model.model_id == "Qwen/Qwen3-4B"
        assert config.memory_agent_model.model_context_window == 32768
        assert config.memory_agent_model.attn_implementation == "sdpa"
        assert config.memory_agent_model.device_map == "auto"

    def test_role_based_config_defaults_to_general_model(self):
        """Test that role configs fall back to general_model."""
        config = ModelConfig(
            memory_agent_model={
                "model_id": "Qwen/Qwen3-4B",
                "model_context_window": 32768,
            },
            general_model={"openai_config": {"api_key": "test-key", "model": "general"}},
        )
        assert config.get_memory_agent_config().model_id == "Qwen/Qwen3-4B"
        assert config.get_manager_openai_config().api_key == "test-key"
        assert config.get_manager_openai_config().model == "general"
        assert config.get_aggregator_openai_config().model == "general"
        assert config.get_question_answer_openai_config().model == "general"
        assert config.get_router_openai_config().model == "general"

    def test_role_overrides_take_precedence(self):
        """Test that optional role overrides override general_model."""
        config = ModelConfig(
            memory_agent_model={"model_id": "Qwen/Qwen3-4B"},
            general_model={"openai_config": {"api_key": "test-key", "model": "general"}},
            aggregator_model={
                "openai_config": {"api_key": "test-key", "model": "aggregator"}
            },
        )
        assert config.get_manager_openai_config().model == "general"
        assert config.get_aggregator_openai_config().model == "aggregator"

    def test_with_quantization_config(self):
        """Test model config with quantization."""
        config = ModelConfig(
            memory_agent_model={
                "model_id": "test-model",
                "quantization_config": {"load_in_4bit": True},
            },
            general_model={"openai_config": {"api_key": "test-key"}},
        )
        assert config.memory_agent_model.quantization_config == {"load_in_4bit": True}

    def test_memory_agent_model_context_window(self):
        """Test role-specific memory agent context window."""
        config = ModelConfig(
            memory_agent_model={
                "model_id": "test-model",
                "model_context_window": 65536,
            },
            general_model={"openai_config": {"api_key": "test-key"}},
        )
        assert config.get_memory_agent_config().model_context_window == 65536


class TestMemoryConfig:
    """Test MemoryConfig validation."""

    def test_default_values(self):
        """Test MemoryConfig default values."""
        config = MemoryConfig()
        assert config.storage_mode == "kv_cache"
        assert config.clean_cache_first is True
        assert config.overlap_ratio == 0.1
        assert config.overlap_mode == "chunk"
        assert config.block_size_ratio == 0.125
        assert config.max_concurrent_gpu_operations == 2
        assert config.max_memory_segments == 5
        assert config.max_blocks == 5

    def test_text_storage_mode(self):
        """Test text storage mode."""
        config = MemoryConfig(storage_mode="text")
        assert config.storage_mode == "text"

    def test_invalid_storage_mode_fails(self):
        """Test that invalid storage_mode raises error."""
        with pytest.raises(ValidationError):
            MemoryConfig(storage_mode="invalid")

    def test_overlap_ratio_validation(self):
        """Test overlap_ratio validation."""
        # Valid values
        MemoryConfig(overlap_ratio=0.0)
        MemoryConfig(overlap_ratio=0.5)

        # Invalid values
        with pytest.raises(ValidationError):
            MemoryConfig(overlap_ratio=-0.1)
        with pytest.raises(ValidationError):
            MemoryConfig(overlap_ratio=0.6)

    def test_block_size_ratio_validation(self):
        """Test block_size_ratio validation."""
        # Valid values
        MemoryConfig(block_size_ratio=0.1)
        MemoryConfig(block_size_ratio=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            MemoryConfig(block_size_ratio=0.0)
        with pytest.raises(ValidationError):
            MemoryConfig(block_size_ratio=1.1)

    def test_block_efficiency_validation(self):
        """Test block_size_ratio + overlap_ratio sum validation at AppConfig level."""
        # Valid combination at memory config level
        MemoryConfig(block_size_ratio=0.5, overlap_ratio=0.3)

        # Invalid: sum > 0.9 - validated at AppConfig level
        with pytest.raises(ValidationError):
            AppConfig(
                model=ModelConfig(
                    memory_agent_model={"model_id": "test"},
                    general_model={"openai_config": {"api_key": "test"}},
                ),
                memory=MemoryConfig(block_size_ratio=0.8, overlap_ratio=0.2),
            )

    def test_token_overlap_mode(self):
        """Test token overlap mode."""
        config = MemoryConfig(overlap_mode="token")
        assert config.overlap_mode == "token"

    def test_invalid_overlap_mode_fails(self):
        """Test that invalid overlap_mode raises error."""
        with pytest.raises(ValidationError):
            MemoryConfig(overlap_mode="invalid")


class TestLocomoEvalConfig:
    """Test LocomoEvalConfig validation."""

    def test_valid_config(self):
        """Test valid LoComo eval configuration."""
        config = LocomoEvalConfig(
            dataset_path="data/locomo.json", output_dir="results/"
        )
        assert config.dataset_path == "data/locomo.json"
        assert config.output_dir == "results/"
        assert config.ratio == 1.0
        assert config.categories == [1, 2, 3, 4, 5]

    def test_custom_categories(self):
        """Test custom categories."""
        config = LocomoEvalConfig(
            dataset_path="data.json", output_dir="out/", categories=[1, 3, 5]
        )
        assert config.categories == [1, 3, 5]

    def test_with_specific_questions(self):
        """Test with specific questions path."""
        config = LocomoEvalConfig(
            dataset_path="data.json",
            output_dir="out/",
            specific_questions_path="questions.json",
        )
        assert config.specific_questions_path == "questions.json"

    def test_invalid_category_raises_error(self):
        """Test invalid category raises validation error."""
        with pytest.raises(ValidationError):
            LocomoEvalConfig(
                dataset_path="data.json",
                output_dir="out/",
                categories=[0, 6],  # Invalid: must be 1-5
            )


class TestHotpotQAEvalConfig:
    """Test HotpotQAEvalConfig validation."""

    def test_valid_config(self):
        """Test valid HotpotQA eval configuration."""
        config = HotpotQAEvalConfig(
            dataset_path="data/hotpotqa.json", output_dir="results/"
        )
        assert config.dataset_path == "data/hotpotqa.json"
        assert config.output_dir == "results/"
        assert config.ratio == 1.0
        assert config.max_tokens_per_chunk == 2048


class TestLoggingConfig:
    """Test LoggingConfig validation."""

    def test_valid_config(self):
        """Test valid logging configuration."""
        config = LoggingConfig(log_dir="logs/")
        assert config.log_dir == "logs/"
        assert config.log_level == "INFO"

    def test_custom_log_level(self):
        """Test custom log level."""
        config = LoggingConfig(log_dir="logs/", log_level="DEBUG")
        assert config.log_level == "DEBUG"


class TestAppConfig:
    """Test full AppConfig validation."""

    def test_valid_full_config(self):
        """Test valid full application configuration."""
        config = AppConfig(
            tokenizer={"model_id": "test-model"},
            model=ModelConfig(
                memory_agent_model={"model_id": "test-model"},
                general_model={"openai_config": {"api_key": "test-key"}},
                question_answer_model={"openai_config": {"api_key": "test-key"}},
            ),
            memory=MemoryConfig(),
            locomo_eval=LocomoEvalConfig(
                dataset_path="data.json", output_dir="out/"
            ),
            hotpotqa_eval=HotpotQAEvalConfig(
                dataset_path="data.json", output_dir="out/"
            ),
            logging=LoggingConfig(log_dir="logs/"),
        )
        assert config.model.memory_agent_model.model_id == "test-model"
        assert config.memory.storage_mode == "kv_cache"

    def test_with_max_memory(self):
        """Test config with max_memory."""
        config = AppConfig(
            tokenizer={"model_id": "test-model"},
            model=ModelConfig(
                memory_agent_model={"model_id": "test-model"},
                general_model={"openai_config": {"api_key": "test-key"}},
                question_answer_model={"openai_config": {"api_key": "test-key"}},
            ),
            max_memory={"0": "20GB", "1": "20GB"},
            memory=MemoryConfig(),
            locomo_eval=LocomoEvalConfig(
                dataset_path="data.json", output_dir="out/"
            ),
            hotpotqa_eval=HotpotQAEvalConfig(
                dataset_path="data.json", output_dir="out/"
            ),
            logging=LoggingConfig(log_dir="logs/"),
        )
        assert config.max_memory == {"0": "20GB", "1": "20GB"}

    def test_minimal_config(self):
        """Test minimal valid configuration (optional fields use defaults)."""
        config = AppConfig(
            model=ModelConfig(
                memory_agent_model={"model_id": "test-model"},
                general_model={"openai_config": {"api_key": "test-key"}},
            ),
        )
        assert config.model.memory_agent_model.model_id == "test-model"
        assert config.memory.storage_mode == "kv_cache"
        assert config.locomo_eval is None
        assert config.hotpotqa_eval is None

    def test_evaluation_uses_general_model_when_qa_override_missing(self):
        """Test evaluation configs can rely on general_model for final answers."""
        config = AppConfig(
            tokenizer={"model_id": "test-model"},
            model=ModelConfig(
                memory_agent_model={"model_id": "test-model"},
                general_model={
                    "openai_config": {"api_key": "test-key", "model": "general-qa"}
                },
            ),
            locomo_eval=LocomoEvalConfig(
                dataset_path="data.json", output_dir="out/"
            ),
        )
        assert config.model.get_question_answer_openai_config().model == "general-qa"

    def test_to_chat_manager_kwargs_uses_general_model_defaults(self):
        """Test runtime kwargs inherit manager/aggregator/router from general_model."""
        config = AppConfig(
            tokenizer={"model_id": "test-model"},
            model=ModelConfig(
                memory_agent_model={"model_id": "test-model"},
                general_model={
                    "openai_config": {"api_key": "test-key", "model": "general"}
                },
            ),
        )
        kwargs = config.to_chat_manager_kwargs()
        assert kwargs["chat_openai_config"]["model"] == "general"
        assert kwargs["aggregator_openai_config"]["model"] == "general"
        assert kwargs["router_openai_config"]["model"] == "general"

    def test_to_chat_manager_kwargs_uses_role_override_when_present(self):
        """Test runtime kwargs prefer role overrides over general_model."""
        config = AppConfig(
            tokenizer={"model_id": "test-model"},
            model=ModelConfig(
                memory_agent_model={"model_id": "test-model"},
                general_model={
                    "openai_config": {"api_key": "test-key", "model": "general"}
                },
                aggregator_model={
                    "openai_config": {"api_key": "test-key", "model": "agg"}
                },
            ),
        )
        kwargs = config.to_chat_manager_kwargs()
        assert kwargs["chat_openai_config"]["model"] == "general"
        assert kwargs["aggregator_openai_config"]["model"] == "agg"


class TestLoadAndValidateConfig:
    """Test load_and_validate_config function."""

    def test_load_valid_config(self):
        """Test loading valid configuration."""
        raw_config = {
            "model": {
                "memory_agent_model": {"model_id": "test-model"},
                "general_model": {"openai_config": {"api_key": "test-key"}},
                "question_answer_model": {"openai_config": {"api_key": "test-key"}},
            },
            "memory": {"storage_mode": "kv_cache"},
            "locomo_eval": {"dataset_path": "data.json", "output_dir": "out/"},
            "hotpotqa_eval": {"dataset_path": "data.json", "output_dir": "out/"},
            "logging": {"log_dir": "logs/"},
        }
        config = load_and_validate_config(raw_config)
        assert isinstance(config, AppConfig)
        assert config.model.memory_agent_model.model_id == "test-model"

    def test_load_invalid_config_raises_error(self):
        """Test that invalid config raises ValidationError."""
        raw_config = {
            "model": {
                "general_model": {"openai_config": {"api_key": "test-key"}},
            },
            "memory": {},
            "locomo_eval": {"dataset_path": "data.json", "output_dir": "out/"},
            "hotpotqa_eval": {"dataset_path": "data.json", "output_dir": "out/"},
            "logging": {"log_dir": "logs/"},
        }
        with pytest.raises(ValidationError):
            load_and_validate_config(raw_config)

    def test_load_minimal_config(self):
        """Test loading minimal configuration."""
        raw_config = {
            "model": {
                "memory_agent_model": {"model_id": "test-model"},
                "general_model": {"openai_config": {"api_key": "test-key"}},
            },
        }
        config = load_and_validate_config(raw_config)
        assert config.model.memory_agent_model.model_id == "test-model"
        assert config.memory.storage_mode == "kv_cache"  # Default


class TestValidateHelpers:
    """Test validation helper functions."""

    def test_validate_openai_config(self):
        """Test validate_openai_config function."""
        from src.config.schema import validate_openai_config

        config = validate_openai_config({"api_key": "test-key"})
        assert config.api_key == "test-key"
        assert config.model == "gpt-4o-mini"

    def test_validate_memory_config(self):
        """Test validate_memory_config function."""
        from src.config.schema import validate_memory_config

        config = validate_memory_config({"storage_mode": "text"})
        assert config.storage_mode == "text"
        assert config.overlap_ratio == 0.1
