"""Tests for config.py module."""
import os
import tempfile

import pytest
import yaml

from config import (
    ConfigurationError,
    build_chat_manager_kwargs,
    ensure_app_config,
    get_config,
    get_config_path,
    load_raw_config,
    load_validated_config,
)


class TestGetConfigPath:
    """Test get_config_path function."""

    def test_get_config_path_returns_string(self):
        """Test that get_config_path returns a string."""
        path = get_config_path()
        assert isinstance(path, str)
        assert path.endswith("config.yaml")


class TestLoadRawConfig:
    """Test load_raw_config function."""

    def test_load_missing_config_returns_empty(self):
        """Test loading missing config returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "nonexistent.yaml")
            result = load_raw_config(config_path)
            assert result == {}

    def test_load_missing_config_can_be_silent(self, caplog):
        """Test missing config warning can be suppressed for import-time init."""
        result = load_raw_config("/tmp/definitely-missing-config.yaml", warn_if_missing=False)
        assert result == {}
        assert "Config file not found" not in caplog.text

    def test_load_valid_config(self):
        """Test loading valid config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            config_data = {"model": {"model_id": "test-model"}}
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            result = load_raw_config(config_path)
            assert result == config_data

    def test_load_empty_config_returns_empty(self):
        """Test loading empty config file returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "empty.yaml")
            with open(config_path, "w") as f:
                f.write("")

            result = load_raw_config(config_path)
            assert result == {}

    def test_load_invalid_yaml_raises_error(self):
        """Test loading invalid YAML raises ConfigurationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "invalid.yaml")
            with open(config_path, "w") as f:
                f.write("invalid: yaml: content: [")

            with pytest.raises(ConfigurationError, match="Invalid YAML"):
                load_raw_config(config_path)

    def test_load_config_with_none_path(self):
        """Test loading config with None path uses default."""
        # This tests the default path behavior
        result = load_raw_config(None)
        # Should either load the default config or return empty
        assert isinstance(result, dict)


class TestLoadValidatedConfig:
    """Test load_validated_config function."""

    def test_load_validated_config_success(self):
        """Test loading and validating a valid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            config_data = {
                "model": {
                    "memory_agent_model": {"model_id": "test-model"},
                    "general_model": {"openai_config": {"api_key": "test-key"}},
                }
            }
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            result = load_validated_config(config_path)
            assert result.model.memory_agent_model.model_id == "test-model"

    def test_load_validated_config_validation_error(self):
        """Test loading invalid config raises ConfigurationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "invalid.yaml")
            config_data = {
                "model": {
                    "general_model": {"openai_config": {"api_key": "test-key"}},
                }
            }
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            with pytest.raises(ConfigurationError, match="validation failed"):
                load_validated_config(config_path)


class TestEnsureAppConfig:
    """Test ensure_app_config function."""

    def test_ensure_app_config_with_app_config(self):
        """Test ensure_app_config with AppConfig instance."""
        from src.config import AppConfig, ModelConfig, OpenAIConfig

        config = AppConfig(
            model=ModelConfig(
                memory_agent_model={"model_id": "test-model"},
                general_model={"openai_config": OpenAIConfig(api_key="test-key")},
            )
        )

        result = ensure_app_config(config)
        assert result is config

    def test_ensure_app_config_with_dict(self):
        """Test ensure_app_config with dict."""
        config_dict = {
            "model": {
                "memory_agent_model": {"model_id": "test-model"},
                "general_model": {"openai_config": {"api_key": "test-key"}},
            }
        }

        result = ensure_app_config(config_dict)
        assert result.model.memory_agent_model.model_id == "test-model"


class TestBuildChatManagerKwargs:
    """Test build_chat_manager_kwargs function."""

    def test_build_chat_manager_kwargs_with_dict(self):
        """Test building kwargs from dict."""
        config_dict = {
            "tokenizer": {"model_id": "test-model"},
            "model": {
                "memory_agent_model": {"model_id": "test-model"},
                "general_model": {"openai_config": {"api_key": "test-key"}},
            },
            "memory": {"storage_mode": "kv_cache"},
        }

        result = build_chat_manager_kwargs(config_dict)
        assert isinstance(result, dict)
        assert "model_id" in result


class TestGetConfig:
    """Test get_config function."""

    def test_get_config_caching(self):
        """Test that get_config caches the result."""
        import config as config_module

        # Reset cache
        config_module._config_cache = None

        result1 = get_config()
        result2 = get_config()

        # Should return the same cached object
        assert result1 is result2

    def test_get_config_reload(self):
        """Test that get_config with reload=True reloads from file."""
        import config as config_module

        # Set a cache
        config_module._config_cache = {"cached": True}

        # Reload should reset the cache
        result = get_config(reload=True)

        # Should be different from the cached value
        assert "cached" not in result


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_configuration_error_message(self):
        """Test ConfigurationError message."""
        error = ConfigurationError("Test error message")
        assert str(error) == "Test error message"

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from Exception."""
        error = ConfigurationError("Test")
        assert isinstance(error, Exception)


class TestGlobalVariables:
    """Test global configuration variables."""

    def test_default_constants_exist(self):
        """Test that default constants are defined."""
        from config import (
            DEFAULT_BLOCK_SIZE_RATIO,
            DEFAULT_OVERLAP_RATIO,
            MAX_CONCURRENT_GPU_OPERATIONS,
        )

        assert isinstance(MAX_CONCURRENT_GPU_OPERATIONS, int)
        assert isinstance(DEFAULT_OVERLAP_RATIO, float)
        assert isinstance(DEFAULT_BLOCK_SIZE_RATIO, float)
