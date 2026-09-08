"""
Tests for OmniMemoryConfig and sub-configurations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch

from omni_memory.core.config import (
    OmniMemoryConfig,
    EmbeddingConfig,
    RetrievalConfig,
    StorageConfig,
    LLMConfig,
    EventConfig,
    EntropyTriggerConfig,
)


# ---------------------------------------------------------------------------
# Sub-config defaults
# ---------------------------------------------------------------------------

class TestSubConfigDefaults:
    def test_embedding_config_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.model_name == "all-MiniLM-L6-v2"
        assert cfg.embedding_dim == 384
        assert cfg.batch_size == 32

    def test_retrieval_config_defaults(self):
        cfg = RetrievalConfig()
        assert cfg.default_top_k == 10
        assert cfg.enable_hybrid_search is True
        assert cfg.enable_graph_traversal is True

    def test_storage_config_defaults(self):
        cfg = StorageConfig()
        assert cfg.base_dir == "./omni_memory_data"
        assert cfg.use_s3 is False

    def test_llm_config_defaults(self):
        cfg = LLMConfig()
        assert cfg.summary_model == "gpt-4o-mini"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 1000

    def test_event_config_defaults(self):
        cfg = EventConfig()
        assert cfg.auto_create_events is True
        assert cfg.event_time_window_seconds == 300.0

    def test_entropy_trigger_config_defaults(self):
        cfg = EntropyTriggerConfig()
        assert cfg.visual_similarity_threshold_high == 0.9
        assert cfg.enable_visual_trigger is True
        assert cfg.enable_audio_trigger is True


# ---------------------------------------------------------------------------
# OmniMemoryConfig
# ---------------------------------------------------------------------------

class TestOmniMemoryConfig:
    def test_default_creation(self):
        cfg = OmniMemoryConfig()
        assert isinstance(cfg.embedding, EmbeddingConfig)
        assert isinstance(cfg.retrieval, RetrievalConfig)
        assert isinstance(cfg.storage, StorageConfig)
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.event, EventConfig)
        assert isinstance(cfg.entropy_trigger, EntropyTriggerConfig)
        assert cfg.debug_mode is False
        assert cfg.log_level == "INFO"
        assert cfg.enable_self_evolution is False

    def test_to_dict(self):
        cfg = OmniMemoryConfig()
        d = cfg.to_dict()
        assert "embedding" in d
        assert "retrieval" in d
        assert "storage" in d
        assert "llm" in d
        assert "event" in d
        assert d["debug_mode"] is False

    def test_from_dict_roundtrip(self):
        original = OmniMemoryConfig()
        original.llm.temperature = 0.7
        original.retrieval.default_top_k = 20
        original.debug_mode = True

        d = original.to_dict()
        restored = OmniMemoryConfig.from_dict(d)

        assert restored.llm.temperature == 0.7
        assert restored.retrieval.default_top_k == 20
        assert restored.debug_mode is True

    def test_set_unified_model(self):
        cfg = OmniMemoryConfig()
        result = cfg.set_unified_model("gpt-4o")
        assert cfg.llm.summary_model == "gpt-4o"
        assert cfg.llm.query_model == "gpt-4o"
        assert cfg.llm.caption_model == "gpt-4o"
        # Should return self for chaining
        assert result is cfg

    def test_enable_evolution(self):
        cfg = OmniMemoryConfig()
        assert cfg.enable_self_evolution is False
        assert cfg.evolution is None

        result = cfg.enable_evolution()
        assert cfg.enable_self_evolution is True
        assert cfg.evolution is not None
        # Should return self for chaining
        assert result is cfg

    def test_env_variable_fallback_for_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}, clear=False):
            cfg = OmniMemoryConfig()
            assert cfg.llm.api_key == "test-key-123"

    def test_env_variable_fallback_for_base_url(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_BASE": "https://custom.api.example.com"},
            clear=False,
        ):
            cfg = OmniMemoryConfig()
            assert cfg.llm.api_base_url == "https://custom.api.example.com"

    def test_explicit_key_overrides_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            cfg = OmniMemoryConfig(llm=LLMConfig(api_key="explicit-key"))
            # Explicit key is set before __post_init__, so it should stay
            assert cfg.llm.api_key == "explicit-key"

    def test_json_roundtrip(self):
        cfg = OmniMemoryConfig()
        cfg.llm.temperature = 0.5
        json_str = cfg.to_json()
        restored = OmniMemoryConfig.from_json(json_str)
        assert restored.llm.temperature == 0.5

    def test_save_and_load_file(self, tmp_path):
        cfg = OmniMemoryConfig()
        cfg.debug_mode = True
        filepath = str(tmp_path / "config.json")
        cfg.save_to_file(filepath)

        loaded = OmniMemoryConfig.from_file(filepath)
        assert loaded.debug_mode is True

    def test_ensure_directories(self, tmp_path):
        cfg = OmniMemoryConfig()
        cfg.storage.base_dir = str(tmp_path / "base")
        cfg.storage.cold_storage_dir = str(tmp_path / "cold")
        cfg.storage.index_dir = str(tmp_path / "index")
        cfg.ensure_directories()

        assert os.path.isdir(str(tmp_path / "base"))
        assert os.path.isdir(str(tmp_path / "cold"))
        assert os.path.isdir(str(tmp_path / "index"))
