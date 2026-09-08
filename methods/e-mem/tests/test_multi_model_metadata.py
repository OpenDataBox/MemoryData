"""Test multi-model metadata handling."""
import os
from unittest.mock import Mock, patch

from src.memory.core.loop_handler import MemoryHandler
from src.memory.kv_block_manager.metadata import (
    load_agents_metadata,
    save_agents_metadata,
)


class TestMultiModelMetadata:
    """Test metadata handling with multiple models and sessions."""
    
    def test_metadata_preserves_other_models(self, temp_kv_dir):
        """Test that saving metadata preserves other sessions' blocks."""
        # Set up session A
        os.environ['EVAL_SESSION_ID'] = 'session-a'
        initial_metadata = [
            {
                "block_id": "block-a-1",
                "timestamp": "20231201_120000",
                "model_id": "model-a",
                "session_id": "session-a",
                "summary": "Summary A",
                "is_active": False,
                "block_used": 1000,
                "chunk_number": 10,
                "block_size_ratio": 0.125
            }
        ]
        save_agents_metadata(initial_metadata)
        
        # Switch to session B
        os.environ['EVAL_SESSION_ID'] = 'session-b'
        try:
            with patch("src.memory.core.loop_handler.Router"), \
                 patch("src.memory.core.loop_handler.AddHandler"):
                handler_b = MemoryHandler(
                    model_id="model-b",
                    openai_config={"api_key": "test"},
                    clean_cache_first=False
                )
                
                mock_agent_b = Mock()
                mock_agent_b.model_id = "model-b"
                mock_agent_b.current_block = Mock()
                mock_agent_b.current_block.block_id = "block-b-1"
                mock_agent_b.current_block.create_timestamp = "20231201_130000"
                mock_agent_b.current_block.block_used = 2000
                mock_agent_b.chunk_number = 20
                mock_agent_b.summary = "Summary B"
                mock_agent_b.saved_chunks = [{"start": 0, "length": 100}]
                mock_agent_b.is_active = False
                mock_agent_b.block_size_ratio = 0.125
                
                handler_b.inactive_memory_agents = [mock_agent_b]
                handler_b._save_metadata()
            
            final_metadata = load_agents_metadata()
            assert len(final_metadata) == 2
            
            model_a_blocks = [m for m in final_metadata if m["model_id"] == "model-a"]
            model_b_blocks = [m for m in final_metadata if m["model_id"] == "model-b"]
            
            assert len(model_a_blocks) == 1
            assert len(model_b_blocks) == 1
            assert model_a_blocks[0]["block_id"] == "block-a-1"
            assert model_b_blocks[0]["block_id"] == "block-b-1"
            
        finally:
            if 'EVAL_SESSION_ID' in os.environ:
                del os.environ['EVAL_SESSION_ID']
    
    def test_metadata_updates_current_model_only(self, temp_kv_dir):
        """Test that updating metadata only affects current session."""
        os.environ['EVAL_SESSION_ID'] = 'session-a'
        initial_metadata = [
            {
                "block_id": "block-a-1",
                "timestamp": "20231201_120000",
                "model_id": "model-a",
                "session_id": "session-a",
                "summary": "Summary A1",
                "is_active": False,
                "block_used": 1000,
                "chunk_number": 10,
                "block_size_ratio": 0.125
            },
            {
                "block_id": "block-b-1",
                "timestamp": "20231201_130000",
                "model_id": "model-b",
                "session_id": "session-b",
                "summary": "Summary B1",
                "is_active": False,
                "block_used": 2000,
                "chunk_number": 20,
                "block_size_ratio": 0.125
            }
        ]
        save_agents_metadata(initial_metadata)
        
        try:
            with patch("src.memory.core.loop_handler.Router"), \
                 patch("src.memory.core.loop_handler.AddHandler"), \
                 patch("src.memory.core.loop_handler.MemoryAgent"):
                handler = MemoryHandler(
                    model_id="model-a",
                    openai_config={"api_key": "test"},
                    clean_cache_first=False
                )
                
                mock_agent_a2 = Mock()
                mock_agent_a2.model_id = "model-a"
                mock_agent_a2.current_block = Mock()
                mock_agent_a2.current_block.block_id = "block-a-2"
                mock_agent_a2.current_block.create_timestamp = "20231201_140000"
                mock_agent_a2.current_block.block_used = 1500
                mock_agent_a2.chunk_number = 15
                mock_agent_a2.summary = "Summary A2"
                mock_agent_a2.saved_chunks = [{"start": 0, "length": 100}]
                mock_agent_a2.is_active = False
                mock_agent_a2.block_size_ratio = 0.125
                
                handler.inactive_memory_agents = [mock_agent_a2]
                handler._save_metadata()
            
            final_metadata = load_agents_metadata()
            
            model_a_blocks = [m for m in final_metadata if m["model_id"] == "model-a"]
            model_b_blocks = [m for m in final_metadata if m["model_id"] == "model-b"]
            
            assert len(model_a_blocks) == 1
            assert len(model_b_blocks) == 1
            assert model_a_blocks[0]["block_id"] == "block-a-2"
            assert model_b_blocks[0]["block_id"] == "block-b-1"
            
        finally:
            if 'EVAL_SESSION_ID' in os.environ:
                del os.environ['EVAL_SESSION_ID']
    
    def test_load_filters_by_model(self, temp_kv_dir):
        """Test that loading only loads current session's blocks."""
        os.environ['EVAL_SESSION_ID'] = 'session-a'
        metadata = [
            {
                "block_id": "block-a-1",
                "timestamp": "20231201_120000",
                "model_id": "model-a",
                "session_id": "session-a",
                "summary": "Summary A",
                "is_active": False,
                "block_used": 1000,
                "chunk_number": 10,
                "block_size_ratio": 0.125
            },
            {
                "block_id": "block-b-1",
                "timestamp": "20231201_130000",
                "model_id": "model-b",
                "session_id": "session-b",
                "summary": "Summary B",
                "is_active": False,
                "block_used": 2000,
                "chunk_number": 20,
                "block_size_ratio": 0.125
            }
        ]
        save_agents_metadata(metadata)
        
        try:
            with patch("src.memory.core.loop_handler.Router"), \
                 patch("src.memory.core.loop_handler.AddHandler"), \
                 patch("src.memory.core.loop_handler.MemoryAgent") as mock_agent_class:
                
                _ = MemoryHandler(
                    model_id="model-a",
                    openai_config={"api_key": "test"},
                    clean_cache_first=False
                )
                
                assert mock_agent_class.call_count == 1
                call_kwargs = mock_agent_class.call_args.kwargs
                assert call_kwargs["load_from_block_id"] == "block-a-1"
            
        finally:
            if 'EVAL_SESSION_ID' in os.environ:
                del os.environ['EVAL_SESSION_ID']
