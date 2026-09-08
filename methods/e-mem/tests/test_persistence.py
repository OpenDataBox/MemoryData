"""Test KV cache persistence functionality."""
import os
import uuid

import pytest

from src.memory.kv_block_manager.metadata import (
    clear_metadata,
    load_agents_metadata,
    save_agents_metadata,
)


@pytest.fixture(autouse=True)
def cleanup(temp_metadata_dir):
    """Cleanup metadata before and after each test using temp directory."""
    # Set unique test session_id for each test
    test_session = f'test_session_{uuid.uuid4().hex[:8]}'
    os.environ['EVAL_SESSION_ID'] = test_session
    
    yield
    
    # Clean up env
    if 'EVAL_SESSION_ID' in os.environ:
        del os.environ['EVAL_SESSION_ID']


def test_save_and_load_metadata():
    """Test saving and loading metadata."""
    test_data = [
        {
            "block_id": "test-uuid-1",
            "timestamp": "20240101_120000",
            "summary": "Test summary 1",
            "is_active": False,
            "block_used": 1000,
            "chunk_number": 5
        },
        {
            "block_id": "test-uuid-2",
            "timestamp": "20240101_130000",
            "summary": None,  # Active agent has no summary
            "is_active": True,
            "block_used": 500,
            "chunk_number": 3
        }
    ]
    
    save_agents_metadata(test_data)
    loaded_data = load_agents_metadata()
    
    assert len(loaded_data) == 2
    assert loaded_data[0]["block_id"] == "test-uuid-1"
    assert not loaded_data[0]["is_active"]
    assert loaded_data[0]["summary"] is not None
    assert loaded_data[1]["is_active"]
    assert loaded_data[1]["summary"] is None  # Active agent has no summary yet


def test_clear_metadata():
    """Test clearing metadata for current session only."""
    import os

    # Get current session from env (set by fixture)
    current_session = os.environ.get('EVAL_SESSION_ID', 'test_session')
    
    test_data = [{
        "block_id": "test",
        "timestamp": "test",
        "model_id": "test-model",
        "session_id": current_session,
        "summary": None,
        "is_active": True,
        "block_used": 0,
        "chunk_number": 0
    }]
    save_agents_metadata(test_data)
    
    clear_metadata()
    loaded_data = load_agents_metadata()
    
    # Should be cleared (belongs to current session)
    assert len(loaded_data) == 0


def test_active_agent_metadata():
    """Test that active agent metadata is saved without summary."""
    active_agent_data = [{
        "block_id": "active-uuid",
        "timestamp": "20240101_140000",
        "summary": None,
        "is_active": True,
        "block_used": 300,
        "chunk_number": 2
    }]
    
    save_agents_metadata(active_agent_data)
    loaded_data = load_agents_metadata()
    
    assert len(loaded_data) == 1
    assert loaded_data[0]["is_active"]
    assert loaded_data[0]["summary"] is None
    assert loaded_data[0]["block_used"] == 300
    assert loaded_data[0]["chunk_number"] == 2


def test_load_nonexistent_metadata():
    """Test loading non-existent metadata returns empty list."""
    # Clear any existing metadata first
    clear_metadata()
    loaded_data = load_agents_metadata()
    assert loaded_data == []
