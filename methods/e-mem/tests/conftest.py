"""Pytest configuration and fixtures."""
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

# Get the tests directory path
TESTS_DIR = Path(__file__).parent
TEST_DATA_DIR = TESTS_DIR / "test_data"


@pytest.fixture
def temp_kv_dir(monkeypatch):
    """Create temporary directory for KV cache in tests folder using environment variable."""
    test_kv_dir = TEST_DATA_DIR / "kv_data"
    test_kv_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable BEFORE any module accesses the path
    monkeypatch.setenv("KV_DATA_DIR", str(test_kv_dir))
    
    yield str(test_kv_dir)
    
    # Cleanup after test
    if test_kv_dir.exists():
        shutil.rmtree(test_kv_dir)


@pytest.fixture
def temp_text_dir(monkeypatch):
    """Create temporary directory for text storage in tests folder using environment variable."""
    test_text_dir = TEST_DATA_DIR / "text_data"
    test_text_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable BEFORE any module accesses the path
    monkeypatch.setenv("TEXT_DATA_DIR", str(test_text_dir))
    
    yield str(test_text_dir)
    
    # Cleanup after test
    if test_text_dir.exists():
        shutil.rmtree(test_text_dir)


@pytest.fixture
def temp_metadata_dir(monkeypatch):
    """Create temporary directory for metadata in tests folder using environment variable."""
    test_kv_dir = TEST_DATA_DIR / "kv_data"
    test_kv_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_file = test_kv_dir / "agents_metadata.json"
    
    # Set environment variable BEFORE any module accesses the path
    monkeypatch.setenv("KV_DATA_DIR", str(test_kv_dir))
    
    yield str(metadata_file)
    
    # Cleanup after test
    if test_kv_dir.exists():
        shutil.rmtree(test_kv_dir)


@pytest.fixture
def temp_all_data_dirs(monkeypatch):
    """Create temporary directories for both KV and text data in tests folder."""
    test_kv_dir = TEST_DATA_DIR / "kv_data"
    test_text_dir = TEST_DATA_DIR / "text_data"
    test_kv_dir.mkdir(parents=True, exist_ok=True)
    test_text_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables BEFORE any module accesses the paths
    monkeypatch.setenv("KV_DATA_DIR", str(test_kv_dir))
    monkeypatch.setenv("TEXT_DATA_DIR", str(test_text_dir))
    
    yield {
        "kv_dir": str(test_kv_dir),
        "text_dir": str(test_text_dir),
        "kv_metadata": str(test_kv_dir / "agents_metadata.json"),
        "text_metadata": str(test_text_dir / "agents_metadata.json"),
    }
    
    # Cleanup after test
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)


@pytest.fixture
def mock_openai_config():
    """Mock OpenAI configuration."""
    return {
        "api_key": "test-key",
        "base_url": "https://test.api.com",
        "model": "gpt-4o-mini"
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "Test response"
    mock_message.tool_calls = None
    mock_response.choices = [Mock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_model():
    """Mock transformer model."""
    mock = Mock()
    mock.device = torch.device("cpu")
    
    # Mock forward pass
    mock_output = Mock()
    mock_output.logits = torch.randn(1, 10, 1000)
    mock_output.past_key_values = [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)) for _ in range(4)]
    mock.return_value = mock_output
    
    return mock


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    mock = Mock()
    mock.eos_token_id = 0
    mock.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock.decode.return_value = "Test decoded text"
    mock.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
    return mock


@pytest.fixture
def sample_cache_state():
    """Sample cache state for testing."""
    return {
        "global_offset": 100,
        "saved_chunks": [
            {
                "cache": [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)) for _ in range(4)],
                "start": 0,
                "length": 50
            },
            {
                "cache": [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)) for _ in range(4)],
                "start": 50,
                "length": 50
            }
        ],
        "chunk_number": 2,
        "model_id": "test-model"
    }


@pytest.fixture
def block_id():
    """Generate test block ID."""
    return uuid.uuid4()


@pytest.fixture
def timestamp():
    """Generate test timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
