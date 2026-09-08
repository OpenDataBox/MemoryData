# Test Suite

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_kv_block.py
```

### Run specific test class
```bash
pytest tests/test_kv_block.py::TestKVBlock
```

### Run specific test method
```bash
pytest tests/test_kv_block.py::TestKVBlock::test_init
```

### Run with coverage
```bash
pytest --cov=src --cov-report=html
```

### Run with verbose output
```bash
pytest -v
```

### Run only unit tests
```bash
pytest -m unit
```

## Test Structure

- `conftest.py` - Shared fixtures and configuration
- `test_kv_block.py` - Tests for KVBlock storage
- `test_base_agent.py` - Tests for BaseAgent with tool calling
- `test_memory_handler.py` - Tests for MemoryHandler components
- `test_router.py` - Tests for Router
- `test_chat_manager.py` - Tests for ChatManager

## Fixtures

Common fixtures available in all tests:

- `temp_kv_dir` - Temporary directory for KV cache
- `mock_openai_config` - Mock OpenAI configuration
- `mock_openai_client` - Mock OpenAI client
- `mock_model` - Mock transformer model
- `mock_tokenizer` - Mock tokenizer
- `sample_cache_state` - Sample cache state
- `block_id` - Test block ID
- `timestamp` - Test timestamp

## Writing New Tests

1. Create test file: `tests/test_<module>.py`
2. Import module to test
3. Create test class: `class Test<Module>:`
4. Write test methods: `def test_<functionality>:`
5. Use fixtures and mocks as needed

Example:
```python
from unittest.mock import Mock, patch
import pytest

class TestMyModule:
    def test_my_function(self, mock_openai_config):
        # Arrange
        with patch("module.dependency") as mock_dep:
            mock_dep.return_value = "expected"
            
            # Act
            result = my_function()
            
            # Assert
            assert result == "expected"
```

## Mocking Guidelines

- Use `unittest.mock.Mock` for simple mocks
- Use `unittest.mock.patch` for patching imports
- Use fixtures for reusable mocks
- Mock external dependencies (OpenAI, transformers)
- Don't mock the code under test

## Coverage Goals

- Aim for >80% code coverage
- Focus on critical paths
- Test error handling
- Test edge cases
