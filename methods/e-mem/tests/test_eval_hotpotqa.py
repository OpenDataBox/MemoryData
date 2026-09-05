"""
Tests for HotpotQA evaluation logic
"""
import json
import os
import tempfile
from unittest.mock import Mock, patch

from evaluation.hotpotqa.eval_hotpotqa import (
    _calculate_f1,
    build_context_chunks_for_sample,
    f1_score,
    load_hotpotqa,
    make_prompt,
    normalize_answer,
    process_sample,
    qa_f1_score,
)


class TestAnswerNormalization:
    def test_normalize_answer_basic(self):
        assert normalize_answer("The Answer") == "answer"
        assert normalize_answer("A simple test") == "simple test"
        assert normalize_answer("An example!") == "example"
    
    def test_normalize_answer_punctuation(self):
        assert normalize_answer("Hello, World!") == "hello world"
        assert normalize_answer("test-case") == "testcase"


class TestF1Score:
    def test_f1_score_exact_match(self):
        pred = ["hello", "world"]
        gold = ["hello", "world"]
        assert f1_score(pred, gold) == 1.0
    
    def test_f1_score_no_match(self):
        pred = ["hello"]
        gold = ["world"]
        assert f1_score(pred, gold) == 0.0
    
    def test_f1_score_partial_match(self):
        pred = ["hello", "world"]
        gold = ["hello", "there"]
        score = f1_score(pred, gold)
        assert 0 < score < 1


class TestQAF1Score:
    def test_qa_f1_score_exact(self):
        assert qa_f1_score("Paris", "Paris") == 1.0
    
    def test_qa_f1_score_case_insensitive(self):
        assert qa_f1_score("Paris", "paris") == 1.0
    
    def test_qa_f1_score_with_articles(self):
        score = qa_f1_score("The Paris", "Paris")
        assert score == 1.0


class TestCalculateF1:
    def test_calculate_f1_single_answer(self):
        pred = "Paris"
        gold = ["Paris"]
        assert _calculate_f1(pred, gold) == 1.0
    
    def test_calculate_f1_multiple_answers(self):
        pred = "Paris"
        gold = ["London", "Paris", "Berlin"]
        assert _calculate_f1(pred, gold) == 1.0
    
    def test_calculate_f1_no_match(self):
        pred = "Tokyo"
        gold = ["Paris", "London"]
        assert _calculate_f1(pred, gold) == 0.0


class TestLoadHotpotQA:
    def test_load_hotpotqa(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {
                    "index": 0,
                    "context": "Test context",
                    "input": "Test question?",
                    "answers": ["Test answer"]
                }
            ]
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = load_hotpotqa(temp_path)
            assert len(result) == 1
            assert result[0]["index"] == 0
            assert result[0]["context"] == "Test context"
            assert result[0]["input"] == "Test question?"
            assert result[0]["answers"] == ["Test answer"]
            assert result[0]["_id"] == "hotpotqa-0"
        finally:
            os.unlink(temp_path)


class TestContextChunking:
    def test_build_context_chunks_small(self):
        sample = {"context": "Short context"}
        chunks = build_context_chunks_for_sample(sample, max_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == "Short context"
    
    def test_build_context_chunks_empty(self):
        sample = {"context": ""}
        chunks = build_context_chunks_for_sample(sample, max_tokens=1000)
        assert len(chunks) == 0
    
    def test_build_context_chunks_large(self):
        # Create a large context that will be split
        large_context = " ".join(["word"] * 5000)
        sample = {"context": large_context}
        chunks = build_context_chunks_for_sample(sample, max_tokens=100)
        assert len(chunks) > 1


class TestPromptGeneration:
    def test_make_prompt(self):
        context = "Paris is the capital of France."
        question = "What is the capital of France?"
        prompt = make_prompt(context, question)
        
        assert "Paris is the capital of France." in prompt
        assert "What is the capital of France?" in prompt
        assert "Context:" in prompt
        assert "Question:" in prompt


class TestProcessSample:
    @patch('evaluation.hotpotqa.eval_hotpotqa.create_chat_manager')
    @patch('evaluation.hotpotqa.eval_hotpotqa.OpenAI')
    def test_process_sample_success(self, mock_openai, mock_create_chat):
        # Setup mocks
        mock_agent = Mock()
        mock_agent.add_memory = Mock()
        mock_agent.search_memory = Mock(return_value="Paris is the capital")
        mock_create_chat.return_value = mock_agent
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Paris"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test data
        sample = {
            "_id": "test-1",
            "index": 0,
            "context": "Paris is the capital of France.",
            "input": "What is the capital of France?",
            "answers": ["Paris"]
        }
        
        config = {
            "tokenizer": {"model_id": "test-model"},
            "memory": {"storage_mode": "text"},
            "model": {
                "memory_agent_model": {
                    "openai_config": {"api_key": "test", "base_url": "http://test"},
                    "model_context_window": 32768,
                },
                "general_model": {"openai_config": {"api_key": "test", "base_url": "http://test"}},
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = process_sample(
                sample, 0, tmpdir,
                "test-key", "http://test", "test-model",
                config, max_tokens=1000
            )
            
            assert result["sample_id"] == "test-1"
            assert result["pred"] == "Paris"
            assert result["f1"] == 1.0
            assert "error" not in result
    
    @patch('evaluation.hotpotqa.eval_hotpotqa.create_chat_manager')
    def test_process_sample_error_handling(self, mock_create_chat):
        # Setup mock to raise error
        mock_create_chat.side_effect = Exception("Test error")
        
        sample = {
            "_id": "test-1",
            "index": 0,
            "context": "Test",
            "input": "Test?",
            "answers": ["Test"]
        }
        
        config = {
            "tokenizer": {"model_id": "test-model"},
            "memory": {"storage_mode": "text"},
            "model": {
                "memory_agent_model": {
                    "openai_config": {"api_key": "test", "base_url": "http://test"},
                    "model_context_window": 32768,
                },
                "general_model": {"openai_config": {"api_key": "test", "base_url": "http://test"}},
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = process_sample(
                sample, 0, tmpdir,
                "test-key", "http://test", "test-model",
                config, max_tokens=1000
            )
            
            assert "error" in result
            assert "Test error" in result["error"]
