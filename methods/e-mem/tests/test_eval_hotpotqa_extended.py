"""Additional tests for HotpotQA evaluation to improve coverage."""
import json
import logging
import os
import tempfile
from unittest.mock import Mock, patch

from evaluation.hotpotqa.eval_hotpotqa import (
    _calculate_f1,
    build_context_chunks_for_sample,
    f1_score,
    force_cleanup_gpu_memory,
    load_hotpotqa,
    make_prompt,
    normalize_answer,
    process_sample,
    qa_f1_score,
    setup_logger,
)


class TestSetupLogger:
    """Test setup_logger function."""

    def test_setup_logger_no_file(self):
        """Test logger setup without file."""
        logger = setup_logger("")
        assert logger is not None
        assert logger.name == "hotpotqa_eval"

    def test_setup_logger_with_file(self):
        """Test logger setup with file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = setup_logger(log_file)

            assert logger is not None
            assert os.path.exists(log_file)

    def test_setup_logger_creates_log_dir(self):
        """Test logger creates log directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs", "nested")
            log_file = os.path.join(log_dir, "test.log")
            setup_logger(log_file)

            assert os.path.exists(log_dir)


class TestForceCleanupGpuMemory:
    """Test force_cleanup_gpu_memory function."""

    def test_force_cleanup_no_cuda(self):
        """Test cleanup when CUDA not available."""
        # Should not raise error
        force_cleanup_gpu_memory()


class TestNormalizeAnswer:
    """Additional tests for normalize_answer."""

    def test_normalize_answer_whitespace(self):
        """Test whitespace normalization."""
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_normalize_answer_mixed_case(self):
        """Test case normalization."""
        assert normalize_answer("HeLLo WoRLD") == "hello world"

    def test_normalize_answer_articles(self):
        """Test article removal."""
        assert normalize_answer("a cat") == "cat"
        assert normalize_answer("an apple") == "apple"
        assert normalize_answer("the world") == "world"

    def test_normalize_answer_combined(self):
        """Test combined normalization."""
        assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"


class TestF1Score:
    """Additional tests for f1_score."""

    def test_f1_score_empty_lists(self):
        """Test F1 with empty lists."""
        score = f1_score([], [])
        assert score == 0

    def test_f1_score_one_empty(self):
        """Test F1 with one empty list."""
        score = f1_score(["hello"], [])
        assert score == 0

    def test_f1_score_repeated_tokens(self):
        """Test F1 with repeated tokens."""
        pred = ["hello", "hello", "world"]
        gold = ["hello", "world"]
        # Precision: 2/3, Recall: 2/2
        # F1 = 2 * (2/3 * 1) / (2/3 + 1) = 2 * 2/3 / 5/3 = 4/5 = 0.8
        score = f1_score(pred, gold)
        assert 0 < score <= 1


class TestQaF1Score:
    """Additional tests for qa_f1_score."""

    def test_qa_f1_score_with_punctuation(self):
        """Test QA F1 with punctuation."""
        score = qa_f1_score("Paris, France", "Paris France")
        assert score == 1.0

    def test_qa_f1_score_different_answers(self):
        """Test QA F1 with different answers."""
        score = qa_f1_score("Paris", "London")
        assert score == 0.0


class TestCalculateF1:
    """Additional tests for _calculate_f1."""

    def test_calculate_f1_empty_gold(self):
        """Test _calculate_f1 with empty gold list."""
        score = _calculate_f1("Paris", [])
        assert score == 0.0

    def test_calculate_f1_best_match(self):
        """Test _calculate_f1 returns best match."""
        pred = "Paris France"
        gold = ["London", "Paris", "Berlin"]
        # Should match "Paris" best
        score = _calculate_f1(pred, gold)
        assert score > 0


class TestLoadHotpotQA:
    """Additional tests for load_hotpotqa."""

    def test_load_hotpotqa_missing_fields(self):
        """Test loading data with missing fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {"context": "Test"},  # Missing index, input, answers
            ]
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result = load_hotpotqa(temp_path)
            assert len(result) == 1
            assert result[0]["index"] == 0  # Default index
            assert result[0]["input"] == ""  # Default empty
            assert result[0]["answers"] == []  # Default empty list
        finally:
            os.unlink(temp_path)

    def test_load_hotpotqa_multiple_items(self):
        """Test loading multiple items."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {"index": 0, "context": "C1", "input": "Q1", "answers": ["A1"]},
                {"index": 1, "context": "C2", "input": "Q2", "answers": ["A2"]},
            ]
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result = load_hotpotqa(temp_path)
            assert len(result) == 2
            assert result[0]["_id"] == "hotpotqa-0"
            assert result[1]["_id"] == "hotpotqa-1"
        finally:
            os.unlink(temp_path)


class TestBuildContextChunks:
    """Additional tests for build_context_chunks_for_sample."""

    def test_build_context_chunks_none_context(self):
        """Test with None context."""
        sample = {"context": None}
        chunks = build_context_chunks_for_sample(sample, max_tokens=1000)
        assert chunks == []

    def test_build_context_chunks_no_context_key(self):
        """Test with missing context key."""
        sample = {}
        chunks = build_context_chunks_for_sample(sample, max_tokens=1000)
        assert chunks == []

    def test_build_context_chunks_custom_logger(self):
        """Test with custom logger."""
        sample = {"context": "Test context"}
        logger = logging.getLogger("test_logger")
        chunks = build_context_chunks_for_sample(sample, max_tokens=1000, logger=logger)
        assert len(chunks) == 1


class TestMakePrompt:
    """Additional tests for make_prompt."""

    def test_make_prompt_structure(self):
        """Test prompt structure."""
        context = "Test context"
        question = "Test question"
        prompt = make_prompt(context, question)

        assert "Instructions:" in prompt
        assert "Analyze the Request" in prompt
        assert "Reasoning" in prompt
        assert "Extraction" in prompt

    def test_make_prompt_special_characters(self):
        """Test prompt with special characters."""
        context = "Context with special: chars! @#$%"
        question = "Question with special: chars! @#$%"
        prompt = make_prompt(context, question)

        assert "Context with special: chars! @#$%" in prompt
        assert "Question with special: chars! @#$%" in prompt


class TestProcessSampleEdgeCases:
    """Edge case tests for process_sample."""

    @patch('evaluation.hotpotqa.eval_hotpotqa.create_chat_manager')
    @patch('evaluation.hotpotqa.eval_hotpotqa.OpenAI')
    def test_process_sample_empty_answers(self, mock_openai, mock_create_chat):
        """Test process_sample with empty answers list."""
        mock_agent = Mock()
        mock_agent.add_memory = Mock()
        mock_agent.search_memory = Mock(return_value="Test summary")
        mock_create_chat.return_value = mock_agent

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        sample = {
            "_id": "test-1",
            "index": 0,
            "context": "Test context",
            "input": "Test question?",
            "answers": []  # Empty answers
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
            assert result["answers"] == []

    @patch('evaluation.hotpotqa.eval_hotpotqa.create_chat_manager')
    @patch('evaluation.hotpotqa.eval_hotpotqa.OpenAI')
    def test_process_sample_empty_context(self, mock_openai, mock_create_chat):
        """Test process_sample with empty context."""
        mock_agent = Mock()
        mock_agent.add_memory = Mock()
        mock_agent.search_memory = Mock(return_value="Test summary")
        mock_create_chat.return_value = mock_agent

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        sample = {
            "_id": "test-1",
            "index": 0,
            "context": "",  # Empty context
            "input": "Test question?",
            "answers": ["Answer"]
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
            assert result["num_chunks"] == 0

    @patch('evaluation.hotpotqa.eval_hotpotqa.create_chat_manager')
    @patch('evaluation.hotpotqa.eval_hotpotqa.OpenAI')
    def test_process_sample_query_error(self, mock_openai, mock_create_chat):
        """Test process_sample handles query error."""
        mock_agent = Mock()
        mock_agent.add_memory = Mock()
        mock_agent.search_memory = Mock(side_effect=Exception("Query error"))
        mock_create_chat.return_value = mock_agent

        sample = {
            "_id": "test-1",
            "index": 0,
            "context": "Test context",
            "input": "Test question?",
            "answers": ["Answer"]
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

    @patch('evaluation.hotpotqa.eval_hotpotqa.create_chat_manager')
    @patch('evaluation.hotpotqa.eval_hotpotqa.OpenAI')
    def test_process_sample_no_pred_answer(self, mock_openai, mock_create_chat):
        """Test process_sample with no predicted answer."""
        mock_agent = Mock()
        mock_agent.add_memory = Mock()
        mock_agent.search_memory = Mock(return_value="Test summary")
        mock_create_chat.return_value = mock_agent

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=""))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        sample = {
            "_id": "test-1",
            "index": 0,
            "context": "Test context",
            "input": "Test question?",
            "answers": ["Answer"]
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

            assert result["f1"] == 0.0


class TestMain:
    @patch("evaluation.hotpotqa.eval_hotpotqa.tqdm", side_effect=lambda items, **_: items)
    @patch("evaluation.hotpotqa.eval_hotpotqa.process_sample")
    @patch("evaluation.hotpotqa.eval_hotpotqa.load_hotpotqa")
    @patch("evaluation.hotpotqa.eval_hotpotqa.setup_logger")
    @patch("evaluation.hotpotqa.eval_hotpotqa.ensure_app_config")
    @patch("evaluation.hotpotqa.eval_hotpotqa.load_raw_config")
    def test_main_uses_benchmark_local_default_config(
        self,
        mock_load_raw_config,
        mock_ensure_app_config,
        mock_setup_logger,
        mock_load_hotpotqa,
        mock_process_sample,
        mock_tqdm,
    ):
        import sys

        from evaluation.hotpotqa.eval_hotpotqa import main

        mock_load_raw_config.return_value = {"model": {}, "hotpotqa_eval": {}, "logging": {}}

        app_config = Mock()
        app_config.model.get_question_answer_openai_config.return_value.model_dump.return_value = {
            "api_key": "test-key",
            "base_url": "http://test",
            "model": "test-model",
        }
        app_config.hotpotqa_eval.dataset_path = "evaluation/eval_data/hotpotqa/eval_50.json"
        app_config.hotpotqa_eval.ratio = 1.0
        app_config.hotpotqa_eval.output_dir = "evaluation/hotpotqa/results"
        app_config.hotpotqa_eval.max_tokens_per_chunk = 1024
        app_config.logging.log_dir = "evaluation/hotpotqa/logs"
        app_config.get_runtime_model_summary.return_value = {
            "memory_agent_model": "test-model",
        }
        mock_ensure_app_config.return_value = app_config

        mock_setup_logger.return_value = Mock()
        mock_load_hotpotqa.return_value = [
            {"_id": "hotpotqa-0", "index": 0, "context": "ctx", "input": "q", "answers": ["a"]}
        ]
        mock_process_sample.return_value = {"sample_id": "hotpotqa-0", "f1": 1.0}

        with patch.object(sys, "argv", ["eval_hotpotqa.py"]):
            with patch("os.makedirs"):
                with patch("builtins.open", create=True):
                    main()

        called_path = mock_load_raw_config.call_args.args[0]
        assert called_path.endswith("evaluation/hotpotqa/config.yaml")
