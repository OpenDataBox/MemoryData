"""Unit tests for evaluation utils."""
import os

import pytest

os.environ.setdefault('HF_HOME', '/mnt/d/AI')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from evaluation.locomo.utils import (
    aggregate_metrics,
    calculate_metrics,
    f1_score,
    normalize_answer,
)


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("The Answer") == "answer"
    
    def test_whitespace(self):
        assert normalize_answer("  A  test  ") == "test"
    
    def test_int_input(self):
        assert normalize_answer(2022) == "2022"
    
    def test_remove_punctuation(self):
        assert normalize_answer("Hello, World!") == "hello world"


class TestF1Score:
    def test_exact_match(self):
        assert f1_score("hello world", "hello world") == 1.0
    
    def test_partial_match(self):
        score = f1_score("hello world", "hello")
        assert 0 < score < 1
    
    def test_no_match(self):
        assert f1_score("hello", "world") == 0
    
    def test_empty_strings(self):
        assert f1_score("", "") == 1


class TestCalculateMetrics:
    def test_with_strings(self):
        metrics = calculate_metrics("The answer is 2022", "2022")
        
        expected_keys = [
            "exact_match", "f1", "rouge1_f", "rouge2_f", "rougeL_f",
            "bleu1", "bleu2", "bleu3", "bleu4", "meteor"
        ]
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        assert 0 <= metrics["exact_match"] <= 1
        assert 0 <= metrics["f1"] <= 1
    
    def test_with_int(self):
        metrics = calculate_metrics("The year is 2022", 2022)
        assert isinstance(metrics, dict)
        assert "f1" in metrics
        assert metrics["f1"] > 0
    
    def test_exact_match(self):
        metrics = calculate_metrics("hello world", "hello world")
        assert metrics["exact_match"] == 1
        assert metrics["f1"] == 1.0
    
    def test_case_insensitive(self):
        metrics = calculate_metrics("Hello World", "hello world")
        assert metrics["exact_match"] == 1
    
    def test_article_removal(self):
        metrics = calculate_metrics("the answer", "answer")
        assert metrics["exact_match"] == 1


class TestAggregateMetrics:
    def test_basic_aggregation(self):
        all_metrics = [
            {"f1": 0.8, "exact_match": 1, "bleu1": 0.7},
            {"f1": 0.6, "exact_match": 0, "bleu1": 0.5},
            {"f1": 0.9, "exact_match": 1, "bleu1": 0.8},
        ]
        all_categories = [1, 1, 2]
        
        results = aggregate_metrics(all_metrics, all_categories)
        
        assert "overall" in results
        assert "f1" in results["overall"]
        assert results["overall"]["f1"]["mean"] == pytest.approx(0.7666666666666666)
        
        assert "category_1" in results
        assert "category_2" in results
        assert results["category_1"]["f1"]["mean"] == 0.7
        assert results["category_2"]["f1"]["mean"] == 0.9
    
    def test_single_value(self):
        all_metrics = [{"f1": 0.8}]
        all_categories = [1]
        
        results = aggregate_metrics(all_metrics, all_categories)
        assert results["overall"]["f1"]["std"] == 0
