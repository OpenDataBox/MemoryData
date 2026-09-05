"""Unit tests for evaluation logic."""
import os
from pathlib import Path

import pytest

os.environ.setdefault('HF_HOME', '/mnt/d/AI')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from evaluation.locomo.load_dataset import QA, load_locomo_dataset
from evaluation.locomo.utils import calculate_metrics


class TestReferenceAnswerSelection:
    def test_normal_category(self):
        qa = QA(
            question="What is the answer?",
            evidence=["D1:1"],
            category=1,
            answer="Normal answer",
            adversarial_answer=None
        )
        
        reference = qa.adversarial_answer or qa.answer if qa.category == 5 else qa.answer
        assert reference == "Normal answer"
    
    def test_adversarial_category(self):
        qa = QA(
            question="What is the answer?",
            evidence=["D1:1"],
            category=5,
            answer=None,
            adversarial_answer="Not mentioned in the conversation"
        )
        
        reference = qa.adversarial_answer or qa.answer if qa.category == 5 else qa.answer
        assert reference == "Not mentioned in the conversation"


class TestIntAnswerHandling:
    def test_int_answer(self):
        qa = QA(
            question="What year?",
            evidence=["D1:1"],
            category=2,
            answer=2022,
            adversarial_answer=None
        )
        
        metrics = calculate_metrics("The year is 2022", qa.answer)
        assert isinstance(metrics, dict)
        assert "f1" in metrics


class TestMetricsCalculation:
    @pytest.mark.parametrize("pred,ref,should_match", [
        ("2022", 2022, True),
        ("hello world", "hello world", True),
        ("Hello World", "hello world", True),
        ("the answer", "answer", True),
        ("completely different", "answer", False),
    ])
    def test_various_inputs(self, pred, ref, should_match):
        metrics = calculate_metrics(pred, ref)
        assert isinstance(metrics, dict)
        assert "exact_match" in metrics
        assert "f1" in metrics
        
        if should_match:
            assert metrics["exact_match"] == 1 or metrics["f1"] > 0.5


class TestCategory5Handling:
    @pytest.fixture
    def dataset_paths(self):
        eval_data_dir = Path(__file__).parent.parent / "evaluation" / "eval_data" / "locomo"
        if not eval_data_dir.exists():
            pytest.skip(f"Dataset directory not found: {eval_data_dir}")
        
        json_files = list(eval_data_dir.glob("*.json"))
        if not json_files:
            pytest.skip(f"No JSON dataset files found in: {eval_data_dir}")
        
        return [str(f) for f in json_files]
    
    def test_category_5_has_reference(self, dataset_paths):
        cat5_found = False
        
        for dataset_path in dataset_paths:
            samples = load_locomo_dataset(dataset_path)
            
            for sample in samples:
                for qa in sample.qa:
                    if qa.category == 5:
                        cat5_found = True
                        reference = qa.adversarial_answer or qa.answer
                        assert reference is not None, f"Category 5 QA has no reference answer in {dataset_path}"
                        
                        metrics = calculate_metrics("Not mentioned", reference)
                        assert isinstance(metrics, dict)
                        break
                if cat5_found:
                    break
            if cat5_found:
                break
        
        assert cat5_found, f"No category 5 samples found in any dataset files: {dataset_paths}"
