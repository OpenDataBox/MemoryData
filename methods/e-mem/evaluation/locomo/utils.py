"""Utility functions for evaluation metrics."""
import re
from collections import Counter
from typing import Dict, Union

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize models
sentence_model = None


def get_sentence_model():
    global sentence_model
    if sentence_model is None:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_model


def extract_answer_from_xml(text: str, category: int) -> str:
    """Extract answer from XML tags for category 1 questions.
    
    Args:
        text: The raw text from the model
        category: The question category
        
    Returns:
        The extracted answer or the original text if no XML tags found
    """
    if category != 1:
        return text
    
    # Try to extract content between <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()    
    # If no <answer> tags found, return the original text
    return text


def normalize_answer(s: Union[str, int, float]) -> str:
    """Normalize answer string."""
    s = str(s).lower()
    # Remove punctuation (commas, periods, question marks, exclamation marks, etc.)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_metrics(prediction: Union[str, int, float], reference: Union[str, int, float]) -> Dict[str, float]:
    """Calculate all evaluation metrics."""
    # Convert to strings
    prediction = str(prediction)
    reference = str(reference)
    
    # Exact match
    exact_match = int(normalize_answer(prediction) == normalize_answer(reference))
    
    # F1 score
    f1 = f1_score(prediction, reference)
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, prediction)
    
    # BLEU scores
    ref_tokens = nltk.word_tokenize(str(reference).lower())
    pred_tokens = nltk.word_tokenize(str(prediction).lower())
    smoothing = SmoothingFunction().method1
    
    bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    # METEOR score
    meteor = meteor_score([ref_tokens], pred_tokens)
    
    # # Sentence similarity
    # model = get_sentence_model()
    # ref_emb = model.encode(str(reference), convert_to_tensor=True)
    # pred_emb = model.encode(str(prediction), convert_to_tensor=True)
    # similarity = pytorch_cos_sim(ref_emb, pred_emb).item()
    
    return {
        "exact_match": exact_match,
        "f1": f1,
        "rouge1_f": rouge_scores['rouge1'].fmeasure,
        "rouge2_f": rouge_scores['rouge2'].fmeasure,
        "rougeL_f": rouge_scores['rougeL'].fmeasure,
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        # "sbert_similarity": similarity
    }


def aggregate_metrics(all_metrics, all_categories):
    """Aggregate metrics by category."""
    import statistics
    from collections import defaultdict
    
    category_metrics = defaultdict(lambda: defaultdict(list))
    overall_metrics = defaultdict(list)
    
    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            category_metrics[category][metric_name].append(value)
            overall_metrics[metric_name].append(value)
    
    results = {}
    
    # Overall metrics
    results['overall'] = {
        metric: {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0
        }
        for metric, values in overall_metrics.items()
    }
    
    # Per-category metrics
    for category, metrics in category_metrics.items():
        results[f'category_{category}'] = {
            metric: {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0
            }
            for metric, values in metrics.items()
        }
    
    return results