"""
Evaluation utilities for memory agent benchmarks.
Adapted from an open-source evaluation utility implementation.
"""

import string
import re
from collections import Counter
try:
    import nltk
except ImportError:
    nltk = None
try:
    import tiktoken
except ImportError:
    tiktoken = None
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None
try:
    from editdistance import eval as edit_distance
except ImportError:
    def edit_distance(left_text, right_text):
        """Fallback Levenshtein distance implementation."""
        left_text = str(left_text)
        right_text = str(right_text)
        if left_text == right_text:
            return 0
        if not left_text:
            return len(right_text)
        if not right_text:
            return len(left_text)

        previous_row = list(range(len(right_text) + 1))
        for left_index, left_char in enumerate(left_text, start=1):
            current_row = [left_index]
            for right_index, right_char in enumerate(right_text, start=1):
                insertion_cost = current_row[right_index - 1] + 1
                deletion_cost = previous_row[right_index] + 1
                substitution_cost = previous_row[right_index - 1] + (left_char != right_char)
                current_row.append(min(insertion_cost, deletion_cost, substitution_cost))
            previous_row = current_row
        return previous_row[-1]

import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _ApproximateEncoding:
    """Offline-safe fallback tokenizer when tiktoken assets are unavailable."""

    def encode(self, text, allowed_special=None, disallowed_special=None):
        return list(text or "")


def _get_encoding_with_fallback(model_name="gpt-4o-mini"):
    """Return a tiktoken encoding when available, otherwise a simple fallback."""
    if tiktoken is not None:
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass

    logger.warning(
        "Falling back to approximate character tokenizer for model '%s'.",
        model_name,
    )
    return _ApproximateEncoding()


def _sent_tokenize_with_fallback(text):
    """Split text into sentences without requiring NLTK downloads."""
    try:
        if nltk is None:
            raise ImportError("nltk is not installed")
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.warning("NLTK punkt not available; using regex sentence splitter.")
            raise
        return nltk.sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
        return [sentence for sentence in sentences if sentence]


# ============================================================================
# TEXT NORMALIZATION AND SCORING UTILITIES
# ============================================================================

def normalize_answer(answer_text):
    """
    Normalize text for evaluation by removing articles, punctuation, and extra whitespace.
    
    Args:
        answer_text: The text to normalize
        
    Returns:
        Normalized text string
    """
    # Apply all normalization steps in sequence
    text = "" if answer_text is None else str(answer_text).lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    
    return text


def _longest_common_subsequence_length(left_tokens, right_tokens):
    """Compute the LCS length for two token sequences."""
    previous_row = [0] * (len(right_tokens) + 1)
    for left_token in left_tokens:
        current_row = [0]
        for right_index, right_token in enumerate(right_tokens, start=1):
            if left_token == right_token:
                current_row.append(previous_row[right_index - 1] + 1)
            else:
                current_row.append(max(previous_row[right_index], current_row[-1]))
        previous_row = current_row
    return previous_row[-1]


class _FallbackRougeScore:
    """Small compatibility wrapper for rouge_score objects."""

    def __init__(self, precision, recall, fmeasure):
        self.precision = precision
        self.recall = recall
        self.fmeasure = fmeasure


class _FallbackRougeScorer:
    """Lightweight ROUGE-L scorer for offline environments."""

    def __init__(self, rouge_types, use_stemmer=True):
        self.rouge_types = rouge_types

    def score(self, target, prediction):
        target_tokens = normalize_answer(target).split()
        prediction_tokens = normalize_answer(prediction).split()
        lcs_length = _longest_common_subsequence_length(target_tokens, prediction_tokens)
        precision = lcs_length / len(prediction_tokens) if prediction_tokens else 0.0
        recall = lcs_length / len(target_tokens) if target_tokens else 0.0
        fmeasure = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        rouge_value = _FallbackRougeScore(precision, recall, fmeasure)
        return {rouge_type: rouge_value for rouge_type in self.rouge_types}


def f1_score(prediction, ground_truth):
    """
    Calculate F1 score between prediction and ground truth.
    
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        Tuple of (f1_score, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # Handle special cases for yes/no/noanswer responses
    special_answers = {'yes', 'no', 'noanswer'}
    if ((normalized_prediction in special_answers or normalized_ground_truth in special_answers) and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    # Tokenize both texts and calculate token overlap
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common_tokens = sum(common_tokens.values())
    
    if num_common_tokens == 0:
        return ZERO_METRIC
    
    # Calculate precision, recall, and F1
    precision = num_common_tokens / len(prediction_tokens)
    recall = num_common_tokens / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall


def drqa_exact_match_score(prediction, ground_truth):
    """
    Check if prediction is an exact match with ground truth after normalization.
    
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        Boolean indicating exact match
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction, ground_truth):
    """
    Check if ground truth is a substring of the prediction after normalization.
    
    Args:
        prediction: The predicted text  
        ground_truth: The ground truth text
        
    Returns:
        Boolean indicating substring match
    """
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def drqa_metric_max_over_ground_truths(metric_function, prediction, ground_truths):
    """
    Calculate the maximum score over multiple ground truth answers.
    
    Args:
        metric_function: Function to calculate score between prediction and single ground truth
        prediction: The predicted text
        ground_truths: List of ground truth answers (can be string, list, or nested list)
        
    Returns:
        Maximum score across all ground truths
    """
    ground_truth_list = _normalize_ground_truth_answers(ground_truths)

    # Calculate score for each ground truth and return maximum
    return max(metric_function(prediction, gt) for gt in ground_truth_list)


def _normalize_ground_truth_answers(ground_truths):
    """Flatten and stringify ground truths so metric code can treat them uniformly."""
    if ground_truths is None:
        return [""]
    if isinstance(ground_truths, str):
        return [ground_truths]
    if isinstance(ground_truths, (list, tuple)):
        normalized = []
        for item in ground_truths:
            normalized.extend(_normalize_ground_truth_answers(item))
        return normalized or [""]
    return [str(ground_truths)]


def parse_output(output_text, answer_prefix="Answer:"):
    """
    Parse model output to extract the answer portion.
    
    Args:
        output_text: The complete model output
        answer_prefix: The prefix that indicates where the answer starts
        
    Returns:
        Extracted answer text or None if not found
    """
    # Try multiple patterns to extract the answer
    extraction_patterns = [
        re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), 
        re.compile(r"(?:^)(.*)(?:\n|$)")
    ]
    
    for pattern in extraction_patterns:
        match = pattern.search(output_text)
        if match:
            extracted_text = match[1].strip()
            # Remove prefix again in case it was repeated
            clean_answer = re.sub(f'^{re.escape(answer_prefix)}', '', extracted_text, flags=re.IGNORECASE).strip()
            return clean_answer
    
    # Should rarely reach here, but return None if no pattern matches
    return None


LABEL_PATTERN = re.compile(r"\blabel\s*[:：]\s*(-?\d+)\b|\b(-?\d+)\b", flags=re.IGNORECASE)
ROLE_ONLY_LINE_PATTERN = re.compile(r"^(?:assistant|user|system)\s*:?\s*$", flags=re.IGNORECASE)
STRICT_LABEL_ONLY_PATTERN = re.compile(
    r"^\s*(?:assistant\s*[:：]\s*)?(?:label|answer|intent|class|category)?\s*[:：]?\s*(-?\d+)\s*$",
    flags=re.IGNORECASE,
)
LABEL_CUE_PATTERN = re.compile(
    r"\b(?:label|answer|intent|class|category)(?:\s+is)?\s*[:：]?\s*(-?\d+)\b",
    flags=re.IGNORECASE,
)
LEADING_LABEL_CUE_PATTERN = re.compile(
    r"^\s*(?:assistant\s*[:：]\s*)?(?:(?:my|the)\s+)?(?:(?:final|correct)\s+)?(?:answer|label|prediction|intent|class|category)\b(?:\s+is)?\s*[:：]?\s*(-?\d+)\b",
    flags=re.IGNORECASE,
)
LEADING_TEXT_ANSWER_CUE_PATTERN = re.compile(
    r"^\s*(?:assistant\s*[:：]\s*)?(?:(?:my|the)\s+)?(?:(?:final|correct)\s+)?answer\b(?:\s+is)?\s*[:：]?\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)
STRICT_MCQ_OPTION_ONLY_PATTERN = re.compile(
    r"^\s*(?:assistant\s*[:：]\s*)?(?:(?:my|the)\s+)?(?:(?:final|correct)\s+)?(?:answer|option|choice)?\s*[:：]?\s*\(?([A-D])\)?\s*$",
    flags=re.IGNORECASE,
)
MCQ_OPTION_CUE_PATTERN = re.compile(
    r"\b(?:answer|option|choice)(?:\s+is)?\s*[:：]?\s*\(?([A-D])\)?\b",
    flags=re.IGNORECASE,
)
LEADING_MCQ_OPTION_PATTERN = re.compile(
    r"^\s*(?:assistant\s*[:：]\s*)?\(?([A-D])\)?(?:[\s\.\):,-]|$)",
    flags=re.IGNORECASE,
)


def flatten_ground_truth_answers(ground_truths):
    """Normalize ground truths to a flat list of strings."""
    if isinstance(ground_truths, str):
        return [ground_truths]
    if not ground_truths:
        return []
    if isinstance(ground_truths[0], list):
        return [gt for gt_sublist in ground_truths for gt in gt_sublist]
    return ground_truths


def extract_first_nonempty_line(text):
    """Return the first non-empty line from text."""
    for line in str(text or "").splitlines():
        candidate = line.strip()
        if candidate and not ROLE_ONLY_LINE_PATTERN.fullmatch(candidate):
            return candidate
    return ""


def extract_first_sentence(text):
    """Return the first sentence-like span from text."""
    stripped_text = str(text or "").strip()
    if not stripped_text:
        return ""
    return re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", stripped_text, maxsplit=1)[0].strip()


def extract_last_nonempty_line(text):
    """Return the last non-empty line from text."""
    for line in reversed(str(text or "").splitlines()):
        candidate = line.strip()
        if candidate and not ROLE_ONLY_LINE_PATTERN.fullmatch(candidate):
            return candidate
    return ""


def count_normalized_words(text):
    """Count whitespace-delimited words after normalization."""
    normalized_text = normalize_answer(text)
    return len(normalized_text.split()) if normalized_text else 0


def extract_label_prediction(text):
    """Extract the first label-like integer from a model prediction."""
    if text is None:
        return None
    match = LABEL_PATTERN.search(str(text))
    if match is None:
        return None
    return match.group(1) or match.group(2)


def build_icl_label_candidates(text):
    """Build concise candidate spans for ICL label extraction."""
    stripped_text = str(text or "").strip()
    lines = [
        line.strip()
        for line in stripped_text.splitlines()
        if line.strip() and not ROLE_ONLY_LINE_PATTERN.fullmatch(line.strip())
    ]

    candidates = []
    seen = set()

    def add(candidate):
        normalized_candidate = str(candidate or "").strip()
        if not normalized_candidate or normalized_candidate in seen:
            return
        seen.add(normalized_candidate)
        candidates.append(normalized_candidate)

    for candidate in reversed(lines[-5:]):
        add(candidate)
        add(extract_first_sentence(candidate))

    if lines:
        add(lines[0])
        add(extract_first_sentence(lines[0]))

    add(extract_last_nonempty_line(stripped_text))
    add(extract_first_nonempty_line(stripped_text))
    if len(stripped_text) <= 80:
        add(stripped_text)

    return candidates


def extract_icl_label_prediction(text):
    """Extract a likely final ICL label without trusting echoed prompt examples."""
    stripped_text = str(text or "").strip()
    label_occurrences = len(re.findall(r"\blabel\s*[:：]", stripped_text, flags=re.IGNORECASE))
    probable_prompt_echo = label_occurrences > 2 or count_normalized_words(stripped_text) > 80

    if probable_prompt_echo:
        tail_candidates = [
            candidate
            for candidate in build_icl_label_candidates(stripped_text)
            if count_normalized_words(candidate) <= 12
        ]
        for candidate in tail_candidates:
            strict_match = STRICT_LABEL_ONLY_PATTERN.fullmatch(candidate)
            if strict_match:
                return strict_match.group(1), candidate, True

        for candidate in tail_candidates:
            label_match = LEADING_LABEL_CUE_PATTERN.search(candidate)
            if label_match:
                return label_match.group(1), candidate, False

        return None, "", False

    candidates = build_icl_label_candidates(text)

    for candidate in candidates:
        strict_match = STRICT_LABEL_ONLY_PATTERN.fullmatch(candidate)
        if strict_match:
            return strict_match.group(1), candidate, True

    for candidate in candidates:
        label_match = LEADING_LABEL_CUE_PATTERN.search(candidate)
        if label_match:
            return label_match.group(1), candidate, False

    return None, "", False


def build_mcq_option_candidates(text):
    """Build concise candidate spans for multiple-choice option extraction."""
    stripped_text = str(text or "").strip()
    lines = [
        line.strip()
        for line in stripped_text.splitlines()
        if line.strip() and not ROLE_ONLY_LINE_PATTERN.fullmatch(line.strip())
    ]

    candidates = []
    seen = set()

    def add(candidate):
        normalized_candidate = str(candidate or "").strip()
        if not normalized_candidate or normalized_candidate in seen:
            return
        seen.add(normalized_candidate)
        candidates.append(normalized_candidate)

    for candidate in reversed(lines[-5:]):
        add(candidate)
        add(extract_first_sentence(candidate))

    if lines:
        add(lines[0])
        add(extract_first_sentence(lines[0]))

    add(extract_last_nonempty_line(stripped_text))
    add(extract_first_nonempty_line(stripped_text))
    if len(stripped_text) <= 120:
        add(stripped_text)

    return candidates


def extract_mcq_option_prediction(text, valid_options=("A", "B", "C", "D")):
    """Extract a likely final multiple-choice option from model output."""
    valid_option_set = {str(option).strip().upper() for option in valid_options}
    if not valid_option_set:
        return None

    for candidate in build_mcq_option_candidates(text):
        strict_match = STRICT_MCQ_OPTION_ONLY_PATTERN.fullmatch(candidate)
        if strict_match:
            option = strict_match.group(1).upper()
            if option in valid_option_set:
                return option

    for candidate in build_mcq_option_candidates(text):
        cue_match = MCQ_OPTION_CUE_PATTERN.search(candidate)
        if cue_match:
            option = cue_match.group(1).upper()
            if option in valid_option_set:
                return option

    for candidate in build_mcq_option_candidates(text):
        leading_match = LEADING_MCQ_OPTION_PATTERN.match(candidate)
        if leading_match:
            option = leading_match.group(1).upper()
            if option in valid_option_set:
                return option

    return None


def _normalize_choice_text(text):
    normalized_text = str(text or "").strip().lower()
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    return normalized_text.strip(" \n\t\r.,:;!?\"'`()[]{}")


def _tokenize_choice_text(text):
    return re.findall(r"[a-z0-9]+", _normalize_choice_text(text))


def extract_mcq_option_from_choice_text(text, eval_metadata):
    """Map a model output back to an option label when it copies choice text."""
    if not eval_metadata:
        return None

    normalized_prediction = _normalize_choice_text(text)
    if not normalized_prediction:
        return None

    normalized_choices = []
    for option in ("A", "B", "C", "D"):
        choice_text = eval_metadata.get(f"choice_{option}")
        normalized_choice = _normalize_choice_text(choice_text)
        if not normalized_choice:
            continue
        if normalized_prediction == normalized_choice:
            return option
        normalized_choices.append((option, normalized_choice))

    if len(normalized_prediction) < 24:
        prediction_tokens = _tokenize_choice_text(normalized_prediction)
        if len(prediction_tokens) < 3:
            return None
    else:
        prediction_tokens = _tokenize_choice_text(normalized_prediction)

    prefix_matches = [
        option
        for option, normalized_choice in normalized_choices
        if normalized_choice.startswith(normalized_prediction)
        or normalized_prediction.startswith(normalized_choice)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    best_option = None
    best_score = 0.0
    second_best = 0.0
    prediction_token_set = set(prediction_tokens)
    for option, normalized_choice in normalized_choices:
        choice_tokens = _tokenize_choice_text(normalized_choice)
        if not choice_tokens:
            continue
        choice_token_set = set(choice_tokens)
        overlap = len(prediction_token_set & choice_token_set)
        if overlap == 0:
            continue
        precision = overlap / max(len(prediction_token_set), 1)
        recall = overlap / max(len(choice_token_set), 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        if normalized_choice in normalized_prediction or normalized_prediction in normalized_choice:
            f1_score += 0.15
        if f1_score > best_score:
            second_best = best_score
            best_score = f1_score
            best_option = option
        elif f1_score > second_best:
            second_best = f1_score

    if best_option and best_score >= 0.55 and (best_score - second_best) >= 0.08:
        return best_option

    return None


def select_best_prediction_candidate(named_candidates, ground_truth_answers):
    """Pick one best candidate and keep all primary metrics tied to that candidate."""
    best_candidate_name = None
    best_candidate_text = ""
    best_metrics = None
    best_score = None

    for candidate_name, candidate_text in named_candidates:
        if candidate_text is None or not str(candidate_text).strip():
            continue

        candidate_metrics = calculate_metrics(candidate_text, ground_truth_answers)
        candidate_score = (
            int(candidate_metrics["exact_match"]),
            int(candidate_metrics["substring_exact_match"]),
            candidate_metrics["f1"],
            candidate_metrics["rougeL_recall"],
            candidate_metrics["rougeL_f1"],
            -count_normalized_words(candidate_text),
        )

        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_candidate_name = candidate_name
            best_candidate_text = candidate_text
            best_metrics = candidate_metrics

    if best_metrics is None:
        return "", "empty_candidate", calculate_metrics("", ground_truth_answers)

    return best_candidate_text, best_candidate_name, best_metrics


def extract_answer_span(text):
    """Extract a concise answer span from common answer cue phrasings."""
    stripped_text = str(text or "").strip()
    if not stripped_text:
        return ""

    match = LEADING_TEXT_ANSWER_CUE_PATTERN.fullmatch(stripped_text)
    if match:
        return match.group(1).strip()
    return ""


# ============================================================================
# TEXT CHUNKING UTILITIES
# ============================================================================

def chunk_text_into_sentences(text, model_name="gpt-4o-mini", chunk_size=4096):
    """
    Split text into chunks of specified token size, preserving sentence boundaries.
    
    Args:
        text: The long text document to be split
        model_name: The tokenizer model name (default: gpt-4o-mini)
        chunk_size: Maximum number of tokens allowed per chunk
        
    Returns:
        List of text chunks, each within the specified token limit
    """
    # Initialize tokenizer with offline-safe fallback
    encoding = _get_encoding_with_fallback(model_name)

    # Split text into sentences
    sentences = _sent_tokenize_with_fallback(text)
    
    text_chunks = []
    current_chunk_sentences = []
    current_chunk_token_count = 0

    for sentence in sentences:
        # Count tokens in current sentence
        # Treat unknown special-token-like markers as plain text so long-context
        # benchmarks do not fail on dataset-specific placeholders.
        sentence_tokens = encoding.encode(sentence, disallowed_special=())
        sentence_token_count = len(sentence_tokens)
        
        # Check if adding this sentence would exceed chunk size
        if current_chunk_token_count + sentence_token_count > chunk_size:
            # Finalize current chunk and start new one
            text_chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_chunk_token_count = sentence_token_count
        else:
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_token_count += sentence_token_count
    
    # Add final chunk if it contains any sentences
    if current_chunk_sentences:
        text_chunks.append(" ".join(current_chunk_sentences))
    
    return text_chunks


def count_tokens(text, model_name="gpt-3.5-turbo"):
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model_name: Model name for tokenizer
        
    Returns:
        Number of tokens in the text
    """
    encoding = _get_encoding_with_fallback(model_name)
    return len(encoding.encode(text, disallowed_special=()))


def create_chunks_use_sent_tokenizer(text, max_tokens=10000):
    """
    Create text chunks using sentence tokenization with token limits.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = _sent_tokenize_with_fallback(text)
    
    chunks = []
    current_chunk_text = ""
    current_token_count = 0
    
    for sentence in sentences:
        sentence_token_count = count_tokens(sentence)
        
        # Start new chunk if adding sentence would exceed limit
        if current_token_count + sentence_token_count > max_tokens and current_chunk_text:
            chunks.append(current_chunk_text.strip())
            current_chunk_text = sentence
            current_token_count = sentence_token_count
        else:
            # Add sentence to current chunk
            if current_chunk_text:
                current_chunk_text += " " + sentence
                current_token_count += sentence_token_count + count_tokens(" ")
            else:
                current_chunk_text = sentence
                current_token_count = sentence_token_count
    
    # Add final chunk
    if current_chunk_text:
        chunks.append(current_chunk_text.strip())
    
    return chunks


# ============================================================================
# METRICS CALCULATION
# ============================================================================

# Initialize ROUGE scorer
if rouge_scorer is not None:
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)
else:
    rouge_scorer_instance = _FallbackRougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)


def calculate_metrics(prediction, ground_truth_answers):
    """
    Calculate comprehensive metrics for prediction evaluation.
    
    Args:
        prediction: The predicted text
        ground_truth_answers: Ground truth answer(s) - can be string or list
        
    Returns:
        Dictionary of calculated metrics
    """
    # Calculate basic metrics using maximum over ground truths
    metrics = {
        "exact_match": drqa_metric_max_over_ground_truths(drqa_exact_match_score, prediction, ground_truth_answers),
        "f1": drqa_metric_max_over_ground_truths(lambda x, y: f1_score(x, y)[0], prediction, ground_truth_answers),
        "substring_exact_match": drqa_metric_max_over_ground_truths(substring_exact_match_score, prediction, ground_truth_answers)
    }

    # Normalize ground truth answers for ROUGE calculation
    answer_list = _normalize_ground_truth_answers(ground_truth_answers)

    # Calculate ROUGE scores
    rouge_scores = [rouge_scorer_instance.score(target=answer, prediction=prediction) for answer in answer_list]
    
    # Extract ROUGE metrics
    for rouge_type in rouge_scorer_instance.rouge_types:
        metrics[rouge_type + "_f1"] = max(score[rouge_type].fmeasure for score in rouge_scores)
        metrics[rouge_type + "_recall"] = max(score[rouge_type].recall for score in rouge_scores)

    return metrics


def calculate_metrics_on_candidates(candidates, ground_truth_answers):
    """Return the best score for each metric across multiple prediction candidates."""
    metric_candidates = [
        calculate_metrics(candidate, ground_truth_answers)
        for candidate in candidates
        if candidate is not None and str(candidate).strip()
    ]
    if not metric_candidates:
        return calculate_metrics("", ground_truth_answers)

    metric_names = metric_candidates[0].keys()
    return {
        metric_name: max(candidate_metrics[metric_name] for candidate_metrics in metric_candidates)
        for metric_name in metric_names
    }


def calculate_locomo_recall_metrics(retrieved_source_id_groups, evidence, requested_recall_k=None):
    """Calculate strict Recall@K for LoCoMo evidence coverage."""
    return _calculate_source_id_recall_metrics(
        prefix="locomo_recall",
        retrieved_source_id_groups=retrieved_source_id_groups,
        target_source_ids=evidence,
        requested_recall_k=requested_recall_k,
    )


def calculate_membench_recall_metrics(retrieved_source_id_groups, target_source_ids, requested_recall_k=None):
    """Calculate strict Recall@K for MemBench target-step coverage."""
    return _calculate_source_id_recall_metrics(
        prefix="membench_recall",
        retrieved_source_id_groups=retrieved_source_id_groups,
        target_source_ids=target_source_ids,
        requested_recall_k=requested_recall_k,
    )


def _calculate_source_id_recall_metrics(prefix, retrieved_source_id_groups, target_source_ids, requested_recall_k=None):
    """Compute strict Recall@K over normalized source-id groups."""
    normalized_targets = {
        str(item).strip()
        for item in (target_source_ids or [])
        if str(item).strip()
    }
    if not normalized_targets:
        return {}

    normalized_groups = []
    for group in retrieved_source_id_groups or []:
        normalized_group = [
            str(item).strip()
            for item in (group or [])
            if str(item).strip()
        ]
        normalized_groups.append(normalized_group)

    try:
        requested_k = int(requested_recall_k) if requested_recall_k is not None else len(normalized_groups)
    except (TypeError, ValueError):
        requested_k = len(normalized_groups)

    if retrieved_source_id_groups is None:
        return {}
    if requested_k <= 0 and not normalized_groups:
        return {}

    k_values = []
    for candidate_k in (1, 5, 10):
        if requested_k >= candidate_k:
            k_values.append(candidate_k)
    if requested_k > 0 and requested_k not in k_values:
        k_values.append(requested_k)
    if not k_values:
        k_values.append(len(normalized_groups))

    recall_metrics = {}
    for k in sorted(set(max(1, value) for value in k_values)):
        covered_source_ids = set()
        for group in normalized_groups[:k]:
            covered_source_ids.update(group)
        recall_metrics[f"{prefix}@{k}"] = len(covered_source_ids & normalized_targets) / len(normalized_targets)

    return recall_metrics


# ============================================================================
# DATASET-SPECIFIC POST-PROCESSING
# ============================================================================

def post_process(output, answer, dataset_config, eval_metadata=None):
    """
    Apply dataset-specific post-processing to model outputs.
    
    Args:
        output: Model output dictionary
        answer: Ground truth answer
        dataset_config: Dataset configuration dictionary
        
    Returns:
        Tuple of (metrics_dict, additional_info_dict)
    """
    sub_dataset_name = str(dataset_config.get('sub_dataset', '') or '').strip().lower()
    
    # Route to appropriate post-processing based on dataset type
    if 'icl' in sub_dataset_name:
        return _process_icl_dataset(output, answer)
    elif 'factconsolidation' in sub_dataset_name:
        return _process_factconsolidation_dataset(output, answer)
    elif 'eventqa' in sub_dataset_name:
        return _process_eventqa_dataset(output, answer)
    elif 'longmemeval' in sub_dataset_name:
        return _process_longmemeval_dataset(output, answer)
    elif 'locomo' in sub_dataset_name:
        return _process_locomo_qa_dataset(output, answer, eval_metadata)
    elif 'membench' in sub_dataset_name or str(dataset_config.get('dataset', '')).lower() == 'membench':
        return _process_membench_dataset(output, answer, eval_metadata)
    elif 'longbench' in sub_dataset_name or str(dataset_config.get('dataset', '')).lower() == 'longbench':
        return _process_longbench_dataset(output, answer, eval_metadata)
    else:
        return default_post_process(output, answer)


def _process_icl_dataset(output, answer):
    """Process in-context learning dataset outputs."""
    prediction = output["output"]
    parsed_label, label_source_text, strict_label_format = extract_icl_label_prediction(prediction)
    evaluation_prediction = parsed_label or ""
    raw_parsed_prediction = parse_output(prediction, answer_prefix="label:")

    metrics = calculate_metrics(evaluation_prediction, answer)
    legacy_prediction = raw_parsed_prediction or prediction
    legacy_metrics = calculate_metrics(legacy_prediction, answer)
    answer_list = flatten_ground_truth_answers(answer)
    ground_truth_label = next(
        (extract_label_prediction(gt) or normalize_answer(gt) for gt in answer_list),
        None,
    )

    metrics["label_accuracy"] = int(parsed_label is not None and parsed_label == ground_truth_label)
    metrics["label_format_valid"] = int(parsed_label is not None)
    metrics["label_strict_format"] = int(strict_label_format)
    metrics["label_strict_accuracy"] = int(metrics["label_accuracy"] and strict_label_format)
    metrics["legacy_exact_match"] = legacy_metrics["exact_match"]
    metrics["legacy_f1"] = legacy_metrics["f1"]

    return metrics, {
        "parsed_output": evaluation_prediction,
        "raw_parsed_output": raw_parsed_prediction,
        "parsed_label": parsed_label,
        "label_source_text": label_source_text,
    }


def _process_factconsolidation_dataset(output, answer):
    """Process conflict-resolution outputs with parsed-answer emphasis."""
    prediction = output["output"]
    parsed_prediction = parse_output(prediction)
    first_line_prediction = extract_first_nonempty_line(prediction)
    first_sentence_prediction = extract_first_sentence(prediction)
    last_line_prediction = extract_last_nonempty_line(prediction)
    answer_cue_prediction = (
        extract_answer_span(parsed_prediction)
        or extract_answer_span(first_line_prediction)
        or extract_answer_span(first_sentence_prediction)
        or extract_answer_span(last_line_prediction)
    )

    selected_prediction, selected_source, metrics = select_best_prediction_candidate([
        ("answer_cue_output", answer_cue_prediction),
        ("parsed_output", parsed_prediction),
        ("first_line_output", first_line_prediction),
        ("first_sentence_output", first_sentence_prediction),
        ("last_line_output", last_line_prediction),
    ], answer)

    legacy_metrics = calculate_metrics(prediction, answer)
    if parsed_prediction is not None:
        parsed_metrics = calculate_metrics(parsed_prediction, answer)
        legacy_metrics = {
            metric_name: max(original_score, parsed_metrics[metric_name])
            for metric_name, original_score in legacy_metrics.items()
        }

    metrics["answer_hit"] = drqa_metric_max_over_ground_truths(
        substring_exact_match_score,
        prediction,
        answer,
    )
    metrics["parsed_answer_hit"] = drqa_metric_max_over_ground_truths(
        substring_exact_match_score,
        parsed_prediction or "",
        answer,
    )
    metrics["first_line_answer_hit"] = drqa_metric_max_over_ground_truths(
        substring_exact_match_score,
        first_line_prediction,
        answer,
    )
    metrics["selected_answer_hit"] = drqa_metric_max_over_ground_truths(
        substring_exact_match_score,
        selected_prediction,
        answer,
    )
    metrics["concise_response"] = int(count_normalized_words(selected_prediction) <= 8)
    metrics["concise_answer_hit"] = int(metrics["selected_answer_hit"] and metrics["concise_response"])
    metrics["legacy_exact_match"] = legacy_metrics["exact_match"]
    metrics["legacy_f1"] = legacy_metrics["f1"]
    metrics["legacy_rougeL_recall"] = legacy_metrics["rougeL_recall"]

    return metrics, {
        "parsed_output": selected_prediction,
        "parsed_output_source": selected_source,
        "raw_parsed_output": parsed_prediction,
        "first_line_output": first_line_prediction,
        "first_sentence_output": first_sentence_prediction,
        "last_line_output": last_line_prediction,
        "answer_cue_output": answer_cue_prediction,
    }


def _process_eventqa_dataset(output, answer):
    """Process EventQA dataset outputs with recall calculation."""
    prediction = output["output"]
    
    # Calculate recall: fraction of answer elements found in prediction
    recall_score = sum(answer_element.lower() in prediction.lower() for answer_element in answer) / len(answer)
    
    # Convert to binary recall (1 if all elements found, 0 otherwise)
    binary_recall = int(recall_score == 1)
    
    # Calculate standard metrics
    parsed_prediction = parse_output(prediction)
    standard_metrics = calculate_metrics(parsed_prediction, answer)
    standard_metrics["eventqa_recall"] = binary_recall

    return standard_metrics, {"parsed_output": parsed_prediction}


def _process_longmemeval_dataset(output, answer):
    """Process LongMemEval open-ended QA outputs."""
    return default_post_process(output, answer)


def _process_locomo_qa_dataset(output, answer, eval_metadata):
    """Process LoCoMo QA outputs with optional strict recall scoring."""
    metrics, additional_info = default_post_process(output, answer)
    recall_metrics = calculate_locomo_recall_metrics(
        output.get("retrieved_source_id_groups"),
        (eval_metadata or {}).get("evidence", []),
        requested_recall_k=output.get("requested_recall_k"),
    )
    metrics.update(recall_metrics)
    return metrics, additional_info


def _process_longbench_dataset(output, answer, eval_metadata):
    """Process LongBench multiple-choice outputs."""
    prediction = output["output"]
    parsed_prediction = extract_mcq_option_prediction(prediction)
    parsed_output = parse_output(prediction)
    parsed_output_option = extract_mcq_option_prediction(parsed_output)
    first_line_option = extract_mcq_option_prediction(extract_first_nonempty_line(prediction))
    last_line_option = extract_mcq_option_prediction(extract_last_nonempty_line(prediction))
    choice_text_option = extract_mcq_option_from_choice_text(prediction, eval_metadata)
    parsed_choice_text_option = extract_mcq_option_from_choice_text(parsed_output, eval_metadata)

    selected_prediction = (
        parsed_prediction
        or parsed_output_option
        or first_line_option
        or last_line_option
        or choice_text_option
        or parsed_choice_text_option
    )

    metrics = calculate_metrics(selected_prediction or "", answer)
    legacy_metrics = calculate_metrics(prediction, answer)
    metrics["legacy_exact_match"] = legacy_metrics["exact_match"]
    metrics["legacy_f1"] = legacy_metrics["f1"]

    additional_info = {
        "parsed_output": selected_prediction or parsed_output or "",
        "pred": selected_prediction,
        "judge": bool(metrics["exact_match"]),
    }
    if eval_metadata:
        additional_info.update({
            "_id": eval_metadata.get("_id"),
            "domain": eval_metadata.get("domain"),
            "sub_domain": eval_metadata.get("sub_domain"),
            "difficulty": eval_metadata.get("difficulty"),
            "length": eval_metadata.get("length"),
        })

    return metrics, additional_info


def _process_membench_dataset(output, answer, eval_metadata):
    """Process MemBench multiple-choice outputs."""
    metrics, additional_info = _process_longbench_dataset(output, answer, eval_metadata=None)
    recall_metrics = calculate_membench_recall_metrics(
        output.get("retrieved_source_id_groups"),
        (eval_metadata or {}).get("target_source_ids", []),
        requested_recall_k=output.get("requested_recall_k"),
    )
    metrics.update(recall_metrics)

    memory_construction_time = float(output.get("memory_construction_time", 0) or 0)
    query_time_len = float(output.get("query_time_len", 0) or 0)
    context_chunk_count = (eval_metadata or {}).get("context_chunk_count")
    try:
        context_chunk_count = int(context_chunk_count)
    except (TypeError, ValueError):
        context_chunk_count = 0

    metrics["membench_total_overhead_time_s"] = memory_construction_time + query_time_len
    metrics["membench_read_time_per_op_s"] = query_time_len
    if context_chunk_count > 0:
        metrics["membench_write_time_per_op_s"] = memory_construction_time / context_chunk_count

    if eval_metadata:
        additional_info.update({
            "sample_id": eval_metadata.get("sample_id"),
            "slice": eval_metadata.get("slice"),
            "branch": eval_metadata.get("branch"),
            "scenario_index": eval_metadata.get("scenario_index"),
            "variant_index": eval_metadata.get("variant_index"),
            "trajectory_tid": eval_metadata.get("trajectory_tid"),
            "ground_truth": eval_metadata.get("ground_truth"),
            "answer_text": eval_metadata.get("answer_text"),
            "question_time": eval_metadata.get("question_time"),
            "target_source_ids": eval_metadata.get("target_source_ids"),
            "context_chunk_count": eval_metadata.get("context_chunk_count"),
            "context_length": eval_metadata.get("context_length"),
            "session_count": eval_metadata.get("session_count"),
        })

    return metrics, additional_info


def default_post_process(output, answer):
    """
    Default post-processing function for model outputs.
    
    Args:
        output: Model output dictionary
        answer: Ground truth answer
        
    Returns:
        Tuple of (metrics_dict, additional_info_dict)
    """
    prediction = output["output"]
    metrics = calculate_metrics(prediction, answer)
    
    # Try parsing output and take maximum scores
    parsed_prediction = parse_output(prediction)
    if parsed_prediction is not None:
        parsed_metrics = calculate_metrics(parsed_prediction, answer)
        metrics = {metric_name: max(original_score, parsed_metrics[metric_name]) 
                  for metric_name, original_score in metrics.items()}
    
    return metrics, {"parsed_output": parsed_prediction}


def recompute_result_metrics(result_record, dataset_config):
    """Recompute task-specific metrics from a stored result record."""
    output = {"output": result_record.get("output", "")}
    if "retrieved_source_id_groups" in result_record:
        output["retrieved_source_id_groups"] = result_record.get("retrieved_source_id_groups")
    if "requested_recall_k" in result_record:
        output["requested_recall_k"] = result_record.get("requested_recall_k")
    calculated_metrics, additional_info = post_process(
        output,
        result_record.get("answer"),
        dataset_config,
        eval_metadata=result_record.get("eval_metadata"),
    )
    return {**additional_info, **calculated_metrics}


# ============================================================================
# METRICS SUMMARIZATION
# ============================================================================

def metrics_summarization(
    output,
    query,
    answer,
    dataset_config,
    metrics,
    results,
    query_id=None,
    context_id=None,
    qa_pair_id=None,
    eval_metadata=None,
):
    """
    Summarize metrics for a single query and update overall metrics and results.
    
    Args:
        output: Model output dictionary
        query: The input query
        answer: Ground truth answer
        dataset_config: Dataset configuration
        metrics: Running metrics dictionary
        results: List of result records
        query_id: Optional query identifier
        context_id: Optional context identifier
        qa_pair_id: Optional qa_pair_id for the question
        
    Returns:
        Tuple of (updated_metrics, updated_results)
    """
    if output is None:
        logger.info("Skipping example because the model returned None")
        return metrics, results
    
    # Calculate dataset-specific metrics
    calculated_metrics, additional_info = post_process(output, answer, dataset_config, eval_metadata=eval_metadata)
    output.update({**additional_info, **calculated_metrics})
    
    # Update running metrics
    for metric_name, metric_value in calculated_metrics.items():
        metrics[metric_name].append(metric_value)

    # Update system metrics
    metrics["input_len"].append(output["input_len"])
    metrics["output_len"].append(output["output_len"])
    metrics["memory_construction_time"].append(output.get("memory_construction_time", 0))
    metrics["query_time_len"].append(output.get("query_time_len", 0))
    
    # Create result record
    result_record = {**output, "answer": answer, 'query': query}
    if query_id is not None:
        result_record["query_id"] = query_id
    if context_id is not None:
        result_record["context_id"] = context_id
    if qa_pair_id is not None:
        result_record["qa_pair_id"] = qa_pair_id
    if eval_metadata is not None:
        result_record["eval_metadata"] = eval_metadata
        for key in (
            "question_id",
            "sample_id",
            "category",
            "source",
            "question_type",
            "question_date",
        ):
            value = eval_metadata.get(key)
            if value is not None and key not in result_record:
                result_record[key] = value
    results.append(result_record)

    # Log debug information if enabled
    if dataset_config['debug']:
        logger.info(f"Input length: {output['input_len']}")
        logger.info(f"Answer: {answer}")
        logger.info(f"Output: {output['output']}")
        logger.info(f"Parsed output: {output['parsed_output']}")
                    
    return metrics, results
