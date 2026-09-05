"""
BM25 scoring utility for hybrid router.

Implements BM25 (Best Matching 25) algorithm for keyword-based text retrieval.
Optimized with:
- Pre-computed document term frequencies (avoid repeated computation during queries)
- Inverted index for faster lookups
- Custom tokenizer support (including Chinese via jieba)
"""

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class BM25Scorer:
    """
    BM25 scorer for keyword-based document retrieval.

    BM25 is a bag-of-words retrieval function that ranks documents based on
    the query terms appearing in each document.

    Optimized for production use with:
    - Pre-computed term frequencies per document
    - Inverted index for efficient query processing
    - Custom tokenizer injection (supports Chinese via jieba)
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        stop_words: Optional[Set[str]] = None,
    ):
        """
        Initialize BM25 scorer.

        Args:
            tokenizer: Custom tokenizer function (e.g., jieba.lcut for Chinese).
                       If None, uses default regex tokenizer for English.
            k1: Term frequency saturation parameter (1.2-2.0 typical).
            b: Document length normalization parameter (0.75 typical).
            epsilon: Floor value for IDF to avoid negative scores.
            stop_words: Set of stop words to filter out.
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.stop_words = stop_words or self._default_stop_words()

        # Allow dependency injection of tokenizer, default to simple regex (English only)
        self.tokenizer = tokenizer if tokenizer else self._default_tokenizer

        # Internal state (set after fit)
        self._n_docs: int = 0
        self._avg_doc_length: float = 0.0
        self._doc_lengths: List[int] = []
        self._idf: Dict[str, float] = {}

        # Optimization 1: Pre-compute term frequencies per document
        # List[Dict[str, int]]
        self._doc_term_freqs: List[Dict[str, int]] = []

        # Optimization 2: Inverted index - maps term to list of doc indices
        # Dict[term, List[doc_index]]
        self._inverted_index: Dict[str, List[int]] = defaultdict(list)

    def _default_stop_words(self) -> Set[str]:
        """Return default English stop words."""
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
            "the", "to", "was", "were", "will", "with", "i", "me", "my",
            "you", "your", "we", "they", "them", "their", "this", "these",
            "those", "what", "which", "who", "whom", "where", "when", "why",
            "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "not", "only", "own", "same",
            "so", "than", "too", "very", "can", "just", "should", "now",
        }

    def _default_tokenizer(self, text: str) -> List[str]:
        """
        Default regex tokenizer for English.

        For Chinese text, inject a custom tokenizer like jieba.lcut or use
        create_multilingual_tokenizer().

        Args:
            text: Input text to tokenize.

        Returns:
            List of lowercase tokens with stop words removed.
        """
        text = text.lower()
        # Extract alphanumeric words
        tokens = re.findall(r'\b[\w]+\b', text)
        # Filter stop words and short ASCII tokens (keep non-ASCII single chars)
        result = []
        for t in tokens:
            if t in self.stop_words:
                continue
            # Only filter short tokens for ASCII (English) words
            if t.isascii() and len(t) <= 1:
                continue
            result.append(t)
        return result

    def fit(self, documents: List[str]) -> "BM25Scorer":
        """
        Fit BM25 on a corpus of documents.

        Performance optimized: pre-computes all document statistics during fit
        to avoid repeated computation during queries.

        Args:
            documents: List of document strings.

        Returns:
            Self for chaining.
        """
        self._n_docs = len(documents)
        self._doc_lengths = []
        self._doc_term_freqs = []
        self._inverted_index = defaultdict(list)

        if self._n_docs == 0:
            logger.warning("BM25 fitted on empty corpus")
            return self

        total_length = 0
        doc_freq_counts: Counter = Counter()  # Term appears in how many documents

        for idx, doc_text in enumerate(documents):
            # 1. Tokenize
            tokens = self.tokenizer(doc_text)
            length = len(tokens)
            self._doc_lengths.append(length)
            total_length += length

            # 2. Pre-compute Term Frequencies per document
            freqs = Counter(tokens)
            self._doc_term_freqs.append(dict(freqs))

            # 3. Update global stats and inverted index
            for term in freqs.keys():
                doc_freq_counts[term] += 1
                self._inverted_index[term].append(idx)

        self._avg_doc_length = total_length / self._n_docs if self._n_docs > 0 else 0

        # 4. Calculate IDF for each term
        self._idf = {}
        for term, freq in doc_freq_counts.items():
            # Lucene/ATIRE smooth IDF formula: guarantees positive IDF
            idf_raw = math.log(1 + (self._n_docs - freq + 0.5) / (freq + 0.5))
            self._idf[term] = max(idf_raw, self.epsilon)

        logger.debug(
            f"BM25 fitted on {self._n_docs} documents, "
            f"vocabulary size: {len(self._idf)}, "
            f"avg doc length: {self._avg_doc_length:.2f}"
        )
        return self

    def score(self, query: str) -> List[float]:
        """
        Score all documents against a query.

        Uses pre-computed term frequencies and inverted index for efficiency.

        Args:
            query: Query string.

        Returns:
            List of BM25 scores for each document.
        """
        query_tokens = self.tokenizer(query)
        scores = [0.0] * self._n_docs

        if not query_tokens or self._n_docs == 0:
            return scores

        # Optimization: Only iterate over query terms (not all documents)
        for term in query_tokens:
            if term not in self._idf:
                continue

            idf = self._idf[term]

            # Optimization: Only iterate over documents containing this term
            # Using inverted index instead of scanning all documents
            target_docs = self._inverted_index.get(term, [])

            for doc_idx in target_docs:
                tf = self._doc_term_freqs[doc_idx].get(term, 0)
                doc_len = self._doc_lengths[doc_idx]

                # BM25 TF formula with length normalization
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self._avg_doc_length
                )

                scores[doc_idx] += idf * (numerator / denominator)

        return scores

    def score_single(self, query: str, document: str) -> float:
        """
        Score a single document against a query (without fitting).

        This is useful for quick ad-hoc scoring without building the full corpus.
        Note: This bypasses pre-computation optimizations.

        Args:
            query: Query string.
            document: Document string.

        Returns:
            BM25-like score (without proper IDF normalization).
        """
        query_tokens = self.tokenizer(query)
        doc_tokens = self.tokenizer(document)

        if not query_tokens or not doc_tokens:
            return 0.0

        doc_term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            tf = doc_term_freqs.get(term, 0)
            if tf == 0:
                continue
            # Simple TF-IDF like scoring
            score += tf / (tf + self.k1)

        return score

    def get_top_k(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Get top-k documents by BM25 score.

        Args:
            query: Query string.
            k: Number of top documents to return.

        Returns:
            List of (document_index, score) tuples sorted by score descending.
        """
        scores = self.score(query)
        # For small lists (<1000), standard sort is efficient enough
        # For larger lists, partial sort (heapq.nlargest) could be used
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:k]

    def get_documents_containing(self, term: str) -> List[int]:
        """
        Get indices of documents containing a term.

        Args:
            term: The term to search for.

        Returns:
            List of document indices containing the term.
        """
        tokenized_term = self.tokenizer(term)
        if not tokenized_term:
            return []
        return self._inverted_index.get(tokenized_term[0], [])


def create_bm25_scorer(
    documents: List[str],
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Scorer:
    """
    Convenience function to create and fit a BM25 scorer.

    Args:
        documents: List of document strings.
        tokenizer: Custom tokenizer function (e.g., jieba.lcut for Chinese).
        k1: Term frequency saturation parameter.
        b: Document length normalization parameter.

    Returns:
        Fitted BM25Scorer instance.

    Example:
        # For English text:
        scorer = create_bm25_scorer(documents)

        # For Chinese text:
        import jieba
        scorer = create_bm25_scorer(documents, tokenizer=jieba.lcut)
    """
    scorer = BM25Scorer(tokenizer=tokenizer, k1=k1, b=b)
    scorer.fit(documents)
    return scorer


def create_multilingual_tokenizer(
    use_jieba: bool = True,
    custom_stop_words: Optional[Set[str]] = None,
) -> Callable[[str], List[str]]:
    """
    Create a tokenizer that handles both English and Chinese text.

    Args:
        use_jieba: Whether to use jieba for Chinese tokenization.
        custom_stop_words: Additional stop words to filter.

    Returns:
        Tokenizer function.

    Example:
        tokenizer = create_multilingual_tokenizer(use_jieba=True)
        scorer = BM25Scorer(tokenizer=tokenizer)

    Note:
        For Chinese text, single-character words are meaningful (e.g., "书", "车").
        The length filter (len > 1) only applies to ASCII/English words.
    """
    stop_words = {
        # English stop words
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "to", "was", "were", "will", "with", "i", "me", "my",
        "you", "your", "we", "they", "them", "their", "this", "these",
        # Chinese stop words (single characters that are meaningless)
        "的", "了", "和", "是", "就", "都", "而", "及", "与", "着",
        "或", "在", "有", "个", "也", "不", "这", "那", "你", "我",
        "他", "她", "它", "们", "啊", "呢", "吧", "吗", "呀", "哦",
        # Chinese stop words (multi-character)
        "一个", "没有", "我们", "你们", "他们", "它们", "这个",
        "那个", "什么", "怎么", "为什么", "哪里", "谁", "如何",
        "可以", "因为", "所以", "但是", "如果", "虽然", "然后",
    }

    if custom_stop_words:
        stop_words.update(custom_stop_words)

    def _should_keep_token(token: str) -> bool:
        """
        Determine if a token should be kept.
        
        - Filter out stop words
        - For ASCII words (English): require length > 1
        - For non-ASCII words (Chinese, etc.): keep single characters
        """
        if not token or token in stop_words:
            return False
        # Only apply length filter to ASCII (English) words
        if token.isascii() and len(token) <= 1:
            return False
        return True

    if use_jieba:
        try:
            import warnings

            # Suppress pkg_resources deprecation warning from jieba
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
                import jieba
            jieba.setLogLevel(logging.WARNING)  # Suppress jieba logs

            def tokenize(text: str) -> List[str]:
                text = text.lower()
                # Use jieba for tokenization (handles both Chinese and English)
                tokens = jieba.lcut(text)
                return [t.strip() for t in tokens if _should_keep_token(t.strip())]

            return tokenize
        except ImportError:
            logger.warning("jieba not installed, falling back to regex tokenizer")

    # Fallback: regex tokenizer for English only
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b[\w]+\b', text)
        return [t for t in tokens if _should_keep_token(t)]

    return tokenize
