"""BM25 search implementation for memory retrieval.

BM25 (Best Matching 25) is a ranking function used for text retrieval.
It scores documents based on:
- TF (Term Frequency): How often a term appears in a document
- IDF (Inverse Document Frequency): How rare a term is across all documents
- Document length normalization: Prevents long documents from dominating

This implementation is optimized for small document collections (<1000 docs)
typical of agent memory usage.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass


def tokenize(text: str) -> list[str]:
    """Simple tokenizer that splits on whitespace and punctuation.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of lowercase tokens
    """
    # Convert to lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


@dataclass
class BM25Params:
    """BM25 algorithm parameters.
    
    k1: Term frequency saturation parameter (1.2-2.0 typical)
        Higher k1 = term frequency counts more
    b: Document length normalization (0-1)
        b=0: no length normalization
        b=1: full length normalization
    """
    k1: float = 1.5
    b: float = 0.75


class BM25Index:
    """BM25 index for efficient text search.
    
    Example usage:
        documents = [
            "[ENV] Docker container has no systemd",
            "[PATH] PostgreSQL: /var/lib/postgresql/16/main",
            "[BUG] shred timeout - solved with xargs -n 10",
        ]
        
        index = BM25Index(documents)
        results = index.search("timeout shred", top_k=2)
        # Returns: [2, 0] (indices sorted by relevance)
    """
    
    def __init__(
        self,
        documents: list[str],
        params: BM25Params | None = None,
    ):
        """Build BM25 index from documents.
        
        Args:
            documents: List of document strings
            params: BM25 parameters (optional)
        """
        self.documents = documents
        self.params = params or BM25Params()
        
        # Tokenize all documents
        self.doc_tokens: list[list[str]] = [tokenize(doc) for doc in documents]
        self.doc_lengths: list[int] = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl: float = sum(self.doc_lengths) / len(documents) if documents else 0
        self.n_docs: int = len(documents)
        
        # Build term frequency index
        self.doc_term_freqs: list[Counter] = [
            Counter(tokens) for tokens in self.doc_tokens
        ]
        
        # Calculate document frequency for each term
        self.doc_freq: Counter = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.doc_freq[term] += 1
    
    def _idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency for a term.
        
        IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        
        Args:
            term: The term to calculate IDF for
            
        Returns:
            IDF score
        """
        df = self.doc_freq.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
    
    def _score_document(self, query_tokens: list[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document given a query.
        
        Args:
            query_tokens: Tokenized query
            doc_idx: Index of the document
            
        Returns:
            BM25 score
        """
        k1 = self.params.k1
        b = self.params.b
        
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        
        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue
            
            tf = term_freqs[term]
            idf = self._idf(term)
            
            # BM25 scoring formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> list[int]:
        """Search for documents matching the query.
        
        Args:
            query: Search query string
            top_k: Maximum number of results
            
        Returns:
            List of document indices sorted by relevance (highest first)
        """
        if not self.documents:
            return []
        
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        
        # Score all documents
        scores: list[tuple[int, float]] = []
        for doc_idx in range(self.n_docs):
            score = self._score_document(query_tokens, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))
        
        # Sort by score (descending) and return top-k indices
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:top_k]]
    
    def search_with_scores(
        self, query: str, top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Search for documents and return indices with scores.
        
        Args:
            query: Search query string
            top_k: Maximum number of results
            
        Returns:
            List of (doc_index, score) tuples sorted by relevance
        """
        if not self.documents:
            return []
        
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        
        # Score all documents
        scores: list[tuple[int, float]] = []
        for doc_idx in range(self.n_docs):
            score = self._score_document(query_tokens, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))
        
        # Sort by score (descending) and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class BM25Search:
    """Wrapper class for searching with automatic text extraction.
    
    This class handles both raw text and structured objects (like MemoryEntry).
    """
    
    def __init__(self, params: BM25Params | None = None):
        """Initialize BM25 search.
        
        Args:
            params: BM25 parameters
        """
        self.params = params or BM25Params()
        self._index: BM25Index | None = None
        self._documents: list[str] = []
    
    def build_index(self, texts: list[str]) -> None:
        """Build or rebuild the search index.
        
        Args:
            texts: List of text strings to index
        """
        self._documents = texts
        self._index = BM25Index(texts, self.params)
    
    def search(self, query: str, top_k: int = 5) -> list[int]:
        """Search the index.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            
        Returns:
            List of document indices
        """
        if self._index is None:
            return []
        return self._index.search(query, top_k)
    
    def search_documents(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[str]:
        """Search documents directly without pre-building index.
        
        Useful for one-off searches on small document sets.
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Maximum number of results
            
        Returns:
            List of matching document strings
        """
        index = BM25Index(documents, self.params)
        indices = index.search(query, top_k)
        return [documents[i] for i in indices]
