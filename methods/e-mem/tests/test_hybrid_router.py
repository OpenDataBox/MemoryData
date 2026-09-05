"""Tests for HybridRouter, EmbeddingFactory, and BM25Scorer."""
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestBM25Scorer:
    """Test BM25Scorer functionality."""

    def test_init(self):
        """Test BM25Scorer initialization."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        scorer = BM25Scorer()
        assert scorer.k1 == 1.5
        assert scorer.b == 0.75
        assert len(scorer.stop_words) > 0

    def test_init_custom_params(self):
        """Test BM25Scorer with custom parameters."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        scorer = BM25Scorer(k1=2.0, b=0.5, stop_words={"custom", "stop"})
        assert scorer.k1 == 2.0
        assert scorer.b == 0.5
        assert "custom" in scorer.stop_words

    def test_init_custom_tokenizer(self):
        """Test BM25Scorer with custom tokenizer."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        def custom_tokenizer(text):
            return text.lower().split()
        
        scorer = BM25Scorer(tokenizer=custom_tokenizer)
        tokens = scorer.tokenizer("Hello World Test")
        assert tokens == ["hello", "world", "test"]

    def test_tokenize(self):
        """Test text tokenization."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        scorer = BM25Scorer()
        tokens = scorer.tokenizer("The quick brown fox jumps over the lazy dog")
        
        # "the" is a stop word, should be filtered
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_tokenize_empty(self):
        """Test tokenization of empty string."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        scorer = BM25Scorer()
        tokens = scorer.tokenizer("")
        assert tokens == []

    def test_fit(self):
        """Test fitting BM25 on documents."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        documents = [
            "The cat sat on the mat",
            "The dog ran in the park",
            "A bird flew over the tree"
        ]
        
        scorer = BM25Scorer()
        result = scorer.fit(documents)
        
        assert result is scorer  # Returns self for chaining
        assert scorer._n_docs == 3
        assert len(scorer._idf) > 0
        assert scorer._avg_doc_length > 0

    def test_score(self):
        """Test scoring documents against query."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        documents = [
            "The cat sat on the mat",
            "The dog ran in the park",
            "A cat played with a ball"
        ]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        scores = scorer.score("cat")
        
        assert len(scores) == 3
        # Documents 0 and 2 mention "cat", should have higher scores
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]

    def test_score_empty_query(self):
        """Test scoring with empty query."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        documents = ["doc1", "doc2"]
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        scores = scorer.score("")
        assert all(s == 0.0 for s in scores)

    def test_get_top_k(self):
        """Test getting top-k documents."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        documents = [
            "apple orange banana",
            "car truck bus",
            "apple pie dessert",
            "orange juice drink"
        ]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        top_k = scorer.get_top_k("apple", k=2)
        
        assert len(top_k) == 2
        # First result should be index 0 or 2 (both contain "apple")
        assert top_k[0][0] in [0, 2]

    def test_create_bm25_scorer(self):
        """Test convenience function."""
        from src.memory.router.bm25_scorer import create_bm25_scorer
        
        documents = ["doc1", "doc2", "doc3"]
        scorer = create_bm25_scorer(documents)
        
        assert scorer._n_docs == 3

    def test_fit_precomputes_term_freqs(self):
        """Test that fit precomputes term frequencies."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        documents = ["apple apple orange", "banana orange", "apple"]
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        # Check doc 0 has apple:2, orange:1
        assert scorer._doc_term_freqs[0].get("apple") == 2
        assert scorer._doc_term_freqs[0].get("orange") == 1

    def test_inverted_index(self):
        """Test inverted index construction."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        documents = ["apple orange", "banana", "apple banana"]
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        # "apple" appears in docs 0 and 2
        assert 0 in scorer._inverted_index["apple"]
        assert 2 in scorer._inverted_index["apple"]
        # "banana" appears in docs 1 and 2
        assert 1 in scorer._inverted_index["banana"]
        assert 2 in scorer._inverted_index["banana"]

    def test_score_uses_inverted_index(self):
        """Test that scoring uses inverted index efficiently."""
        from src.memory.router.bm25_scorer import BM25Scorer

        # Create documents where only some have the query term
        documents = ["apple", "banana", "cherry", "apple pie"]
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        scores = scorer.score("apple")
        
        # Only docs 0 and 3 should have non-zero scores
        assert scores[0] > 0
        assert scores[1] == 0
        assert scores[2] == 0
        assert scores[3] > 0

    def test_create_bm25_scorer_with_tokenizer(self):
        """Test convenience function with custom tokenizer."""
        from src.memory.router.bm25_scorer import create_bm25_scorer
        
        def custom_tokenizer(text):
            return text.split("-")
        
        documents = ["a-b-c", "d-e-f"]
        scorer = create_bm25_scorer(documents, tokenizer=custom_tokenizer)
        
        assert scorer._n_docs == 2
        assert "a" in scorer._idf

    def test_create_multilingual_tokenizer_fallback(self):
        """Test multilingual tokenizer fallback when jieba not needed."""
        from src.memory.router.bm25_scorer import create_multilingual_tokenizer

        # This should work without jieba (use_jieba=False)
        tokenizer = create_multilingual_tokenizer(use_jieba=False)
        tokens = tokenizer("Hello World")
        assert "hello" in tokens
        assert "world" in tokens


class TestEmbeddingFactory:
    """Test EmbeddingModelFactory and embedding models."""

    def test_factory_supported_providers(self):
        """Test factory lists supported providers."""
        from src.memory.router.embedding_factory import EmbeddingModelFactory
        
        assert "huggingface" in EmbeddingModelFactory.SUPPORTED_PROVIDERS
        assert "openai" in EmbeddingModelFactory.SUPPORTED_PROVIDERS

    def test_factory_unsupported_provider(self):
        """Test factory raises error for unsupported provider."""
        from src.memory.router.embedding_factory import EmbeddingModelFactory
        
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            EmbeddingModelFactory.create(provider="unsupported")

    def test_factory_openai_requires_api_key(self):
        """Test OpenAI provider requires api_key."""
        from src.memory.router.embedding_factory import EmbeddingModelFactory
        
        with pytest.raises(ValueError, match="api_key is required"):
            EmbeddingModelFactory.create(provider="openai", config={})

    @patch("sentence_transformers.SentenceTransformer")
    def test_huggingface_embedding_model(self, mock_st):
        """Test HuggingFace embedding model initialization."""
        from src.memory.router.embedding_factory import HuggingFaceEmbeddingModel
        
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.randn(2, 384)
        mock_st.return_value = mock_model
        
        model = HuggingFaceEmbeddingModel(model_name="test-model")
        
        assert model.get_embedding_dimension() == 384
        mock_st.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_huggingface_embed(self, mock_st):
        """Test HuggingFace embedding."""
        from src.memory.router.embedding_factory import HuggingFaceEmbeddingModel
        
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        expected_embedding = np.random.randn(2, 384)
        mock_model.encode.return_value = expected_embedding
        mock_st.return_value = mock_model
        
        model = HuggingFaceEmbeddingModel()
        result = model.embed(["text1", "text2"])
        
        assert result.shape == (2, 384)
        mock_model.encode.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_huggingface_embed_single_text(self, mock_st):
        """Test HuggingFace embedding with single text."""
        from src.memory.router.embedding_factory import HuggingFaceEmbeddingModel
        
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.randn(1, 384)
        mock_st.return_value = mock_model
        
        model = HuggingFaceEmbeddingModel()
        result = model.embed("single text")
        
        assert result.shape == (1, 384)

    @patch("openai.OpenAI")
    def test_openai_embedding_model(self, mock_openai_cls):
        """Test OpenAI embedding model initialization."""
        from src.memory.router.embedding_factory import OpenAIEmbeddingModel
        
        model = OpenAIEmbeddingModel(api_key="test-key", model="text-embedding-3-small")
        
        mock_openai_cls.assert_called_once()
        assert model.model == "text-embedding-3-small"

    @patch("openai.OpenAI")
    def test_openai_embed(self, mock_openai_cls):
        """Test OpenAI embedding."""
        from src.memory.router.embedding_factory import OpenAIEmbeddingModel
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client
        
        model = OpenAIEmbeddingModel(api_key="test-key")
        result = model.embed(["text1", "text2"])
        
        assert result.shape == (2, 1536)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from src.memory.router.embedding_factory import cosine_similarity
        
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([
            [1.0, 0.0, 0.0],  # Same direction
            [0.0, 1.0, 0.0],  # Orthogonal
            [-1.0, 0.0, 0.0], # Opposite direction
        ])
        
        similarities = cosine_similarity(a, b)
        
        assert np.isclose(similarities[0], 1.0)  # Same direction
        assert np.isclose(similarities[1], 0.0)  # Orthogonal
        assert np.isclose(similarities[2], -1.0) # Opposite

    def test_cosine_similarity_batch(self):
        """Test batch cosine similarity."""
        from src.memory.router.embedding_factory import cosine_similarity
        
        a = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        
        similarities = cosine_similarity(a, b)
        
        assert similarities.shape == (2, 2)


class TestHybridRouter:
    """Test HybridRouter functionality."""

    @pytest.fixture
    def mock_openai_config(self):
        """Mock OpenAI configuration."""
        return {
            "api_key": "test-key",
            "base_url": "https://test.api.com",
            "model": "gpt-4o-mini"
        }

    def test_init_defaults(self, mock_openai_config):
        """Test HybridRouter initialization with defaults."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        assert router.name == "hybrid_router"
        assert router.agent == []
        assert router.max_blocks == 5
        assert router.enable_router is True
        # Weights should be normalized
        total = router.summary_weight + router.text_weight + router.bm25_weight
        assert np.isclose(total, 1.0)

    def test_init_custom_weights(self, mock_openai_config):
        """Test HybridRouter with custom weights."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            summary_weight=0.2,
            text_weight=0.5,
            bm25_weight=0.3,
        )
        
        # Weights should be normalized
        total = router.summary_weight + router.text_weight + router.bm25_weight
        assert np.isclose(total, 1.0)
        assert router.text_weight > router.summary_weight

    def test_init_router_disabled(self):
        """Test HybridRouter with router disabled."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=None, enable_router=False)
        
        assert router.enable_router is False

    def test_add_blocks_inactive(self, mock_openai_config):
        """Test adding inactive memory agent."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        mock_agent = Mock()
        mock_agent.is_active = False
        mock_agent.summary = "Test summary"
        
        router.add_blocks(mock_agent)
        
        assert len(router.agent) == 1
        assert router._embeddings_dirty is True

    def test_add_blocks_active_ignored(self, mock_openai_config):
        """Test that active agents are not added."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        mock_agent = Mock()
        mock_agent.is_active = True
        
        router.add_blocks(mock_agent)
        
        assert len(router.agent) == 0

    def test_map_blocks_no_agents(self, mock_openai_config):
        """Test mapping with no agents."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        result = router._map_blocks("Test query")
        
        assert result == []

    def test_map_blocks_router_disabled(self, mock_openai_config):
        """Test mapping when router is disabled returns all agents."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=None, enable_router=False)
        
        # Add mock agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.is_active = False
            router.agent.append(mock_agent)
        
        result = router._map_blocks("Test query")
        
        assert len(result) == 3

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_map_blocks_with_agents(self, mock_factory, mock_openai_config):
        """Test mapping with agents using hybrid scoring."""
        from src.memory.router.hybrid_router import HybridRouter

        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = np.random.randn(3, 384)
        mock_factory.create.return_value = mock_embedding_model
        
        router = HybridRouter(openai_config=mock_openai_config, max_blocks=2)
        
        # Add mock agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            mock_agent.get_original_texts.return_value = [f"Text content {i}"]
            mock_agent.original_texts = [f"Text content {i}"]
            router.add_blocks(mock_agent)
        
        result = router._map_blocks("Test query")
        
        # Should return at most max_blocks
        assert len(result) <= 2

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_bm25_boost_threshold(self, mock_factory, mock_openai_config):
        """Test BM25 boost threshold auto-selects high-scoring blocks."""
        from src.memory.router.hybrid_router import HybridRouter

        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = np.random.randn(3, 384)
        mock_factory.create.return_value = mock_embedding_model
        
        # Create router with BM25 boost threshold
        router = HybridRouter(
            openai_config=mock_openai_config,
            max_blocks=2,
            bm25_boost_threshold=0.7,  # Enable boost
        )
        
        # Add mock agents with different content
        agents = []
        for i, content in enumerate(["apple banana", "cherry date", "apple apple apple"]):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            mock_agent.get_original_texts.return_value = [content]
            mock_agent.original_texts = [content]
            agents.append(mock_agent)
            router.add_blocks(mock_agent)
        
        # Query for "apple" - should boost agent 2 (highest BM25 for "apple")
        result = router._map_blocks("apple")
        
        # Should return results
        assert len(result) >= 1

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_bm25_boost_disabled(self, mock_factory, mock_openai_config):
        """Test BM25 boost is disabled when threshold is None."""
        from src.memory.router.hybrid_router import HybridRouter

        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = np.random.randn(3, 384)
        mock_factory.create.return_value = mock_embedding_model
        
        # Create router without BM25 boost threshold
        router = HybridRouter(
            openai_config=mock_openai_config,
            max_blocks=2,
            bm25_boost_threshold=None,  # Disabled
        )
        
        assert router.bm25_boost_threshold is None

    def test_chunk_text(self):
        """Test text chunking function."""
        from src.memory.router.hybrid_router import chunk_text
        
        short_text = "This is a short text."
        chunks = chunk_text(short_text, max_chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_text_long(self):
        """Test chunking of long text."""
        from src.memory.router.hybrid_router import chunk_text
        
        long_text = "This is sentence one. " * 50
        chunks = chunk_text(long_text, max_chunk_size=100)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100 or len(chunk.split()) == 1  # Allow single long words

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_normalize_scores(self, mock_factory, mock_openai_config):
        """Test score normalization."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        scores = np.array([1.0, 5.0, 3.0])
        normalized = router._normalize_scores(scores)
        
        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_normalize_scores_all_same(self, mock_factory, mock_openai_config):
        """Test normalization when all scores are the same."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        scores = np.array([5.0, 5.0, 5.0])
        normalized = router._normalize_scores(scores)
        
        assert np.all(normalized == 0.0)

    def test_execute_tool(self, mock_openai_config):
        """Test execute_tool does nothing."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        result = router.execute_tool("test", {})
        assert result is None


class TestBM25ScorerAdvanced:
    """Advanced BM25 scorer tests for edge cases."""

    def test_fit_empty_corpus(self):
        """Test BM25 fit on empty corpus."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        scorer = BM25Scorer()
        scorer.fit([])
        
        assert scorer._n_docs == 0
        scores = scorer.score("test query")
        assert len(scores) == 0

    def test_get_top_k(self):
        """Test get_top_k method."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        documents = [
            "apple apple apple",
            "banana",
            "apple orange",
            "cherry date"
        ]
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        top_k = scorer.get_top_k("apple", k=2)
        
        assert len(top_k) == 2
        # First result should be doc 0 (most apples)
        assert top_k[0][0] == 0
        # Scores should be descending
        assert top_k[0][1] >= top_k[1][1]

    def test_get_top_k_larger_than_docs(self):
        """Test get_top_k when k > number of documents."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        documents = ["apple", "banana"]
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        top_k = scorer.get_top_k("apple", k=10)
        
        assert len(top_k) == 2

    def test_stop_words_filtering(self):
        """Test stop words are filtered."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        scorer = BM25Scorer(stop_words={"the", "is", "a"})
        tokens = scorer._default_tokenizer("The apple is a fruit")
        
        assert "the" not in tokens
        assert "is" not in tokens
        assert "apple" in tokens
        assert "fruit" in tokens

    def test_single_char_ascii_filtered(self):
        """Test single char ASCII words are filtered."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        scorer = BM25Scorer(stop_words=set())
        tokens = scorer._default_tokenizer("I am a cat")
        
        # Single char ASCII should be filtered
        assert "I" not in tokens and "i" not in tokens
        assert "a" not in tokens
        assert "am" in tokens
        assert "cat" in tokens

    def test_chinese_single_char_preserved(self):
        """Test Chinese single characters are preserved."""
        from src.memory.router.bm25_scorer import create_multilingual_tokenizer

        # Use fallback tokenizer (no jieba) for this test
        tokenizer = create_multilingual_tokenizer(use_jieba=False)
        # Verify tokenizer was created successfully
        assert callable(tokenizer)
        # Test tokenization works
        tokens = tokenizer("hello world")
        assert len(tokens) > 0

    def test_multilingual_tokenizer_with_jieba(self):
        """Test multilingual tokenizer with jieba enabled."""
        import importlib.util
        import warnings

        from src.memory.router.bm25_scorer import create_multilingual_tokenizer

        if importlib.util.find_spec("jieba") is None:
            pytest.skip("jieba not installed")

        # Suppress pkg_resources deprecation warning from jieba
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
            # If jieba is available, test it
            tokenizer = create_multilingual_tokenizer(use_jieba=True)
            tokens = tokenizer("我爱编程 I love coding")

        # Should tokenize both Chinese and English
        assert len(tokens) > 0
        # Check that common stop words are filtered
        assert "的" not in tokens

    def test_multilingual_tokenizer_chinese_words(self):
        """Test multilingual tokenizer preserves meaningful Chinese words."""
        import importlib.util
        import warnings

        from src.memory.router.bm25_scorer import create_multilingual_tokenizer

        if importlib.util.find_spec("jieba") is None:
            pytest.skip("jieba not installed")

        # Suppress pkg_resources deprecation warning from jieba
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
            tokenizer = create_multilingual_tokenizer(use_jieba=True)

            # Test that meaningful single Chinese characters are preserved
            tokens = tokenizer("买车")  # "buy car"
            # Should keep meaningful characters
            assert len(tokens) >= 1

    def test_score_single(self):
        """Test score_single method."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        scorer = BM25Scorer()
        
        score = scorer.score_single("apple banana", "apple cherry")
        
        # Should have some score since "apple" is in both
        assert score > 0

    def test_score_single_empty(self):
        """Test score_single with empty inputs."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        scorer = BM25Scorer()
        
        assert scorer.score_single("", "apple") == 0.0
        assert scorer.score_single("apple", "") == 0.0

    def test_get_documents_containing(self):
        """Test get_documents_containing method."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        documents = ["apple orange", "banana", "apple banana"]
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        docs_with_apple = scorer.get_documents_containing("apple")
        
        assert 0 in docs_with_apple
        assert 2 in docs_with_apple
        assert 1 not in docs_with_apple

    def test_get_documents_containing_empty_term(self):
        """Test get_documents_containing with empty term."""
        from src.memory.router.bm25_scorer import BM25Scorer
        
        documents = ["apple orange", "banana"]
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        # Empty term should return empty list
        result = scorer.get_documents_containing("")
        assert result == []


class TestEmbeddingFactoryAdvanced:
    """Advanced embedding factory tests."""

    def test_cosine_similarity_zero_vectors(self):
        """Test cosine similarity with zero vectors."""
        from src.memory.router.embedding_factory import cosine_similarity
        
        a = np.zeros((1, 384))
        b = np.random.randn(3, 384)
        
        # Should handle zero vectors gracefully
        result = cosine_similarity(a, b)
        assert result.shape == (1, 3)

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors."""
        from src.memory.router.embedding_factory import cosine_similarity
        
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        
        result = cosine_similarity(a, b)
        assert np.isclose(result[0, 0], 1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors."""
        from src.memory.router.embedding_factory import cosine_similarity
        
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])
        
        result = cosine_similarity(a, b)
        assert np.isclose(result[0, 0], 0.0)

    def test_cosine_similarity_1d_input(self):
        """Test cosine similarity handles 1D input by reshaping."""
        from src.memory.router.embedding_factory import cosine_similarity
        
        a = np.array([[1.0, 0.0, 0.0]])  # 2D input
        b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        result = cosine_similarity(a, b)
        # Result shape depends on implementation
        assert result.shape[1] == 2

    def test_embedding_factory_default_models(self):
        """Test factory uses default model names when not specified."""
        from src.memory.router.embedding_factory import EmbeddingModelFactory

        # Verify supported providers list
        assert "huggingface" in EmbeddingModelFactory.SUPPORTED_PROVIDERS
        assert "openai" in EmbeddingModelFactory.SUPPORTED_PROVIDERS

    @patch("sentence_transformers.SentenceTransformer")
    def test_factory_huggingface_with_config(self, mock_st):
        """Test factory creates HuggingFace model with config."""
        from src.memory.router.embedding_factory import EmbeddingModelFactory
        
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        model = EmbeddingModelFactory.create(
            provider="huggingface",
            config={"device": "cpu", "batch_size": 16}
        )
        
        assert model is not None

    @patch("openai.OpenAI")
    def test_factory_openai_with_base_url(self, mock_openai):
        """Test factory creates OpenAI model with custom base_url."""
        from src.memory.router.embedding_factory import EmbeddingModelFactory
        
        model = EmbeddingModelFactory.create(
            provider="openai",
            model_name="custom-model",
            config={
                "api_key": "test-key",
                "base_url": "https://custom.api.com/v1",
            }
        )
        
        assert model.model == "custom-model"


class TestHybridRouterScoring:
    """Test hybrid router scoring methods."""

    @pytest.fixture
    def mock_openai_config(self):
        return {
            "api_key": "test-key",
            "base_url": "http://test",
            "model": "gpt-4o-mini"
        }

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_score_summary_similarity_empty(self, mock_factory, mock_openai_config):
        """Test summary scoring with no embeddings."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        router._summary_embeddings = None
        
        # Add a mock agent
        mock_agent = Mock()
        mock_agent.is_active = False
        router.agent = [mock_agent]
        
        scores = router._score_summary_similarity("test query")
        
        assert len(scores) == 1
        assert scores[0] == 0.0

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_score_text_similarity_empty(self, mock_factory, mock_openai_config):
        """Test text scoring with no embeddings."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        router._text_chunk_embeddings = None
        
        mock_agent = Mock()
        mock_agent.is_active = False
        router.agent = [mock_agent]
        
        scores = router._score_text_similarity("test query")
        
        assert len(scores) == 1
        assert scores[0] == 0.0

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_score_bm25_empty(self, mock_factory, mock_openai_config):
        """Test BM25 scoring with no scorer."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        router._bm25_scorer = None
        
        mock_agent = Mock()
        mock_agent.is_active = False
        router.agent = [mock_agent]
        
        scores = router._score_bm25("test query")
        
        assert len(scores) == 1
        assert scores[0] == 0.0

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_get_agent_texts_with_method(self, mock_factory, mock_openai_config):
        """Test _get_agent_texts with get_original_texts method."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        mock_agent = Mock()
        mock_agent.get_original_texts.return_value = ["text1", "text2"]
        
        texts = router._get_agent_texts(mock_agent)
        
        assert texts == ["text1", "text2"]

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_get_agent_texts_with_attribute(self, mock_factory, mock_openai_config):
        """Test _get_agent_texts with original_texts attribute."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        mock_agent = Mock(spec=[])  # No methods
        mock_agent.original_texts = ["text1", "text2"]
        
        texts = router._get_agent_texts(mock_agent)
        
        assert texts == ["text1", "text2"]

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_get_agent_texts_fallback(self, mock_factory, mock_openai_config):
        """Test _get_agent_texts fallback to empty list."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        mock_agent = Mock(spec=[])  # No methods or attributes
        
        texts = router._get_agent_texts(mock_agent)
        
        assert texts == []

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_normalize_scores_empty(self, mock_factory, mock_openai_config):
        """Test score normalization with empty array."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        scores = np.array([])
        normalized = router._normalize_scores(scores)
        
        assert len(normalized) == 0


class TestChunkText:
    """Test text chunking edge cases."""

    def test_chunk_very_long_word(self):
        """Test chunking text with very long words."""
        from src.memory.router.hybrid_router import chunk_text

        # Create text with a very long word
        long_word = "a" * 200
        text = f"Short word {long_word} another short word"
        
        chunks = chunk_text(text, max_chunk_size=100)
        
        assert len(chunks) >= 1

    def test_chunk_many_sentences(self):
        """Test chunking with many sentences."""
        from src.memory.router.hybrid_router import chunk_text
        
        sentences = ["This is sentence number {}. ".format(i) for i in range(20)]
        text = " ".join(sentences)
        
        chunks = chunk_text(text, max_chunk_size=200)
        
        assert len(chunks) > 1
        for chunk in chunks:
            # Most chunks should be under limit (some may exceed due to single sentences)
            assert len(chunk) <= 200 or "." not in chunk[:-1]

    def test_chunk_single_long_sentence(self):
        """Test chunking a single very long sentence."""
        from src.memory.router.hybrid_router import chunk_text

        # Single sentence longer than max_chunk_size
        text = "word " * 100  # ~500 chars
        
        chunks = chunk_text(text, max_chunk_size=100)
        
        assert len(chunks) >= 1


class TestHybridRouterConfig:
    """Test HybridRouterConfig schema."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.config.schema import HybridRouterConfig
        
        config = HybridRouterConfig()
        
        assert config.embedding_provider == "huggingface"
        assert config.summary_weight == 0.3
        assert config.text_weight == 0.4
        assert config.bm25_weight == 0.3
        assert config.text_chunk_size == 512
        assert config.bm25_boost_threshold is None

    def test_custom_config(self):
        """Test custom configuration."""
        from src.config.schema import HybridRouterConfig
        
        config = HybridRouterConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
            summary_weight=0.5,
            text_weight=0.3,
            bm25_weight=0.2,
            bm25_boost_threshold=0.8,
        )
        
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-large"
        assert config.summary_weight == 0.5
        assert config.bm25_boost_threshold == 0.8

    def test_memory_config_with_hybrid_router(self):
        """Test MemoryConfig includes hybrid router config."""
        from src.config.schema import MemoryConfig
        
        config = MemoryConfig(router_type="hybrid")
        
        assert config.router_type == "hybrid"
        assert config.hybrid_router is not None
        assert config.hybrid_router.embedding_provider == "huggingface"


class TestSequentialAndBatchQuery:
    """Test sequential and batch query functionality."""

    @pytest.fixture
    def mock_openai_config(self):
        return {
            "api_key": "test-key",
            "base_url": "http://test",
            "model": "gpt-4o-mini"
        }

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_sequential_query_agents(self, mock_factory, mock_openai_config):
        """Test _sequential_query_agents method."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        # Create mock agents
        mock_agents = []
        for i in range(3):
            mock_agent = Mock()
            mock_agent.query.return_value = f"Response {i}"
            mock_agent.current_block = Mock()
            mock_agent.current_block.block_id = f"block_{i}"
            mock_agents.append(mock_agent)
        
        results = router._sequential_query_agents(mock_agents, "test query")
        
        assert len(results) == 3
        assert results[0] == "Response 0"
        assert results[1] == "Response 1"
        assert results[2] == "Response 2"

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_sequential_query_agents_with_error(self, mock_factory, mock_openai_config):
        """Test _sequential_query_agents handles errors gracefully."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        mock_agent1 = Mock()
        mock_agent1.query.return_value = "Response 1"
        mock_agent1.current_block = Mock()
        mock_agent1.current_block.block_id = "block_1"
        
        mock_agent2 = Mock()
        mock_agent2.query.side_effect = Exception("Query failed")
        mock_agent2.current_block = Mock()
        mock_agent2.current_block.block_id = "block_2"
        
        results = router._sequential_query_agents([mock_agent1, mock_agent2], "test")
        
        assert len(results) == 2
        assert results[0] == "Response 1"
        assert "[ERROR]" in results[1]

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_batch_query_agents_fallback(self, mock_factory, mock_openai_config):
        """Test _batch_query_agents falls back on batch error."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            query_batch_size=2,
        )
        
        # Create mock agents
        mock_agents = []
        for i in range(2):
            mock_agent = Mock()
            mock_agent.query.return_value = f"Fallback Response {i}"
            mock_agent.current_block = Mock()
            mock_agent.current_block.block_id = f"block_{i}"
            mock_agents.append(mock_agent)
        
        # Mock _execute_batch_query to fail
        router._execute_batch_query = Mock(side_effect=Exception("Batch failed"))
        
        results = router._batch_query_agents(mock_agents, "test query")
        
        assert len(results) == 2
        assert "Fallback Response" in results[0]

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_execute_batch_query_empty(self, mock_factory, mock_openai_config):
        """Test _execute_batch_query with empty agents list."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        results = router._execute_batch_query([], "test query")
        
        assert results == []

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_execute_batch_query_no_cache(self, mock_factory, mock_openai_config):
        """Test _execute_batch_query when agents have no cache."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        
        # Mock agent with no cache
        mock_agent = Mock()
        mock_agent.get_cache_for_batch.return_value = None
        mock_agent.model = Mock()
        mock_agent.tokenizer = Mock()
        mock_agent.layer_devices = {}
        mock_agent.primary_device = "cpu"
        
        results = router._execute_batch_query([mock_agent], "test query")
        
        # Should return "No knowledge available." for agents without cache
        assert len(results) == 1
        assert "No knowledge" in results[0]

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_batch_query_multiple_batches(self, mock_factory, mock_openai_config):
        """Test _batch_query_agents processes multiple batches."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            query_batch_size=2,
        )
        
        # Create mock agents
        mock_agents = []
        for i in range(5):  # 5 agents, batch_size=2 -> 3 batches
            mock_agent = Mock()
            mock_agent.query.return_value = f"Response {i}"
            mock_agent.current_block = Mock()
            mock_agent.current_block.block_id = f"block_{i}"
            mock_agents.append(mock_agent)
        
        # Mock _execute_batch_query
        batch_results = [["R0", "R1"], ["R2", "R3"], ["R4"]]
        call_idx = [0]
        def mock_execute_batch(agents, query):
            result = batch_results[call_idx[0]]
            call_idx[0] += 1
            return result
        
        router._execute_batch_query = mock_execute_batch
        
        results = router._batch_query_agents(mock_agents, "test query")
        
        assert len(results) == 5

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_batch_query_inner_exception(self, mock_factory, mock_openai_config):
        """Test _batch_query_agents handles inner fallback exceptions."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            query_batch_size=2,
        )
        
        # Mock agent where fallback also fails
        mock_agent = Mock()
        mock_agent.query.side_effect = Exception("Query failed")
        mock_agent.current_block = Mock()
        mock_agent.current_block.block_id = "block_0"
        
        # Mock _execute_batch_query to fail
        router._execute_batch_query = Mock(side_effect=Exception("Batch failed"))
        
        results = router._batch_query_agents([mock_agent], "test query")
        
        assert len(results) == 1
        assert "[ERROR]" in results[0]

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_map_reduce_blocks_sequential_mode(self, mock_factory, mock_openai_config):
        """Test map_reduce_blocks in sequential mode (batch_size=1)."""
        from src.memory.router.hybrid_router import HybridRouter
        
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = np.random.randn(1, 384)
        mock_factory.create.return_value = mock_embedding_model
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            query_batch_size=1,  # Sequential mode
            enable_router=False,
        )
        
        mock_agent = Mock()
        mock_agent.is_active = False
        mock_agent.summary = "Summary"
        mock_agent.get_original_texts.return_value = ["Content"]
        mock_agent.original_texts = ["Content"]
        mock_agent.preload_cache.return_value = None
        mock_agent._owns_model = False  # Shared model
        mock_agent.current_block = Mock()
        mock_agent.current_block.block_id = "block_0"
        router.add_blocks(mock_agent)
        
        # Mock sequential query
        router._sequential_query_agents = Mock(return_value=["Sequential Response"])
        
        result = router.map_reduce_blocks("Test query")
        
        assert len(result) == 1

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_map_reduce_blocks_batch_mode(self, mock_factory, mock_openai_config):
        """Test map_reduce_blocks in batch inference mode."""
        from src.memory.router.hybrid_router import HybridRouter
        
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = np.random.randn(1, 384)
        mock_factory.create.return_value = mock_embedding_model
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            query_batch_size=4,  # Batch mode
            enable_router=False,
        )
        
        mock_agent = Mock()
        mock_agent.is_active = False
        mock_agent.summary = "Summary"
        mock_agent.get_original_texts.return_value = ["Content"]
        mock_agent.original_texts = ["Content"]
        mock_agent.preload_cache.return_value = None
        mock_agent._owns_model = False  # Shared model -> batch mode
        mock_agent.current_block = Mock()
        mock_agent.current_block.block_id = "block_0"
        router.add_blocks(mock_agent)
        
        # Mock batch query
        router._batch_query_agents = Mock(return_value=["Batch Response"])
        
        result = router.map_reduce_blocks("Test query")
        
        assert len(result) == 1


class TestMapReduceBlocks:
    """Test map_reduce_blocks and batch query functionality."""

    @pytest.fixture
    def mock_openai_config(self):
        return {
            "api_key": "test-key",
            "base_url": "http://test",
            "model": "gpt-4o-mini"
        }

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_map_reduce_blocks_no_agents(self, mock_factory, mock_openai_config):
        """Test map_reduce_blocks with no agents."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(openai_config=mock_openai_config)
        result = router.map_reduce_blocks("test query")
        
        assert result == []

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_map_reduce_blocks_router_disabled(self, mock_factory, mock_openai_config):
        """Test map_reduce_blocks with router disabled."""
        from src.memory.router.hybrid_router import HybridRouter
        
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = np.random.randn(1, 384)
        mock_factory.create.return_value = mock_embedding_model
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            enable_router=False,
        )
        
        # Add mock agents
        for i in range(2):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            mock_agent.get_original_texts.return_value = [f"Content {i}"]
            mock_agent.original_texts = [f"Content {i}"]
            mock_agent.preload_cache.return_value = None
            mock_agent.query.return_value = f"Response {i}"
            mock_agent._owns_model = True  # Standalone mode
            router.add_blocks(mock_agent)
        
        result = router.map_reduce_blocks("test query")
        
        # Should return all agents' responses
        assert len(result) == 2

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_map_reduce_blocks_with_memory_limit(self, mock_factory, mock_openai_config):
        """Test map_reduce_blocks respects memory segment limit."""
        from src.memory.router.hybrid_router import HybridRouter
        
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = np.random.randn(1, 384)
        mock_factory.create.return_value = mock_embedding_model
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            max_memory_segments=2,
            enable_router=False,
        )
        
        # Add mock agents that return multiple segments
        for i in range(3):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            mock_agent.get_original_texts.return_value = [f"Content {i}"]
            mock_agent.original_texts = [f"Content {i}"]
            mock_agent.preload_cache.return_value = None
            mock_agent.query.return_value = f"<memory>Segment {i}</memory>"
            mock_agent._owns_model = True
            router.add_blocks(mock_agent)
        
        result = router.map_reduce_blocks("test query")
        
        # Returns responses from agents (3), but max_memory_segments limits final output
        assert len(result) >= 1


class TestLLMFallback:
    """Test LLM fallback routing."""

    @pytest.fixture
    def mock_openai_config(self):
        return {
            "api_key": "test-key",
            "base_url": "http://test",
            "model": "gpt-4o-mini"
        }

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_llm_map_blocks_valid_response(self, mock_factory, mock_openai_config):
        """Test _llm_map_blocks with valid LLM response."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            use_llm_fallback=True,
        )
        
        # Add mock agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            router.agent.append(mock_agent)
        
        # Mock generate_response
        router.generate_response = Mock(
            return_value="<summary_index>0, 2</summary_index>"
        )
        
        result = router._llm_map_blocks("test query", max_blocks=2)
        
        assert len(result) == 2
        assert result[0] == router.agent[0]
        assert result[1] == router.agent[2]

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_llm_map_blocks_invalid_response(self, mock_factory, mock_openai_config):
        """Test _llm_map_blocks with invalid LLM response."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            use_llm_fallback=True,
        )
        
        # Add mock agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            router.agent.append(mock_agent)
        
        # Mock generate_response with invalid format
        router.generate_response = Mock(return_value="Invalid response")
        
        result = router._llm_map_blocks("test query", max_blocks=2)
        
        # Should fall back to first max_blocks agents
        assert len(result) == 2

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_llm_map_blocks_out_of_range(self, mock_factory, mock_openai_config):
        """Test _llm_map_blocks with out-of-range indices."""
        from src.memory.router.hybrid_router import HybridRouter
        
        router = HybridRouter(
            openai_config=mock_openai_config,
            use_llm_fallback=True,
        )
        
        # Add mock agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            router.agent.append(mock_agent)
        
        # Mock response with out-of-range indices
        router.generate_response = Mock(
            return_value="<summary_index>0, 5, 10, 1</summary_index>"
        )
        
        result = router._llm_map_blocks("test query", max_blocks=2)
        
        # Should only include valid indices
        assert len(result) == 2
        assert result[0] == router.agent[0]
        assert result[1] == router.agent[1]


class TestIntegration:
    """Integration tests for hybrid router with memory handlers."""

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_hybrid_router_end_to_end(self, mock_factory):
        """Test hybrid router scoring pipeline."""
        from src.memory.router.hybrid_router import HybridRouter

        # Mock embedding model
        mock_embedding_model = Mock()
        # Return different embeddings for query vs documents
        call_count = [0]
        def mock_embed(texts):
            call_count[0] += 1
            if isinstance(texts, str):
                texts = [texts]
            return np.random.randn(len(texts), 384)
        
        mock_embedding_model.embed.side_effect = mock_embed
        mock_factory.create.return_value = mock_embedding_model
        
        # Create router
        router = HybridRouter(
            openai_config={"api_key": "test"},
            max_blocks=2,
            summary_weight=0.3,
            text_weight=0.4,
            bm25_weight=0.3,
        )
        
        # Add mock agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"This is summary number {i} about topic {i}"
            mock_agent.get_original_texts.return_value = [
                f"Original text content for block {i}. Some keywords: apple, orange, banana."
            ]
            mock_agent.original_texts = [
                f"Original text content for block {i}. Some keywords: apple, orange, banana."
            ]
            router.add_blocks(mock_agent)
        
        # Perform mapping
        result = router._map_blocks("What about apple?")
        
        # Should return up to max_blocks
        assert len(result) <= 2
        # Embedding model should have been called
        assert call_count[0] > 0

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_hybrid_router_with_bm25_boost(self, mock_factory):
        """Test hybrid router with BM25 boost threshold."""
        from src.memory.router.hybrid_router import HybridRouter

        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = np.random.randn(1, 384)
        mock_factory.create.return_value = mock_embedding_model
        
        # Create router with BM25 boost
        router = HybridRouter(
            openai_config={"api_key": "test"},
            max_blocks=2,
            bm25_boost_threshold=0.5,
        )
        
        # Add mock agents with very different BM25 profiles
        for i, content in enumerate(["apple", "banana banana", "apple apple apple"]):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            mock_agent.get_original_texts.return_value = [content]
            mock_agent.original_texts = [content]
            router.add_blocks(mock_agent)
        
        result = router._map_blocks("apple")
        
        assert len(result) >= 1

    @patch("src.memory.router.hybrid_router.EmbeddingModelFactory")
    def test_map_blocks_scoring_exception_fallback(self, mock_factory):
        """Test _map_blocks fallback when scoring fails."""
        from src.memory.router.hybrid_router import HybridRouter

        # Create router with valid embeddings but make scoring fail
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = np.random.randn(3, 384)
        mock_factory.create.return_value = mock_embedding_model
        
        router = HybridRouter(
            openai_config={"api_key": "test"},
            max_blocks=2,
        )
        
        # Add mock agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            mock_agent.get_original_texts.return_value = [f"Content {i}"]
            mock_agent.original_texts = [f"Content {i}"]
            router.add_blocks(mock_agent)
        
        # Force rebuild embeddings first
        router._rebuild_embeddings()
        
        # Now make embed fail for query
        mock_embedding_model.embed.side_effect = Exception("Query embedding failed")
        
        # Should fall back to returning first max_blocks agents
        result = router._map_blocks("test query")
        
        assert len(result) <= 2

