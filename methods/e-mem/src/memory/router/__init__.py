from importlib import import_module

__all__ = [
    "Router",
    "HybridRouter",
    "BM25Scorer",
    "create_bm25_scorer",
    "BaseEmbeddingModel",
    "EmbeddingModelFactory",
    "HuggingFaceEmbeddingModel",
    "OpenAIEmbeddingModel",
    "cosine_similarity",
]

_EXPORTS = {
    "Router": (".router", "Router"),
    "HybridRouter": (".hybrid_router", "HybridRouter"),
    "BM25Scorer": (".bm25_scorer", "BM25Scorer"),
    "create_bm25_scorer": (".bm25_scorer", "create_bm25_scorer"),
    "BaseEmbeddingModel": (".embedding_factory", "BaseEmbeddingModel"),
    "EmbeddingModelFactory": (".embedding_factory", "EmbeddingModelFactory"),
    "HuggingFaceEmbeddingModel": (".embedding_factory", "HuggingFaceEmbeddingModel"),
    "OpenAIEmbeddingModel": (".embedding_factory", "OpenAIEmbeddingModel"),
    "cosine_similarity": (".embedding_factory", "cosine_similarity"),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
