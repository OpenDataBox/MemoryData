"""Paper-facing method names used across plans, tables, and figures.

This registry keeps documentation aligned with the paper's display names instead
of local folder names. Base methods and common experimental variants are both
listed here so future tables can reuse one canonical source of truth.
"""

from __future__ import annotations

PAPER_METHOD_NAMES = {
    "long_context": "Long Context",
    "bm25": "BM25",
    "embedding_rag": "Dense Retrieval",
    "graph_rag": "GraphRAG",
    "hipporag": "HippoRAG2",
    "self_rag": "Self-RAG",
    "mem0": "Mem0",
    "cognee": "Cognee",
    "zep": "Zep",
    "zep_local": "Zep-Local",
    "letta": "Letta",
    "simplemem": "SimpleMem",
    "lightmem": "LightMem",
    "memagent": "MemAgent",
    "a_mem": "A-MEM",
    "MemOS": "MemOS",
    "memorag": "MemoRAG",
    "raptor": "RAPTOR",
}

PAPER_METHOD_VARIANT_NAMES = {
    "mem0-vector": "Mem0 (Vector Only)",
    "mem0-graph": "Mem0 (Triplet Graph)",
    "mem0-vector-miniLM": "Mem0 (Vector Only, MiniLM)",
    "mem0-qdrant": "Mem0 (Qdrant Only)",
    "mem0-qdrant+graph": "Mem0 (Qdrant + Graph)",
    "mem0-top10": "Mem0 (Top-10)",
    "mem0-top50": "Mem0 (Top-50)",
    "mem0-top100": "Mem0 (Top-100)",
    "memorag-beacon2": "MemoRAG (Beacon Ratio = 2)",
    "memorag-beacon4": "MemoRAG (Beacon Ratio = 4)",
    "memorag-beacon8": "MemoRAG (Beacon Ratio = 8)",
    "lightmem-user-raw": "LightMem (User-Only Raw)",
    "lightmem-hybrid-raw": "LightMem (Hybrid Raw)",
    "lightmem-user-summary": "LightMem (User-Only Summary)",
    "lightmem-user-compressed": "LightMem (User-Only Compressed)",
    "lightmem-on-disk": "LightMem (On-Disk)",
    "lightmem-in-memory": "LightMem (In-Memory)",
    "lightmem-bm25": "LightMem (BM25)",
    "lightmem-direct": "LightMem (Direct Ingest)",
    "lightmem-pipeline": "LightMem (Pipeline Ingest)",
    "lightmem-update-offline": "LightMem (Offline Update)",
    "lightmem-update-online": "LightMem (Online Update)",
    "simplemem-default": "SimpleMem (Default)",
    "simplemem-semantic-only": "SimpleMem (Semantic Only)",
    "simplemem-hybrid-no-planning": "SimpleMem (Hybrid, No Planning)",
    "simplemem-hybrid-planning-reflect1": "SimpleMem (Hybrid, Planning + 1 Reflection)",
    "simplemem-window40-overlap0": "SimpleMem (Window = 40, Overlap = 0)",
    "simplemem-window60-overlap0": "SimpleMem (Window = 60, Overlap = 0)",
    "simplemem-window60-overlap2": "SimpleMem (Window = 60, Overlap = 2)",
    "zep-default": "Zep (All Scopes)",
    "zep-cloud": "Zep (Cloud)",
    "zep-facts-only": "Zep (Facts Only)",
    "zep-entities-only": "Zep (Entities Only)",
    "zep-episodes-only": "Zep (Episodes Only)",
    "embedding_rag-chunk1024": "Dense Retrieval (1K Chunks)",
    "embedding_rag-chunk2048": "Dense Retrieval (2K Chunks)",
    "embedding_rag-chunk4096": "Dense Retrieval (4K Chunks)",
    "embedding_rag-precompressed": "Dense Retrieval (+ LLMLingua-2)",
    "embedding_rag-contriever": "Dense Retrieval (Contriever)",
    "embedding_rag-text_embedding_3_small": "Dense Retrieval (text-embedding-3-small)",
    "embedding_rag-text_embedding_3_large": "Dense Retrieval (text-embedding-3-large)",
    "embedding_rag-qwen3_embedding_4b": "Dense Retrieval (Qwen3-Embedding-4B)",
    "embedding_rag-text_embedding_3_small_ext": "Dense Retrieval (text-embedding-3-small-ext)",
    "memagent-default": "MemAgent (Default: 5K / 120K)",
    "memagent-chunk2000": "MemAgent (Chunk = 2K)",
    "memagent-ctx60k": "MemAgent (Context = 60K)",
    "a_mem-qwen3emb": "A-MEM (Qwen3-Embedding-4B)",
}

PAPER_METHOD_GROUPS = {
    "long_context": "control",
    "bm25": "retrieval_baseline",
    "embedding_rag": "retrieval_baseline",
    "graph_rag": "retrieval_baseline",
    "hipporag": "retrieval_baseline",
    "self_rag": "retrieval_baseline",
    "a_mem": "flat_text_memory",
    "simplemem": "flat_text_memory",
    "lightmem": "flat_text_memory",
    "mem0": "structured_memory",
    "cognee": "structured_memory",
    "zep": "structured_memory",
    "zep_local": "structured_memory",
    "memagent": "recurrent_memory",
    "letta": "paging_system",
    "MemOS": "paging_system",
    "memorag": "compressed_memory",
    "raptor": "retrieval_baseline",
}


def get_paper_method_name(method_key: str) -> str:
    """Return the paper-facing display name for a method or variant key."""

    if method_key in PAPER_METHOD_VARIANT_NAMES:
        return PAPER_METHOD_VARIANT_NAMES[method_key]
    return PAPER_METHOD_NAMES.get(method_key, method_key)


__all__ = [
    "PAPER_METHOD_GROUPS",
    "PAPER_METHOD_NAMES",
    "PAPER_METHOD_VARIANT_NAMES",
    "get_paper_method_name",
]
