"""Runtime config for the vendored upstream SimpleMem repository."""

import os


OPENAI_API_KEY = os.environ.get("SIMPLEMEM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("SIMPLEMEM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
LLM_MODEL = os.environ.get("SIMPLEMEM_MODEL", "qwen-plus")

EMBEDDING_API_KEY = os.environ.get("SIMPLEMEM_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_BASE_URL = (
    os.environ.get("SIMPLEMEM_EMBEDDING_BASE_URL")
    or os.environ.get("OPENAI_EMBEDDING_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
)
EMBEDDING_MODEL = os.environ.get(
    "SIMPLEMEM_EMBEDDING_MODEL",
    "Qwen3-Embedding-4B",
)
EMBEDDING_DIMENSION = int(os.environ.get("SIMPLEMEM_EMBEDDING_DIMENSION", "0"))
EMBEDDING_CONTEXT_LENGTH = int(os.environ.get("SIMPLEMEM_EMBEDDING_CONTEXT_LENGTH", "32768"))

ENABLE_THINKING = os.environ.get("SIMPLEMEM_ENABLE_THINKING", "false").lower() == "true"
USE_STREAMING = os.environ.get("SIMPLEMEM_USE_STREAMING", "false").lower() == "true"
USE_JSON_FORMAT = os.environ.get("SIMPLEMEM_USE_JSON_FORMAT", "false").lower() == "true"

WINDOW_SIZE = int(os.environ.get("SIMPLEMEM_WINDOW_SIZE", "40"))
OVERLAP_SIZE = int(os.environ.get("SIMPLEMEM_OVERLAP_SIZE", "2"))

SEMANTIC_TOP_K = int(os.environ.get("SIMPLEMEM_SEMANTIC_TOP_K", "25"))
KEYWORD_TOP_K = int(os.environ.get("SIMPLEMEM_KEYWORD_TOP_K", "5"))
STRUCTURED_TOP_K = int(os.environ.get("SIMPLEMEM_STRUCTURED_TOP_K", "5"))

LANCEDB_PATH = os.environ.get("SIMPLEMEM_LANCEDB_PATH", "./lancedb_data")
MEMORY_TABLE_NAME = os.environ.get("SIMPLEMEM_MEMORY_TABLE_NAME", "memory_entries")

ENABLE_PARALLEL_PROCESSING = os.environ.get("SIMPLEMEM_ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"
MAX_PARALLEL_WORKERS = int(os.environ.get("SIMPLEMEM_MAX_PARALLEL_WORKERS", "4"))
ENABLE_PARALLEL_RETRIEVAL = os.environ.get("SIMPLEMEM_ENABLE_PARALLEL_RETRIEVAL", "true").lower() == "true"
MAX_RETRIEVAL_WORKERS = int(os.environ.get("SIMPLEMEM_MAX_RETRIEVAL_WORKERS", "3"))
ENABLE_PLANNING = os.environ.get("SIMPLEMEM_ENABLE_PLANNING", "true").lower() == "true"
ENABLE_REFLECTION = os.environ.get("SIMPLEMEM_ENABLE_REFLECTION", "true").lower() == "true"
MAX_REFLECTION_ROUNDS = int(os.environ.get("SIMPLEMEM_MAX_REFLECTION_ROUNDS", "2"))

JUDGE_API_KEY = OPENAI_API_KEY
JUDGE_BASE_URL = OPENAI_BASE_URL
JUDGE_MODEL = LLM_MODEL
JUDGE_ENABLE_THINKING = False
JUDGE_USE_STREAMING = False
JUDGE_TEMPERATURE = 0.3
