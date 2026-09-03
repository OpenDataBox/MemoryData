"""
Storage management for Omni-Memory.

Implements cold storage for heavy multimodal data and hot storage for
lightweight metadata and embeddings.
"""

from omni_memory.storage.cold_storage import ColdStorageManager
from omni_memory.storage.mau_store import MAUStore
from omni_memory.storage.vector_store import VectorStore

__all__ = [
    "ColdStorageManager",
    "MAUStore",
    "VectorStore",
]
