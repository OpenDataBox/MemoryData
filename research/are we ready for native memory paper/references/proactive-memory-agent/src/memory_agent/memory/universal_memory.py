"""Universal Memory class for storing status, knowledge, and procedural memories.

Memory types:
- status: Working memory for current task progress (always in context)
- knowledge: Environmental facts observed by the agent [ENV], [PATH], [FACT]
- procedural: Historical solutions and experiences [BUG], [PERF]
"""

import json
import shortuuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MemoryEntry:
    """A single memory entry with ID, content, and metadata."""
    
    id: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    last_accessed: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=data["id"],
            content=data["content"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
        )


class UniversalMemory:
    """Universal memory system for ReAct agents.
    
    Provides three types of memory:
    - status: Current task progress state (~512 tokens, updated frequently)
    - knowledge: Environmental facts observed by the agent
    - procedural: Historical solutions and bug fixes
    
    Example usage:
        memory = UniversalMemory()
        
        # Update working status
        memory.update_status("Step 3: Running optimization...")
        
        # Save knowledge
        kid = memory.save_knowledge("[ENV] Docker container has no systemd")
        
        # Save procedural memory
        pid = memory.save_procedural("[BUG] shred timeout - solved with xargs -n 10")
        
        # Search memories
        results = memory.search("timeout", top_k=5)
        
        # Delete outdated memory
        memory.delete(kid)
    """
    
    def __init__(self, max_status_length: int = 2000):
        """Initialize empty memory stores.
        
        Args:
            max_status_length: Maximum character length for status (default: 2000)
        """
        self._status: str = ""
        self._knowledge: list[MemoryEntry] = []
        self._procedural: list[MemoryEntry] = []
        self._max_status_length = max_status_length
        
        # Lazy-loaded BM25 indices
        self._knowledge_index = None
        self._procedural_index = None
    
    @property
    def status(self) -> str:
        """Get current working status."""
        return self._status
    
    @status.setter
    def status(self, value: str) -> None:
        """Set working status with truncation if needed."""
        if len(value) > self._max_status_length:
            value = value[:self._max_status_length] + "..."
        self._status = value
    
    @property
    def knowledge(self) -> list[MemoryEntry]:
        """Get all knowledge entries."""
        return self._knowledge
    
    @property
    def procedural(self) -> list[MemoryEntry]:
        """Get all procedural entries."""
        return self._procedural
    
    def _generate_id(self) -> str:
        """Generate a short unique ID for memory entries."""
        return shortuuid.uuid()[:8]
    
    def _invalidate_indices(self, memory_type: str) -> None:
        """Invalidate BM25 index after modification."""
        if memory_type == "knowledge":
            self._knowledge_index = None
        elif memory_type == "procedural":
            self._procedural_index = None
    
    def update_status(self, content: str) -> None:
        """Update working status.
        
        Args:
            content: New status content (will be truncated if too long)
        """
        self.status = content
    
    def save_knowledge(self, content: str) -> str:
        """Save an environmental fact to knowledge memory.
        
        Recommended tags:
        - [ENV] Environmental characteristics (e.g., "[ENV] No systemd in container")
        - [PATH] Key paths (e.g., "[PATH] PostgreSQL: /var/lib/postgresql/16/main")
        - [FACT] General facts (e.g., "[FACT] Task goal is to maximize accuracy")
        
        Args:
            content: Knowledge content to save
        
        Returns:
            The generated memory ID
        """
        entry = MemoryEntry(
            id=self._generate_id(),
            content=content,
        )
        self._knowledge.append(entry)
        self._invalidate_indices("knowledge")
        return entry.id
    
    def save_procedural(self, content: str) -> str:
        """Save a solution record to procedural memory.
        
        Recommended tags:
        - [BUG] Bug fix records (e.g., "[BUG] OOM solved by reducing batch_size to 16")
        - [PERF] Performance improvements (e.g., "[PERF] LightGBM features improved acc by 5%")
        
        Args:
            content: Procedural content to save
        
        Returns:
            The generated memory ID
        """
        entry = MemoryEntry(
            id=self._generate_id(),
            content=content,
        )
        self._procedural.append(entry)
        self._invalidate_indices("procedural")
        return entry.id
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by ID.
        
        Args:
            memory_id: ID of the memory to delete
        
        Returns:
            True if deleted, False if not found
        """
        # Try knowledge first
        for i, entry in enumerate(self._knowledge):
            if entry.id == memory_id:
                self._knowledge.pop(i)
                self._invalidate_indices("knowledge")
                return True
        
        # Try procedural
        for i, entry in enumerate(self._procedural):
            if entry.id == memory_id:
                self._procedural.pop(i)
                self._invalidate_indices("procedural")
                return True
        
        return False
    
    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory entry by ID.
        
        Args:
            memory_id: ID of the memory to get
        
        Returns:
            MemoryEntry if found, None otherwise
        """
        for entry in self._knowledge:
            if entry.id == memory_id:
                # access_count tracking disabled: only BM25 path incremented it,
                # but context is now LLM-generated (Phase 2), making counts misleading.
                return entry

        for entry in self._procedural:
            if entry.id == memory_id:
                return entry
        
        return None
    
    def search_knowledge(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search knowledge memory using BM25.
        
        Args:
            query: Search query
            top_k: Maximum number of results
        
        Returns:
            List of relevant MemoryEntry objects
        """
        if not self._knowledge:
            return []
        
        from .bm25_search import BM25Index
        
        if self._knowledge_index is None:
            documents = [entry.content for entry in self._knowledge]
            self._knowledge_index = BM25Index(documents)
        
        indices = self._knowledge_index.search(query, top_k=top_k)
        
        results = []
        for idx in indices:
            entry = self._knowledge[idx]
            results.append(entry)

        return results

    def search_procedural(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search procedural memory using BM25.
        
        Args:
            query: Search query
            top_k: Maximum number of results
        
        Returns:
            List of relevant MemoryEntry objects
        """
        if not self._procedural:
            return []
        
        from .bm25_search import BM25Index
        
        if self._procedural_index is None:
            documents = [entry.content for entry in self._procedural]
            self._procedural_index = BM25Index(documents)
        
        indices = self._procedural_index.search(query, top_k=top_k)
        
        results = []
        for idx in indices:
            entry = self._procedural[idx]
            results.append(entry)

        return results
    
    def search(self, query: str, top_k: int = 5) -> dict[str, list[MemoryEntry]]:
        """Search both knowledge and procedural memories.
        
        Args:
            query: Search query
            top_k: Maximum number of results per memory type
        
        Returns:
            Dict with 'knowledge' and 'procedural' lists
        """
        return {
            "knowledge": self.search_knowledge(query, top_k),
            "procedural": self.search_procedural(query, top_k),
        }
    
    def clear(self, keep_status: bool = False) -> None:
        """Clear all memories.
        
        Args:
            keep_status: If True, keep the current status
        """
        if not keep_status:
            self._status = ""
        self._knowledge.clear()
        self._procedural.clear()
        self._invalidate_indices("knowledge")
        self._invalidate_indices("procedural")
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize memory to dictionary."""
        return {
            "status": self._status,
            "knowledge": [entry.to_dict() for entry in self._knowledge],
            "procedural": [entry.to_dict() for entry in self._procedural],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UniversalMemory":
        """Deserialize memory from dictionary."""
        memory = cls()
        memory._status = data.get("status", "")
        memory._knowledge = [
            MemoryEntry.from_dict(entry) for entry in data.get("knowledge", [])
        ]
        memory._procedural = [
            MemoryEntry.from_dict(entry) for entry in data.get("procedural", [])
        ]
        return memory
    
    def save_to_file(self, path: Path | str) -> None:
        """Save memory to a JSON file.
        
        Args:
            path: Path to save the memory file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, path: Path | str) -> "UniversalMemory":
        """Load memory from a JSON file.
        
        Args:
            path: Path to the memory file
        
        Returns:
            UniversalMemory instance
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def render_context(self, include_empty: bool = False) -> str:
        """Render memory as XML context for injection into prompts.

        Args:
            include_empty: Whether to include empty sections

        Returns:
            Formatted XML string
        """
        sections = []

        # NOTE: Status section is excluded to prevent premature completion signals.
        # Status values like "Task completed successfully" cause the action agent to stop
        # iterating before the task is truly finished. Only knowledge/procedural injected.
        # if self._status or include_empty:
        #     sections.append(f"<status>\n{self._status or '(no status)'}\n</status>")
        
        # Knowledge section
        if self._knowledge:
            knowledge_lines = [
                f"[{entry.id}] {entry.content}" for entry in self._knowledge
            ]
            sections.append(
                f"<knowledge>\n{chr(10).join(knowledge_lines)}\n</knowledge>"
            )
        elif include_empty:
            sections.append("<knowledge>\n(no knowledge entries)\n</knowledge>")
        
        # Procedural section
        if self._procedural:
            procedural_lines = [
                f"[{entry.id}] {entry.content}" for entry in self._procedural
            ]
            sections.append(
                f"<procedural>\n{chr(10).join(procedural_lines)}\n</procedural>"
            )
        elif include_empty:
            sections.append("<procedural>\n(no procedural entries)\n</procedural>")
        
        return "\n\n".join(sections)
    
    def render_context_with_search(
        self,
        query: str,
        knowledge_top_k: int = 5,
        procedural_top_k: int = 5,
    ) -> str:
        """Render memory context with BM25-filtered results.

        Args:
            query: Search query (typically the current observation)
            knowledge_top_k: Max knowledge entries to include
            procedural_top_k: Max procedural entries to include

        Returns:
            Formatted XML string with relevant memories
        """
        sections = []

        # NOTE: Status section is excluded to prevent premature completion signals.
        # Status values like "Task completed successfully" cause the action agent to stop
        # iterating before the task is truly finished. Only knowledge/procedural injected.
        # sections.append(f"<status>\n{self._status or '(no status)'}\n</status>")
        
        # Search and include relevant knowledge
        if self._knowledge:
            relevant_k = self.search_knowledge(query, top_k=knowledge_top_k)
            if relevant_k:
                knowledge_lines = [
                    f"[{entry.id}] {entry.content}" for entry in relevant_k
                ]
                sections.append(
                    f"<knowledge>\n{chr(10).join(knowledge_lines)}\n</knowledge>"
                )
        
        # Search and include relevant procedural
        if self._procedural:
            relevant_p = self.search_procedural(query, top_k=procedural_top_k)
            if relevant_p:
                procedural_lines = [
                    f"[{entry.id}] {entry.content}" for entry in relevant_p
                ]
                sections.append(
                    f"<procedural>\n{chr(10).join(procedural_lines)}\n</procedural>"
                )
        
        return "\n\n".join(sections)
    
    def __len__(self) -> int:
        """Return total number of memory entries (excluding status)."""
        return len(self._knowledge) + len(self._procedural)
    
    def __repr__(self) -> str:
        return (
            f"UniversalMemory(status_len={len(self._status)}, "
            f"knowledge={len(self._knowledge)}, procedural={len(self._procedural)})"
        )
