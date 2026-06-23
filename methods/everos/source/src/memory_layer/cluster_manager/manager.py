"""ClusterManager - Core component for automatic memcell clustering.

This module provides pure computation logic for clustering memcells.
Storage is managed by the caller, not by ClusterManager itself.

Design:
- ClusterManager is a pure computation component
- Input: memcell + current state
- Output: cluster_id + updated state
- Caller is responsible for loading/saving state
"""

import asyncio
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

from memory_layer.cluster_manager.config import ClusterManagerConfig
from core.observation.logger import get_logger

logger = get_logger(__name__)

# Try to import vectorize service
try:
    from agentic_layer.vectorize_service import get_vectorize_service
    VECTORIZE_SERVICE_AVAILABLE = True
except ImportError:
    VECTORIZE_SERVICE_AVAILABLE = False
    logger.warning("Vectorize service not available, clustering will be limited")


class ClusterState:
    """Internal state for a single group's clustering."""
    
    def __init__(self):
        """Initialize empty cluster state."""
        self.event_ids: List[str] = []
        self.timestamps: List[float] = []
        self.vectors: List[np.ndarray] = []
        self.cluster_ids: List[str] = []
        self.eventid_to_cluster: Dict[str, str] = {}
        self.next_cluster_idx: int = 0
        
        # Centroid-based clustering state
        self.cluster_centroids: Dict[str, np.ndarray] = {}
        self.cluster_counts: Dict[str, int] = {}
        self.cluster_last_ts: Dict[str, Optional[float]] = {}
    
    def assign_new_cluster(self, event_id: str) -> str:
        """Assign a new cluster ID to an event."""
        cluster_id = f"cluster_{self.next_cluster_idx:03d}"
        self.next_cluster_idx += 1
        self.eventid_to_cluster[event_id] = cluster_id
        self.cluster_ids.append(cluster_id)
        return cluster_id
    
    def add_to_cluster(
        self,
        event_id: str,
        cluster_id: str,
        vector: np.ndarray,
        timestamp: Optional[float]
    ) -> None:
        """Add an event to an existing cluster."""
        self.eventid_to_cluster[event_id] = cluster_id
        self.cluster_ids.append(cluster_id)
        self._update_cluster_centroid(cluster_id, vector, timestamp)
    
    def _update_cluster_centroid(
        self,
        cluster_id: str,
        vector: np.ndarray,
        timestamp: Optional[float]
    ) -> None:
        """Update cluster centroid with new vector."""
        if vector is None or vector.size == 0:
            if timestamp is not None:
                prev_ts = self.cluster_last_ts.get(cluster_id)
                self.cluster_last_ts[cluster_id] = max(prev_ts or timestamp, timestamp)
            return
        
        count = self.cluster_counts.get(cluster_id, 0)
        if count <= 0:
            self.cluster_centroids[cluster_id] = vector.astype(np.float32, copy=False)
            self.cluster_counts[cluster_id] = 1
        else:
            current_centroid = self.cluster_centroids[cluster_id]
            if current_centroid.dtype != np.float32:
                current_centroid = current_centroid.astype(np.float32)
            new_centroid = (current_centroid * float(count) + vector) / float(count + 1)
            self.cluster_centroids[cluster_id] = new_centroid.astype(np.float32, copy=False)
            self.cluster_counts[cluster_id] = count + 1
        
        if timestamp is not None:
            prev_ts = self.cluster_last_ts.get(cluster_id)
            self.cluster_last_ts[cluster_id] = max(prev_ts or timestamp, timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "event_ids": self.event_ids,
            "timestamps": self.timestamps,
            "cluster_ids": self.cluster_ids,
            "eventid_to_cluster": self.eventid_to_cluster,
            "next_cluster_idx": self.next_cluster_idx,
            "cluster_centroids": {
                cid: centroid.tolist()
                for cid, centroid in self.cluster_centroids.items()
            },
            "cluster_counts": self.cluster_counts,
            "cluster_last_ts": self.cluster_last_ts,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ClusterState":
        """Create ClusterState from dictionary."""
        state = ClusterState()
        state.event_ids = list(data.get("event_ids", []))
        state.timestamps = list(data.get("timestamps", []))
        state.cluster_ids = list(data.get("cluster_ids", []))
        state.eventid_to_cluster = dict(data.get("eventid_to_cluster", {}))
        state.next_cluster_idx = int(data.get("next_cluster_idx", 0))
        
        centroids = data.get("cluster_centroids", {}) or {}
        state.cluster_centroids = {
            k: np.array(v, dtype=np.float32) for k, v in centroids.items()
        }
        state.cluster_counts = {
            k: int(v) for k, v in (data.get("cluster_counts", {}) or {}).items()
        }
        state.cluster_last_ts = {
            k: float(v) for k, v in (data.get("cluster_last_ts", {}) or {}).items()
        }
        
        return state


class ClusterManager:
    """Automatic clustering manager - pure computation component.
    
    ClusterManager handles incremental clustering of memcells based on semantic
    similarity (embeddings) and temporal proximity.
    
    IMPORTANT: This is a pure computation component. The caller is responsible
    for loading/saving cluster state.
    
    Usage:
        ```python
        cluster_mgr = ClusterManager(config)
        
        # Caller loads state (from InMemory / MongoDB / file)
        state_dict = await storage.load(group_id)
        state = ClusterState.from_dict(state_dict) if state_dict else ClusterState()
        
        # Pure computation
        cluster_id, updated_state = await cluster_mgr.cluster_memcell(memcell, state)
        
        # Caller saves state
        await storage.save(group_id, updated_state.to_dict())
        ```
    """
    
    def __init__(self, config: Optional[ClusterManagerConfig] = None):
        """Initialize ClusterManager.
        
        Args:
            config: Clustering configuration (uses defaults if None)
        """
        self.config = config or ClusterManagerConfig()
        self._callbacks: List[Callable] = []
        
        # Vectorize service
        self._vectorize_service = None
        if VECTORIZE_SERVICE_AVAILABLE:
            try:
                self._vectorize_service = get_vectorize_service()
            except Exception as e:
                logger.warning(f"Failed to initialize vectorize service: {e}")
        
        # Statistics
        self._stats = {
            "total_memcells": 0,
            "clustered_memcells": 0,
            "new_clusters": 0,
            "failed_embeddings": 0,
        }
    
    def on_cluster_assigned(self, callback: Callable[[str, Dict[str, Any], str], None]) -> None:
        """Register a callback for cluster assignment events.
        
        Callback signature:
            callback(group_id: str, memcell: Dict[str, Any], cluster_id: str) -> None
        """
        self._callbacks.append(callback)
    
    async def cluster_memcell(
        self,
        memcell: Dict[str, Any],
        state: ClusterState,
    ) -> Tuple[Optional[str], ClusterState]:
        """Cluster a memcell and return updated state.
        
        Pure computation method - no storage operations.
        Caller is responsible for loading state before and saving it after.
        
        Args:
            memcell: Memcell dictionary with event_id, timestamp, episode/summary
            state: Current cluster state for the group
        
        Returns:
            Tuple of (cluster_id, updated_state):
            - cluster_id: Assigned cluster ID, or None if failed
            - state: Updated ClusterState (same object, mutated)
        """
        self._stats["total_memcells"] += 1
        
        # Extract key fields
        event_id = str(memcell.get("event_id", ""))
        if not event_id:
            logger.warning("Memcell missing event_id, skipping clustering")
            return None, state
        
        timestamp = self._parse_timestamp(memcell.get("timestamp"))
        text = self._extract_text(memcell)
        
        # Get embedding
        vector = await self._get_embedding(text)
        if vector is None or vector.size == 0:
            logger.warning(f"Failed to get embedding for event {event_id}, creating singleton cluster")
            cluster_id = state.assign_new_cluster(event_id)
            state.event_ids.append(event_id)
            state.timestamps.append(timestamp or 0.0)
            state.vectors.append(np.zeros((1,), dtype=np.float32))
            self._stats["new_clusters"] += 1
            self._stats["failed_embeddings"] += 1
            return cluster_id, state
        
        # Find best matching cluster
        cluster_id = self._find_best_cluster(state, vector, timestamp)
        
        # Add to cluster
        if cluster_id is None:
            cluster_id = state.assign_new_cluster(event_id)
            state._update_cluster_centroid(cluster_id, vector, timestamp)
            self._stats["new_clusters"] += 1
        else:
            state.add_to_cluster(event_id, cluster_id, vector, timestamp)
        
        # Update state
        state.event_ids.append(event_id)
        state.timestamps.append(timestamp or 0.0)
        state.vectors.append(vector)
        
        self._stats["clustered_memcells"] += 1
        
        return cluster_id, state
    
    def _find_best_cluster(
        self,
        state: ClusterState,
        vector: np.ndarray,
        timestamp: Optional[float]
    ) -> Optional[str]:
        """Find the best matching cluster for a vector."""
        if not state.cluster_centroids:
            return None
        
        best_similarity = -1.0
        best_cluster_id = None
        
        vector_norm = np.linalg.norm(vector) + 1e-9
        
        for cluster_id, centroid in state.cluster_centroids.items():
            if centroid is None or centroid.size == 0:
                continue
            
            # Check time constraint
            if timestamp is not None:
                last_ts = state.cluster_last_ts.get(cluster_id)
                if last_ts is not None:
                    time_diff = abs(timestamp - last_ts)
                    if time_diff > self.config.max_time_gap_seconds:
                        continue
            
            # Compute cosine similarity
            centroid_norm = np.linalg.norm(centroid) + 1e-9
            similarity = float((centroid @ vector) / (centroid_norm * vector_norm))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id
        
        if best_similarity >= self.config.similarity_threshold:
            return best_cluster_id
        
        return None
    
    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        if not self._vectorize_service:
            logger.warning("Vectorize service not available")
            return None
        
        try:
            vector_arr = await self._vectorize_service.get_embedding(text)
            if vector_arr is not None:
                return np.array(vector_arr, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
        
        return None
    
    def _extract_text(self, memcell: Dict[str, Any]) -> str:
        """Extract representative text from memcell.
        
        Priority: episode > summary > original_data
        """
        episode = memcell.get("episode")
        if isinstance(episode, str) and episode.strip():
            return episode.strip()
        
        summary = memcell.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        
        lines = []
        original_data = memcell.get("original_data")
        if isinstance(original_data, list):
            for item in original_data:
                if isinstance(item, dict):
                    content = item.get("content") or item.get("summary")
                    if content:
                        text = str(content).strip()
                        if text:
                            lines.append(text)
        
        return "\n".join(lines) if lines else str(memcell.get("event_id", ""))
    
    def _parse_timestamp(self, timestamp: Any) -> Optional[float]:
        """Parse timestamp to float seconds."""
        if timestamp is None:
            return None
        
        try:
            if isinstance(timestamp, (int, float)):
                val = float(timestamp)
                if val > 10_000_000_000:
                    val = val / 1000.0
                return val
            elif isinstance(timestamp, str):
                from common_utils.datetime_utils import from_iso_format
                dt = from_iso_format(timestamp)
                return dt.timestamp()
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {timestamp}: {e}")
        
        return None
    
    async def _notify_callbacks(
        self,
        group_id: str,
        memcell: Dict[str, Any],
        cluster_id: str
    ) -> None:
        """Notify all registered callbacks of cluster assignment."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(group_id, memcell, cluster_id)
                else:
                    callback(group_id, memcell, cluster_id)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        return dict(self._stats)
