"""Cluster Manager - Pure computation component for memcell clustering.

This module provides ClusterManager, a pure computation component that clusters
memcells based on semantic similarity and temporal proximity.

IMPORTANT: This is a pure computation component. The caller is responsible
for loading/saving cluster state.

Usage:
    from memory_layer.cluster_manager import ClusterManager, ClusterManagerConfig, ClusterState
    
    # Initialize
    config = ClusterManagerConfig(
        similarity_threshold=0.65,
        max_time_gap_days=7,
    )
    cluster_mgr = ClusterManager(config)
    
    # Caller loads state (from InMemory / MongoDB / file)
    state_dict = await storage.load_cluster_state(group_id)
    state = ClusterState.from_dict(state_dict) if state_dict else ClusterState()
    
    # Pure computation
    cluster_id, state = await cluster_mgr.cluster_memcell(memcell, state)
    
    # Caller saves state
    await storage.save_cluster_state(group_id, state.to_dict())
"""

from memory_layer.cluster_manager.config import ClusterManagerConfig
from memory_layer.cluster_manager.manager import ClusterManager, ClusterState

__all__ = [
    "ClusterManager",
    "ClusterManagerConfig",
    "ClusterState",
]
