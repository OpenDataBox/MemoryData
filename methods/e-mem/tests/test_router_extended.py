"""Additional tests for router.py to improve coverage."""
from unittest.mock import Mock

import pytest

from src.memory.router.router import Router


class TestRouterEdgeCases:
    """Test Router edge cases."""

    def test_router_with_disabled_router(self):
        """Test Router with router disabled."""
        router = Router(
            openai_config=None,
            enable_router=False,
            max_memory_segments=5,
            max_blocks=5,
        )

        assert router is not None
        assert router.enable_router is False

    def test_router_add_blocks_with_inactive_agent(self):
        """Test Router add_blocks with inactive agent."""
        router = Router(
            openai_config=None,
            enable_router=False,
            max_memory_segments=5,
            max_blocks=5,
        )

        mock_agent = Mock()
        mock_agent.summary = "Test summary"
        mock_agent.original_texts = ["text1", "text2"]
        mock_agent.is_active = False

        router.add_blocks(mock_agent)
        assert len(router.agent) == 1

    def test_router_add_blocks_with_active_agent_skipped(self):
        """Test Router add_blocks skips active agent."""
        router = Router(
            openai_config=None,
            enable_router=False,
            max_memory_segments=5,
            max_blocks=5,
        )

        mock_agent = Mock()
        mock_agent.summary = "Test summary"
        mock_agent.is_active = True

        router.add_blocks(mock_agent)
        assert len(router.agent) == 0  # Active agent should be skipped

    def test_router_map_blocks_no_agents(self):
        """Test Router _map_blocks with no agents."""
        router = Router(
            openai_config=None,
            enable_router=False,
            max_memory_segments=5,
            max_blocks=5,
        )

        result = router._map_blocks("test query")
        assert result == []

    def test_router_map_blocks_disabled_router(self):
        """Test Router _map_blocks with disabled router."""
        router = Router(
            openai_config=None,
            enable_router=False,
            max_memory_segments=5,
            max_blocks=5,
        )

        mock_agent = Mock()
        mock_agent.summary = "Test summary"
        mock_agent.original_texts = ["text1", "text2"]
        mock_agent.is_active = False
        router.add_blocks(mock_agent)

        result = router._map_blocks("test query")
        # Should return all blocks when router is disabled
        assert len(result) == 1

    def test_router_init_requires_openai_config_when_enabled(self):
        """Test Router raises error when openai_config missing with enabled router."""
        with pytest.raises(NotImplementedError, match="Please provide openai_config"):
            Router(
                openai_config=None,
                enable_router=True,
            )

    def test_router_map_reduce_blocks_no_agents(self):
        """Test Router map_reduce_blocks with no agents."""
        router = Router(
            openai_config=None,
            enable_router=False,
            max_memory_segments=5,
            max_blocks=5,
        )

        result = router.map_reduce_blocks("test query")
        assert result == []

    def test_router_map_reduce_blocks_with_agent(self):
        """Test Router map_reduce_blocks with agent."""
        router = Router(
            openai_config=None,
            enable_router=False,
            max_memory_segments=5,
            max_blocks=5,
        )

        mock_agent = Mock()
        mock_agent.summary = "Test summary"
        mock_agent.original_texts = ["text1", "text2"]
        mock_agent.is_active = False
        mock_agent.query.return_value = "Test response"
        router.add_blocks(mock_agent)

        result = router.map_reduce_blocks("test query")
        assert len(result) == 1
        assert result[0] == "Test response"

    def test_router_execute_tool(self):
        """Test Router execute_tool method."""
        router = Router(
            openai_config=None,
            enable_router=False,
            max_memory_segments=5,
            max_blocks=5,
        )

        # execute_tool should return None for unknown tools
        result = router.execute_tool("unknown_tool", {})
        assert result is None

    def test_router_max_blocks_parameter(self):
        """Test Router respects max_blocks parameter when router is disabled."""
        router = Router(
            openai_config=None,
            enable_router=False,
            max_memory_segments=5,
            max_blocks=2,
        )

        # Add 3 agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.summary = f"Summary {i}"
            mock_agent.original_texts = [f"text{i}"]
            mock_agent.is_active = False
            router.add_blocks(mock_agent)

        # When router is disabled, it returns all blocks regardless of max_blocks
        result = router._map_blocks("test query", max_blocks=2)
        # Disabled router returns all blocks
        assert len(result) == 3
