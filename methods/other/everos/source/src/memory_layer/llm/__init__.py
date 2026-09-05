"""
LLM providers module for memory layer.

This module provides LLM providers for the memory layer functionality.
"""

from memory_layer.llm.openai_provider import OpenAIProvider
from memory_layer.llm.protocol import LLMProvider

__all__ = ["LLMProvider", "OpenAIProvider"]


def create_provider(provider_type: str, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: Type of provider ("openai")
        **kwargs: Provider-specific arguments

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider_type is not supported
    """
    provider_type = provider_type.lower()

    if provider_type == "openai":
        return OpenAIProvider(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider type: {provider_type}. Supported types: 'openai'"
        )


def create_provider_from_env(provider_type: str, **kwargs) -> LLMProvider:
    """
    Create LLM provider from environment variables.

    Args:
        provider_type: Type of provider ("openai")
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider_type is not supported
    """
    provider_type = provider_type.lower()

    if provider_type == "openai":
        return OpenAIProvider.from_env(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider type: {provider_type}. Supported types: 'openai'"
        )
