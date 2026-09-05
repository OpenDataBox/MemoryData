"""Configuration loader for memory system with Pydantic validation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import yaml

if TYPE_CHECKING:
    from src.config import AppConfig

logger = logging.getLogger(__name__)

# Default values
MAX_CONCURRENT_GPU_OPERATIONS: int = 2
DEFAULT_OVERLAP_RATIO: float = 0.1
DEFAULT_BLOCK_SIZE_RATIO: float = 0.125

# Cached config instance
_config_cache: Optional[Dict[str, Any]] = None


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


def get_config_path() -> str:
    """Get the configuration file path."""
    return os.path.join(os.path.dirname(__file__), "config.yaml")


def load_raw_config(
    config_path: Optional[str] = None,
    *,
    warn_if_missing: bool = True,
) -> Dict[str, Any]:
    """
    Load raw configuration from YAML file.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Configuration dictionary.

    Raises:
        ConfigurationError: If config file cannot be loaded.
    """
    if config_path is None:
        config_path = get_config_path()

    if not os.path.exists(config_path):
        if warn_if_missing:
            logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            return config_data if config_data else {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in config file: {e}") from e
    except OSError as e:
        raise ConfigurationError(f"Cannot read config file: {e}") from e


def load_validated_config(config_path: Optional[str] = None) -> "AppConfig":
    """
    Load and validate configuration using Pydantic.

    Args:
        config_path: Optional path to config file.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigurationError: If validation fails.
    """
    from pydantic import ValidationError

    from src.config import load_and_validate_config

    raw_config = load_raw_config(config_path)

    try:
        return load_and_validate_config(raw_config)
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            error_messages.append(f"  - {loc}: {msg}")
        raise ConfigurationError(
            "Configuration validation failed:\n" + "\n".join(error_messages)
        ) from e


def ensure_app_config(config: Union["AppConfig", Dict[str, Any]]) -> "AppConfig":
    """Return a validated AppConfig instance from a dict or AppConfig."""
    from src.config import AppConfig, load_and_validate_config

    if isinstance(config, AppConfig):
        return config
    return load_and_validate_config(config)


def build_chat_manager_kwargs(
    config: Union["AppConfig", Dict[str, Any]]
) -> Dict[str, Any]:
    """Build normalized create_chat_manager kwargs from a config object."""
    app_config = ensure_app_config(config)
    return app_config.to_chat_manager_kwargs()


def get_config(reload: bool = False) -> Dict[str, Any]:
    """
    Get configuration with caching.

    Args:
        reload: Force reload from file.

    Returns:
        Configuration dictionary.
    """
    global _config_cache

    if _config_cache is None or reload:
        _config_cache = load_raw_config()

    return _config_cache


def _update_globals_from_config() -> None:
    """Update global configuration values from config file."""
    global MAX_CONCURRENT_GPU_OPERATIONS, DEFAULT_OVERLAP_RATIO, DEFAULT_BLOCK_SIZE_RATIO

    config_data = load_raw_config(warn_if_missing=False)
    if not config_data:
        return

    try:
        app_config = ensure_app_config(config_data)
    except Exception as exc:  # pragma: no cover - defensive fallback on import
        logger.warning("Failed to validate config during global initialization: %s", exc)
        return

    MAX_CONCURRENT_GPU_OPERATIONS = app_config.memory.max_concurrent_gpu_operations
    DEFAULT_OVERLAP_RATIO = app_config.memory.overlap_ratio
    DEFAULT_BLOCK_SIZE_RATIO = app_config.memory.block_size_ratio


# Initialize on module load
_update_globals_from_config()
