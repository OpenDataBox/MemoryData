"""
Milvus Tenant Configuration Utility Functions

This module provides utility functions related to tenant Milvus configuration, used to extract Milvus configuration from tenant information.
"""

import os
from typing import Optional, Dict, Any
from hashlib import md5

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant
from core.tenants.tenant_config import get_tenant_config

logger = get_logger(__name__)


def get_tenant_milvus_config() -> Optional[Dict[str, Any]]:
    """
    Retrieve Milvus configuration from the current tenant context

    Extract Milvus-related configuration from the storage_info of tenant information.
    If tenant configuration is incomplete or missing, supplement it from environment variables.

    Example configuration structure:
    {
        "host": "localhost",
        "port": 19530,
        "user": "admin",
        "password": "password"
    }

    Note:
        Tenant isolation in Milvus is achieved through Collection names (handled in TenantAwareCollection),
        not using db_name level isolation.

    Returns:
        Milvus configuration dictionary, or None if not available

    Examples:
        >>> config = get_tenant_milvus_config()
        >>> if config:
        ...     print(f"Milvus URI: {config['host']}:{config['port']}")
    """
    tenant_info = get_current_tenant()
    if not tenant_info:
        logger.debug(
            "⚠️ Tenant context not set, unable to retrieve tenant Milvus configuration"
        )
        return None

    # Retrieve Milvus configuration from tenant's storage_info
    # Support two configuration key names: milvus or milvus_config
    milvus_config = tenant_info.get_storage_info("milvus")
    if milvus_config is None:
        milvus_config = tenant_info.get_storage_info("milvus_config")

    # Retrieve environment variable configuration as fallback
    env_fallback_config = load_milvus_config_from_env()

    if not milvus_config:
        # No Milvus information in tenant configuration, use environment variable configuration
        final_config = {
            "host": env_fallback_config.get("host", "localhost"),
            "port": env_fallback_config.get("port", 19530),
            "user": env_fallback_config.get("user", ""),
            "password": env_fallback_config.get("password", ""),
        }
        logger.info(
            "✅ Tenant [%s] configuration missing Milvus information, using environment variable configuration: host=%s, port=%s",
            tenant_info.tenant_id,
            final_config["host"],
            final_config["port"],
        )
        return final_config

    # Compatibility logic: if tenant configuration is missing certain fields, supplement from environment variables
    final_config = {
        "host": milvus_config.get("host")
        or env_fallback_config.get("host", "localhost"),
        "port": milvus_config.get("port") or env_fallback_config.get("port", 19530),
        "user": milvus_config.get("user") or env_fallback_config.get("user", ""),
        "password": milvus_config.get("password")
        or env_fallback_config.get("password", ""),
    }

    logger.debug(
        "✅ Retrieved Milvus configuration from tenant [%s]: host=%s, port=%s",
        tenant_info.tenant_id,
        final_config["host"],
        final_config["port"],
    )

    return final_config


def get_milvus_connection_cache_key(config: Dict[str, Any]) -> str:
    """
    Generate cache key based on Milvus connection configuration

    Use the hash value of connection parameters (host, port, user, password) as the cache key.
    Note: db_name is not included in the cache key generation, as the same connection can access different databases.

    Args:
        config: Milvus connection configuration dictionary

    Returns:
        Cache key string (MD5 hash value)

    Examples:
        >>> config = {"host": "localhost", "port": 19530, "user": "admin", "password": "pwd"}
        >>> cache_key = get_milvus_connection_cache_key(config)
    """
    # Use connection parameters to generate a unique cache key (excluding db_name, as one connection can access multiple databases)
    key_parts = [
        str(config.get("host", "")),
        str(config.get("port", "")),
        str(config.get("user", "")),
        str(config.get("password", "")),
    ]
    key_str = "|".join(key_parts)
    cache_key = md5(key_str.encode()).hexdigest()[:16]
    return cache_key


def load_milvus_config_from_env() -> Dict[str, Any]:
    """
    Load default Milvus configuration from environment variables

    Read the following environment variables:
    - MILVUS_HOST: Milvus host address, default localhost
    - MILVUS_PORT: Milvus port, default 19530
    - MILVUS_USER: Username (optional)
    - MILVUS_PASSWORD: Password (optional)

    Note:
        MILVUS_DB_NAME is not used; tenant isolation is achieved through Collection names

    Returns:
        Milvus configuration dictionary

    Examples:
        >>> config = load_milvus_config_from_env()
        >>> print(f"Milvus URI: {config['host']}:{config['port']}")
    """
    config = {
        "host": os.getenv("MILVUS_HOST", "localhost"),
        "port": int(os.getenv("MILVUS_PORT", "19530")),
        "user": os.getenv("MILVUS_USER", ""),
        "password": os.getenv("MILVUS_PASSWORD", ""),
    }

    logger.debug(
        "Loaded default Milvus configuration from environment variables: host=%s, port=%s",
        config["host"],
        config["port"],
    )

    return config


def get_tenant_aware_collection_name(original_name: str) -> str:
    """
    Generate tenant-aware Collection name

    Add tenant prefix to the Collection name based on the current tenant context.
    If in non-tenant mode or without tenant context, return the original name.

    Naming rules:
    - Add tenant prefix: {tenant_id}_{original_name}
    - Replace special characters: replace "-" and "." with "_" to comply with Milvus naming conventions

    Args:
        original_name: Original Collection name

    Returns:
        str: Tenant-aware Collection name

    Examples:
        >>> # In tenant mode
        >>> set_current_tenant(TenantInfo(tenant_id="tenant-001", ...))
        >>> get_tenant_aware_collection_name("my_collection")
        'tenant_001_my_collection'

        >>> # In non-tenant mode or without tenant context
        >>> get_tenant_aware_collection_name("my_collection")
        'my_collection'
    """
    try:

        # Check if in non-tenant mode
        config = get_tenant_config()
        if config.non_tenant_mode:
            return original_name

        # Retrieve current tenant information
        tenant_info = get_current_tenant()
        if not tenant_info:
            return original_name

        # Generate tenant prefix (replace special characters to comply with Milvus naming conventions)
        tenant_prefix = tenant_info.tenant_id.replace("-", "_").replace(".", "_")

        # Return tenant-aware table name
        return f"{tenant_prefix}_{original_name}"

    except Exception as e:
        logger.warning(
            "Failed to generate tenant-aware Collection name, using original name [%s]: %s",
            original_name,
            e,
        )
        return original_name
