# -*- coding: utf-8 -*-
"""
Request tenant provider module

This module defines the request tenant information model and provider interface,
used to retrieve tenant-related information from HTTP requests for constructing
Redis keys and other tenant-specific identifiers.

In the open-source version, the default implementation returns a simple key prefix.
Enterprise version can provide implementations that extract tenant info from HTTP headers
and construct more complex key prefixes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.di.decorators import component


@dataclass
class RequestTenantInfo:
    """
    Request tenant information data class

    Contains tenant-related information used for constructing Redis keys
    and other tenant-specific identifiers.

    This is a minimal interface - enterprise implementations can extend
    with additional fields internally, but the public interface only exposes
    what's needed for key construction.

    Attributes:
        tenant_key_prefix: Prefix for Redis keys (enterprise version may include org/space info)
    """

    tenant_key_prefix: str = field(default="default")

    def build_status_key(self, base_prefix: str, request_id: str) -> str:
        """
        Build complete Redis key for request status

        Args:
            base_prefix: Base prefix for the key (e.g., "request_status")
            request_id: Request ID

        Returns:
            str: Complete Redis key
        """
        return f"{base_prefix}:{self.tenant_key_prefix}:{request_id}"


class RequestTenantProvider(ABC):
    """
    Request tenant provider interface

    This interface defines standard methods for retrieving request tenant information
    from HTTP request.

    Using DI mechanism:
    - Multiple implementations can be registered
    - Use primary=True to mark the default implementation
    - Obtain instances through the container
    """

    @abstractmethod
    def get_tenant_info_from_request(self, request: Any) -> RequestTenantInfo:
        """
        Extract tenant information from HTTP request

        Args:
            request: HTTP request object (e.g., FastAPI Request)

        Returns:
            RequestTenantInfo: Tenant information extracted from request headers
        """
        raise NotImplementedError


@component("default_request_tenant_provider", primary=True)
class DefaultRequestTenantProvider(RequestTenantProvider):
    """
    Default request tenant provider implementation

    This implementation provides default tenant information for tenant-less scenarios.
    In the open-source version, tenant_key_prefix defaults to "default".

    Enterprise version should provide an implementation that:
    - Extracts tenant info from HTTP headers
    - Constructs appropriate tenant_key_prefix (e.g., "{org_id}:{space_id}")
    """

    def get_tenant_info_from_request(self, request: Any) -> RequestTenantInfo:
        """
        Extract tenant information from HTTP request (default implementation)

        In the open-source version, this returns default values regardless of request.

        Args:
            request: HTTP request object (ignored in default implementation)

        Returns:
            RequestTenantInfo: Default tenant information
        """
        return RequestTenantInfo(tenant_key_prefix="default")
