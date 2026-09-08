"""
Tenant information service module

This module defines the tenant information service interface and its default implementation,
used to retrieve tenant information based on tenant_id (typically single_tenant_id from config).

Uses DI mechanism to manage TenantInfoService implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional

from core.tenants.tenant_models import TenantInfo, TenantDetail
from core.di.decorators import component


class TenantInfoService(ABC):
    """
    Tenant information service interface

    This interface defines standard methods for retrieving tenant information.
    Different implementations can retrieve tenant information from various data sources (e.g., database, API, configuration files).

    Using DI mechanism:
    - Multiple implementations can be registered
    - Use primary=True to mark the default implementation
    - Obtain instances through the container
    """

    @abstractmethod
    def get_tenant_info(self, tenant_id: str) -> Optional[TenantInfo]:
        """
        Retrieve tenant information by tenant ID

        Args:
            tenant_id: Unique identifier of the tenant

        Returns:
            Tenant information object, or None if not found
        """
        raise NotImplementedError


@component("default_tenant_info_service")
class DefaultTenantInfoService(TenantInfoService):
    """
    Default tenant information service implementation

    This implementation provides basic tenant information containing only the tenant_id,
    without detailed information such as storage configurations. Suitable for simple scenarios or as the default implementation.

    Uses the @component decorator to register into the DI container and mark as primary.
    """

    def get_tenant_info(self, tenant_id: str) -> Optional[TenantInfo]:
        """
        Create a basic tenant information object by tenant ID

        This implementation creates a TenantInfo object containing only the tenant_id,
        without any storage configuration information.

        Args:
            tenant_id: Unique identifier of the tenant

        Returns:
            TenantInfo object containing basic information

        Examples:
            >>> from core.di.container import get_container
            >>> service = get_container().get_bean_by_type(TenantInfoService)
            >>> tenant_info = service.get_tenant_info("tenant_001")
            >>> print(tenant_info.tenant_id)
            tenant_001
        """
        # Return None if tenant_id is empty
        if not tenant_id:
            return None

        # Create basic tenant information containing only tenant_id
        return TenantInfo(
            tenant_id=tenant_id, tenant_detail=TenantDetail(), origin_tenant_data={}
        )
