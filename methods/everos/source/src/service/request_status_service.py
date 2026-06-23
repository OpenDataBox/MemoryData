# -*- coding: utf-8 -*-
"""
Request status service

Responsible for writing request status to Redis (using Hash structure) and providing read functionality.
Used to track the status of requests moved to the background.

Redis Key format: request_status:{tenant_key_prefix}:{request_id}
TTL: 1 hour
"""

from typing import Any, Dict, Optional

from core.component.redis_provider import RedisProvider
from core.di import service
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger
from core.tenants.request_tenant_provider import RequestTenantInfo

logger = get_logger(__name__)

# Redis key prefix
REQUEST_STATUS_KEY_PREFIX = "request_status"

# TTL: 1 hour (in seconds)
REQUEST_STATUS_TTL = 60 * 60


@service("request_status_service")
class RequestStatusService:
    """
    Request status service

    Responsibilities:
    - Write request status to Redis (using Hash structure for extensibility)
    - Provide functionality to read specific request status
    - Set a TTL of 1 hour

    Redis Hash structure example:
    request_status:{tenant_key_prefix}:{request_id} = {
        "status": "start|success|failed",
        "url": "request URL",
        "method": "GET|POST|...",
        "http_code": "200",
        "time_ms": "123",
        "error_message": "error message (if any)",
        "start_time": "start timestamp",
        "end_time": "end timestamp"
    }
    """

    def __init__(self):
        """Initialize service"""
        # Lazy load RedisProvider to avoid circular dependency
        self._redis_provider: Optional[RedisProvider] = None

    def _get_redis_provider(self) -> RedisProvider:
        """
        Get Redis Provider (lazy loading)

        Returns:
            RedisProvider: Redis provider instance
        """
        if self._redis_provider is None:
            self._redis_provider = get_bean_by_type(RedisProvider)
        return self._redis_provider

    def _build_key(self, tenant_info: RequestTenantInfo, request_id: str) -> str:
        """
        Build Redis Key

        Args:
            tenant_info: Request tenant information
            request_id: Request ID

        Returns:
            str: Redis key
        """
        return tenant_info.build_status_key(REQUEST_STATUS_KEY_PREFIX, request_id)

    async def update_request_status(
        self,
        tenant_info: RequestTenantInfo,
        request_id: str,
        status: str,
        url: Optional[str] = None,
        method: Optional[str] = None,
        http_code: Optional[int] = None,
        time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        timestamp: Optional[int] = None,
    ) -> bool:
        """
        Update request status to Redis

        Args:
            tenant_info: Request tenant information
            request_id: Request ID
            status: Request status (start/success/failed)
            url: Request URL (optional)
            method: HTTP method (optional)
            http_code: HTTP status code (optional)
            time_ms: Request duration in milliseconds (optional)
            error_message: Error message (optional)
            timestamp: Timestamp (optional)

        Returns:
            bool: Whether the update was successful
        """
        if not request_id:
            logger.warning(
                "Missing request_id, skipping request status update: tenant_key_prefix=%s",
                tenant_info.tenant_key_prefix,
            )
            return False

        try:
            redis_provider = self._get_redis_provider()
            client = await redis_provider.get_client()

            key = self._build_key(tenant_info, request_id)

            # Build fields to update
            fields: Dict[str, str] = {"status": status}

            if url is not None:
                fields["url"] = url
            if method is not None:
                fields["method"] = method
            if http_code is not None:
                fields["http_code"] = str(http_code)
            if time_ms is not None:
                fields["time_ms"] = str(time_ms)
            if error_message is not None:
                fields["error_message"] = error_message
            if timestamp is not None:
                # Set different time fields based on status
                if status == "start":
                    fields["start_time"] = str(timestamp)
                else:
                    fields["end_time"] = str(timestamp)

            # Use Pipeline to combine hset + expire operations (reduce network round trips)
            pipe = client.pipeline()
            pipe.hset(key, mapping=fields)
            pipe.expire(key, REQUEST_STATUS_TTL)
            await pipe.execute()

            logger.debug(
                "Request status updated to Redis: key=%s, status=%s", key, status
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to update request status to Redis: tenant_key_prefix=%s, req=%s, error=%s",
                tenant_info.tenant_key_prefix,
                request_id,
                str(e),
            )
            return False

    async def get_request_status(
        self, tenant_info: RequestTenantInfo, request_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get request status

        Args:
            tenant_info: Request tenant information
            request_id: Request ID

        Returns:
            Optional[Dict[str, Any]]: Request status information, returns None if not exists
        """
        if not request_id:
            logger.warning(
                "Missing request_id, cannot get request status: tenant_key_prefix=%s",
                tenant_info.tenant_key_prefix,
            )
            return None

        try:
            redis_provider = self._get_redis_provider()
            client = await redis_provider.get_client()

            key = self._build_key(tenant_info, request_id)

            # Use Pipeline to combine hgetall + ttl operations (reduce network round trips)
            pipe = client.pipeline()
            pipe.hgetall(key)
            pipe.ttl(key)
            results = await pipe.execute()

            data = results[0]
            ttl = results[1]

            if not data:
                logger.debug("Request status does not exist: key=%s", key)
                return None

            # Convert data types
            result: Dict[str, Any] = {"request_id": request_id}

            for field, value in data.items():
                if field in ("http_code", "time_ms", "start_time", "end_time"):
                    # Convert numeric fields to int
                    try:
                        result[field] = int(value)
                    except (ValueError, TypeError):
                        result[field] = value
                else:
                    result[field] = value

            # Add remaining TTL
            if ttl > 0:
                result["ttl_seconds"] = ttl

            logger.debug("Successfully retrieved request status: key=%s", key)
            return result

        except Exception as e:
            logger.error(
                "Failed to get request status: tenant_key_prefix=%s, req=%s, error=%s",
                tenant_info.tenant_key_prefix,
                request_id,
                str(e),
            )
            return None

    async def delete_request_status(
        self, tenant_info: RequestTenantInfo, request_id: str
    ) -> bool:
        """
        Delete request status

        Args:
            tenant_info: Request tenant information
            request_id: Request ID

        Returns:
            bool: Whether deletion was successful
        """
        if not request_id:
            return False

        try:
            redis_provider = self._get_redis_provider()
            key = self._build_key(tenant_info, request_id)
            deleted = await redis_provider.delete(key)
            logger.debug("Request status deleted: key=%s, deleted=%d", key, deleted)
            return deleted > 0

        except Exception as e:
            logger.error(
                "Failed to delete request status: tenant_key_prefix=%s, req=%s, error=%s",
                tenant_info.tenant_key_prefix,
                request_id,
                str(e),
            )
            return False
