# -*- coding: utf-8 -*-
"""
Request status controller

Provides API for querying the processing status of requests.
Mainly used for tracking requests that are moved to background processing.
"""

from typing import Optional

from fastapi import Header, HTTPException, Query, Request as FastAPIRequest

from core.di.decorators import component
from core.di.utils import get_bean_by_type
from core.interface.controller.base_controller import BaseController, get
from core.observation.logger import get_logger
from core.tenants.request_tenant_provider import RequestTenantProvider
from service.request_status_service import RequestStatusService
from infra_layer.adapters.input.api.dto.status_dto import RequestStatusResponse


logger = get_logger(__name__)


@component(name="statusController")
class StatusController(BaseController):
    """
    Request status controller

    Provides API for querying the processing status of requests, mainly used for tracking requests that are moved to background processing.
    """

    def __init__(self, request_status_service: RequestStatusService):
        """
        Initialize the controller

        Args:
            request_status_service: Request status service (via dependency injection)
        """
        super().__init__(
            prefix="/api/v1/stats",
            tags=["Stats - Request Status"],
            default_auth="none",  # Adjust authentication strategy as needed
        )
        self.request_status_service = request_status_service
        self._request_tenant_provider: Optional[RequestTenantProvider] = None
        logger.info("StatusController initialized")

    def _get_request_tenant_provider(self) -> RequestTenantProvider:
        """Get RequestTenantProvider (lazy loading)"""
        if self._request_tenant_provider is None:
            self._request_tenant_provider = get_bean_by_type(RequestTenantProvider)
        return self._request_tenant_provider

    @get(
        "/request",
        response_model=RequestStatusResponse,
        summary="Query request status",
        description="""
        Query the processing status of a specific request

        ## Function description:
        - Query request status by request_id
        - Return request processing progress (start/success/failed)
        - Support viewing request duration, HTTP status code, and other information

        ## Parameter passing method:
        Pass parameters via Query Parameter (recommended):
        - request_id: Request ID

        Or via HTTP Header (deprecated, will be removed in future versions):
        - X-Request-Id: Request ID

        If both are provided, Query Parameter takes precedence.

        ## Use cases:
        - Tracking status of background requests
        - Client polling for request completion status

        ## Note:
        - Request status data has a TTL of 1 hour; it will no longer be queryable after expiration

        ## API path:
        GET /api/v1/stats/request?request_id=xxx
        """,
        responses={
            200: {
                "description": "Query successful",
                "content": {
                    "application/json": {
                        "example": {
                            "success": True,
                            "found": True,
                            "data": {
                                "request_id": "req-789",
                                "status": "success",
                                "url": "/api/memory/memorize",
                                "method": "POST",
                                "http_code": 200,
                                "time_ms": 1500,
                                "start_time": 1702400000000,
                                "end_time": 1702400001500,
                                "ttl_seconds": 3500,
                            },
                            "message": None,
                        }
                    }
                },
            },
            400: {
                "description": "Parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": "Missing required parameter: request_id (query param) or X-Request-Id (header)"
                        }
                    }
                },
            },
        },
    )
    async def get_request_status(
        self,
        request: FastAPIRequest,
        request_id: Optional[str] = Query(None, description="Request ID (recommended)"),
        x_request_id: Optional[str] = Header(
            None,
            alias="X-Request-Id",
            description="Request ID (deprecated, use request_id query param instead)",
        ),
    ) -> RequestStatusResponse:
        """
        Query request status

        Pass parameters via Query Parameter (recommended):
        - request_id: Request ID

        Or via HTTP Header (deprecated):
        - X-Request-Id: Request ID

        Returns:
            RequestStatusResponse: Request status response
        """
        # Parameter validation - query param takes precedence over header
        effective_request_id = request_id or x_request_id
        if not effective_request_id:
            raise HTTPException(
                status_code=400,
                detail="Missing required parameter: request_id (query param) or X-Request-Id (header)",
            )

        try:
            # Get tenant information from request via RequestTenantProvider
            tenant_provider = self._get_request_tenant_provider()
            tenant_info = tenant_provider.get_tenant_info_from_request(request)

            # Query status
            data = await self.request_status_service.get_request_status(
                tenant_info, effective_request_id
            )

            if data is None:
                return RequestStatusResponse(
                    success=True,
                    found=False,
                    data=None,
                    message="Request status does not exist or has expired",
                )

            return RequestStatusResponse(
                success=True, found=True, data=data, message=None
            )

        except Exception as e:
            logger.error(
                "Exception when querying request status: req=%s, error=%s",
                effective_request_id,
                str(e),
                exc_info=True,
            )
            return RequestStatusResponse(
                success=False, found=False, data=None, message=f"Query failed: {str(e)}"
            )
