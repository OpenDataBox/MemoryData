"""
Global exception handler

Provides a unified exception handling mechanism for FastAPI applications, ensuring all HTTP exceptions
(including exceptions raised by middleware) are properly handled and returned to the client.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from core.observation.logger import get_logger
from common_utils.datetime_utils import to_iso_format, get_now_with_timezone
from core.constants.errors import ErrorCode, ErrorStatus

logger = get_logger(__name__)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler

    Handles all exceptions uniformly, including HTTPException and other exceptions,
    ensuring they are properly formatted and returned to the client.

    Args:
        request: FastAPI request object
        exc: Exception object

    Returns:
        JSONResponse: Formatted error response
    """
    # Handle HTTP exceptions
    if isinstance(exc, HTTPException):
        logger.warning(
            "HTTP exception: %s %s - Status code: %d, Detail: %s",
            request.method,
            str(request.url),
            exc.status_code,
            exc.detail,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": ErrorStatus.FAILED.value,
                "code": ErrorCode.HTTP_ERROR.value,
                "message": exc.detail,
                "timestamp": to_iso_format(get_now_with_timezone()),
                "path": str(request.url.path),
            },
        )

    # Handle other exceptions
    logger.error(
        "Unhandled exception: %s %s - Exception type: %s, Detail: %s",
        request.method,
        str(request.url),
        type(exc).__name__,
        str(exc),
        exc_info=True,
    )

    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": ErrorStatus.FAILED.value,
            "code": ErrorCode.SYSTEM_ERROR.value,
            "message": "Internal server error",
            "timestamp": to_iso_format(get_now_with_timezone()),
            "path": str(request.url.path),
        },
    )
