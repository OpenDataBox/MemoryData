from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import uuid
from fastapi import Request

from core.di.decorators import component
from core.observation.logger import get_logger
from core.context.context import get_current_app_info, get_current_request

logger = get_logger(__name__)

# Default blocking wait timeout (seconds)
DEFAULT_BLOCKING_TIMEOUT = 5.0


class AppLogicProvider(ABC):
    """
    Application logic provider interface

    Responsible for extracting application-level context information from requests and handling application logic.
    Provides hooks for the request lifecycle:
    - should_process_request(): Determines whether the request needs processing (used for filtering)
    - setup_app_context(): Extracts and sets application context (called first by middleware)
    - on_request_begin(): Called when the request begins (business logic, e.g., event dispatching)
    - on_request_complete(): Called when the request ends (optional implementation)

    Helper methods (retrieve from context):
    - get_current_request_id(): Get the current request's request_id
    - get_current_request(): Get the current request object
    - get_current_app_info(): Get current application information
    """

    def should_process_request(self, request: Request) -> bool:
        """
        Determine whether the request needs business logic processing

        Used to filter requests and decide whether to execute:
        - on_request_begin() callback
        - on_request_complete() callback

        Note: setup_app_context() is not affected by this method and is called on every request.

        Subclasses can override this method to implement custom filtering logic,
        for example, only processing requests under /api/ routes.

        Args:
            request: FastAPI request object

        Returns:
            bool: True means process, False means skip
        """
        # Default: process all requests
        return True

    @abstractmethod
    def setup_app_context(self, request: Request) -> Dict[str, Any]:
        """
        Extract and set application context

        Extract all context-related data from the request, for example:
        - Record request start time
        - Extract request_id, hash_key
        - Set tenant context

        This method is called first by middleware, before on_request_begin.

        Args:
            request: FastAPI request object

        Returns:
            Dict[str, Any]: app_info dictionary containing context data
        """
        raise NotImplementedError

    async def validate_request(self, request: Request) -> None:
        """
        Validate request (optional implementation)

        This method is called after setup_app_context() and before on_request_begin().
        If validation fails, raise HTTPException to abort the request.

        Use cases:
        - Quota validation
        - Permission checks
        - Rate limiting
        - Custom business validations

        Args:
            request: FastAPI request object

        Raises:
            HTTPException: When validation fails (e.g., 429 for quota exceeded)

        Example:
            >>> async def validate_request(self, request: Request) -> None:
            ...     if not self._check_quota(request):
            ...         raise HTTPException(
            ...             status_code=429,
            ...             detail="Quota exceeded"
            ...         )
        """
        # Default implementation is empty; subclasses may optionally override
        _ = request  # Avoid unused parameter warning

    async def on_request_begin(self, request: Request) -> None:
        """
        Callback method when request begins

        Used to handle business logic at the start of a request, for example:
        - Dispatching request start event

        Note: Context data has already been set by setup_app_context(),
        and can be retrieved via self.get_current_app_info().

        Args:
            request: FastAPI request object
        """
        # Default implementation is empty; subclasses may optionally override
        _ = request  # Avoid unused parameter warning

    async def on_request_complete(
        self, request: Request, http_code: int, error_message: Optional[str] = None
    ) -> None:
        """
        Callback method when request completes (optional implementation)

        Subclasses can override this method to handle post-request logic,
        for example: logging request details, dispatching events, etc.

        Note: Current app_info set in setup_app_context() or on_request_begin() can be retrieved via self.get_current_app_info().

        Args:
            request: FastAPI request object
            http_code: HTTP response status code
            error_message: Error message (optional)
        """
        # Default implementation is empty; subclasses may optionally override
        _ = (request, http_code, error_message)  # Avoid unused parameter warning

    def get_current_request_id(self) -> str:
        """
        Get the current request's request_id

        Retrieve app_info from context, then extract request_id.

        Returns:
            str: Current request's request_id, returns "unknown" if not set
        """
        app_info = get_current_app_info()
        if app_info:
            return app_info.get("request_id", "unknown")
        return "unknown"

    def get_current_request(self) -> Optional[Request]:
        """
        Get the current request object

        Retrieve request from context.

        Returns:
            Optional[Request]: Current request object, returns None if not set
        """
        return get_current_request()

    def get_current_app_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current application information

        Retrieve app_info from context.

        Returns:
            Optional[Dict[str, Any]]: Current application information, returns None if not set
        """
        return get_current_app_info()


@component(name="app_logic_provider")
class AppLogicProviderImpl(AppLogicProvider):
    """Application logic provider implementation, responsible for extracting application-level context information from requests"""

    def setup_app_context(self, request: Request) -> Dict[str, Any]:
        """
        Extract and set application context

        Args:
            request: FastAPI request object

        Returns:
            Dict[str, Any]: app_info dictionary containing context data
        """
        app_info: Dict[str, Any] = {}

        # Get request_id from request headers, prioritize X-Request-Id, fallback to lowercase
        request_id = request.headers.get('X-Request-Id') or request.headers.get(
            'x-request-id'
        )
        if not request_id:
            request_id = str(uuid.uuid4())

        app_info['request_id'] = request_id

        return app_info
