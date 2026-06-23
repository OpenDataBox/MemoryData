"""
Exception handling module

This module defines all custom exception classes and error codes used in the project.
Follows a unified exception handling specification, facilitating error tracking and debugging.
"""

from enum import Enum
from typing import Optional, Dict, Any
from core.constants.errors import ErrorCode


class BaseException(Exception):
    """Base exception class

    Base class for all custom exceptions, providing a unified exception handling interface.
    Includes error code, error message, and optional details.
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        Initialize base exception

        Args:
            code: Error code
            message: Error message
            details: Optional dictionary of detailed information
            original_exception: Original exception object
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception

    def __str__(self) -> str:
        """Return string representation of the exception"""
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation of the exception"""
        details_str = f", details={self.details}" if self.details else ""
        original_str = (
            f", original={self.original_exception}" if self.original_exception else ""
        )
        return f"{self.__class__.__name__}(code='{self.code}', message='{self.message}'{details_str}{original_str})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for easy serialization"""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "exception_type": self.__class__.__name__,
        }


class AgentException(BaseException):
    """Base class for Agent-related exceptions

    Base class for all exceptions related to Agent execution.
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(code, message, details, original_exception)


class ValidationException(BaseException):
    """Data validation exception

    Raised when input data validation fails.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if field:
            message = f"Field '{field}': {message}"

        super().__init__(
            code=ErrorCode.VALIDATION_ERROR.value, message=message, details=details
        )


class ResourceNotFoundException(BaseException):
    """Resource not found exception

    Raised when the requested resource does not exist.
    """

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_FOUND.value, message=message, details=details
        )


class ConfigurationException(BaseException):
    """Configuration exception

    Raised when system configuration is incorrect or missing.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if config_key:
            message = f"Configuration error for '{config_key}': {message}"

        super().__init__(
            code=ErrorCode.CONFIGURATION_ERROR.value, message=message, details=details
        )


class DatabaseException(BaseException):
    """Database exception

    Raised when a database operation fails.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        if operation:
            message = f"Database {operation} failed: {message}"

        super().__init__(
            code=ErrorCode.DATABASE_ERROR.value,
            message=message,
            details=details,
            original_exception=original_exception,
        )


class ExternalServiceException(BaseException):
    """External service exception

    Raised when calling an external service fails.
    """

    def __init__(
        self,
        service_name: str,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        if status_code:
            message = f"{service_name} service error (HTTP {status_code}): {message}"
        else:
            message = f"{service_name} service error: {message}"

        super().__init__(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR.value,
            message=message,
            details=details,
            original_exception=original_exception,
        )


class AuthenticationException(BaseException):
    """Authentication exception

    Raised when user authentication fails.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            code=ErrorCode.AUTHENTICATION_ERROR.value,
            message=message,
            details=details,
            original_exception=original_exception,
        )


class LLMOutputParsingException(AgentException):
    """LLM output parsing exception

    Raised when the content returned by LLM cannot be parsed correctly.
    """

    def __init__(
        self,
        message: str,
        llm_output: Optional[str] = None,
        expected_format: Optional[str] = None,
        attempt_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        if expected_format:
            message = f"LLM output parsing failed, expected format: {expected_format}, error: {message}"
        if attempt_count:
            message = f"{message} [Attempt {attempt_count}]"

        # Add LLM output to details
        if details is None:
            details = {}
        if llm_output:
            details["llm_output"] = llm_output[
                :500
            ]  # Limit length to avoid being too long

        super().__init__(
            code=ErrorCode.LLM_OUTPUT_PARSING_ERROR.value,
            message=message,
            details=details,
            original_exception=original_exception,
        )


def create_exception_from_error_code(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    original_exception: Optional[Exception] = None,
) -> BaseException:
    """
    Create corresponding exception object based on error code

    Args:
        error_code: Error code enumeration
        message: Error message
        details: Optional detailed information
        original_exception: Original exception object

    Returns:
        Corresponding exception object
    """
    return BaseException(
        code=error_code.value,
        message=message,
        details=details,
        original_exception=original_exception,
    )


# Long Job System Errors - Long job system error classes
from core.longjob.longjob_error import (
    FatalError,
    BusinessLogicError,
    LongJobError,
    JobNotFoundError,
    JobAlreadyExistsError,
    JobStateError,
    ManagerShutdownError,
    MaxConcurrentJobsError,
)

# Export long job system error classes
__all__ = [
    # Error codes and base exception
    'ErrorCode',
    'BaseException',
    'AgentException',
    'ValidationException',
    'ResourceNotFoundException',
    'ConfigurationException',
    'DatabaseException',
    'ExternalServiceException',
    'AuthenticationException',
    'LLMOutputParsingException',
    'create_exception_from_error_code',
    # Long job system error classes
    'FatalError',
    'BusinessLogicError',
    'LongJobError',
    'JobNotFoundError',
    'JobAlreadyExistsError',
    'JobStateError',
    'ManagerShutdownError',
    'MaxConcurrentJobsError',
]
