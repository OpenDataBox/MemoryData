"""
This module provides utilities for marking functions, classes, and parameters
as deprecated. It includes decorators for deprecation, a function to issue
warnings, and utilities to check deprecation status.
"""

import functools
import warnings

from collections.abc import Callable
from typing import Any, TypeVar


warnings.simplefilter("default", DeprecationWarning)


F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)


def deprecated(
    reason: str | None = None,
    version: str | None = None,
    alternative: str | None = None,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 2,
) -> Callable[[F], F]:
    """
    Decorator to mark functions as deprecated.

    Args:
        reason: Optional reason for deprecation
        version: Version when the function was deprecated
        alternative: Suggested alternative function/method
        category: Warning category to use
        stacklevel: Stack level for the warning

    Example:
        @deprecated(reason="Use new_function instead", version="1.2.0")
        def old_function():
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build deprecation message
            msg_parts = [f"Function '{func.__name__}' is deprecated"]

            if version:
                msg_parts.append(f"since version {version}")

            if reason:
                msg_parts.append(f"- {reason}")

            if alternative:
                msg_parts.append(f"Use '{alternative}' instead")

            message = ". ".join(msg_parts) + "."

            warnings.warn(message, category=category, stacklevel=stacklevel)
            return func(*args, **kwargs)

        # Mark the wrapper as deprecated for introspection
        wrapper.__deprecated__ = True
        wrapper.__deprecation_info__ = {
            "reason": reason,
            "version": version,
            "alternative": alternative,
            "category": category,
        }

        return wrapper

    return decorator


def deprecated_class(
    reason: str | None = None,
    version: str | None = None,
    alternative: str | None = None,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 2,
) -> Callable[[C], C]:
    """
    Decorator to mark classes as deprecated.

    Args:
        reason: Optional reason for deprecation
        version: Version when the class was deprecated
        alternative: Suggested alternative class
        category: Warning category to use
        stacklevel: Stack level for the warning

    Example:
        @deprecated_class(reason="Use NewClass instead", version="1.2.0")
        class OldClass:
            pass
    """

    def decorator(cls: C) -> C:
        # Store original __init__
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Build deprecation message
            msg_parts = [f"Class '{cls.__name__}' is deprecated"]

            if version:
                msg_parts.append(f"since version {version}")

            if reason:
                msg_parts.append(f"- {reason}")

            if alternative:
                msg_parts.append(f"Use '{alternative}' instead")

            message = ". ".join(msg_parts) + "."

            warnings.warn(message, category=category, stacklevel=stacklevel)
            original_init(self, *args, **kwargs)

        # Replace __init__
        cls.__init__ = new_init

        # Mark the class as deprecated for introspection
        cls.__deprecated__ = True
        cls.__deprecation_info__ = {
            "reason": reason,
            "version": version,
            "alternative": alternative,
            "category": category,
        }

        return cls

    return decorator


def deprecated_parameter(
    parameter_name: str,
    reason: str | None = None,
    version: str | None = None,
    alternative: str | None = None,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 2,
) -> Callable[[F], F]:
    """
    Decorator to mark specific parameters as deprecated.

    Args:
        parameter_name: Name of the deprecated parameter
        reason: Optional reason for deprecation
        version: Version when the parameter was deprecated
        alternative: Suggested alternative parameter
        category: Warning category to use
        stacklevel: Stack level for the warning

    Example:
        @deprecated_parameter("old_param", alternative="new_param", version="1.2.0")
        def my_function(new_param=None, old_param=None):
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if deprecated parameter is used
            if parameter_name in kwargs:
                # Build deprecation message
                msg_parts = [
                    f"Parameter '{parameter_name}' in function '{func.__name__}' is deprecated"
                ]

                if version:
                    msg_parts.append(f"since version {version}")

                if reason:
                    msg_parts.append(f"- {reason}")

                if alternative:
                    msg_parts.append(f"Use parameter '{alternative}' instead")

                message = ". ".join(msg_parts) + "."

                warnings.warn(message, category=category, stacklevel=stacklevel)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def warn_deprecated(
    item_name: str,
    item_type: str = "feature",
    reason: str | None = None,
    version: str | None = None,
    alternative: str | None = None,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 2,
) -> None:
    """
    Issue a deprecation warning for any item.

    Args:
        item_name: Name of the deprecated item
        item_type: Type of item (e.g., "function", "class", "parameter", "feature")
        reason: Optional reason for deprecation
        version: Version when the item was deprecated
        alternative: Suggested alternative
        category: Warning category to use
        stacklevel: Stack level for the warning

    Example:
        warn_deprecated("old_method", "method", version="1.2.0", alternative="new_method")
    """
    # Build deprecation message
    msg_parts = [f"{item_type.capitalize()} '{item_name}' is deprecated"]

    if version:
        msg_parts.append(f"since version {version}")

    if reason:
        msg_parts.append(f"- {reason}")

    if alternative:
        msg_parts.append(f"Use '{alternative}' instead")

    message = ". ".join(msg_parts) + "."

    warnings.warn(message, category=category, stacklevel=stacklevel)


def is_deprecated(obj: Any) -> bool:
    """
    Check if an object is marked as deprecated.

    Args:
        obj: Object to check

    Returns:
        True if the object is deprecated, False otherwise
    """
    return getattr(obj, "__deprecated__", False)


def get_deprecation_info(obj: Any) -> dict | None:
    """
    Get deprecation information for an object.

    Args:
        obj: Object to get deprecation info for

    Returns:
        Dictionary with deprecation info or None if not deprecated
    """
    if is_deprecated(obj):
        return getattr(obj, "__deprecation_info__", None)
    return None
