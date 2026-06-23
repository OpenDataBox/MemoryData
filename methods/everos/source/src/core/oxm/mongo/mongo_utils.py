"""
MongoDB ObjectId utility functions

Provides utility functions for ObjectId generation and conversion, usable without connecting to a database.
"""

from datetime import datetime
from typing import Tuple

from bson.objectid import ObjectId


def generate_object_id() -> Tuple[ObjectId, str, datetime]:
    """
    Generate a new MongoDB ObjectId (does not require database connection)

    Returns:
        Tuple[ObjectId, str, datetime]: Returns a tuple containing:
            - The ObjectId object itself
            - String representation of the ObjectId (suitable for API responses or frontend storage)
            - Timestamp when the ID was generated

    Example:
        >>> obj_id, id_str, gen_time = generate_object_id()
        >>> print(f"ObjectId object: {obj_id}")
        >>> print(f"String representation: {id_str}")
        >>> print(f"Generation time: {gen_time}")
    """
    new_id = ObjectId()
    return new_id, str(new_id), new_id.generation_time


def generate_object_id_str() -> str:
    """
    Generate a new MongoDB ObjectId and return its string representation

    Returns:
        str: String representation of the ObjectId (24-character hexadecimal string)

    Example:
        >>> id_str = generate_object_id_str()
        >>> print(f"ObjectId string: {id_str}")  # e.g.: "507f1f77bcf86cd799439011"
    """
    return str(ObjectId())
