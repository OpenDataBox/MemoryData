"""
Group chat format mapping module

Converts open-source group chat format (GroupChatFormat) to the format required by the memorize interface

⚠️ Important note:
This is the only adaptation layer from GroupChatFormat to internal format. All related conversion logic must be centralized in this file.
It is forbidden to add format conversion logic in other modules (such as controller, service, etc.) to maintain the single responsibility of the adaptation layer.

Main functions:
1. Format validation: validate_group_chat_format_input() - Validates whether input data conforms to GroupChatFormat specifications
2. Format conversion: convert_group_chat_format_to_memorize_input() - Converts GroupChatFormat to internal format
3. Simple message conversion: convert_simple_message_to_memorize_input() - Converts V1 simple message format
4. Field extraction: extract_message_core_fields() - Extracts core message fields from request body (for logging/storage)
5. Utility: normalize_refer_list() - Normalizes refer_list format to string list

Usage example:
    from infra_layer.adapters.input.api.mapper.group_chat_converter import (
        validate_group_chat_format_input,
        convert_group_chat_format_to_memorize_input,
        extract_message_core_fields,
    )

    # Validate format
    if validate_group_chat_format_input(data):
        # Convert to internal format
        memorize_input = convert_group_chat_format_to_memorize_input(data)
        # Use memorize_input to call memory storage service

    # Extract core fields for logging
    core_fields = extract_message_core_fields(request_body)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from common_utils.datetime_utils import from_iso_format


def convert_group_chat_format_to_memorize_input(
    group_chat_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert GroupChatFormat data to memorize interface input format

    Args:
        group_chat_data: Dictionary containing GroupChatFormat data, including:
            - version: Format version number
            - conversation_meta: Conversation metadata (name, description, group_id, user_details, etc.)
            - conversation_list: List of messages

    Returns:
        Dict[str, Any]: Input format suitable for memorize interface, containing:
            - messages: Converted message list
            - group_id: Group ID (optional)
            - raw_data_type: Data type (default is "Conversation")
            - current_time: Current time (using the time of the last message)

    Raises:
        ValueError: When required fields are missing
    """
    # Validate required fields
    if "conversation_meta" not in group_chat_data:
        raise ValueError("Missing required field: conversation_meta")
    if "conversation_list" not in group_chat_data:
        raise ValueError("Missing required field: conversation_list")

    conversation_meta = group_chat_data["conversation_meta"]
    conversation_list = group_chat_data["conversation_list"]

    if not conversation_list:
        raise ValueError("conversation_list cannot be empty")

    # Extract group information
    group_id = conversation_meta.get("group_id")
    group_name = conversation_meta.get(
        "name"
    )  # Extract group name from conversation_meta
    user_details = conversation_meta.get("user_details", {})
    default_timezone = conversation_meta.get("default_timezone", "UTC")

    # Extract list of all user IDs
    user_id_list = list(user_details.keys())

    # Convert message list
    messages = []
    for msg in conversation_list:
        converted_msg = _convert_message_to_internal_format(
            msg,
            group_id=group_id,
            user_id_list=user_id_list,
            user_details=user_details,
            default_timezone=default_timezone,
        )
        messages.append(converted_msg)

    # Get the time of the last message as current_time
    current_time = None
    if messages:
        last_msg = conversation_list[-1]
        create_time_str = last_msg.get("create_time")
        if create_time_str:
            current_time = _parse_datetime_with_timezone(
                create_time_str, default_timezone
            )

    # Build memorize interface input format
    result = {"messages": messages, "raw_data_type": "Conversation"}

    # Add optional fields
    if group_id:
        result["group_id"] = group_id
    if group_name:
        result["group_name"] = group_name
    if current_time:
        result["current_time"] = current_time.isoformat()

    return result


def _convert_message_to_internal_format(
    message: Dict[str, Any],
    group_id: Optional[str] = None,
    user_id_list: Optional[List[str]] = None,
    user_details: Optional[Dict[str, Any]] = None,
    default_timezone: str = "UTC",
) -> Dict[str, Any]:
    """
    Convert a single message from GroupChatFormat to internal format

    Args:
        message: Single message in GroupChatFormat, containing:
            - message_id: Message ID
            - create_time: Creation time (ISO 8601 format)
            - sender: Sender user ID
            - sender_name: Sender name (optional)
            - type: Message type
            - content: Message content
            - refer_list: Reference message list (optional)
        group_id: Group ID
        user_id_list: List of user IDs
        user_details: Dictionary of user details
        default_timezone: Default timezone

    Returns:
        Dict[str, Any]: Converted message format, containing _id, fullName, receiverId, roomId,
                       userIdList, referList, content, createTime, createBy, updateTime, orgId, etc.
    """
    # Extract basic fields
    message_id = message.get("message_id")
    sender_id = message.get("sender")
    sender_name = message.get("sender_name")
    create_time = message.get("create_time")
    content = message.get("content", "")
    refer_list = message.get("refer_list", [])

    # Message type: Currently only supports text messages, fixed as 1 (TEXT)
    msg_type = 1

    # If sender_name is empty, try to get it from user_details
    if not sender_name and sender_id and user_details:
        user_detail = user_details.get(sender_id, {})
        sender_name = user_detail.get("full_name", sender_id)

    # Convert refer_list format using the public normalize function
    converted_refer_list = normalize_refer_list(refer_list) if refer_list else []

    # Parse time (use default timezone if no timezone info)
    parsed_create_time = _parse_datetime_with_timezone(create_time, default_timezone)
    create_time_iso = (
        parsed_create_time.isoformat() if parsed_create_time else create_time
    )

    # Build internal format
    # Note: This format needs to match the expected input of convert_single_message_to_raw_data
    internal_format = {
        "_id": message_id,
        "fullName": sender_name or sender_id,
        "receiverId": None,  # Group chat messages have no individual receiver
        "roomId": group_id,
        "userIdList": user_id_list or [],
        "referList": converted_refer_list,
        "content": content,
        "createTime": create_time_iso,
        "createBy": sender_id,
        "updateTime": create_time_iso,  # Use createTime as updateTime
        "orgId": None,  # No orgId in GroupChatFormat, set to None
        "msgType": msg_type,
    }

    # Add extra information to extra field (if exists)
    extra = message.get("extra")
    if extra:
        internal_format["extra"] = extra

    return internal_format


def _parse_datetime_with_timezone(
    datetime_str: Optional[str], default_timezone: str = "UTC"
) -> Optional[datetime]:
    """
    Parse datetime string with timezone

    Args:
        datetime_str: Datetime string in ISO 8601 format
        default_timezone: Default timezone to use if no timezone info in string

    Returns:
        datetime object, or None if parsing fails
    """
    if not datetime_str:
        return None

    try:
        # Try to parse using from_iso_format
        # If no timezone info, use provided default timezone
        tz = ZoneInfo(default_timezone)
        return from_iso_format(datetime_str, tz)
    except (ValueError, TypeError, KeyError) as e:
        # Parsing failed, return None
        print(f"Failed to parse datetime: {datetime_str}, error: {e}")
        return None


def convert_simple_message_to_memorize_input(
    message_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert simple direct single message format to memorize interface input format

    This is the simple format used by V1 memorize interface, without complex GroupChatFormat structure.

    Args:
        message_data: Simple single message data, containing:
            - group_id (optional): Group ID
            - group_name (optional): Group name
            - message_id (required): Message ID
            - create_time (required): Creation time
            - sender (required): Sender user ID
            - sender_name (optional): Sender name
            - content (required): Message content
            - refer_list (optional): List of referenced message IDs

    Returns:
        Dict[str, Any]: Input format suitable for memorize interface

    Raises:
        ValueError: When required fields are missing
    """
    # Extract fields
    group_id = message_data.get("group_id")
    group_name = message_data.get("group_name")
    message_id = message_data.get("message_id")
    create_time = message_data.get("create_time")
    sender = message_data.get("sender")
    sender_name = message_data.get("sender_name", sender)
    content = message_data.get("content", "")
    refer_list = message_data.get("refer_list", [])

    # Validate required fields
    if not message_id:
        raise ValueError("Missing required field: message_id")
    if not create_time:
        raise ValueError("Missing required field: create_time")
    if not sender:
        raise ValueError("Missing required field: sender")
    if not content:
        raise ValueError("Missing required field: content")

    # Build internal format
    # Note: V1 simple format refer_list is typically already string list,
    # but we still normalize it for consistency
    internal_message = {
        "_id": message_id,
        "fullName": sender_name,
        "receiverId": None,
        "roomId": group_id,
        "userIdList": [],
        "referList": normalize_refer_list(refer_list) if refer_list else [],
        "content": content,
        "createTime": create_time,
        "createBy": sender,
        "updateTime": create_time,
        "orgId": None,
        "msgType": 1,  # TEXT
    }

    # Build memorize interface input format
    result = {
        "messages": [internal_message],
        "raw_data_type": "Conversation",
        "split_ratio": 0,  # All as new messages
    }

    # Add optional fields
    if group_id:
        result["group_id"] = group_id
    if group_name:
        result["group_name"] = group_name
    if create_time:
        result["current_time"] = create_time

    return result


def normalize_refer_list(refer_list: List[Any]) -> List[str]:
    """
    Normalize refer_list format to a list of message IDs

    GroupChatFormat supports two formats:
    1. String list: ["msg_id_1", "msg_id_2"]
    2. MessageReference object list: [{"message_id": "msg_id_1", ...}, ...]

    Args:
        refer_list: Original reference list

    Returns:
        List[str]: Normalized list of message IDs
    """
    normalized: List[str] = []
    for refer in refer_list:
        if isinstance(refer, str):
            normalized.append(refer)
        elif isinstance(refer, dict):
            ref_msg_id = refer.get("message_id")
            if ref_msg_id:
                normalized.append(str(ref_msg_id))
    return normalized


def extract_message_core_fields(body_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract core message fields from memorize request body

    This function extracts the essential message fields from different request formats,
    providing a unified interface for downstream consumers (e.g., MemoryRequestLog storage).

    Supports two formats:
    1. V1 simple message format: message_id, create_time, sender, sender_name, content, refer_list
    2. GroupChatFormat: extracts from conversation_list (first message)

    Args:
        body_data: Parsed request body dictionary

    Returns:
        Dict[str, Any]: Dictionary containing core message fields:
            - message_id: Message ID
            - message_create_time: Message creation time (ISO format string)
            - sender: Sender user ID
            - sender_name: Sender display name
            - content: Message content
            - group_name: Group name
            - refer_list: Normalized list of referenced message IDs
    """
    result: Dict[str, Any] = {
        "message_id": None,
        "message_create_time": None,
        "sender": None,
        "sender_name": None,
        "content": None,
        "group_name": None,
        "refer_list": None,
    }

    if not body_data:
        return result

    # Try to extract group_name from top level
    result["group_name"] = body_data.get("group_name")

    # Check if it's GroupChatFormat (has conversation_list)
    conversation_list = body_data.get("conversation_list")
    if conversation_list and isinstance(conversation_list, list) and len(conversation_list) > 0:
        # GroupChatFormat: extract from the first message in conversation_list
        # Note: If all messages need to be saved, caller should iterate and call this for each
        first_msg = conversation_list[0]
        result["message_id"] = first_msg.get("message_id")
        result["message_create_time"] = first_msg.get("create_time")
        result["sender"] = first_msg.get("sender")
        result["sender_name"] = first_msg.get("sender_name")
        result["content"] = first_msg.get("content")

        # Extract and normalize refer_list
        refer_list = first_msg.get("refer_list", [])
        if refer_list:
            result["refer_list"] = normalize_refer_list(refer_list)

        # Try to get group_name from conversation_meta if not set
        conversation_meta = body_data.get("conversation_meta", {})
        if not result["group_name"]:
            result["group_name"] = conversation_meta.get("name")

        # If sender_name is missing, try to get from user_details
        if not result["sender_name"] and result["sender"]:
            user_details = conversation_meta.get("user_details", {})
            user_detail = user_details.get(result["sender"], {})
            result["sender_name"] = user_detail.get("full_name")

    else:
        # V1 simple message format: extract from top level
        result["message_id"] = body_data.get("message_id")
        result["message_create_time"] = body_data.get("create_time")
        result["sender"] = body_data.get("sender")
        result["sender_name"] = body_data.get("sender_name") or body_data.get("sender")
        result["content"] = body_data.get("content")

        # Extract and normalize refer_list
        refer_list = body_data.get("refer_list", [])
        if refer_list:
            result["refer_list"] = normalize_refer_list(refer_list)

    return result


def validate_group_chat_format_input(data: Dict[str, Any]) -> bool:
    """
    Validate whether input data conforms to GroupChatFormat specification

    Args:
        data: Input data dictionary

    Returns:
        bool: Whether the data conforms to the specification
    """
    # Check required top-level fields
    if "conversation_meta" not in data or "conversation_list" not in data:
        return False

    meta = data["conversation_meta"]
    if "name" not in meta or "user_details" not in meta:
        return False

    # Check message list
    conversation_list = data["conversation_list"]
    if not isinstance(conversation_list, list):
        return False

    user_ids = set(meta["user_details"].keys())

    # Validate each message
    for msg in conversation_list:
        # Check required fields
        required_fields = ["message_id", "create_time", "sender", "type", "content"]
        for field in required_fields:
            if field not in msg:
                return False

        # Check if sender is in user_details
        if msg.get("sender") not in user_ids:
            return False

        # Check refer_list format (if exists)
        refer_list = msg.get("refer_list", [])
        if refer_list:
            for refer in refer_list:
                if isinstance(refer, dict):
                    # MessageReference object must have message_id
                    if "message_id" not in refer:
                        return False
                elif not isinstance(refer, str):
                    # Must be string or dictionary
                    return False

    return True
