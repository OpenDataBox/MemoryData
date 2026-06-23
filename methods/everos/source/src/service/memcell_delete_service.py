"""
MemCell Delete Service - Handle soft delete logic for MemCell

Provides multiple deletion methods:
- Delete by single event_id
- Batch delete by user_id
- Batch delete by group_id
"""

from typing import Optional
from core.di.decorators import component
from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)

logger = get_logger(__name__)


@component("memcell_delete_service")
class MemCellDeleteService:
    """MemCell soft delete service"""

    def __init__(self, memcell_repository: MemCellRawRepository):
        """
        Initialize deletion service

        Args:
            memcell_repository: MemCell data repository
        """
        self.memcell_repository = memcell_repository
        logger.info("MemCellDeleteService initialized")

    async def delete_by_event_id(
        self, event_id: str, deleted_by: Optional[str] = None
    ) -> bool:
        """
        Soft delete a single MemCell by event_id

        Args:
            event_id: The event_id of MemCell
            deleted_by: Identifier of the deleter (optional)

        Returns:
            bool: Returns True if deletion succeeds, False if not found or already deleted

        Example:
            >>> service = MemCellDeleteService(repo)
            >>> success = await service.delete_by_event_id("507f1f77bcf86cd799439011", "admin")
        """
        logger.info(
            "Deleting MemCell by event_id: event_id=%s, deleted_by=%s",
            event_id,
            deleted_by,
        )

        try:
            result = await self.memcell_repository.delete_by_event_id(
                event_id=event_id, deleted_by=deleted_by
            )

            if result:
                logger.info(
                    "Successfully deleted MemCell: event_id=%s, deleted_by=%s",
                    event_id,
                    deleted_by,
                )
            else:
                logger.warning(
                    "MemCell not found or already deleted: event_id=%s", event_id
                )

            return result

        except Exception as e:
            logger.error(
                "Failed to delete MemCell by event_id: event_id=%s, error=%s",
                event_id,
                e,
                exc_info=True,
            )
            raise

    async def delete_by_user_id(
        self, user_id: str, deleted_by: Optional[str] = None
    ) -> int:
        """
        Batch soft delete all MemCells of a user by user_id

        Args:
            user_id: User ID
            deleted_by: Identifier of the deleter (optional)

        Returns:
            int: Number of deleted records

        Example:
            >>> service = MemCellDeleteService(repo)
            >>> count = await service.delete_by_user_id("user_123", "admin")
            >>> print(f"Deleted {count} records")
        """
        logger.info(
            "Deleting MemCells by user_id: user_id=%s, deleted_by=%s",
            user_id,
            deleted_by,
        )

        try:
            count = await self.memcell_repository.delete_by_user_id(
                user_id=user_id, deleted_by=deleted_by
            )

            logger.info(
                "Successfully deleted MemCells by user_id: user_id=%s, deleted_by=%s, count=%d",
                user_id,
                deleted_by,
                count,
            )

            return count

        except Exception as e:
            logger.error(
                "Failed to delete MemCells by user_id: user_id=%s, error=%s",
                user_id,
                e,
                exc_info=True,
            )
            raise

    async def delete_by_group_id(
        self, group_id: str, deleted_by: Optional[str] = None
    ) -> int:
        """
        Batch soft delete all MemCells of a group by group_id

        Args:
            group_id: Group ID
            deleted_by: Identifier of the deleter (optional)

        Returns:
            int: Number of deleted records

        Example:
            >>> service = MemCellDeleteService(repo)
            >>> count = await service.delete_by_group_id("group_456", "admin")
            >>> print(f"Deleted {count} records")
        """
        logger.info(
            "Deleting MemCells by group_id: group_id=%s, deleted_by=%s",
            group_id,
            deleted_by,
        )

        try:
            # Use repository's delete_many method
            from infra_layer.adapters.out.persistence.document.memory.memcell import (
                MemCell,
            )

            result = await MemCell.delete_many(
                {"group_id": group_id}, deleted_by=deleted_by
            )

            count = result.modified_count if result else 0

            logger.info(
                "Successfully deleted MemCells by group_id: group_id=%s, deleted_by=%s, count=%d",
                group_id,
                deleted_by,
                count,
            )

            return count

        except Exception as e:
            logger.error(
                "Failed to delete MemCells by group_id: group_id=%s, error=%s",
                group_id,
                e,
                exc_info=True,
            )
            raise

    async def delete_by_combined_criteria(
        self,
        event_id: Optional[str] = None,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> dict:
        """
        Delete MemCell based on combined criteria (multiple conditions must all be satisfied)

        Args:
            event_id: The event_id of MemCell (one of the combined conditions)
            user_id: User ID (one of the combined conditions)
            group_id: Group ID (one of the combined conditions)

        Returns:
            dict: Dictionary containing deletion results
                - filters: List of filter conditions used
                - count: Number of deleted records
                - success: Whether the operation succeeded

        Example:
            >>> service = MemCellDeleteService(repo)
            >>> # Delete records of a specific user in a specific group
            >>> result = await service.delete_by_combined_criteria(
            ...     user_id="user_123",
            ...     group_id="group_456",
            ... )
            >>> print(result)
            {'filters': ['user_id', 'group_id'], 'count': 5, 'success': True}
        """
        from core.oxm.constants import MAGIC_ALL
        from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell

        # Build filter conditions
        filter_dict = {}
        filters_used = []

        if event_id and event_id != MAGIC_ALL:
            from bson import ObjectId

            try:
                filter_dict["_id"] = ObjectId(event_id)
                filters_used.append("event_id")
            except Exception as e:
                logger.error("Invalid event_id format: %s, error: %s", event_id, e)
                return {
                    "filters": [],
                    "count": 0,
                    "success": False,
                    "error": f"Invalid event_id format: {event_id}",
                }

        if user_id and user_id != MAGIC_ALL:
            filter_dict["user_id"] = user_id
            filters_used.append("user_id")

        if group_id and group_id != MAGIC_ALL:
            filter_dict["group_id"] = group_id
            filters_used.append("group_id")

        # If no filter conditions are provided
        if not filter_dict:
            logger.warning("No deletion criteria provided (all are MAGIC_ALL)")
            return {
                "filters": [],
                "count": 0,
                "success": False,
                "error": "No deletion criteria provided",
            }

        logger.info(
            "Deleting MemCells with combined criteria: filters=%s", filters_used
        )

        try:
            # Use delete_many to batch soft delete
            result = await MemCell.delete_many(filter_dict)
            count = result.modified_count if result else 0

            logger.info(
                "Successfully deleted MemCells: filters=%s, count=%d",
                filters_used,
                count,
            )

            return {"filters": filters_used, "count": count, "success": count > 0}

        except Exception as e:
            logger.error(
                "Failed to delete MemCells with combined criteria: filters=%s, error=%s",
                filters_used,
                e,
                exc_info=True,
            )
            raise
