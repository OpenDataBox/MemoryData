from typing import List, Optional
from pymongo.asynchronous.client_session import AsyncClientSession
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.behavior_history import (
    BehaviorHistory,
)
from core.observation.logger import get_logger
from core.di.decorators import repository
from common_utils.datetime_utils import get_now_with_timezone

logger = get_logger(__name__)


@repository("behavior_history_raw_repository", primary=True)
class BehaviorHistoryRawRepository(BaseRepository[BehaviorHistory]):
    """
    Behavior history raw data repository

    Provides CRUD operations and query capabilities for user behavior history data.
    """

    def __init__(self):
        super().__init__(BehaviorHistory)

    # ==================== Basic CRUD Operations ====================

    async def get_by_user_id(
        self,
        user_id: str,
        limit: int = 100,
        session: Optional[AsyncClientSession] = None,
    ) -> List[BehaviorHistory]:
        """Get behavior history list by user ID"""
        try:
            results = (
                await self.model.find({"user_id": user_id}, session=session)
                .sort("-timestamp")
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ Successfully retrieved behavior history by user ID: %s, count=%d",
                user_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve behavior history by user ID: %s", e)
            return []

    async def get_by_time_range(
        self,
        start_timestamp: int,
        end_timestamp: int,
        user_id: Optional[str] = None,
        limit: int = 100,
        session: Optional[AsyncClientSession] = None,
    ) -> List[BehaviorHistory]:
        """Get behavior history list by time range"""
        try:
            query = {"timestamp": {"$gte": start_timestamp, "$lte": end_timestamp}}
            if user_id:
                query["user_id"] = user_id

            results = (
                await self.model.find(query, session=session)
                .sort("-timestamp")
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ Successfully retrieved behavior history by time range: start=%d, end=%d, count=%d",
                start_timestamp,
                end_timestamp,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve behavior history by time range: %s", e)
            return []

    async def append_behavior(
        self,
        behavior_history: BehaviorHistory,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[BehaviorHistory]:
        """Append new behavior history"""
        try:
            await behavior_history.insert(session=session)
            logger.debug(
                "✅ Successfully appended behavior history: user_id=%s, timestamp=%s",
                behavior_history.user_id,
                behavior_history.timestamp,
            )
            return behavior_history
        except Exception as e:
            logger.error("❌ Failed to append behavior history: %s", e)
            return None

    async def delete_by_user_and_timestamp(
        self, user_id: str, timestamp: int, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """Delete behavior history by user ID and timestamp"""
        try:
            result = await self.model.find_one(
                {"user_id": user_id, "timestamp": timestamp}, session=session
            )
            if not result:
                logger.warning(
                    "⚠️  Behavior history not found for deletion: user_id=%s, timestamp=%d",
                    user_id,
                    timestamp,
                )
                return False

            await result.delete(session=session)
            logger.info(
                "✅ Successfully deleted behavior history by user ID and timestamp: %s, %d",
                user_id,
                timestamp,
            )
            return True
        except Exception as e:
            logger.error(
                "❌ Failed to delete behavior history by user ID and timestamp: %s", e
            )
            return False

    # ==================== Batch Operations ====================

    async def get_recent_behaviors(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 100,
        session: Optional[AsyncClientSession] = None,
    ) -> List[BehaviorHistory]:
        """Get user's behavior history from the recent N hours"""
        try:
            current_timestamp = int(get_now_with_timezone().timestamp())
            start_timestamp = current_timestamp - (hours * 3600)

            results = (
                await self.model.find(
                    {"user_id": user_id, "timestamp": {"$gte": start_timestamp}},
                    session=session,
                )
                .sort("-timestamp")
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ Successfully retrieved recent behavior history: user_id=%s, hours=%d, count=%d",
                user_id,
                hours,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve recent behavior history: %s", e)
            return []

    # ==================== Statistics Methods ====================

    async def count_by_user(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """Count the number of behavior history records for a user"""
        try:
            count = await self.model.find({"user_id": user_id}, session=session).count()
            logger.debug(
                "✅ Successfully counted behavior history for user: user_id=%s, count=%d",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count behavior history for user: %s", e)
            return 0

    async def count_by_time_range(
        self,
        start_timestamp: int,
        end_timestamp: int,
        user_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """Count the number of behavior history records within a time range"""
        try:
            query = {"timestamp": {"$gte": start_timestamp, "$lte": end_timestamp}}
            if user_id:
                query["user_id"] = user_id

            count = await self.model.find(query, session=session).count()
            logger.debug(
                "✅ Successfully counted behavior history within time range: start=%d, end=%d, count=%d",
                start_timestamp,
                end_timestamp,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count behavior history within time range: %s", e)
            return 0
