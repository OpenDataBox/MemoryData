from datetime import datetime
from typing import List, Optional, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession
from beanie import PydanticObjectId
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.relationship import (
    Relationship,
)
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("relationship_raw_repository", primary=True)
class RelationshipRawRepository(BaseRepository[Relationship]):
    """
    Relationship repository for raw data

    Provides CRUD operations and query capabilities for entity relationship data.
    """

    def __init__(self):
        super().__init__(Relationship)

    # ==================== Basic CRUD Operations ====================

    async def get_by_entity_ids(
        self,
        source_entity_id: str,
        target_entity_id: str,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[Relationship]:
        """Get relationship by source and target entity IDs"""
        try:
            result = await self.model.find_one(
                {
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                },
                session=session,
            )
            if result:
                logger.debug(
                    "✅ Successfully retrieved relationship by entity IDs: %s -> %s",
                    source_entity_id,
                    target_entity_id,
                )
            else:
                logger.debug(
                    "⚠️  Relationship not found: source=%s, target=%s",
                    source_entity_id,
                    target_entity_id,
                )
            return result
        except Exception as e:
            logger.error("❌ Failed to retrieve relationship by entity IDs: %s", e)
            return None

    async def get_by_source_entity(
        self,
        source_entity_id: str,
        limit: int = 100,
        session: Optional[AsyncClientSession] = None,
    ) -> List[Relationship]:
        """Get all relationships by source entity ID"""
        try:
            results = (
                await self.model.find(
                    {"source_entity_id": source_entity_id}, session=session
                )
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ Successfully retrieved relationships by source entity ID: %s, count=%d",
                source_entity_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve relationships by source entity ID: %s", e
            )
            return []

    async def get_by_target_entity(
        self,
        target_entity_id: str,
        limit: int = 100,
        session: Optional[AsyncClientSession] = None,
    ) -> List[Relationship]:
        """Get all relationships by target entity ID"""
        try:
            results = (
                await self.model.find(
                    {"target_entity_id": target_entity_id}, session=session
                )
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ Successfully retrieved relationships by target entity ID: %s, count=%d",
                target_entity_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve relationships by target entity ID: %s", e
            )
            return []

    async def update_by_entity_ids(
        self,
        source_entity_id: str,
        target_entity_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[Relationship]:
        """Update relationship by entity IDs"""
        try:
            existing_doc = await self.model.find_one(
                {
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                },
                session=session,
            )
            if not existing_doc:
                logger.warning(
                    "⚠️  Relationship to update not found: source=%s, target=%s",
                    source_entity_id,
                    target_entity_id,
                )
                return None

            for key, value in update_data.items():
                setattr(existing_doc, key, value)
            await existing_doc.save(session=session)
            logger.debug(
                "✅ Successfully updated relationship by entity IDs: %s -> %s",
                source_entity_id,
                target_entity_id,
            )
            return existing_doc
        except Exception as e:
            logger.error("❌ Failed to update relationship by entity IDs: %s", e)
            return None

    async def delete_by_entity_ids(
        self,
        source_entity_id: str,
        target_entity_id: str,
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """Delete relationship by entity IDs"""
        try:
            result = await self.model.find_one(
                {
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                },
                session=session,
            )
            if not result:
                logger.warning(
                    "⚠️  Relationship to delete not found: source=%s, target=%s",
                    source_entity_id,
                    target_entity_id,
                )
                return False

            await result.delete(session=session)
            logger.info(
                "✅ Successfully deleted relationship by entity IDs: %s -> %s",
                source_entity_id,
                target_entity_id,
            )
            return True
        except Exception as e:
            logger.error("❌ Failed to delete relationship by entity IDs: %s", e)
            return False

    # ==================== Batch Operations ====================

    async def get_relationships_by_entity(
        self,
        entity_id: str,
        limit: int = 100,
        session: Optional[AsyncClientSession] = None,
    ) -> List[Relationship]:
        """Get all relationships associated with the specified entity (as source or target)"""
        try:
            results = (
                await self.model.find(
                    {
                        "$or": [
                            {"source_entity_id": entity_id},
                            {"target_entity_id": entity_id},
                        ]
                    },
                    session=session,
                )
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ Successfully retrieved relationships for entity: entity=%s, count=%d",
                entity_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve relationships for entity: %s", e)
            return []

    # ==================== Statistics Methods ====================

    async def count_by_entity(
        self, entity_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """Count the number of relationships associated with the specified entity"""
        try:
            count = await self.model.find(
                {
                    "$or": [
                        {"source_entity_id": entity_id},
                        {"target_entity_id": entity_id},
                    ]
                },
                session=session,
            ).count()
            logger.debug(
                "✅ Successfully counted entity relationships: entity=%s, count=%d",
                entity_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count entity relationships: %s", e)
            return 0

    async def count_all(self, session: Optional[AsyncClientSession] = None) -> int:
        """Count all relationships"""
        try:
            count = await self.model.count()
            logger.debug("✅ Successfully counted all relationships: count=%d", count)
            return count
        except Exception as e:
            logger.error("❌ Failed to count all relationships: %s", e)
            return 0
