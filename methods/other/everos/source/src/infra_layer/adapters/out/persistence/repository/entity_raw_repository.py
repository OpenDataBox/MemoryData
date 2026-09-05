from datetime import datetime
from typing import List, Optional, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession
from beanie import PydanticObjectId
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.entity import Entity
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("entity_raw_repository", primary=True)
class EntityRawRepository(BaseRepository[Entity]):
    """
    Entity repository for raw data

    Provides CRUD operations and query capabilities for entity data.
    """

    def __init__(self):
        super().__init__(Entity)

    # ==================== Basic CRUD Operations ====================

    async def get_by_entity_id(
        self, entity_id: str, session: Optional[AsyncClientSession] = None
    ) -> Optional[Entity]:
        """Get entity by entity ID"""
        try:
            result = await self.model.find_one(
                {"entity_id": entity_id}, session=session
            )
            if result:
                logger.debug(
                    "✅ Successfully retrieved entity by entity ID: %s", entity_id
                )
            else:
                logger.debug("⚠️  Entity not found: entity_id=%s", entity_id)
            return result
        except Exception as e:
            logger.error("❌ Failed to retrieve entity by entity ID: %s", e)
            return None

    async def get_by_alias(
        self, alias: str, session: Optional[AsyncClientSession] = None
    ) -> List[Entity]:
        """Get list of entities by alias"""
        try:
            results = await self.model.find(
                {"aliases": {"$in": [alias]}}, session=session
            ).to_list()
            logger.debug(
                "✅ Successfully retrieved entities by alias: alias=%s, count=%d",
                alias,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve entities by alias: %s", e)
            return []

    async def update_by_entity_id(
        self,
        entity_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[Entity]:
        """Update entity by entity ID"""
        try:
            existing_doc = await self.model.find_one(
                {"entity_id": entity_id}, session=session
            )
            if not existing_doc:
                logger.warning("⚠️  Entity to update not found: entity_id=%s", entity_id)
                return None

            for key, value in update_data.items():
                setattr(existing_doc, key, value)
            await existing_doc.save(session=session)
            logger.debug("✅ Successfully updated entity by entity ID: %s", entity_id)
            return existing_doc
        except Exception as e:
            logger.error("❌ Failed to update entity by entity ID: %s", e)
            return None

    async def delete_by_entity_id(
        self, entity_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """Delete entity by entity ID"""
        try:
            result = await self.model.find_one(
                {"entity_id": entity_id}, session=session
            )
            if not result:
                logger.warning("⚠️  Entity to delete not found: entity_id=%s", entity_id)
                return False

            await result.delete(session=session)
            logger.debug("✅ Successfully deleted entity by entity ID: %s", entity_id)
            return True
        except Exception as e:
            logger.error("❌ Failed to delete entity by entity ID: %s", e)
            return False

    # ==================== Batch Operations ====================

    async def get_all_entities(
        self, limit: int = 100, session: Optional[AsyncClientSession] = None
    ) -> List[Entity]:
        """Get all entities"""
        try:
            results = await self.model.find({}, session=session).limit(limit).to_list()
            logger.debug(
                "✅ Successfully retrieved all entities: count=%d", len(results)
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve all entities: %s", e)
            return []

    async def get_entities_by_ids(
        self, entity_ids: List[str], session: Optional[AsyncClientSession] = None
    ) -> List[Entity]:
        """Batch get entities by list of entity IDs"""
        try:
            results = await self.model.find(
                {"entity_id": {"$in": entity_ids}}, session=session
            ).to_list()
            logger.debug(
                "✅ Successfully batch retrieved entities by ID list: count=%d",
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to batch retrieve entities by ID list: %s", e)
            return []

    # ==================== Statistics Methods ====================

    async def count_by_type(
        self, entity_type: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """Count the number of entities of specified type"""
        try:
            count = await self.model.find(
                {"type": entity_type}, session=session
            ).count()
            logger.debug(
                "✅ Successfully counted entities: type=%s, count=%d",
                entity_type,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count entities: %s", e)
            return 0

    async def count_all(self, session: Optional[AsyncClientSession] = None) -> int:
        """Count all entities"""
        try:
            count = await self.model.count()
            logger.debug("✅ Successfully counted all entities: count=%d", count)
            return count
        except Exception as e:
            logger.error("❌ Failed to count all entities: %s", e)
            return 0
