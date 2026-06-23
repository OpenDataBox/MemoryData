"""Foresight and event log synchronization service

Responsible for writing unified foresight and event logs into Milvus / Elasticsearch.
"""

from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord,
)
from infra_layer.adapters.out.search.elasticsearch.converter.foresight_converter import (
    ForesightConverter,
)
from infra_layer.adapters.out.search.milvus.converter.foresight_milvus_converter import (
    ForesightMilvusConverter,
)
from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
    EventLogRecord,
)
from infra_layer.adapters.out.search.elasticsearch.converter.event_log_converter import (
    EventLogConverter,
)
from infra_layer.adapters.out.search.milvus.converter.event_log_milvus_converter import (
    EventLogMilvusConverter,
)
from infra_layer.adapters.out.search.repository.foresight_milvus_repository import (
    ForesightMilvusRepository,
)
from infra_layer.adapters.out.search.repository.event_log_milvus_repository import (
    EventLogMilvusRepository,
)
from infra_layer.adapters.out.search.repository.foresight_es_repository import (
    ForesightEsRepository,
)
from infra_layer.adapters.out.search.repository.event_log_es_repository import (
    EventLogEsRepository,
)
from core.di import get_bean_by_type, service
from common_utils.datetime_utils import get_now_with_timezone

logger = logging.getLogger(__name__)


@service(name="memory_sync_service", primary=True)
class MemorySyncService:
    """Foresight and event log synchronization service"""

    def __init__(
        self,
        foresight_milvus_repo: Optional[ForesightMilvusRepository] = None,
        eventlog_milvus_repo: Optional[EventLogMilvusRepository] = None,
        foresight_es_repo: Optional[ForesightEsRepository] = None,
        eventlog_es_repo: Optional[EventLogEsRepository] = None,
    ):
        """Initialize synchronization service

        Args:
            foresight_milvus_repo: Foresight Milvus repository instance (optional, obtained from DI if not provided)
            eventlog_milvus_repo: Event log Milvus repository instance (optional, obtained from DI if not provided)
            foresight_es_repo: Foresight ES repository instance (optional, obtained from DI if not provided)
            eventlog_es_repo: Event log ES repository instance (optional, obtained from DI if not provided)
        """
        self.foresight_milvus_repo = foresight_milvus_repo or get_bean_by_type(
            ForesightMilvusRepository
        )
        self.eventlog_milvus_repo = eventlog_milvus_repo or get_bean_by_type(
            EventLogMilvusRepository
        )
        self.foresight_es_repo = foresight_es_repo or get_bean_by_type(
            ForesightEsRepository
        )
        self.eventlog_es_repo = eventlog_es_repo or get_bean_by_type(
            EventLogEsRepository
        )

        logger.info("MemorySyncService initialization completed")

    @staticmethod
    def _normalize_datetime(value: Optional[datetime | str]) -> Optional[datetime]:
        """Convert str/None to datetime (supports date-only strings)"""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d")
                except ValueError:
                    logger.warning("Unable to parse date string: %s", value)
                    return None
        return None

    async def sync_foresight(
        self,
        foresight: ForesightRecord,
        sync_to_es: bool = True,
        sync_to_milvus: bool = True,
    ) -> Dict[str, int]:
        """Synchronize a single foresight to Milvus/ES

        Args:
            foresight: ForesightRecord document object
            sync_to_es: Whether to sync to ES (default True)
            sync_to_milvus: Whether to sync to Milvus (default True)

        Returns:
            Synchronization statistics {"foresight": 1}
        """
        stats = {"foresight": 0, "es_records": 0}

        try:
            # Read embedding from MongoDB, skip if not exists
            if not foresight.vector:
                logger.warning(
                    f"Foresight {foresight.id} has no embedding, skipping sync"
                )
                return stats

            # Sync to Milvus
            if sync_to_milvus:
                # Use converter to generate Milvus entity
                milvus_entity = ForesightMilvusConverter.from_mongo(foresight)
                await self.foresight_milvus_repo.insert(milvus_entity, flush=False)
                stats["foresight"] += 1
                logger.debug(f"Foresight synced to Milvus: {foresight.id}")

            # Sync to ES
            if sync_to_es:
                # Use converter to generate correct ES document (including jieba tokenized search_content)
                es_doc = ForesightConverter.from_mongo(foresight)
                await self.foresight_es_repo.create(es_doc)
                stats["es_records"] += 1
                logger.debug(f"Foresight synced to ES: {foresight.id}")

        except Exception as e:
            logger.error(f"Failed to sync foresight: {e}", exc_info=True)
            raise

        return stats

    async def sync_event_log(
        self,
        event_log: EventLogRecord,
        sync_to_es: bool = True,
        sync_to_milvus: bool = True,
    ) -> Dict[str, int]:
        """Synchronize a single event log to Milvus/ES

        Args:
            event_log: EventLogRecord document object
            sync_to_es: Whether to sync to ES (default True)
            sync_to_milvus: Whether to sync to Milvus (default True)

        Returns:
            Synchronization statistics {"event_log": 1}
        """
        stats = {"event_log": 0, "es_records": 0}

        try:
            # Read existing vector from MongoDB
            if not event_log.vector:
                logger.warning(
                    f"Event log {event_log.id} has no embedding, skipping sync"
                )
                return stats

            # Sync to Milvus
            if sync_to_milvus:
                # Use converter to generate Milvus entity
                milvus_entity = EventLogMilvusConverter.from_mongo(event_log)
                await self.eventlog_milvus_repo.insert(milvus_entity, flush=False)
                stats["event_log"] += 1
                logger.debug(f"Event log synced to Milvus: {event_log.id}")

            # Sync to ES
            if sync_to_es:
                # Use converter to generate correct ES document (including jieba tokenized search_content)
                es_doc = EventLogConverter.from_mongo(event_log)
                await self.eventlog_es_repo.create(es_doc)
                stats["es_records"] += 1
                logger.debug(f"Event log synced to ES: {event_log.id}")

        except Exception as e:
            logger.error(f"Failed to sync event log: {e}", exc_info=True)
            raise

        return stats

    async def sync_batch_foresights(
        self,
        foresights: List[ForesightRecord],
        sync_to_es: bool = True,
        sync_to_milvus: bool = True,
    ) -> Dict[str, int]:
        """Batch synchronize foresights

        Args:
            foresights: List of ForesightRecord
            sync_to_es: Whether to sync to ES (default True)
            sync_to_milvus: Whether to sync to Milvus (default True)

        Returns:
            Synchronization statistics
        """
        total_stats = {"foresight": 0, "es_records": 0}

        for foresight_mem in foresights:
            try:
                stats = await self.sync_foresight(
                    foresight_mem, sync_to_es=sync_to_es, sync_to_milvus=sync_to_milvus
                )
                total_stats["foresight"] += stats.get("foresight", 0)
                total_stats["es_records"] += stats.get("es_records", 0)
            except Exception as e:
                logger.error(
                    f"Failed to batch sync foresight: {foresight_mem.id}, error: {e}",
                    exc_info=True,
                )
                # Do not silently swallow exceptions

        logger.info(
            f"✅ Foresight Milvus flush completed: {total_stats['foresight']} records"
        )

        return total_stats

    async def sync_batch_event_logs(
        self,
        event_logs: List[EventLogRecord],
        sync_to_es: bool = True,
        sync_to_milvus: bool = True,
    ) -> Dict[str, int]:
        """Batch synchronize event logs

        Args:
            event_logs: List of EventLogRecord
            sync_to_es: Whether to sync to ES (default True)
            sync_to_milvus: Whether to sync to Milvus (default True)

        Returns:
            Synchronization statistics
        """
        total_stats = {"event_log": 0, "es_records": 0}

        for evt_log in event_logs:
            try:
                stats = await self.sync_event_log(
                    evt_log, sync_to_es=sync_to_es, sync_to_milvus=sync_to_milvus
                )
                total_stats["event_log"] += stats.get("event_log", 0)
                total_stats["es_records"] += stats.get("es_records", 0)
            except Exception as e:
                logger.error(
                    f"Failed to batch sync event log: {evt_log.id}, error: {e}",
                    exc_info=True,
                )
                # Do not silently swallow exceptions, let it surface
                raise

        logger.info(
            f"✅ Event log Milvus flush completed: {total_stats['event_log']} records"
        )

        return total_stats
