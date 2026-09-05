"""
Tenant-aware Milvus Collection Management Class with Suffix and Alias Mechanism

This module combines the functionalities of TenantAwareCollection and MilvusCollectionWithSuffix:
1. Tenant awareness: Automatically selects the correct connection and table name based on tenant context
2. Dynamic table names: Supports dynamic suffix setting via suffix parameter or environment variables
3. Alias mechanism: Real table names include timestamps, accessed via alias
"""

from typing import Optional
from pymilvus import connections, Collection
from pymilvus.client.types import ConsistencyLevel

from core.observation.logger import get_logger
from core.oxm.milvus.milvus_collection_base import (
    MilvusCollectionWithSuffix,
    generate_new_collection_name,
)
from core.tenants.tenantize.oxm.milvus.tenant_aware_collection import (
    TenantAwareCollection,
)
from core.tenants.tenantize.oxm.milvus.config_utils import (
    get_tenant_aware_collection_name,
)
from pymilvus import utility

logger = get_logger(__name__)


class TenantAwareMilvusCollectionWithSuffix(MilvusCollectionWithSuffix):
    """
    Tenant-aware Milvus Collection Management Class with Suffix and Alias Mechanism

    Inherits from MilvusCollectionWithSuffix, adding tenant awareness capabilities:
    1. Automatically selects the correct Milvus connection based on tenant context
    2. Supports tenant-aware table names (automatically adds tenant prefix)
    3. Retains all functionalities of MilvusCollectionWithSuffix (suffix, alias, creation management, etc.)

    Key features:
    - Tenant isolation: Different tenants use different connections and table names
    - Dynamic table names: Supports suffix and environment variables
    - Alias mechanism: Real table names include timestamps, accessed via alias
    - Version management: Can create new versions and perform gradual switching
    - Tenant prefix: All operations automatically add tenant prefix (e.g., tenant_001_movies)

    Table naming rules:
    - Original base name: movies
    - With suffix: movies_production
    - With tenant prefix: tenant_001_movies_production (alias)
    - Real name: tenant_001_movies_production-20231015123456789000

    Usage:
    1. Subclass definition:
       - _COLLECTION_NAME: Base name of the Collection (required)
       - _SCHEMA: Schema definition of the Collection (required)
       - _INDEX_CONFIGS: List of index configurations (optional)
       - _DB_USING: Milvus connection alias (optional, will be overridden by tenant-aware connection)

    2. Instantiation:
       mgr = TenantAwareMovieCollection(suffix="customer_a")
       # Within tenant context:
       # - Uses tenant's Milvus connection
       # - Alias: tenant_001_movies_customer_a
       # - Real name: tenant_001_movies_customer_a-20231015123456789000

    3. Initialization:
       with tenant_context(tenant_info):
           mgr.ensure_all()  # One-click initialization

    4. Usage:
       with tenant_context(tenant_info):
           mgr.collection.insert([...])
           mgr.collection.search(...)

    Example:
        class MovieCollection(TenantAwareMilvusCollectionWithSuffix):
            _COLLECTION_NAME = "movies"
            _SCHEMA = CollectionSchema(fields=[...])
            _INDEX_CONFIGS = [
                IndexConfig(field_name="embedding", index_type="IVF_FLAT", ...),
            ]

        # Multi-tenant scenario usage
        tenant_a = TenantInfo(tenant_id="tenant_001", ...)
        tenant_b = TenantInfo(tenant_id="tenant_002", ...)

        mgr = MovieCollection(suffix="production")

        # Tenant A operations
        with tenant_context(tenant_a):
            mgr.ensure_all()
            mgr.collection.insert([...])

        # Tenant B operations
        with tenant_context(tenant_b):
            mgr.ensure_all()
            mgr.collection.insert([...])
    """

    def __init__(self, suffix: Optional[str] = None):
        """
        Initialize the tenant-aware Collection manager

        Args:
            suffix: Collection name suffix; if not provided, read from environment variable

        Note:
            - Save the original _alias_name (without tenant prefix)
            - The actual table name will dynamically add tenant prefix at runtime
        """
        super().__init__(suffix=suffix)
        # Save the original alias name (without tenant prefix)
        # Used in the name property to dynamically compute tenant-aware names
        self._original_alias_name = self._alias_name

    @property
    def name(self) -> str:
        """
        Get the tenant-aware Collection name (alias)

        Override parent class's name property to dynamically add tenant prefix.
        This ensures all places using self.name automatically get tenant-aware table names.

        Returns:
            str: Tenant-aware alias name

        Example:
            Original alias: movies_production
            Tenant A: tenant_001_movies_production
            Tenant B: tenant_002_movies_production
        """
        return TenantAwareCollection.get_tenant_aware_name(self._original_alias_name)

    @property
    def using(self) -> str:
        """
        Get the tenant-aware connection alias
        """
        return TenantAwareCollection._get_tenant_aware_using()

    def ensure_connection_registered(self) -> None:
        """
        Ensure the tenant-aware connection is registered
        """
        TenantAwareCollection._ensure_connection_registered(self.using)

    def load_collection(self) -> TenantAwareCollection:
        """
        Load or create a tenant-aware Collection

        Override parent class method, using TenantAwareCollection instead of regular Collection.
        This ensures all Collection operations are tenant-aware.

        Args:
            name: Collection name (alias name, already includes tenant prefix)

        Returns:
            TenantAwareCollection instance

        Note:
            - Use TenantAwareCollection to automatically handle tenant connections
            - Maintain MilvusCollectionWithSuffix's alias mechanism
            - If alias does not exist, create a new timestamped Collection
            - The name parameter should already be tenant-aware (passed via self.name)
        """
        using = self.using
        origin_alias_name = self._original_alias_name
        tenant_aware_alias_name = get_tenant_aware_collection_name(origin_alias_name)
        new_real_name = generate_new_collection_name(origin_alias_name)
        tenant_aware_new_real_name = get_tenant_aware_collection_name(new_real_name)

        # First check if alias exists (using tenant-aware connection)
        # Note: TenantAwareCollection automatically handles the using parameter
        self.ensure_connection_registered()

        if not utility.has_collection(tenant_aware_alias_name, using=using):
            # Collection does not exist, create a new tenant-aware Collection
            logger.info(
                "Collection '%s' does not exist, creating new tenant-aware Collection: %s",
                origin_alias_name,
                tenant_aware_new_real_name,
            )

            # Create tenant-aware Collection
            # Use native Collection, need to explicitly pass using parameter
            _coll = Collection(
                name=tenant_aware_new_real_name,
                schema=self._SCHEMA,
                consistency_level=ConsistencyLevel.Bounded,
                using=using,
            )

            # Create alias pointing to new Collection
            # Note: First delete any existing old alias
            try:
                utility.drop_alias(tenant_aware_alias_name, using=using)
            except Exception:
                pass  # alias does not exist, ignore

            utility.create_alias(
                collection_name=tenant_aware_new_real_name,
                alias=tenant_aware_alias_name,
                using=using,
            )
            logger.info(
                "Alias '%s' -> '%s' created",
                tenant_aware_alias_name,
                tenant_aware_new_real_name,
            )

        # Uniformly load tenant-aware Collection via alias
        coll = TenantAwareCollection(
            name=origin_alias_name,
            schema=self._SCHEMA,
            consistency_level=ConsistencyLevel.Bounded,
        )

        return coll

    def ensure_create(self) -> None:
        """
        Ensure Collection has been created

        Override parent class method, using tenant-aware alias name.

        This method triggers lazy loading of Collection; if alias does not exist, creates a new Collection.
        """
        if self._collection_instance is None:
            # Use tenant-aware alias name
            self._collection_instance = self.load_collection()
        logger.info("Collection '%s' is ready", self.name)

    def create_new_collection(self) -> TenantAwareCollection:
        """
        Create a new tenant-aware real Collection (without switching alias)

        Override parent class method, using TenantAwareCollection and tenant-aware names.

        Returns:
            New tenant-aware Collection instance (with indexes created and loaded)

        Note:
            - Use native Collection for creation (need to explicitly pass using parameter)
            - New Collection name includes tenant prefix and timestamp
            - Return TenantAwareCollection instance to ensure tenant isolation
            - Automatically create indexes and load into memory
        """
        if not self._SCHEMA:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define '_SCHEMA' to support collection creation"
            )

        # Use tenant-aware alias name
        using = self.using
        origin_alias_name = self._original_alias_name
        tenant_aware_alias_name = get_tenant_aware_collection_name(origin_alias_name)
        new_real_name = generate_new_collection_name(origin_alias_name)
        tenant_aware_new_real_name = get_tenant_aware_collection_name(new_real_name)

        # Create new tenant-aware collection
        # Use native Collection, need to explicitly pass using parameter
        _coll = Collection(
            name=tenant_aware_new_real_name,
            schema=self._SCHEMA,
            consistency_level=ConsistencyLevel.Bounded,
            using=using,
        )

        logger.info(
            "New tenant-aware Collection created: %s", tenant_aware_new_real_name
        )

        # Create indexes for new collection and load
        try:
            self._create_indexes_for_collection(_coll)
            _coll.load()
            logger.info(
                "Indexes created and loading completed for new Collection '%s'",
                new_real_name,
            )
        except Exception as e:
            logger.warning("Error creating indexes for new collection: %s", e)
            raise

        # Return TenantAwareCollection instance, using original alias name
        # Note: Use _original_alias_name here, TenantAwareCollection will automatically add tenant prefix
        new_coll = TenantAwareCollection(
            name=new_real_name,
            schema=self._SCHEMA,
            consistency_level=ConsistencyLevel.Bounded,
        )

        return new_coll

    def switch_alias(
        self, new_collection: TenantAwareCollection, drop_old: bool = False
    ) -> None:
        """
        Switch alias to specified new collection, optionally delete old collection

        Override parent class method, using tenant-aware alias name.

        Args:
            new_collection: New Collection instance
            drop_old: Whether to delete old collection (default False)

        Note:
            - Use tenant-aware alias name for switching
            - Prefer alter_alias, fall back to drop/create if failed
            - Refresh class-level cache after switching
        """
        # Use tenant-aware alias name
        using = self.using
        origin_alias_name = self._original_alias_name
        tenant_aware_alias_name = get_tenant_aware_collection_name(origin_alias_name)
        tenant_aware_new_real_name = new_collection.name

        # Get old collection real name (if exists)
        old_real_name: Optional[str] = None
        try:
            conn = connections._fetch_handler(using)
            desc = conn.describe_alias(tenant_aware_alias_name)
            old_real_name = (
                desc.get("collection_name") if isinstance(desc, dict) else None
            )
        except Exception:
            old_real_name = None

        # Alias switching
        try:
            conn = connections._fetch_handler(using)
            conn.alter_alias(tenant_aware_new_real_name, tenant_aware_alias_name)
            logger.info(
                "Alias '%s' switched to '%s'",
                tenant_aware_alias_name,
                tenant_aware_new_real_name,
            )
        except Exception as e:
            logger.warning("alter_alias failed, trying drop/create: %s", e)
            try:
                utility.drop_alias(tenant_aware_alias_name, using=using)
            except Exception:
                pass
            utility.create_alias(
                collection_name=tenant_aware_new_real_name,
                alias=tenant_aware_alias_name,
                using=using,
            )
            logger.info(
                "Alias '%s' -> '%s' created",
                tenant_aware_alias_name,
                tenant_aware_new_real_name,
            )

        # Optionally delete old collection (after switching completes)
        if drop_old and old_real_name:
            try:
                utility.drop_collection(old_real_name, using=using)
                logger.info("Old collection deleted: %s", old_real_name)
            except Exception as e:
                logger.warning(
                    "Failed to delete old collection (can handle manually): %s", e
                )

        # Refresh class-level cache to alias collection
        try:
            self.__class__._collection_instance = TenantAwareCollection(
                name=origin_alias_name,
                schema=self._SCHEMA,
                consistency_level=ConsistencyLevel.Bounded,
            )
        except Exception:
            pass

    def exists(self) -> bool:
        """
        Check if Collection exists (via alias)

        Override parent class method, using tenant-aware name and using.

        Returns:
            bool: Whether Collection exists
        """
        name = self.name
        using = self.using
        return utility.has_collection(name, using=using)

    def drop(self) -> None:
        """
        Delete current Collection (including alias and real Collection)

        Override parent class method, using tenant-aware name, using, and TenantAwareCollection.

        Note:
            - Use tenant-aware connection alias
            - Use TenantAwareCollection to ensure tenant isolation
            - Delete the real Collection (not the alias)
        """
        using = self.using
        name = self.name
        try:
            utility.drop_collection(name, using=using)
            logger.info("Collection '%s' deleted", name)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Collection '%s' does not exist or deletion failed: %s", name, e
            )
