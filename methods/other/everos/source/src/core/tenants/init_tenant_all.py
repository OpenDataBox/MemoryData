"""
Tenant database initialization module

This module is used to initialize MongoDB, Milvus, and Elasticsearch databases for a specific tenant.
The tenant ID is specified via the environment variable TENANT_SINGLE_TENANT_ID:
1. Create tenant information and set tenant context
2. Invoke MongoDB lifespan startup logic
3. Invoke Milvus lifespan startup logic
4. Invoke Elasticsearch lifespan startup logic

Usage:
    Called via manage.py:
    export TENANT_SINGLE_TENANT_ID=tenant_001
    python src/manage.py tenant-init

Note:
    - Environment variable TENANT_SINGLE_TENANT_ID must be set, otherwise an error will be raised
    - Database names are automatically generated based on the tenant ID (format: {tenant_id}_memsys)
    - Database connection configurations are obtained from default environment variables
"""

from core.observation.logger import get_logger
from core.tenants.tenant_config import get_tenant_config
from core.lifespan.mongodb_lifespan import MongoDBLifespanProvider
from core.lifespan.milvus_lifespan import MilvusLifespanProvider
from core.lifespan.elasticsearch_lifespan import ElasticsearchLifespanProvider

logger = get_logger(__name__)


async def init_mongodb() -> bool:
    """
    Initialize tenant's MongoDB database

    Args:
        tenant_info: Tenant information

    Returns:
        Whether initialization was successful
    """
    logger.info("=" * 60)
    logger.info("Starting initialization of tenant's MongoDB database...")
    logger.info("=" * 60)

    try:
        # Create MongoDB lifespan provider
        mongodb_provider = MongoDBLifespanProvider()

        # Create a mock FastAPI app object (only needs state attribute)
        class MockApp:
            class State:
                pass

            state = State()

        mock_app = MockApp()

        # Call startup logic
        await mongodb_provider.startup(mock_app)

        logger.info("=" * 60)
        logger.info("✅ Tenant's MongoDB database initialized successfully")
        logger.info("=" * 60)

        # Close connections
        await mongodb_provider.shutdown(mock_app)

        return True

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ Failed to initialize tenant's MongoDB database: %s", e)
        logger.error("=" * 60)
        return False


async def init_milvus() -> bool:
    """
    Initialize tenant's Milvus database

    Args:
        tenant_info: Tenant information

    Returns:
        Whether initialization was successful
    """
    logger.info("=" * 60)
    logger.info("Starting initialization of tenant's Milvus database...")
    logger.info("=" * 60)

    try:
        # Create Milvus lifespan provider
        milvus_provider = MilvusLifespanProvider()

        # Create a mock FastAPI app object (only needs state attribute)
        class MockApp:
            class State:
                pass

            state = State()

        mock_app = MockApp()

        # Call startup logic
        await milvus_provider.startup(mock_app)

        logger.info("=" * 60)
        logger.info("✅ Tenant's Milvus database initialized successfully")
        logger.info("=" * 60)

        # Close connections
        await milvus_provider.shutdown(mock_app)

        return True

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ Failed to initialize tenant's Milvus database: %s", e)
        logger.error("=" * 60)
        return False


async def init_elasticsearch() -> bool:
    """
    Initialize tenant's Elasticsearch database

    Returns:
        Whether initialization was successful
    """
    logger.info("=" * 60)
    logger.info("Starting initialization of tenant's Elasticsearch database...")
    logger.info("=" * 60)

    try:
        # Create Elasticsearch lifespan provider
        es_provider = ElasticsearchLifespanProvider()

        # Create a mock FastAPI app object (only needs state attribute)
        class MockApp:
            class State:
                pass

            state = State()

        mock_app = MockApp()

        # Call startup logic
        await es_provider.startup(mock_app)

        logger.info("=" * 60)
        logger.info("✅ Tenant's Elasticsearch database initialized successfully")
        logger.info("=" * 60)

        # Close connections
        await es_provider.shutdown(mock_app)

        return True

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ Failed to initialize tenant's Elasticsearch database: %s", e)
        logger.error("=" * 60)
        return False


async def run_tenant_init() -> bool:
    """
    Execute tenant database initialization

    Read tenant ID from environment variable TENANT_SINGLE_TENANT_ID.
    If this environment variable is not set, raise an error.

    Returns:
        Whether all initializations were successful

    Raises:
        ValueError: If the TENANT_SINGLE_TENANT_ID environment variable is not set

    Examples:
        export TENANT_SINGLE_TENANT_ID=tenant_001
        python src/manage.py tenant-init
    """
    logger.info("*" * 60)
    logger.info("Tenant Database Initialization Tool")
    logger.info("*" * 60)

    # Get tenant ID from configuration
    tenant_config = get_tenant_config()
    tenant_id = tenant_config.single_tenant_id

    # If tenant ID is not configured, raise error
    if not tenant_id:
        error_msg = (
            "Tenant ID is not set!\n"
            "Please set the environment variable TENANT_SINGLE_TENANT_ID, for example:\n"
            "  export TENANT_SINGLE_TENANT_ID=tenant_001\n"
            "  python src/manage.py tenant-init"
        )
        logger.error(error_msg)
        raise ValueError("Environment variable TENANT_SINGLE_TENANT_ID is not set")

    logger.info("Tenant ID: %s", tenant_id)
    logger.info("*" * 60)

    # Initialize MongoDB
    mongodb_success = await init_mongodb()

    # Initialize Milvus
    milvus_success = await init_milvus()

    # Initialize Elasticsearch
    es_success = await init_elasticsearch()

    # Output summary
    logger.info("")
    logger.info("*" * 60)
    logger.info("Initialization Result Summary")
    logger.info("*" * 60)
    logger.info("Tenant ID: %s", tenant_id)
    logger.info("MongoDB: %s", "✅ Success" if mongodb_success else "❌ Failure")
    logger.info("Milvus: %s", "✅ Success" if milvus_success else "❌ Failure")
    logger.info("Elasticsearch: %s", "✅ Success" if es_success else "❌ Failure")
    logger.info("*" * 60)

    # Return whether all were successful
    return mongodb_success and milvus_success and es_success
