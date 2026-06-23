"""
Application startup module

Responsible for various initialization operations when the application starts
"""

import importlib

# Import dependency injection related modules
from core.observation.logger import get_logger
from core.addons.addons_registry import ADDONS_REGISTRY
from core.addons.addonize.di_setup import (
    setup_dependency_injection,
    print_registered_beans,
)

# Recommended usage: get logger once at the top of the module, then use directly (high performance)
logger = get_logger(__name__)


def _ensure_local_addon_loaded():
    """
    Load the local addon module when entry points are unavailable.

    This allows `python src/run.py` to work from a source checkout even when the
    project has not been installed into the active environment yet.
    """
    if ADDONS_REGISTRY.count() > 0:
        return

    for module_name in ("addon", "src.addon"):
        try:
            importlib.import_module(module_name)
            logger.info("📦 Loaded local addon fallback: %s", module_name)
            return
        except ImportError:
            continue

    logger.warning(
        "⚠️ No addon entry points were found and local addon fallback could not be loaded"
    )


def setup_all(load_entrypoints: bool = True):
    """
    Set up all components

    Args:
        load_entrypoints (bool): Whether to load addons from entry points. Default is True

    Returns:
        ComponentScanner: Configured component scanner
    """
    # 0. Load addons entry points (if enabled)
    if load_entrypoints:
        logger.info("🔌 Loading addons entry points...")
        ADDONS_REGISTRY.load_entrypoints()

    _ensure_local_addon_loaded()

    # Get all addons
    all_addons = ADDONS_REGISTRY.get_all()
    logger.info("📦 Loaded %d addons in total", len(all_addons))

    # 1. Set up dependency injection
    scanner = setup_dependency_injection(all_addons)

    # 2. Set up asynchronous tasks
    # setup_async_tasks(all_addons)

    return scanner


if __name__ == "__main__":
    # Start dependency injection
    setup_all()

    # Print registered Bean information
    print_registered_beans()

    # Print registered tasks
    from core.addons.addonize.asynctasks_setup import print_registered_tasks

    print_registered_tasks()

    logger.info("\n✨ Application startup completed!")
