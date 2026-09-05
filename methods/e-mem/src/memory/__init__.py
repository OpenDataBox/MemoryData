from importlib import import_module

__all__ = ["MemoryHandler", "KVBlock", "clear_cache", "MemoryAgent", "Router"]

_EXPORTS = {
    "MemoryHandler": (".core", "MemoryHandler"),
    "KVBlock": (".kv_block_manager", "KVBlock"),
    "clear_cache": (".kv_block_manager", "clear_cache"),
    "MemoryAgent": (".memory_agent", "MemoryAgent"),
    "Router": (".router", "Router"),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
