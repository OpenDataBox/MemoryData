from importlib import import_module

__all__ = ["KVBlock", "clear_cache", "KV_DATA_DIR"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(".block", __name__), name)
    globals()[name] = value
    return value
