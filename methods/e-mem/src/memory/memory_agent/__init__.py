from importlib import import_module

__all__ = ["MemoryAgent"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(".agent", __name__), name)
    globals()[name] = value
    return value
