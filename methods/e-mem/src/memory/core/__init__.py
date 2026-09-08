from importlib import import_module

__all__ = ["MemoryHandler", "AddHandler", "QueryHandler"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(".loop_handler", __name__), name)
    globals()[name] = value
    return value
