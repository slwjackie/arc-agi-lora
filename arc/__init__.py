__all__ = ["ARCSolver"]


def __getattr__(name):
    if name == "ARCSolver":
        from .arc import ARCSolver
        return ARCSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
