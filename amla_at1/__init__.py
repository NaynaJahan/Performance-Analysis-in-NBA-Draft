# from amla_at1 import config  # noqa: F401


try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("amla-at1")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except Exception:
    __version__ = "0.0.0"