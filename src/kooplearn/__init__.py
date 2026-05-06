from importlib.metadata import PackageNotFoundError, version

from kooplearn.structs import DynamicalModes

try:
    __version__ = version("kooplearn")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "DynamicalModes",
    "__version__",
]
