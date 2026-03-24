"""dynaris: A JAX-powered Python library for Dynamic Linear Models (DLMs)."""

from dynaris.core import (
    FilterProtocol,
    FilterResult,
    GaussianState,
    SmootherProtocol,
    SmootherResult,
    StateSpaceModel,
)

__version__ = "0.1.0"

__all__ = [
    "FilterProtocol",
    "FilterResult",
    "GaussianState",
    "SmootherProtocol",
    "SmootherResult",
    "StateSpaceModel",
    "__version__",
]
