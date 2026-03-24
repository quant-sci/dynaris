"""dynaris: A JAX-powered Python library for Dynamic Linear Models (DLMs)."""

from dynaris.core import (
    FilterProtocol,
    FilterResult,
    GaussianState,
    SmootherProtocol,
    SmootherResult,
    StateSpaceModel,
)
from dynaris.filters import KalmanFilter, kalman_filter
from dynaris.smoothers import RTSSmoother, rts_smooth

__version__ = "0.1.0"

__all__ = [
    "FilterProtocol",
    "FilterResult",
    "GaussianState",
    "KalmanFilter",
    "RTSSmoother",
    "SmootherProtocol",
    "SmootherResult",
    "StateSpaceModel",
    "__version__",
    "kalman_filter",
    "rts_smooth",
]
