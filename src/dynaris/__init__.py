"""dynaris: A JAX-powered Python library for Dynamic Linear Models (DLMs)."""

from dynaris.core import (
    FilterProtocol,
    FilterResult,
    GaussianState,
    SmootherProtocol,
    SmootherResult,
    StateSpaceModel,
)
from dynaris.dlm import (
    Autoregressive,
    Cycle,
    LocalLevel,
    LocalLinearTrend,
    Regression,
    Seasonal,
)
from dynaris.filters import KalmanFilter, kalman_filter
from dynaris.smoothers import RTSSmoother, rts_smooth

__version__ = "0.1.0"

__all__ = [
    "Autoregressive",
    "Cycle",
    "FilterProtocol",
    "FilterResult",
    "GaussianState",
    "KalmanFilter",
    "LocalLevel",
    "LocalLinearTrend",
    "RTSSmoother",
    "Regression",
    "Seasonal",
    "SmootherProtocol",
    "SmootherResult",
    "StateSpaceModel",
    "__version__",
    "kalman_filter",
    "rts_smooth",
]
