"""dynaris: A JAX-powered Python library for Dynamic Linear Models (DLMs)."""

from dynaris.core import (
    FilterProtocol,
    FilterResult,
    GaussianState,
    NonlinearSSM,
    SmootherProtocol,
    SmootherResult,
    StateSpaceModel,
)
from dynaris.dlm import (
    DLM,
    Autoregressive,
    Cycle,
    LocalLevel,
    LocalLinearTrend,
    Regression,
    Seasonal,
)
from dynaris.filters import (
    ExtendedKalmanFilter,
    KalmanFilter,
    UnscentedKalmanFilter,
    ekf_filter,
    kalman_filter,
    ukf_filter,
)
from dynaris.smoothers import RTSSmoother, rts_smooth

__version__ = "0.1.0"

__all__ = [
    "DLM",
    "Autoregressive",
    "Cycle",
    "ExtendedKalmanFilter",
    "FilterProtocol",
    "FilterResult",
    "GaussianState",
    "KalmanFilter",
    "LocalLevel",
    "LocalLinearTrend",
    "NonlinearSSM",
    "RTSSmoother",
    "Regression",
    "Seasonal",
    "SmootherProtocol",
    "SmootherResult",
    "StateSpaceModel",
    "UnscentedKalmanFilter",
    "__version__",
    "ekf_filter",
    "kalman_filter",
    "rts_smooth",
    "ukf_filter",
]
