"""dynaris: A JAX-powered Python library for Dynamic Linear Models (DLMs)."""

from dynaris.core import (
    SSM,
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
    ParticleFilter,
    UnscentedKalmanFilter,
    ekf_filter,
    kalman_filter,
    particle_filter,
    ukf_filter,
)
from dynaris.models import (
    BearingsTracking,
    LorenzAttractor,
    StochasticVolatility,
    transform_returns,
)
from dynaris.smoothers import RTSSmoother, rts_smooth

__version__ = "0.1.0"

__all__ = [
    "DLM",
    "SSM",
    "Autoregressive",
    "BearingsTracking",
    "Cycle",
    "ExtendedKalmanFilter",
    "FilterProtocol",
    "FilterResult",
    "GaussianState",
    "KalmanFilter",
    "LocalLevel",
    "LocalLinearTrend",
    "LorenzAttractor",
    "NonlinearSSM",
    "ParticleFilter",
    "RTSSmoother",
    "Regression",
    "Seasonal",
    "SmootherProtocol",
    "SmootherResult",
    "StateSpaceModel",
    "StochasticVolatility",
    "UnscentedKalmanFilter",
    "__version__",
    "ekf_filter",
    "kalman_filter",
    "particle_filter",
    "rts_smooth",
    "transform_returns",
    "ukf_filter",
]
