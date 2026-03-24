"""Multi-step-ahead forecasting for linear-Gaussian state-space models."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.results import FilterResult, SmootherResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState
from dynaris.filters.kalman import kalman_filter

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class ForecastResult(NamedTuple):
    """Result of multi-step-ahead forecasting.

    Attributes:
        mean: Forecast means in observation space, shape (steps, obs_dim).
        covariance: Forecast covariances in obs space,
            shape (steps, obs_dim, obs_dim).
        state_mean: Forecast state means, shape (steps, state_dim).
        state_covariance: Forecast state covariances,
            shape (steps, state_dim, state_dim).
    """

    mean: Array  # (steps, m)
    covariance: Array  # (steps, m, m)
    state_mean: Array  # (steps, n)
    state_covariance: Array  # (steps, n, n)


# ---------------------------------------------------------------------------
# Core forecast function
# ---------------------------------------------------------------------------


@jax.jit(static_argnums=(2,))
def forecast(
    model: StateSpaceModel,
    last_state: GaussianState,
    steps: int,
) -> ForecastResult:
    """Multi-step-ahead forecast from a given state belief.

    Iteratively applies the predict step without observations.

    Args:
        model: Linear-Gaussian state-space model.
        last_state: The state belief to forecast from (e.g. last filtered
            or smoothed state).
        steps: Number of steps ahead to forecast.

    Returns:
        ForecastResult with forecast means and covariances.
    """

    def _forecast_step(
        state: GaussianState, _: None
    ) -> tuple[GaussianState, tuple[Array, Array, Array, Array]]:
        # Predict next state: a = G @ m, R = G @ C @ G' + W
        pred_mean = model.G @ state.mean
        pred_cov = model.G @ state.cov @ model.G.T + model.W

        # Observation-space forecast: f = F' @ a, Q = F' @ R @ F + V
        obs_mean = model.F @ pred_mean
        obs_cov = model.F @ pred_cov @ model.F.T + model.V

        new_state = GaussianState(mean=pred_mean, cov=pred_cov)
        return new_state, (obs_mean, obs_cov, pred_mean, pred_cov)

    _, (obs_means, obs_covs, state_means, state_covs) = jax.lax.scan(
        _forecast_step, last_state, None, length=steps
    )

    return ForecastResult(
        mean=obs_means,
        covariance=obs_covs,
        state_mean=state_means,
        state_covariance=state_covs,
    )


def forecast_from_filter(
    model: StateSpaceModel,
    filter_result: FilterResult,
    steps: int,
) -> ForecastResult:
    """Forecast from the last filtered state.

    Args:
        model: The state-space model used for filtering.
        filter_result: Output of a Kalman filter pass.
        steps: Number of steps ahead.

    Returns:
        ForecastResult.
    """
    last = GaussianState(
        mean=filter_result.filtered_states[-1],
        cov=filter_result.filtered_covariances[-1],
    )
    return forecast(model, last, steps)  # type: ignore[no-any-return]


def forecast_from_smoother(
    model: StateSpaceModel,
    smoother_result: SmootherResult,
    steps: int,
) -> ForecastResult:
    """Forecast from the last smoothed state.

    Args:
        model: The state-space model used for smoothing.
        smoother_result: Output of an RTS smoother pass.
        steps: Number of steps ahead.

    Returns:
        ForecastResult.
    """
    last = GaussianState(
        mean=smoother_result.smoothed_states[-1],
        cov=smoother_result.smoothed_covariances[-1],
    )
    return forecast(model, last, steps)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Confidence bands
# ---------------------------------------------------------------------------


def confidence_bands(
    means: Array,
    covariances: Array,
    level: float = 0.95,
) -> tuple[Array, Array]:
    """Compute confidence bands from means and covariances.

    Works for both filtered/smoothed state estimates and forecasts.

    Args:
        means: Mean values, shape (T,) or (T, dim).
        covariances: Covariance matrices, shape (T,), (T, dim, dim),
            or matching means shape.
        level: Confidence level (default 0.95 for 95% intervals).

    Returns:
        Tuple of (lower, upper) bounds with the same shape as means.
    """
    from scipy import stats

    z = float(stats.norm.ppf((1.0 + level) / 2.0))

    means = jnp.asarray(means)
    covariances = jnp.asarray(covariances)

    if covariances.ndim >= 2 and covariances.shape[-1] == covariances.shape[-2]:
        # (..., d, d) covariance matrices -> extract diagonal variances
        std_devs = jnp.sqrt(jnp.diagonal(covariances, axis1=-2, axis2=-1))
    else:
        # Already variances, same shape as means
        std_devs = jnp.sqrt(covariances)

    lower = means - z * std_devs
    upper = means + z * std_devs
    return lower, upper


# ---------------------------------------------------------------------------
# Batch forecasting via vmap
# ---------------------------------------------------------------------------


def forecast_batch(
    model: StateSpaceModel,
    observations_batch: Array,
    steps: int,
) -> ForecastResult:
    """Batch forecast across multiple series in parallel using jax.vmap.

    Args:
        model: A single state-space model (same for all series).
        observations_batch: Batch of observation sequences,
            shape (n_series, T, obs_dim).
        steps: Number of steps ahead to forecast.

    Returns:
        ForecastResult with batch dimensions prepended:
            mean shape (n_series, steps, obs_dim), etc.
    """

    def _single_forecast(obs: Array) -> ForecastResult:
        fr = kalman_filter(model, obs)
        last = GaussianState(
            mean=fr.filtered_states[-1],
            cov=fr.filtered_covariances[-1],
        )
        return forecast(model, last, steps)  # type: ignore[no-any-return]

    return jax.vmap(_single_forecast)(observations_batch)


def fit_batch(
    model: StateSpaceModel,
    observations_batch: Array,
) -> FilterResult:
    """Fit (filter) multiple series in parallel using jax.vmap.

    Args:
        model: A single state-space model (same for all series).
        observations_batch: Batch of observation sequences,
            shape (n_series, T, obs_dim).

    Returns:
        FilterResult with batch dimensions prepended:
            filtered_states shape (n_series, T, state_dim), etc.
    """

    def _single_filter(obs: Array) -> FilterResult:
        return kalman_filter(model, obs)  # type: ignore[no-any-return]

    return jax.vmap(_single_filter)(observations_batch)
