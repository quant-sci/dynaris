"""Tests for the forecasting module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState
from dynaris.dlm.components import LocalLevel, LocalLinearTrend, Seasonal
from dynaris.filters.kalman import kalman_filter
from dynaris.forecast.forecast import (
    ForecastResult,
    confidence_bands,
    fit_batch,
    forecast,
    forecast_batch,
    forecast_from_filter,
    forecast_from_smoother,
)
from dynaris.smoothers.rts import rts_smooth

NILE = jnp.array([
    1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140,
    995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140,
    1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840,
    874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969,
    831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821,
    768, 845, 864, 862, 698, 845, 744, 796, 1040, 759,
    781, 865, 845, 944, 984, 897, 822, 1010, 771, 676,
    649, 846, 812, 742, 801, 1040, 860, 874, 848, 890,
    744, 749, 838, 1050, 918, 986, 797, 923, 975, 815,
    1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740,
], dtype=jnp.float32)


# ===================================================================
# Multi-step-ahead forecast
# ===================================================================


def test_forecast_shapes() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    state = GaussianState(mean=jnp.array([850.0]), cov=jnp.array([[5000.0]]))
    result = forecast(model, state, steps=12)

    assert isinstance(result, ForecastResult)
    assert result.mean.shape == (12, 1)
    assert result.covariance.shape == (12, 1, 1)
    assert result.state_mean.shape == (12, 1)
    assert result.state_covariance.shape == (12, 1, 1)


def test_forecast_finite() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    state = GaussianState(mean=jnp.array([850.0]), cov=jnp.array([[5000.0]]))
    result = forecast(model, state, steps=12)
    assert jnp.all(jnp.isfinite(result.mean))
    assert jnp.all(jnp.isfinite(result.covariance))


def test_forecast_uncertainty_grows() -> None:
    """Forecast uncertainty should increase with horizon."""
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    state = GaussianState(mean=jnp.array([850.0]), cov=jnp.array([[5000.0]]))
    result = forecast(model, state, steps=20)

    variances = result.covariance[:, 0, 0]
    # Each step should have >= variance of previous step
    for i in range(1, 20):
        assert float(variances[i]) >= float(variances[i - 1]) - 1e-3


def test_forecast_local_level_constant_mean() -> None:
    """For a local-level model, forecast mean should be constant."""
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    state = GaussianState(mean=jnp.array([850.0]), cov=jnp.array([[5000.0]]))
    result = forecast(model, state, steps=10)

    # Mean should be constant for random walk
    np.testing.assert_allclose(
        result.mean[:, 0], 850.0 * jnp.ones(10), atol=1e-4
    )


def test_forecast_trend_increasing() -> None:
    """For a trend model with positive slope, forecast should increase."""
    model = LocalLinearTrend(sigma_level=1.0, sigma_slope=0.1, sigma_obs=2.0)
    # State: [level=100, slope=5]
    state = GaussianState(
        mean=jnp.array([100.0, 5.0]),
        cov=jnp.eye(2) * 10.0,
    )
    result = forecast(model, state, steps=10)

    # Forecast means should be increasing
    means = result.mean[:, 0]
    for i in range(1, 10):
        assert float(means[i]) > float(means[i - 1])


def test_forecast_jit() -> None:
    """Forecast function should be JIT-compatible."""
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    state = GaussianState(mean=jnp.array([850.0]), cov=jnp.array([[5000.0]]))
    r1 = forecast(model, state, steps=5)
    r2 = forecast(model, state, steps=5)
    np.testing.assert_allclose(r1.mean, r2.mean, atol=1e-5)


# ===================================================================
# Forecast from filter / smoother
# ===================================================================


def test_forecast_from_filter() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    obs = NILE.reshape(-1, 1)
    fr = kalman_filter(model, obs)
    result = forecast_from_filter(model, fr, steps=12)

    assert result.mean.shape == (12, 1)
    assert jnp.all(jnp.isfinite(result.mean))


def test_forecast_from_smoother() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    obs = NILE.reshape(-1, 1)
    fr = kalman_filter(model, obs)
    sr = rts_smooth(model, fr)
    result = forecast_from_smoother(model, sr, steps=12)

    assert result.mean.shape == (12, 1)
    assert jnp.all(jnp.isfinite(result.mean))


def test_smoother_forecast_lower_initial_uncertainty() -> None:
    """Smoother-based forecast should start with <= uncertainty vs filter."""
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    obs = NILE.reshape(-1, 1)
    fr = kalman_filter(model, obs)
    sr = rts_smooth(model, fr)

    fc_filter = forecast_from_filter(model, fr, steps=5)
    fc_smooth = forecast_from_smoother(model, sr, steps=5)

    # At step 1, smoother-based variance should be <= filter-based
    assert float(fc_smooth.covariance[0, 0, 0]) <= float(
        fc_filter.covariance[0, 0, 0]
    ) + 1e-3


# ===================================================================
# Confidence bands
# ===================================================================


def test_confidence_bands_shapes() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    state = GaussianState(mean=jnp.array([850.0]), cov=jnp.array([[5000.0]]))
    result = forecast(model, state, steps=10)

    lower, upper = confidence_bands(result.mean, result.covariance)
    assert lower.shape == result.mean.shape
    assert upper.shape == result.mean.shape


def test_confidence_bands_ordering() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    state = GaussianState(mean=jnp.array([850.0]), cov=jnp.array([[5000.0]]))
    result = forecast(model, state, steps=10)

    lower, upper = confidence_bands(result.mean, result.covariance)
    assert jnp.all(lower <= result.mean + 1e-5)
    assert jnp.all(upper >= result.mean - 1e-5)
    assert jnp.all(upper > lower)


def test_confidence_bands_wider_at_lower_confidence() -> None:
    """99% bands should be wider than 90% bands."""
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    state = GaussianState(mean=jnp.array([850.0]), cov=jnp.array([[5000.0]]))
    result = forecast(model, state, steps=5)

    l90, u90 = confidence_bands(result.mean, result.covariance, level=0.90)
    l99, u99 = confidence_bands(result.mean, result.covariance, level=0.99)

    width_90 = u90 - l90
    width_99 = u99 - l99
    assert jnp.all(width_99 > width_90)


def test_confidence_bands_on_filtered_states() -> None:
    """Confidence bands work on filtered states too."""
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    obs = NILE.reshape(-1, 1)
    fr = kalman_filter(model, obs)

    lower, upper = confidence_bands(
        fr.filtered_states, fr.filtered_covariances
    )
    assert lower.shape == fr.filtered_states.shape
    assert jnp.all(jnp.isfinite(lower))
    assert jnp.all(jnp.isfinite(upper))


# ===================================================================
# Batch forecasting via vmap
# ===================================================================


def test_forecast_batch_shapes() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    # 3 series of length 50
    key = jax.random.PRNGKey(0)
    batch = jax.random.normal(key, (3, 50, 1)) * 100 + 800

    result = forecast_batch(model, batch, steps=10)
    assert result.mean.shape == (3, 10, 1)
    assert result.covariance.shape == (3, 10, 1, 1)
    assert result.state_mean.shape == (3, 10, 1)


def test_forecast_batch_finite() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    key = jax.random.PRNGKey(1)
    batch = jax.random.normal(key, (5, 30, 1)) * 100 + 800

    result = forecast_batch(model, batch, steps=5)
    assert jnp.all(jnp.isfinite(result.mean))


def test_forecast_batch_matches_individual() -> None:
    """Batch forecast should match individual forecasts."""
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    key = jax.random.PRNGKey(2)
    batch = jax.random.normal(key, (3, 30, 1)) * 100 + 800

    batch_result = forecast_batch(model, batch, steps=5)

    # Compare with individual forecasts
    for i in range(3):
        fr = kalman_filter(model, batch[i])
        individual = forecast_from_filter(model, fr, steps=5)
        np.testing.assert_allclose(
            batch_result.mean[i], individual.mean, atol=1e-4
        )


# ===================================================================
# fit_batch
# ===================================================================


def test_fit_batch_shapes() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    key = jax.random.PRNGKey(3)
    batch = jax.random.normal(key, (4, 40, 1)) * 100 + 800

    result = fit_batch(model, batch)
    assert result.filtered_states.shape == (4, 40, 1)
    assert result.filtered_covariances.shape == (4, 40, 1, 1)
    assert result.log_likelihood.shape == (4,)


def test_fit_batch_matches_individual() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    key = jax.random.PRNGKey(4)
    batch = jax.random.normal(key, (3, 30, 1)) * 100 + 800

    batch_result = fit_batch(model, batch)

    for i in range(3):
        individual = kalman_filter(model, batch[i])
        np.testing.assert_allclose(
            batch_result.filtered_states[i],
            individual.filtered_states,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            float(batch_result.log_likelihood[i]),
            float(individual.log_likelihood),
            atol=1e-2,
        )


# ===================================================================
# Composed model forecast
# ===================================================================


def test_composed_model_forecast() -> None:
    """Forecasting works with composed (trend + seasonal) models."""
    model = (
        LocalLinearTrend(sigma_level=1.0, sigma_slope=0.1, sigma_obs=0.0)
        + Seasonal(period=12, sigma_seasonal=0.5, sigma_obs=2.0)
    )
    t = jnp.arange(60, dtype=jnp.float32)
    obs = (
        t * 0.3
        + 5.0 * jnp.sin(2.0 * jnp.pi * t / 12.0)
        + jax.random.normal(jax.random.PRNGKey(5), (60,)) * 2.0
    ).reshape(-1, 1)

    fr = kalman_filter(model, obs)
    result = forecast_from_filter(model, fr, steps=24)

    assert result.mean.shape == (24, 1)
    assert jnp.all(jnp.isfinite(result.mean))
    assert jnp.all(jnp.isfinite(result.covariance))
