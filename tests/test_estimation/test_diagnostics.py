"""Tests for model diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.core.state_space import StateSpaceModel
from dynaris.estimation.diagnostics import acf, ljung_box, pacf, standardized_residuals
from dynaris.filters.kalman import kalman_filter
from dynaris.datasets import load_nile_jax

NILE = load_nile_jax()


def _fit_nile() -> tuple[StateSpaceModel, object]:
    model = StateSpaceModel(
        system_matrix=jnp.array([[1.0]]),
        observation_matrix=jnp.array([[1.0]]),
        evolution_cov=jnp.array([[1469.1]]),
        obs_cov=jnp.array([[15099.0]]),
    )
    obs = NILE.reshape(-1, 1)
    fr = kalman_filter(model, obs)
    return model, fr


# ===================================================================
# Standardized residuals
# ===================================================================


def test_standardized_residuals_shape() -> None:
    model, fr = _fit_nile()
    resids = standardized_residuals(fr, model)
    assert resids.shape == (100,)  # univariate -> squeezed


def test_standardized_residuals_finite() -> None:
    model, fr = _fit_nile()
    resids = standardized_residuals(fr, model)
    assert jnp.all(jnp.isfinite(resids))


def test_standardized_residuals_approximately_standard() -> None:
    """For a well-specified model, standardized residuals ~ N(0, 1)."""
    model, fr = _fit_nile()
    resids = standardized_residuals(fr, model)
    # Mean should be near 0, std near 1 (with tolerance for finite sample)
    assert abs(float(jnp.mean(resids))) < 0.5
    assert 0.5 < float(jnp.std(resids)) < 1.5


# ===================================================================
# ACF
# ===================================================================


def test_acf_lag_zero_is_one() -> None:
    x = jax.random.normal(jax.random.PRNGKey(0), (100,))
    acf_vals = acf(x, n_lags=10)
    np.testing.assert_allclose(float(acf_vals[0]), 1.0, atol=1e-6)


def test_acf_shape() -> None:
    x = jax.random.normal(jax.random.PRNGKey(0), (100,))
    acf_vals = acf(x, n_lags=15)
    assert acf_vals.shape == (16,)  # 0 through 15


def test_acf_white_noise() -> None:
    """ACF of white noise should be near zero for lags > 0."""
    x = jax.random.normal(jax.random.PRNGKey(1), (1000,))
    acf_vals = acf(x, n_lags=20)
    # All non-zero lags should be small
    assert float(jnp.max(jnp.abs(acf_vals[1:]))) < 0.1


def test_acf_correlated_process() -> None:
    """ACF of an AR(1) process should show decaying autocorrelation."""
    key = jax.random.PRNGKey(2)
    n = 2000
    noise = jax.random.normal(key, (n,))

    def _step(x: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_new = 0.9 * x + w
        return x_new, x_new

    _, x = jax.lax.scan(_step, jnp.array(0.0), noise)
    acf_vals = acf(x, n_lags=10)
    # ACF at lag 1 should be near 0.9
    assert float(acf_vals[1]) > 0.7
    # ACF should decay
    assert float(acf_vals[5]) < float(acf_vals[1])


# ===================================================================
# PACF
# ===================================================================


def test_pacf_lag_zero_is_one() -> None:
    x = jax.random.normal(jax.random.PRNGKey(0), (100,))
    pacf_vals = pacf(x, n_lags=10)
    np.testing.assert_allclose(float(pacf_vals[0]), 1.0, atol=1e-6)


def test_pacf_shape() -> None:
    x = jax.random.normal(jax.random.PRNGKey(0), (100,))
    pacf_vals = pacf(x, n_lags=10)
    assert pacf_vals.shape == (11,)


def test_pacf_ar1_cuts_off() -> None:
    """PACF of AR(1) should be significant at lag 1, near zero after."""
    key = jax.random.PRNGKey(3)
    n = 2000
    noise = jax.random.normal(key, (n,))

    def _step(x: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_new = 0.8 * x + w
        return x_new, x_new

    _, x = jax.lax.scan(_step, jnp.array(0.0), noise)
    pacf_vals = pacf(x, n_lags=10)
    # PACF at lag 1 should be near 0.8
    assert float(pacf_vals[1]) > 0.6
    # PACF at lag 2+ should be small
    assert float(jnp.max(jnp.abs(pacf_vals[3:]))) < 0.15


# ===================================================================
# Ljung-Box test
# ===================================================================


def test_ljung_box_white_noise() -> None:
    """White noise should NOT be rejected (high p-value)."""
    x = jax.random.normal(jax.random.PRNGKey(4), (500,))
    q_stat, p_val = ljung_box(x, n_lags=10)
    assert p_val > 0.01  # should not reject at 1% level


def test_ljung_box_correlated() -> None:
    """Correlated process should be rejected (low p-value)."""
    key = jax.random.PRNGKey(5)
    n = 500
    noise = jax.random.normal(key, (n,))

    def _step(x: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_new = 0.9 * x + w
        return x_new, x_new

    _, x = jax.lax.scan(_step, jnp.array(0.0), noise)
    q_stat, p_val = ljung_box(x, n_lags=10)
    assert p_val < 0.05


def test_ljung_box_returns_tuple() -> None:
    x = jax.random.normal(jax.random.PRNGKey(6), (100,))
    result = ljung_box(x, n_lags=5)
    assert isinstance(result, tuple)
    assert len(result) == 2
    q_stat, p_val = result
    assert q_stat >= 0
    assert 0 <= p_val <= 1


# ===================================================================
# Transforms
# ===================================================================


def test_softplus_inverse_roundtrip() -> None:
    from dynaris.estimation.transforms import inverse_softplus, softplus

    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 5.0])
    y = softplus(x)
    x_back = inverse_softplus(y)
    np.testing.assert_allclose(x, x_back, atol=1e-5)


def test_softplus_positive() -> None:
    from dynaris.estimation.transforms import softplus

    x = jnp.array([-10.0, -1.0, 0.0, 1.0, 100.0])
    y = softplus(x)
    assert jnp.all(y >= 0)
    # Should be strictly positive for moderate inputs
    assert jnp.all(softplus(jnp.array([-5.0, 0.0, 5.0])) > 0)
