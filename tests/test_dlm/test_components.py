"""Tests for DLM components."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dynaris.dlm.components import (
    Autoregressive,
    Cycle,
    LocalLevel,
    LocalLinearTrend,
    Regression,
    Seasonal,
)
from dynaris.filters.kalman import kalman_filter
from dynaris.smoothers.rts import rts_smooth

# ===================================================================
# LocalLevel
# ===================================================================


def test_local_level_shapes() -> None:
    m = LocalLevel(sigma_level=2.0, sigma_obs=3.0)
    assert m.state_dim == 1
    assert m.obs_dim == 1
    assert float(m.W[0, 0]) == 4.0
    assert float(m.V[0, 0]) == 9.0


def test_local_level_filter_on_synthetic() -> None:
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    true_states = jnp.cumsum(jax.random.normal(k1, (100,)) * 1.0)
    obs = (true_states + jax.random.normal(k2, (100,)) * 5.0).reshape(-1, 1)

    model = LocalLevel(sigma_level=1.0, sigma_obs=5.0)
    result = kalman_filter(model, obs)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    # Filtered should track the true states reasonably
    corr = jnp.corrcoef(
        jnp.stack([result.filtered_states[:, 0], true_states])
    )[0, 1]
    assert float(corr) > 0.8


# ===================================================================
# LocalLinearTrend
# ===================================================================


def test_local_linear_trend_shapes() -> None:
    m = LocalLinearTrend(sigma_level=1.0, sigma_slope=0.5, sigma_obs=2.0)
    assert m.state_dim == 2
    assert m.obs_dim == 1
    # G = [[1, 1], [0, 1]]
    np.testing.assert_allclose(m.G, [[1, 1], [0, 1]])
    # F = [[1, 0]]
    np.testing.assert_allclose(m.F, [[1, 0]])


def test_local_linear_trend_filter_on_trend_data() -> None:
    """Filter should detect an upward linear trend."""
    t = jnp.arange(100, dtype=jnp.float32)
    obs = (0.5 * t + jax.random.normal(jax.random.PRNGKey(1), (100,)) * 2.0)
    obs = obs.reshape(-1, 1)

    model = LocalLinearTrend(sigma_level=1.0, sigma_slope=0.1, sigma_obs=2.0)
    result = kalman_filter(model, obs)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    # The slope estimate at the end should be positive
    final_slope = float(result.filtered_states[-1, 1])
    assert final_slope > 0.0


# ===================================================================
# Seasonal (dummy form)
# ===================================================================


def test_seasonal_dummy_shapes() -> None:
    m = Seasonal(period=12, form="dummy")
    assert m.state_dim == 11  # period - 1
    assert m.obs_dim == 1
    assert m.G.shape == (11, 11)
    # First row should be all -1
    np.testing.assert_allclose(m.G[0, :], -jnp.ones(11))


def test_seasonal_dummy_filter_on_periodic_data() -> None:
    """Filter should detect seasonality in periodic synthetic data."""
    period = 12
    t = jnp.arange(120, dtype=jnp.float32)
    seasonal_pattern = 10.0 * jnp.sin(2.0 * jnp.pi * t / period)
    obs = (seasonal_pattern
           + jax.random.normal(jax.random.PRNGKey(2), (120,)) * 1.0)
    obs = obs.reshape(-1, 1)

    model = Seasonal(period=period, sigma_seasonal=1.0, sigma_obs=1.0)
    result = kalman_filter(model, obs)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


# ===================================================================
# Seasonal (Fourier form)
# ===================================================================


def test_seasonal_fourier_shapes_even() -> None:
    m = Seasonal(period=12, form="fourier")
    # 12 harmonics: 5 pairs of 2 + 1 scalar = 11
    assert m.state_dim == 11
    assert m.obs_dim == 1


def test_seasonal_fourier_shapes_odd() -> None:
    m = Seasonal(period=7, form="fourier")
    # 7: 3 pairs of 2 = 6
    assert m.state_dim == 6
    assert m.obs_dim == 1


def test_seasonal_fourier_filter() -> None:
    period = 12
    t = jnp.arange(120, dtype=jnp.float32)
    pattern = 5.0 * jnp.cos(2.0 * jnp.pi * t / period)
    obs = (pattern
           + jax.random.normal(jax.random.PRNGKey(3), (120,)) * 1.0)
    obs = obs.reshape(-1, 1)

    model = Seasonal(period=period, sigma_seasonal=1.0,
                     sigma_obs=1.0, form="fourier")
    result = kalman_filter(model, obs)
    assert jnp.all(jnp.isfinite(result.filtered_states))


def test_seasonal_invalid_form() -> None:
    try:
        Seasonal(period=12, form="invalid")
        assert False, "Should have raised ValueError"  # noqa: B011
    except ValueError as e:
        assert "invalid" in str(e)


# ===================================================================
# Regression
# ===================================================================


def test_regression_shapes() -> None:
    m = Regression(n_regressors=3, sigma_coeff=0.1, sigma_obs=1.0)
    assert m.state_dim == 3
    assert m.obs_dim == 1
    np.testing.assert_allclose(m.G, jnp.eye(3))


def test_regression_static_coefficients() -> None:
    """With sigma_coeff=0, W is zero (static coefficients)."""
    m = Regression(n_regressors=2, sigma_coeff=0.0)
    np.testing.assert_allclose(m.W, jnp.zeros((2, 2)))


# ===================================================================
# Autoregressive
# ===================================================================


def test_autoregressive_shapes() -> None:
    m = Autoregressive(order=3)
    assert m.state_dim == 3
    assert m.obs_dim == 1
    # Companion form: first row zeros (no coefficients), shift below
    np.testing.assert_allclose(m.G[0, :], [0, 0, 0])
    np.testing.assert_allclose(m.G[1:, :-1], jnp.eye(2))


def test_autoregressive_with_coefficients() -> None:
    phi = jnp.array([0.8, -0.2])
    m = Autoregressive(order=2, coefficients=phi)
    np.testing.assert_allclose(m.G[0, :], [0.8, -0.2])
    np.testing.assert_allclose(m.G[1, :], [1.0, 0.0])


def test_autoregressive_order_1() -> None:
    phi = jnp.array([0.9])
    m = Autoregressive(order=1, coefficients=phi)
    assert m.state_dim == 1
    np.testing.assert_allclose(m.G, [[0.9]])


def test_autoregressive_filter_on_ar_data() -> None:
    """Filter an AR(1) process."""
    key = jax.random.PRNGKey(4)
    n = 200
    phi = 0.8
    noise = jax.random.normal(key, (n,)) * 1.0

    # Generate AR(1) data
    def _ar_step(x: Array, w: Array) -> tuple[Array, Array]:
        x_new = phi * x + w
        return x_new, x_new

    _, states = jax.lax.scan(_ar_step, jnp.array(0.0), noise)
    obs = (states + jax.random.normal(
        jax.random.PRNGKey(5), (n,)
    ) * 0.5).reshape(-1, 1)

    model = Autoregressive(
        order=1, coefficients=jnp.array([phi]),
        sigma_ar=1.0, sigma_obs=0.5,
    )
    result = kalman_filter(model, obs)
    assert jnp.all(jnp.isfinite(result.filtered_states))


# ===================================================================
# Cycle
# ===================================================================


def test_cycle_shapes() -> None:
    m = Cycle(period=20.0, damping=0.95, sigma_cycle=1.0)
    assert m.state_dim == 2
    assert m.obs_dim == 1


def test_cycle_undamped_rotation() -> None:
    """Undamped cycle: transition matrix should be a rotation."""
    m = Cycle(period=10.0, damping=1.0)
    freq = 2.0 * jnp.pi / 10.0
    expected_f = jnp.array([
        [jnp.cos(freq), jnp.sin(freq)],
        [-jnp.sin(freq), jnp.cos(freq)],
    ])
    np.testing.assert_allclose(m.G, expected_f, atol=1e-6)


def test_cycle_damped() -> None:
    """Damped cycle: eigenvalues should have magnitude = damping."""
    rho = 0.9
    m = Cycle(period=8.0, damping=rho)
    eigenvalues = jnp.linalg.eigvals(m.G)
    magnitudes = jnp.abs(eigenvalues)
    np.testing.assert_allclose(magnitudes, rho, atol=1e-5)


def test_cycle_filter_on_cyclic_data() -> None:
    period = 20.0
    t = jnp.arange(200, dtype=jnp.float32)
    true_cycle = 3.0 * jnp.cos(2.0 * jnp.pi * t / period)
    obs = (true_cycle
           + jax.random.normal(jax.random.PRNGKey(6), (200,)) * 1.0)
    obs = obs.reshape(-1, 1)

    model = Cycle(period=period, damping=1.0, sigma_cycle=1.0, sigma_obs=1.0)
    result = kalman_filter(model, obs)
    assert jnp.all(jnp.isfinite(result.filtered_states))


# ===================================================================
# Composition via + operator
# ===================================================================


def test_compose_level_plus_seasonal() -> None:
    m1 = LocalLevel(sigma_level=1.0, sigma_obs=0.5)
    m2 = Seasonal(period=4, sigma_seasonal=0.1, sigma_obs=0.5)
    composed = m1 + m2
    # State dim: 1 (level) + 3 (seasonal, period-1) = 4
    assert composed.state_dim == 4
    assert composed.obs_dim == 1
    assert composed.G.shape == (4, 4)
    # Obs noise adds: 0.5^2 + 0.5^2 = 0.5
    np.testing.assert_allclose(composed.V, [[0.5]], atol=1e-6)


def test_compose_trend_plus_seasonal() -> None:
    m1 = LocalLinearTrend(sigma_level=1.0, sigma_slope=0.1, sigma_obs=0.0)
    m2 = Seasonal(period=12, sigma_seasonal=0.1, sigma_obs=1.0)
    composed = m1 + m2
    # State dim: 2 (trend) + 11 (seasonal) = 13
    assert composed.state_dim == 13
    assert composed.obs_dim == 1


def test_compose_three_components() -> None:
    m = (
        LocalLinearTrend(sigma_level=1.0, sigma_slope=0.1, sigma_obs=0.0)
        + Seasonal(period=12, sigma_seasonal=0.1, sigma_obs=0.0)
        + Cycle(period=40.0, sigma_cycle=0.5, sigma_obs=1.0)
    )
    # State dim: 2 + 11 + 2 = 15
    assert m.state_dim == 15
    assert m.obs_dim == 1


def test_composed_model_is_valid_pytree() -> None:
    from jax import tree_util

    m = LocalLevel() + Seasonal(period=4)
    leaves, treedef = tree_util.tree_flatten(m)
    m2 = tree_util.tree_unflatten(treedef, leaves)
    assert m2.state_dim == m.state_dim


def test_composed_model_filter_on_synthetic() -> None:
    """Filter a composed trend + seasonal model on synthetic data."""
    t = jnp.arange(120, dtype=jnp.float32)
    trend = 0.3 * t
    seasonal = 5.0 * jnp.sin(2.0 * jnp.pi * t / 12.0)
    noise = jax.random.normal(jax.random.PRNGKey(7), (120,)) * 2.0
    obs = (trend + seasonal + noise).reshape(-1, 1)

    model = (
        LocalLinearTrend(sigma_level=1.0, sigma_slope=0.1, sigma_obs=0.0)
        + Seasonal(period=12, sigma_seasonal=1.0, sigma_obs=2.0)
    )
    result = kalman_filter(model, obs)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


def test_composed_model_smooth_on_synthetic() -> None:
    """Smoother works on composed models."""
    t = jnp.arange(60, dtype=jnp.float32)
    obs = (t * 0.2
           + 3.0 * jnp.sin(2.0 * jnp.pi * t / 12.0)
           + jax.random.normal(jax.random.PRNGKey(8), (60,)) * 1.0)
    obs = obs.reshape(-1, 1)

    model = (
        LocalLinearTrend(sigma_level=0.5, sigma_slope=0.05, sigma_obs=0.0)
        + Seasonal(period=12, sigma_seasonal=0.5, sigma_obs=1.0)
    )
    fr = kalman_filter(model, obs)
    sr = rts_smooth(model, fr)
    assert jnp.all(jnp.isfinite(sr.smoothed_states))
    # Smoothed variance should be <= filtered
    assert jnp.all(
        jnp.diag(sr.smoothed_covariances[0])
        <= jnp.diag(sr.filtered_covariances[0]) + 1e-5
    )
