"""Tests for the pure NumPy Kalman filter backend."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dynaris.backends.numpy_backend import NumpyFilterResult, kalman_filter_numpy
from dynaris.datasets import load_nile_jax
from dynaris.dlm.components import LocalLevel
from dynaris.filters.kalman import kalman_filter

NILE = np.asarray(load_nile_jax())


def _local_level_matrices(
    sigma_level: float = 38.0, sigma_obs: float = 123.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([[1.0]]),  # F
        np.array([[1.0]]),  # G
        np.array([[sigma_obs**2]]),  # V
        np.array([[sigma_level**2]]),  # W
    )


def test_numpy_filter_returns_result() -> None:
    f, g, v, w = _local_level_matrices()
    obs = NILE.reshape(-1, 1)
    result = kalman_filter_numpy(f, g, v, w, obs)
    assert isinstance(result, NumpyFilterResult)


def test_numpy_filter_shapes() -> None:
    f, g, v, w = _local_level_matrices()
    obs = NILE.reshape(-1, 1)
    result = kalman_filter_numpy(f, g, v, w, obs)

    assert result.filtered_states.shape == (100, 1)
    assert result.filtered_covariances.shape == (100, 1, 1)
    assert result.predicted_states.shape == (100, 1)
    assert result.predicted_covariances.shape == (100, 1, 1)


def test_numpy_filter_finite() -> None:
    f, g, v, w = _local_level_matrices()
    obs = NILE.reshape(-1, 1)
    result = kalman_filter_numpy(f, g, v, w, obs)

    assert np.all(np.isfinite(result.filtered_states))
    assert np.isfinite(result.log_likelihood)


def test_numpy_filter_negative_ll() -> None:
    f, g, v, w = _local_level_matrices()
    obs = NILE.reshape(-1, 1)
    result = kalman_filter_numpy(f, g, v, w, obs)
    assert result.log_likelihood < 0.0


def test_numpy_matches_jax() -> None:
    """NumPy backend should produce similar results to JAX backend."""
    sigma_level, sigma_obs = 38.0, 123.0
    f_np, g_np, v_np, w_np = _local_level_matrices(sigma_level, sigma_obs)
    obs_np = NILE.reshape(-1, 1)

    numpy_result = kalman_filter_numpy(f_np, g_np, v_np, w_np, obs_np)

    jax_model = LocalLevel(sigma_level=sigma_level, sigma_obs=sigma_obs)
    jax_result = kalman_filter(jax_model, jnp.array(obs_np))

    np.testing.assert_allclose(
        numpy_result.filtered_states,
        np.asarray(jax_result.filtered_states),
        atol=1e-2,
    )
    np.testing.assert_allclose(
        numpy_result.log_likelihood,
        float(jax_result.log_likelihood),
        atol=1.0,
    )


def test_numpy_filter_missing_obs() -> None:
    f, g, v, w = _local_level_matrices()
    obs = NILE.reshape(-1, 1).copy()
    obs[10, 0] = np.nan
    obs[20, 0] = np.nan

    result = kalman_filter_numpy(f, g, v, w, obs)
    assert np.all(np.isfinite(result.filtered_states))
    # At NaN indices, filtered should equal predicted
    np.testing.assert_allclose(result.filtered_states[10], result.predicted_states[10], atol=1e-10)


def test_numpy_filter_multivariate() -> None:
    """Test with 2D state and observation."""
    n = 2
    f = np.eye(n)
    g = np.eye(n) * 0.99
    v = np.eye(n) * 1.0
    w = np.eye(n) * 0.1
    obs = np.random.default_rng(42).standard_normal((50, n))

    result = kalman_filter_numpy(f, g, v, w, obs)
    assert result.filtered_states.shape == (50, 2)
    assert np.all(np.isfinite(result.filtered_states))
