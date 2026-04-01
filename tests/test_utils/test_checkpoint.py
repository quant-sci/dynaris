"""Tests for memory-efficient checkpointed Kalman filter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.core.types import GaussianState
from dynaris.datasets import load_nile_jax
from dynaris.dlm.components import LocalLevel
from dynaris.filters.kalman import kalman_filter
from dynaris.utils.checkpoint import checkpointed_kalman_filter

NILE = load_nile_jax()


def test_checkpointed_matches_standard() -> None:
    """Checkpointed filter should produce identical results to standard."""
    model = LocalLevel(sigma_level=38.0, sigma_obs=123.0)
    obs = NILE.reshape(-1, 1)

    standard = kalman_filter(model, obs)
    checkpointed = checkpointed_kalman_filter(model, obs)

    np.testing.assert_allclose(checkpointed.filtered_states, standard.filtered_states, atol=1e-4)
    np.testing.assert_allclose(checkpointed.log_likelihood, standard.log_likelihood, atol=1e-2)


def test_checkpointed_shapes() -> None:
    model = LocalLevel()
    obs = NILE.reshape(-1, 1)
    result = checkpointed_kalman_filter(model, obs)

    assert result.filtered_states.shape == (100, 1)
    assert result.filtered_covariances.shape == (100, 1, 1)
    assert result.predicted_states.shape == (100, 1)
    assert result.log_likelihood.shape == ()


def test_checkpointed_finite() -> None:
    model = LocalLevel(sigma_level=38.0, sigma_obs=123.0)
    obs = NILE.reshape(-1, 1)
    result = checkpointed_kalman_filter(model, obs)

    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


def test_checkpointed_grad() -> None:
    """Verify autodiff works through checkpointed filter."""
    obs = NILE[:30].reshape(-1, 1)

    def neg_ll(log_sigma: jax.Array) -> jax.Array:
        model = LocalLevel(sigma_level=jnp.exp(log_sigma), sigma_obs=123.0)
        return -checkpointed_kalman_filter(model, obs).log_likelihood

    g = jax.grad(neg_ll)(jnp.log(jnp.array(38.0)))
    assert jnp.isfinite(g)


def test_checkpointed_with_custom_initial_state() -> None:
    model = LocalLevel(sigma_level=38.0, sigma_obs=123.0)
    obs = NILE.reshape(-1, 1)
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)

    standard = kalman_filter(model, obs, initial_state=init)
    checkpointed = checkpointed_kalman_filter(model, obs, initial_state=init)

    np.testing.assert_allclose(checkpointed.filtered_states, standard.filtered_states, atol=1e-4)
