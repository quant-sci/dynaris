"""Tests for MLE estimation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dynaris.core.state_space import StateSpaceModel
from dynaris.estimation.mle import MLEResult, fit_mle
from dynaris.datasets import load_nile_jax

# Nile river data (truncated for speed)
NILE = load_nile_jax()


def _local_level_factory(params: jnp.ndarray) -> StateSpaceModel:
    """Model factory: params = [log(sigma_level^2), log(sigma_obs^2)]."""
    return StateSpaceModel(
        system_matrix=jnp.array([[1.0]]),
        observation_matrix=jnp.array([[1.0]]),
        evolution_cov=jnp.exp(params[0:1, None]) * jnp.eye(1),
        obs_cov=jnp.exp(params[1:2, None]) * jnp.eye(1),
    )


def test_mle_returns_result() -> None:
    obs = NILE.reshape(-1, 1)
    init_params = jnp.array([jnp.log(1600.0), jnp.log(15000.0)])
    result = fit_mle(_local_level_factory, obs, init_params)
    assert isinstance(result, MLEResult)
    assert result.converged
    assert result.n_iterations > 0


def test_mle_improves_likelihood() -> None:
    obs = NILE.reshape(-1, 1)
    # Start from poor initial values
    init_params = jnp.array([jnp.log(1.0), jnp.log(1.0)])
    result = fit_mle(_local_level_factory, obs, init_params)

    # MLE log-likelihood should be better than initial
    from dynaris.filters.kalman import kalman_filter

    init_model = _local_level_factory(init_params)
    init_ll = float(kalman_filter(init_model, obs).log_likelihood)
    assert result.log_likelihood > init_ll


def test_mle_nile_known_values() -> None:
    """MLE on Nile data should recover known variance estimates.

    Known MLE values (Durbin & Koopman):
      sigma_level^2 ~ 1469
      sigma_obs^2 ~ 15099
      log-likelihood ~ -632.5
    """
    obs = NILE.reshape(-1, 1)
    init_params = jnp.array([jnp.log(1000.0), jnp.log(10000.0)])
    result = fit_mle(_local_level_factory, obs, init_params)

    assert result.converged
    # Log-likelihood near the known optimum (-632.5 at MLE);
    # with diffuse prior the value is slightly lower (~-641).
    assert result.log_likelihood > -645

    # Recovered variances
    q = float(jnp.exp(result.params[0]))
    r = float(jnp.exp(result.params[1]))
    # Should be in the right ballpark
    assert 500 < q < 5000, f"Q={q}"
    assert 5000 < r < 30000, f"R={r}"


def test_mle_result_has_filter_result() -> None:
    obs = NILE[:30].reshape(-1, 1)
    init_params = jnp.array([jnp.log(1600.0), jnp.log(15000.0)])
    result = fit_mle(_local_level_factory, obs, init_params)
    assert result.filter_result.filtered_states.shape == (30, 1)
    assert jnp.isfinite(result.filter_result.log_likelihood)
