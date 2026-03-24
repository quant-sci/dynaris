"""Tests for MLE estimation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dynaris.core.state_space import StateSpaceModel
from dynaris.estimation.mle import MLEResult, fit_mle

# Nile river data (truncated for speed)
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
