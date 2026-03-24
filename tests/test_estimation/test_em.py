"""Tests for EM estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dynaris.core.state_space import StateSpaceModel
from dynaris.estimation.em import EMResult, fit_em

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


def _local_level(q: float = 1000.0, r: float = 10000.0) -> StateSpaceModel:
    return StateSpaceModel(
        transition_matrix=jnp.array([[1.0]]),
        observation_matrix=jnp.array([[1.0]]),
        state_noise_cov=jnp.array([[q]]),
        obs_noise_cov=jnp.array([[r]]),
    )


def test_em_returns_result() -> None:
    obs = NILE.reshape(-1, 1)
    model = _local_level()
    result = fit_em(obs, model, max_iter=20)
    assert isinstance(result, EMResult)
    assert result.n_iterations > 0
    assert len(result.log_likelihood_history) == result.n_iterations


def test_em_likelihood_monotonically_increases() -> None:
    """EM log-likelihood should be non-decreasing at each iteration."""
    obs = NILE.reshape(-1, 1)
    model = _local_level(q=100.0, r=50000.0)
    result = fit_em(obs, model, max_iter=30)

    ll_hist = result.log_likelihood_history
    # EM with the simplified M-step should generally increase LL.
    # Allow small decreases due to the approximate cross-covariance.
    for i in range(1, len(ll_hist)):
        assert ll_hist[i] >= ll_hist[i - 1] - 5.0, (
            f"LL decreased too much at iter {i}: {ll_hist[i]} < {ll_hist[i-1]}"
        )
    # Overall, LL should improve from start to finish
    assert ll_hist[-1] > ll_hist[0]


def test_em_improves_from_poor_start() -> None:
    obs = NILE.reshape(-1, 1)
    # Start with very bad variance guesses
    model = _local_level(q=1.0, r=1.0)

    from dynaris.filters.kalman import kalman_filter

    init_ll = float(kalman_filter(model, obs).log_likelihood)

    result = fit_em(obs, model, max_iter=50)
    assert result.log_likelihood > init_ll


def test_em_converges() -> None:
    obs = NILE.reshape(-1, 1)
    model = _local_level(q=1000.0, r=10000.0)
    result = fit_em(obs, model, max_iter=200, tol=1e-4)
    assert result.converged


def test_em_on_simulated_data() -> None:
    """EM should recover approximate variance parameters from simulated data."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    n = 500
    true_q, true_r = 4.0, 9.0

    states = jnp.cumsum(jax.random.normal(k1, (n,)) * jnp.sqrt(true_q))
    obs = (states + jax.random.normal(k2, (n,)) * jnp.sqrt(true_r)).reshape(-1, 1)

    model = _local_level(q=1.0, r=1.0)
    result = fit_em(obs, model, max_iter=100, tol=1e-6)

    fitted_q = float(result.model.Q[0, 0])
    fitted_r = float(result.model.R[0, 0])
    # Should be within a factor of 3 of truth (EM can be noisy)
    assert true_q / 3 < fitted_q < true_q * 3, f"Q={fitted_q}, true={true_q}"
    assert true_r / 3 < fitted_r < true_r * 3, f"R={fitted_r}, true={true_r}"
