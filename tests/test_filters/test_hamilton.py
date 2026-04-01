"""Tests for the Hamilton filter for Markov-switching models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.core.results import SwitchingFilterResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.switching import MarkovSwitchingSSM
from dynaris.core.types import GaussianState
from dynaris.datasets import load_nile_jax
from dynaris.dlm.components import LocalLevel
from dynaris.filters.hamilton import hamilton_filter
from dynaris.filters.kalman import kalman_filter

NILE = load_nile_jax()


def _two_regime_model() -> MarkovSwitchingSSM:
    return MarkovSwitchingSSM(
        models=(
            LocalLevel(sigma_level=20.0, sigma_obs=80.0),
            LocalLevel(sigma_level=60.0, sigma_obs=200.0),
        ),
        transition_matrix=jnp.array([[0.95, 0.05], [0.10, 0.90]]),
        initial_probs=jnp.array([0.5, 0.5]),
    )


# ---------------------------------------------------------------------------
# Shape and type tests
# ---------------------------------------------------------------------------


def test_hamilton_filter_shapes() -> None:
    model = _two_regime_model()
    observations = NILE.reshape(-1, 1)
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)
    result = hamilton_filter(model, observations, initial_state=init)

    assert isinstance(result, SwitchingFilterResult)
    assert result.filtered_states.shape == (100, 1)
    assert result.filtered_covariances.shape == (100, 1, 1)
    assert result.predicted_states.shape == (100, 1)
    assert result.predicted_covariances.shape == (100, 1, 1)
    assert result.log_likelihood.shape == ()
    assert result.regime_filtered_probs.shape == (100, 2)
    assert result.regime_predicted_probs.shape == (100, 2)
    assert result.regime_filtered_states.shape == (100, 2, 1)
    assert result.regime_filtered_covs.shape == (100, 2, 1, 1)
    assert result.regime_predicted_states.shape == (100, 2, 1)
    assert result.regime_predicted_covs.shape == (100, 2, 1, 1)


def test_hamilton_filter_finite() -> None:
    model = _two_regime_model()
    observations = NILE.reshape(-1, 1)
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)
    result = hamilton_filter(model, observations, initial_state=init)

    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.all(jnp.isfinite(result.filtered_covariances))
    assert jnp.isfinite(result.log_likelihood)
    assert jnp.all(jnp.isfinite(result.regime_filtered_probs))


def test_hamilton_filter_negative_log_likelihood() -> None:
    model = _two_regime_model()
    observations = NILE.reshape(-1, 1)
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)
    result = hamilton_filter(model, observations, initial_state=init)
    assert float(result.log_likelihood) < 0.0


def test_regime_probs_sum_to_one() -> None:
    model = _two_regime_model()
    observations = NILE.reshape(-1, 1)
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)
    result = hamilton_filter(model, observations, initial_state=init)

    sums = jnp.sum(result.regime_filtered_probs, axis=1)
    np.testing.assert_allclose(sums, jnp.ones(100), atol=1e-4)


# ---------------------------------------------------------------------------
# K=1 matches Kalman filter
# ---------------------------------------------------------------------------


def test_single_regime_matches_kalman() -> None:
    """With K=1, Hamilton filter should match standard Kalman filter."""
    single_model = MarkovSwitchingSSM(
        models=(LocalLevel(sigma_level=40.0, sigma_obs=120.0),),
        transition_matrix=jnp.array([[1.0]]),
        initial_probs=jnp.array([1.0]),
    )
    linear_model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)

    ham_result = hamilton_filter(single_model, observations, initial_state=init)
    kf_result = kalman_filter(linear_model, observations, initial_state=init)

    np.testing.assert_allclose(ham_result.filtered_states, kf_result.filtered_states, atol=1e-3)
    np.testing.assert_allclose(ham_result.log_likelihood, kf_result.log_likelihood, atol=1.0)


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


def test_hamilton_filter_detects_regime_shift() -> None:
    """Simulate data from a 2-regime process and verify detection."""
    key = jax.random.PRNGKey(42)
    n = 200

    # Regime 0: low noise, regime 1: high noise
    regime_0_obs = 100.0 + jax.random.normal(key, (n // 2,)) * 5.0
    key, _ = jax.random.split(key)
    regime_1_obs = 100.0 + jax.random.normal(key, (n // 2,)) * 50.0
    observations = jnp.concatenate([regime_0_obs, regime_1_obs]).reshape(-1, 1)

    model = MarkovSwitchingSSM(
        models=(
            LocalLevel(sigma_level=1.0, sigma_obs=5.0),
            LocalLevel(sigma_level=5.0, sigma_obs=50.0),
        ),
        transition_matrix=jnp.array([[0.98, 0.02], [0.02, 0.98]]),
        initial_probs=jnp.array([0.9, 0.1]),
    )

    init = GaussianState(mean=jnp.array([100.0]), cov=jnp.eye(1) * 100.0)
    result = hamilton_filter(model, observations, initial_state=init)

    # In the second half (high noise), regime 1 probability should be higher
    avg_prob_regime1_first_half = float(jnp.mean(result.regime_filtered_probs[20:100, 1]))
    avg_prob_regime1_second_half = float(jnp.mean(result.regime_filtered_probs[120:, 1]))
    assert avg_prob_regime1_second_half > avg_prob_regime1_first_half


# ---------------------------------------------------------------------------
# Missing observations
# ---------------------------------------------------------------------------


def test_hamilton_filter_with_missing_obs() -> None:
    model = _two_regime_model()
    observations = NILE.reshape(-1, 1)
    observations = observations.at[10, 0].set(jnp.nan)
    observations = observations.at[20, 0].set(jnp.nan)

    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)
    result = hamilton_filter(model, observations, initial_state=init)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_hamilton_filter_jit() -> None:
    model = _two_regime_model()
    observations = NILE[:20].reshape(-1, 1)
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)

    r1 = hamilton_filter(model, observations, initial_state=init)
    r2 = hamilton_filter(model, observations, initial_state=init)
    np.testing.assert_allclose(r1.log_likelihood, r2.log_likelihood, atol=1e-5)


def test_grad_through_hamilton_filter() -> None:
    """Verify autodiff works through the Hamilton filter log-likelihood."""
    observations = NILE[:30].reshape(-1, 1)
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)

    def neg_ll(log_w: jax.Array) -> jax.Array:
        w = jnp.exp(log_w) * jnp.eye(1)
        model = MarkovSwitchingSSM(
            models=(
                StateSpaceModel(
                    system_matrix=jnp.eye(1),
                    observation_matrix=jnp.eye(1),
                    evolution_cov=w,
                    obs_cov=jnp.eye(1) * 6400.0,
                ),
                StateSpaceModel(
                    system_matrix=jnp.eye(1),
                    observation_matrix=jnp.eye(1),
                    evolution_cov=w * 9.0,
                    obs_cov=jnp.eye(1) * 40000.0,
                ),
            ),
            transition_matrix=jnp.array([[0.9, 0.1], [0.1, 0.9]]),
            initial_probs=jnp.array([0.5, 0.5]),
        )
        result = hamilton_filter(model, observations, initial_state=init)
        return -result.log_likelihood

    grad_fn = jax.grad(neg_ll)
    g = grad_fn(jnp.log(jnp.array(20.0)))
    assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# Three-regime model
# ---------------------------------------------------------------------------


def test_three_regime_model() -> None:
    model = MarkovSwitchingSSM(
        models=(LocalLevel(1, 5), LocalLevel(3, 15), LocalLevel(5, 25)),
        transition_matrix=jnp.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        initial_probs=jnp.array([1 / 3, 1 / 3, 1 / 3]),
    )
    observations = NILE[:30].reshape(-1, 1)
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)
    result = hamilton_filter(model, observations, initial_state=init)

    assert result.regime_filtered_probs.shape == (30, 3)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    sums = jnp.sum(result.regime_filtered_probs, axis=1)
    np.testing.assert_allclose(sums, jnp.ones(30), atol=1e-4)
