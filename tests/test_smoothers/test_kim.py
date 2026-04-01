"""Tests for the Kim smoother for Markov-switching models."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dynaris.core.results import SwitchingSmootherResult
from dynaris.core.switching import MarkovSwitchingSSM
from dynaris.core.types import GaussianState
from dynaris.datasets import load_nile_jax
from dynaris.dlm.components import LocalLevel
from dynaris.filters.hamilton import hamilton_filter
from dynaris.filters.kalman import kalman_filter
from dynaris.smoothers.kim import kim_smooth
from dynaris.smoothers.rts import rts_smooth

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


_INIT = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)


# ---------------------------------------------------------------------------
# Shape and type tests
# ---------------------------------------------------------------------------


def test_kim_smoother_shapes() -> None:
    model = _two_regime_model()
    observations = NILE.reshape(-1, 1)
    fr = hamilton_filter(model, observations, initial_state=_INIT)
    result = kim_smooth(model, fr)

    assert isinstance(result, SwitchingSmootherResult)
    assert result.smoothed_states.shape == (100, 1)
    assert result.smoothed_covariances.shape == (100, 1, 1)
    assert result.regime_smoothed_probs.shape == (100, 2)
    assert result.regime_smoothed_states.shape == (100, 2, 1)
    assert result.regime_smoothed_covs.shape == (100, 2, 1, 1)


def test_kim_smoother_finite() -> None:
    model = _two_regime_model()
    observations = NILE.reshape(-1, 1)
    fr = hamilton_filter(model, observations, initial_state=_INIT)
    result = kim_smooth(model, fr)

    assert jnp.all(jnp.isfinite(result.smoothed_states))
    assert jnp.all(jnp.isfinite(result.smoothed_covariances))
    assert jnp.all(jnp.isfinite(result.regime_smoothed_probs))


def test_smoothed_regime_probs_sum_to_one() -> None:
    model = _two_regime_model()
    observations = NILE.reshape(-1, 1)
    fr = hamilton_filter(model, observations, initial_state=_INIT)
    result = kim_smooth(model, fr)

    sums = jnp.sum(result.regime_smoothed_probs, axis=1)
    np.testing.assert_allclose(sums, jnp.ones(100), atol=1e-4)


def test_last_smoothed_equals_last_filtered() -> None:
    model = _two_regime_model()
    observations = NILE.reshape(-1, 1)
    fr = hamilton_filter(model, observations, initial_state=_INIT)
    result = kim_smooth(model, fr)

    np.testing.assert_allclose(result.smoothed_states[-1], result.filtered_states[-1], atol=1e-4)


# ---------------------------------------------------------------------------
# K=1 matches RTS smoother
# ---------------------------------------------------------------------------


def test_single_regime_matches_rts() -> None:
    """With K=1, Kim smoother should match standard RTS smoother."""
    single_model = MarkovSwitchingSSM(
        models=(LocalLevel(sigma_level=40.0, sigma_obs=120.0),),
        transition_matrix=jnp.array([[1.0]]),
        initial_probs=jnp.array([1.0]),
    )
    linear_model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)

    ham_result = hamilton_filter(single_model, observations, initial_state=_INIT)
    kim_result = kim_smooth(single_model, ham_result)

    kf_result = kalman_filter(linear_model, observations, initial_state=_INIT)
    rts_result = rts_smooth(linear_model, kf_result)

    np.testing.assert_allclose(kim_result.smoothed_states, rts_result.smoothed_states, atol=1e-2)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_kim_smoother_jit() -> None:
    model = _two_regime_model()
    observations = NILE[:20].reshape(-1, 1)
    fr = hamilton_filter(model, observations, initial_state=_INIT)

    r1 = kim_smooth(model, fr)
    r2 = kim_smooth(model, fr)
    np.testing.assert_allclose(r1.smoothed_states, r2.smoothed_states, atol=1e-5)


# ---------------------------------------------------------------------------
# Three-regime model
# ---------------------------------------------------------------------------


def test_three_regime_smoother() -> None:
    model = MarkovSwitchingSSM(
        models=(LocalLevel(1, 5), LocalLevel(3, 15), LocalLevel(5, 25)),
        transition_matrix=jnp.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        initial_probs=jnp.array([1 / 3, 1 / 3, 1 / 3]),
    )
    observations = NILE[:30].reshape(-1, 1)
    fr = hamilton_filter(model, observations, initial_state=_INIT)
    result = kim_smooth(model, fr)

    assert result.regime_smoothed_probs.shape == (30, 3)
    sums = jnp.sum(result.regime_smoothed_probs, axis=1)
    np.testing.assert_allclose(sums, jnp.ones(30), atol=1e-4)
