"""Tests for the RTS smoother."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.core.protocols import SmootherProtocol
from dynaris.core.results import SmootherResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState
from dynaris.filters.kalman import kalman_filter
from dynaris.smoothers.rts import RTSSmoother, rts_smooth


# Nile data (same as filter tests)
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


def _local_level_model(
    sigma_level: float = 1.0, sigma_obs: float = 1.0
) -> StateSpaceModel:
    return StateSpaceModel(
        system_matrix=jnp.array([[1.0]]),
        observation_matrix=jnp.array([[1.0]]),
        evolution_cov=jnp.array([[sigma_level**2]]),
        obs_cov=jnp.array([[sigma_obs**2]]),
    )


def _run_filter_and_smooth(
    sigma_level: float = 40.0, sigma_obs: float = 120.0
) -> tuple[StateSpaceModel, SmootherResult]:
    model = _local_level_model(sigma_level, sigma_obs)
    observations = NILE.reshape(-1, 1)
    fr = kalman_filter(model, observations)
    sr = rts_smooth(model, fr)
    return model, sr


# ---------------------------------------------------------------------------
# Basic shape and type tests
# ---------------------------------------------------------------------------


def test_smoother_result_shapes() -> None:
    _, sr = _run_filter_and_smooth()
    assert isinstance(sr, SmootherResult)
    assert sr.smoothed_states.shape == (100, 1)
    assert sr.smoothed_covariances.shape == (100, 1, 1)
    assert sr.filtered_states.shape == (100, 1)
    assert sr.log_likelihood.shape == ()


def test_smoother_result_finite() -> None:
    _, sr = _run_filter_and_smooth()
    assert jnp.all(jnp.isfinite(sr.smoothed_states))
    assert jnp.all(jnp.isfinite(sr.smoothed_covariances))


# ---------------------------------------------------------------------------
# Smoother properties
# ---------------------------------------------------------------------------


def test_smoothed_variance_leq_filtered() -> None:
    """Smoothed covariance should be <= filtered covariance at every step."""
    _, sr = _run_filter_and_smooth()
    smoothed_var = sr.smoothed_covariances[:, 0, 0]
    filtered_var = sr.filtered_covariances[:, 0, 0]
    # Smoother uses future data, so variance should not increase
    assert jnp.all(smoothed_var <= filtered_var + 1e-5)


def test_last_smoothed_equals_last_filtered() -> None:
    """At the last time step, smoothed == filtered."""
    _, sr = _run_filter_and_smooth()
    np.testing.assert_allclose(
        sr.smoothed_states[-1], sr.filtered_states[-1], atol=1e-5
    )
    np.testing.assert_allclose(
        sr.smoothed_covariances[-1],
        sr.filtered_covariances[-1],
        atol=1e-5,
    )


def test_smoother_changes_earlier_estimates() -> None:
    """Smoothed states should differ from filtered for earlier time steps."""
    _, sr = _run_filter_and_smooth()
    # Compare early time steps - smoothed and filtered should differ
    diff = jnp.abs(sr.smoothed_states[:50] - sr.filtered_states[:50])
    assert float(jnp.max(diff)) > 1.0


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_rts_smooth_jit() -> None:
    model = _local_level_model(40.0, 120.0)
    observations = NILE[:20].reshape(-1, 1)
    fr = kalman_filter(model, observations)
    # rts_smooth is @jax.jit, calling twice checks tracing
    sr1 = rts_smooth(model, fr)
    sr2 = rts_smooth(model, fr)
    np.testing.assert_allclose(
        sr1.smoothed_states, sr2.smoothed_states, atol=1e-5
    )


def test_grad_through_smoother() -> None:
    """Autodiff should work through filter + smoother."""
    observations = NILE[:20].reshape(-1, 1)

    def smoothed_state_loss(log_q: jax.Array) -> jax.Array:
        model = StateSpaceModel(
            system_matrix=jnp.array([[1.0]]),
            observation_matrix=jnp.array([[1.0]]),
            evolution_cov=jnp.exp(log_q) * jnp.eye(1),
            obs_cov=jnp.array([[15000.0]]),
        )
        fr = kalman_filter(model, observations)
        sr = rts_smooth(model, fr)
        return jnp.sum(sr.smoothed_states**2)

    grad_fn = jax.grad(smoothed_state_loss)
    g = grad_fn(jnp.log(jnp.array(1600.0)))
    assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# Multivariate test
# ---------------------------------------------------------------------------


def test_rts_smooth_multivariate() -> None:
    model = StateSpaceModel(
        system_matrix=jnp.eye(2) * 0.99,
        observation_matrix=jnp.eye(2),
        evolution_cov=jnp.eye(2) * 0.1,
        obs_cov=jnp.eye(2) * 1.0,
    )
    key = jax.random.PRNGKey(42)
    observations = jax.random.normal(key, (50, 2))
    fr = kalman_filter(model, observations)
    sr = rts_smooth(model, fr)

    assert sr.smoothed_states.shape == (50, 2)
    assert sr.smoothed_covariances.shape == (50, 2, 2)
    assert jnp.all(jnp.isfinite(sr.smoothed_states))


# ---------------------------------------------------------------------------
# Simulated data — smoother should outperform filter
# ---------------------------------------------------------------------------


def test_smoother_more_accurate_than_filter() -> None:
    """Smoother uses future data, so MSE should be <= filter MSE."""
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    n_steps = 200

    true_states = jnp.cumsum(jax.random.normal(k1, (n_steps,)) * 1.0)
    observations = (
        true_states + jax.random.normal(k2, (n_steps,)) * 5.0
    ).reshape(-1, 1)

    model = _local_level_model(sigma_level=1.0, sigma_obs=5.0)
    fr = kalman_filter(model, observations)
    sr = rts_smooth(model, fr)

    filter_mse = float(
        jnp.mean((fr.filtered_states[:, 0] - true_states) ** 2)
    )
    smoother_mse = float(
        jnp.mean((sr.smoothed_states[:, 0] - true_states) ** 2)
    )
    assert smoother_mse <= filter_mse + 1e-3


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_rts_smoother_satisfies_protocol() -> None:
    s = RTSSmoother()
    assert isinstance(s, SmootherProtocol)


def test_rts_smoother_class_smooth() -> None:
    s = RTSSmoother()
    model = _local_level_model(40.0, 120.0)
    observations = NILE[:10].reshape(-1, 1)
    fr = kalman_filter(model, observations)
    sr = s.smooth(model, fr)
    assert isinstance(sr, SmootherResult)
    assert sr.smoothed_states.shape == (10, 1)
