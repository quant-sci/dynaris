"""Comparative tests: all filters on the same models with appropriate tolerances.

Verifies that Kalman, EKF, UKF, and Particle filters produce consistent
results on identical problems, with tolerances reflecting each filter's
approximation quality.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dynaris.core.nonlinear import NonlinearSSM
from dynaris.core.results import FilterResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState
from dynaris.datasets import load_nile_jax
from dynaris.filters.ekf import ekf_filter
from dynaris.filters.kalman import kalman_filter
from dynaris.filters.particle import particle_filter
from dynaris.filters.ukf import ukf_filter

NILE = load_nile_jax()

# ---------------------------------------------------------------------------
# Shared model helpers
# ---------------------------------------------------------------------------

_SIGMA_LEVEL = 40.0
_SIGMA_OBS = 120.0


def _linear_ssm() -> StateSpaceModel:
    return StateSpaceModel(
        system_matrix=jnp.array([[1.0]]),
        observation_matrix=jnp.array([[1.0]]),
        evolution_cov=jnp.array([[_SIGMA_LEVEL**2]]),
        obs_cov=jnp.array([[_SIGMA_OBS**2]]),
    )


def _linear_nonlinear() -> NonlinearSSM:
    return NonlinearSSM(
        transition_fn=lambda x: x,
        observation_fn=lambda x: x,
        transition_cov=jnp.array([[_SIGMA_LEVEL**2]]),
        observation_cov=jnp.array([[_SIGMA_OBS**2]]),
        state_dim=1,
        obs_dim=1,
    )


_INIT = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)


# ---------------------------------------------------------------------------
# All filters return FilterResult with correct shapes
# ---------------------------------------------------------------------------


def test_all_filters_return_filter_result() -> None:
    observations = NILE[:30].reshape(-1, 1)
    lin_model = _linear_ssm()
    nl_model = _linear_nonlinear()

    kf = kalman_filter(lin_model, observations, initial_state=_INIT)
    ekf = ekf_filter(nl_model, observations, initial_state=_INIT)
    ukf = ukf_filter(nl_model, observations, initial_state=_INIT, alpha=1.0)
    pf = particle_filter(
        nl_model,
        observations,
        n_particles=200,
        key=jax.random.PRNGKey(0),
        initial_state=_INIT,
    )

    for name, result in [("KF", kf), ("EKF", ekf), ("UKF", ukf), ("PF", pf)]:
        assert isinstance(result, FilterResult), f"{name} did not return FilterResult"
        assert result.filtered_states.shape == (30, 1), f"{name} filtered shape wrong"
        assert result.filtered_covariances.shape == (30, 1, 1), f"{name} cov shape wrong"
        assert result.predicted_states.shape == (30, 1), f"{name} predicted shape wrong"
        assert result.log_likelihood.shape == (), f"{name} LL shape wrong"


# ---------------------------------------------------------------------------
# Linear model: EKF ≈ KF (exact), UKF ≈ KF (close), PF ≈ KF (loose)
# ---------------------------------------------------------------------------


def test_linear_model_ekf_matches_kalman() -> None:
    """EKF on a linear model should match Kalman filter exactly."""
    observations = NILE.reshape(-1, 1)
    kf = kalman_filter(_linear_ssm(), observations, initial_state=_INIT)
    ekf = ekf_filter(_linear_nonlinear(), observations, initial_state=_INIT)

    np.testing.assert_allclose(ekf.filtered_states, kf.filtered_states, atol=1e-4)
    np.testing.assert_allclose(ekf.log_likelihood, kf.log_likelihood, atol=1e-2)


def test_linear_model_ukf_matches_kalman() -> None:
    """UKF on a linear model should closely match Kalman filter."""
    observations = NILE.reshape(-1, 1)
    kf = kalman_filter(_linear_ssm(), observations, initial_state=_INIT)
    ukf = ukf_filter(_linear_nonlinear(), observations, initial_state=_INIT, alpha=1.0)

    np.testing.assert_allclose(ukf.filtered_states[10:], kf.filtered_states[10:], atol=0.5)
    np.testing.assert_allclose(ukf.log_likelihood, kf.log_likelihood, atol=5.0)


def test_linear_model_pf_matches_kalman() -> None:
    """PF on a linear model should roughly match Kalman filter."""
    observations = NILE.reshape(-1, 1)
    kf = kalman_filter(_linear_ssm(), observations, initial_state=_INIT)
    pf = particle_filter(
        _linear_nonlinear(),
        observations,
        n_particles=5000,
        key=jax.random.PRNGKey(42),
        initial_state=_INIT,
    )

    np.testing.assert_allclose(pf.filtered_states[20:], kf.filtered_states[20:], atol=30.0)


# ---------------------------------------------------------------------------
# Nonlinear model: all nonlinear filters track
# ---------------------------------------------------------------------------


def test_nonlinear_all_filters_track() -> None:
    """EKF, UKF, and PF should all track a nonlinear state."""
    key = jax.random.PRNGKey(42)
    k1, k2, k_filter = jax.random.split(key, 3)
    n_steps = 200

    def f(x: Array) -> Array:
        return 0.95 * x + 0.1 * jnp.sin(x)

    def h(x: Array) -> Array:
        return x

    sigma_q, sigma_r = 0.5, 1.0
    state_noise = jax.random.normal(k1, (n_steps,)) * sigma_q
    obs_noise = jax.random.normal(k2, (n_steps,)) * sigma_r

    state = jnp.array([5.0])
    states_list = []
    for t in range(n_steps):
        state = f(state) + state_noise[t : t + 1]
        states_list.append(state)
    true_states = jnp.concatenate(states_list)
    observations = (true_states + obs_noise).reshape(-1, 1)

    model = NonlinearSSM(
        transition_fn=f,
        observation_fn=h,
        transition_cov=jnp.array([[sigma_q**2]]),
        observation_cov=jnp.array([[sigma_r**2]]),
        state_dim=1,
        obs_dim=1,
    )
    init = GaussianState(mean=jnp.array([5.0]), cov=jnp.array([[1.0]]))

    ekf_res = ekf_filter(model, observations, initial_state=init)
    ukf_res = ukf_filter(model, observations, initial_state=init)
    pf_res = particle_filter(
        model,
        observations,
        n_particles=1000,
        key=k_filter,
        initial_state=init,
    )

    for name, result in [("EKF", ekf_res), ("UKF", ukf_res), ("PF", pf_res)]:
        filtered = result.filtered_states[:, 0]
        corr = float(jnp.corrcoef(jnp.stack([filtered, true_states]))[0, 1])
        assert corr > 0.6, f"{name} correlation {corr:.3f} too low"


# ---------------------------------------------------------------------------
# All filters handle missing observations
# ---------------------------------------------------------------------------


def test_all_filters_handle_nan() -> None:
    observations = NILE.reshape(-1, 1)
    observations = observations.at[10, 0].set(jnp.nan)
    observations = observations.at[30, 0].set(jnp.nan)

    kf = kalman_filter(_linear_ssm(), observations, initial_state=_INIT)
    ekf = ekf_filter(_linear_nonlinear(), observations, initial_state=_INIT)
    ukf = ukf_filter(_linear_nonlinear(), observations, initial_state=_INIT, alpha=1.0)
    pf = particle_filter(
        _linear_nonlinear(),
        observations,
        n_particles=200,
        key=jax.random.PRNGKey(0),
        initial_state=_INIT,
    )

    for name, result in [("KF", kf), ("EKF", ekf), ("UKF", ukf), ("PF", pf)]:
        assert jnp.all(jnp.isfinite(result.filtered_states)), (
            f"{name} has non-finite states with NaN obs"
        )
        assert jnp.isfinite(result.log_likelihood), f"{name} has non-finite LL with NaN obs"


# ---------------------------------------------------------------------------
# All filters produce negative log-likelihood on real data
# ---------------------------------------------------------------------------


def test_all_filters_negative_ll() -> None:
    observations = NILE.reshape(-1, 1)

    kf = kalman_filter(_linear_ssm(), observations, initial_state=_INIT)
    ekf = ekf_filter(_linear_nonlinear(), observations, initial_state=_INIT)
    ukf = ukf_filter(_linear_nonlinear(), observations, initial_state=_INIT, alpha=1.0)
    pf = particle_filter(
        _linear_nonlinear(),
        observations,
        n_particles=500,
        key=jax.random.PRNGKey(0),
        initial_state=_INIT,
    )

    for name, result in [("KF", kf), ("EKF", ekf), ("UKF", ukf), ("PF", pf)]:
        assert float(result.log_likelihood) < 0.0, f"{name} log-likelihood should be negative"
