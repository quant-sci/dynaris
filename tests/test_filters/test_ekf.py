"""Tests for the Extended Kalman Filter."""

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
from dynaris.filters.ekf import ExtendedKalmanFilter, ekf_filter, predict, update
from dynaris.filters.kalman import kalman_filter

NILE = load_nile_jax()


# ---------------------------------------------------------------------------
# Helper: linear model expressed as NonlinearSSM (for comparison with Kalman)
# ---------------------------------------------------------------------------


def _linear_nonlinear_model(
    sigma_level: float = 1.0, sigma_obs: float = 1.0
) -> NonlinearSSM:
    """Local-level model as a NonlinearSSM (identity transition/observation)."""
    Q = jnp.array([[sigma_level**2]])
    R = jnp.array([[sigma_obs**2]])

    def f(x: Array) -> Array:
        return x

    def h(x: Array) -> Array:
        return x

    return NonlinearSSM(
        transition_fn=f,
        observation_fn=h,
        transition_cov=Q,
        observation_cov=R,
        state_dim=1,
        obs_dim=1,
    )


def _linear_ssm(sigma_level: float = 1.0, sigma_obs: float = 1.0) -> StateSpaceModel:
    """Equivalent linear model for Kalman filter comparison."""
    return StateSpaceModel(
        system_matrix=jnp.array([[1.0]]),
        observation_matrix=jnp.array([[1.0]]),
        evolution_cov=jnp.array([[sigma_level**2]]),
        obs_cov=jnp.array([[sigma_obs**2]]),
    )


# ---------------------------------------------------------------------------
# Predict step tests
# ---------------------------------------------------------------------------


def test_predict_identity_transition() -> None:
    model = _linear_nonlinear_model()
    state = GaussianState(mean=jnp.array([5.0]), cov=jnp.array([[2.0]]))
    pred = predict(state, model)
    # Identity transition: mean unchanged, cov = P + Q
    np.testing.assert_allclose(pred.mean, [5.0], atol=1e-6)
    np.testing.assert_allclose(pred.cov, [[3.0]], atol=1e-6)


def test_predict_nonlinear_transition() -> None:
    """Test with a nonlinear transition: f(x) = x + 0.1 * sin(x)."""
    Q = jnp.array([[0.5]])

    def f(x: Array) -> Array:
        return x + 0.1 * jnp.sin(x)

    model = NonlinearSSM(
        transition_fn=f,
        observation_fn=lambda x: x,
        transition_cov=Q,
        observation_cov=jnp.array([[1.0]]),
        state_dim=1,
        obs_dim=1,
    )
    state = GaussianState(mean=jnp.array([1.0]), cov=jnp.array([[0.5]]))
    pred = predict(state, model)
    # Mean should be f(1.0) = 1.0 + 0.1 * sin(1.0)
    expected_mean = 1.0 + 0.1 * float(jnp.sin(1.0))
    np.testing.assert_allclose(pred.mean, [expected_mean], atol=1e-5)
    assert jnp.all(jnp.isfinite(pred.cov))


# ---------------------------------------------------------------------------
# Update step tests
# ---------------------------------------------------------------------------


def test_update_reduces_uncertainty() -> None:
    model = _linear_nonlinear_model(sigma_level=1.0, sigma_obs=1.0)
    predicted = GaussianState(mean=jnp.array([0.0]), cov=jnp.array([[10.0]]))
    obs = jnp.array([5.0])
    filtered, ll = update(predicted, obs, model)
    assert float(filtered.cov[0, 0]) < 10.0
    assert float(filtered.mean[0]) > 0.0
    assert jnp.isfinite(ll)


def test_update_nan_skips() -> None:
    model = _linear_nonlinear_model()
    predicted = GaussianState(mean=jnp.array([3.0]), cov=jnp.array([[2.0]]))
    obs = jnp.array([jnp.nan])
    filtered, ll = update(predicted, obs, model)
    np.testing.assert_allclose(filtered.mean, predicted.mean)
    np.testing.assert_allclose(filtered.cov, predicted.cov)
    assert float(ll) == 0.0


# ---------------------------------------------------------------------------
# EKF matches Kalman on linear models
# ---------------------------------------------------------------------------


def test_ekf_matches_kalman_on_linear_model() -> None:
    """When the model is linear, EKF should produce identical results to Kalman."""
    sigma_level, sigma_obs = 40.0, 120.0
    nl_model = _linear_nonlinear_model(sigma_level, sigma_obs)
    lin_model = _linear_ssm(sigma_level, sigma_obs)

    observations = NILE.reshape(-1, 1)

    # Use same initial state for both
    init = GaussianState(
        mean=jnp.zeros(1),
        cov=jnp.eye(1) * 1e6,
    )

    ekf_result = ekf_filter(nl_model, observations, initial_state=init)
    kf_result = kalman_filter(lin_model, observations, initial_state=init)

    np.testing.assert_allclose(
        ekf_result.filtered_states, kf_result.filtered_states, atol=1e-4
    )
    np.testing.assert_allclose(
        ekf_result.filtered_covariances, kf_result.filtered_covariances, atol=1e-3
    )
    np.testing.assert_allclose(
        ekf_result.log_likelihood, kf_result.log_likelihood, atol=1e-2
    )


# ---------------------------------------------------------------------------
# Full filter scan tests
# ---------------------------------------------------------------------------


def test_ekf_filter_shapes() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = ekf_filter(model, observations)

    assert isinstance(result, FilterResult)
    assert result.filtered_states.shape == (100, 1)
    assert result.filtered_covariances.shape == (100, 1, 1)
    assert result.predicted_states.shape == (100, 1)
    assert result.predicted_covariances.shape == (100, 1, 1)
    assert result.log_likelihood.shape == ()
    assert result.observations.shape == (100, 1)


def test_ekf_filter_finite() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = ekf_filter(model, observations)

    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.all(jnp.isfinite(result.filtered_covariances))
    assert jnp.isfinite(result.log_likelihood)


def test_ekf_filter_negative_log_likelihood() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = ekf_filter(model, observations)
    assert float(result.log_likelihood) < 0.0


def test_ekf_filter_with_missing_obs() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    observations = observations.at[10, 0].set(jnp.nan)
    observations = observations.at[20, 0].set(jnp.nan)

    result = ekf_filter(model, observations)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)

    # At NaN points, predicted == filtered
    np.testing.assert_allclose(
        result.filtered_states[10], result.predicted_states[10], atol=1e-5
    )


# ---------------------------------------------------------------------------
# Nonlinear model tests
# ---------------------------------------------------------------------------


def test_ekf_nonlinear_tracking() -> None:
    """EKF should track a nonlinear state through noisy observations."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    n_steps = 200

    # Mild nonlinear transition: x_t = 0.95 * x_{t-1} + 0.1 * sin(x_{t-1})
    def f(x: Array) -> Array:
        return 0.95 * x + 0.1 * jnp.sin(x)

    def h(x: Array) -> Array:
        return x

    # Simulate true states and observations
    sigma_q, sigma_r = 0.5, 1.0
    state_noise = jax.random.normal(k1, (n_steps,)) * sigma_q
    obs_noise = jax.random.normal(k2, (n_steps,)) * sigma_r

    # Build true state sequence
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
    result = ekf_filter(model, observations, initial_state=init)

    # Filtered states should track the true states well
    filtered = result.filtered_states[:, 0]
    correlation = jnp.corrcoef(jnp.stack([filtered, true_states]))[0, 1]
    assert float(correlation) > 0.7, f"Correlation {correlation} too low"
    assert jnp.all(jnp.isfinite(result.filtered_states))


def test_ekf_2d_nonlinear() -> None:
    """Test EKF with a 2D nonlinear model (polar-to-cartesian observation)."""

    def f(x: Array) -> Array:
        # Near-constant velocity
        return x * 0.99

    def h(x: Array) -> Array:
        # Observe range and bearing (nonlinear observation)
        r = jnp.sqrt(x[0] ** 2 + x[1] ** 2)
        theta = jnp.arctan2(x[1], x[0])
        return jnp.array([r, theta])

    model = NonlinearSSM(
        transition_fn=f,
        observation_fn=h,
        transition_cov=jnp.eye(2) * 0.1,
        observation_cov=jnp.eye(2) * 0.01,
        state_dim=2,
        obs_dim=2,
    )

    key = jax.random.PRNGKey(7)
    # Simulate observations from a known trajectory
    true_state = jnp.array([3.0, 4.0])
    obs_list = []
    for t in range(50):
        true_state = f(true_state) + jax.random.normal(key, (2,)) * 0.01
        key, _ = jax.random.split(key)
        obs = h(true_state) + jax.random.normal(key, (2,)) * 0.1
        key, _ = jax.random.split(key)
        obs_list.append(obs)
    observations = jnp.stack(obs_list)

    init = GaussianState(mean=jnp.array([3.0, 4.0]), cov=jnp.eye(2) * 1.0)
    result = ekf_filter(model, observations, initial_state=init)

    assert result.filtered_states.shape == (50, 2)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_ekf_filter_jit() -> None:
    """Verify ekf_filter is JIT-compiled without errors."""
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:20].reshape(-1, 1)
    r1 = ekf_filter(model, observations)
    r2 = ekf_filter(model, observations)
    np.testing.assert_allclose(r1.log_likelihood, r2.log_likelihood, atol=1e-5)


def test_grad_through_ekf() -> None:
    """Verify autodiff works through the EKF log-likelihood."""
    observations = NILE[:20].reshape(-1, 1)

    def neg_ll(log_sigma_level: Array, log_sigma_obs: Array) -> Array:
        Q = jnp.exp(log_sigma_level) * jnp.eye(1)
        R = jnp.exp(log_sigma_obs) * jnp.eye(1)
        model = NonlinearSSM(
            transition_fn=lambda x: x,
            observation_fn=lambda x: x,
            transition_cov=Q,
            observation_cov=R,
            state_dim=1,
            obs_dim=1,
        )
        result = ekf_filter(model, observations)
        return -result.log_likelihood

    grad_fn = jax.grad(neg_ll, argnums=(0, 1))
    g1, g2 = grad_fn(jnp.log(jnp.array(1600.0)), jnp.log(jnp.array(15000.0)))
    assert jnp.isfinite(g1)
    assert jnp.isfinite(g2)


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------


def test_ekf_class_scan() -> None:
    ekf = ExtendedKalmanFilter()
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:10].reshape(-1, 1)
    result = ekf.scan(model, observations)
    assert isinstance(result, FilterResult)
    assert result.filtered_states.shape == (10, 1)


# ---------------------------------------------------------------------------
# NonlinearSSM tests
# ---------------------------------------------------------------------------


def test_nonlinear_ssm_repr() -> None:
    model = _linear_nonlinear_model()
    assert "NonlinearSSM" in repr(model)
    assert "state_dim=1" in repr(model)


def test_nonlinear_ssm_initial_state() -> None:
    model = _linear_nonlinear_model()
    init = model.initial_state()
    assert init.mean.shape == (1,)
    assert init.cov.shape == (1, 1)
    np.testing.assert_allclose(init.mean, [0.0])
    assert float(init.cov[0, 0]) == 1e6


def test_nonlinear_ssm_pytree() -> None:
    """NonlinearSSM should be a valid JAX pytree."""
    model = _linear_nonlinear_model(sigma_level=2.0, sigma_obs=3.0)
    leaves, treedef = jax.tree_util.tree_flatten(model)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(reconstructed.Q, model.Q)
    np.testing.assert_allclose(reconstructed.R, model.R)
    assert reconstructed.state_dim == model.state_dim
