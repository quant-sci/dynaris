"""Tests for the Unscented Kalman Filter."""

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
from dynaris.filters.kalman import kalman_filter
from dynaris.filters.ukf import (
    UnscentedKalmanFilter,
    compute_weights,
    sigma_points,
    ukf_filter,
    predict,
    update,
)

NILE = load_nile_jax()


# ---------------------------------------------------------------------------
# Helper: linear model as NonlinearSSM (for comparison with Kalman)
# ---------------------------------------------------------------------------


def _linear_nonlinear_model(
    sigma_level: float = 1.0, sigma_obs: float = 1.0
) -> NonlinearSSM:
    """Local-level model as a NonlinearSSM."""
    return NonlinearSSM(
        transition_fn=lambda x: x,
        observation_fn=lambda x: x,
        transition_cov=jnp.array([[sigma_level**2]]),
        observation_cov=jnp.array([[sigma_obs**2]]),
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
# Sigma-point and weight tests
# ---------------------------------------------------------------------------


def test_compute_weights_shape() -> None:
    w = compute_weights(n=3)
    assert w.wm.shape == (7,)
    assert w.wc.shape == (7,)


def test_compute_weights_sum_with_large_alpha() -> None:
    """With alpha=1, weights sum to 1."""
    w = compute_weights(n=3, alpha=1.0, kappa=0.0)
    np.testing.assert_allclose(jnp.sum(w.wm), 1.0, atol=1e-6)


def test_compute_weights_custom_params() -> None:
    w = compute_weights(n=2, alpha=0.5, beta=2.0, kappa=1.0)
    assert w.wm.shape == (5,)


def test_sigma_points_shape() -> None:
    state = GaussianState(mean=jnp.zeros(3), cov=jnp.eye(3))
    w = compute_weights(n=3)
    pts = sigma_points(state, w.lam)
    assert pts.shape == (7, 3)


def test_sigma_points_center_is_mean() -> None:
    mean = jnp.array([1.0, 2.0])
    state = GaussianState(mean=mean, cov=jnp.eye(2) * 0.5)
    w = compute_weights(n=2)
    pts = sigma_points(state, w.lam)
    np.testing.assert_allclose(pts[0], mean, atol=1e-6)


def test_sigma_points_symmetric() -> None:
    state = GaussianState(mean=jnp.array([3.0]), cov=jnp.array([[2.0]]))
    w = compute_weights(n=1)
    pts = sigma_points(state, w.lam)
    # Points 1 and 2 should be equidistant from the mean
    np.testing.assert_allclose(
        pts[1] - state.mean, -(pts[2] - state.mean), atol=1e-6
    )


def test_sigma_points_weighted_mean_recovers_mean() -> None:
    """Weighted mean of sigma points should recover the original mean."""
    mean = jnp.array([2.0, -1.0])
    cov = jnp.array([[1.0, 0.3], [0.3, 0.5]])
    state = GaussianState(mean=mean, cov=cov)
    w = compute_weights(n=2, alpha=1.0)
    pts = sigma_points(state, w.lam)
    recovered = jnp.sum(w.wm[:, None] * pts, axis=0)
    np.testing.assert_allclose(recovered, mean, atol=1e-5)


# ---------------------------------------------------------------------------
# Predict step tests
# ---------------------------------------------------------------------------


def test_predict_identity_transition() -> None:
    model = _linear_nonlinear_model()
    state = GaussianState(mean=jnp.array([5.0]), cov=jnp.array([[2.0]]))
    w = compute_weights(n=1, alpha=1.0)
    pred = predict(state, model, w)
    # Identity transition: mean unchanged, cov = P + Q
    np.testing.assert_allclose(pred.mean, [5.0], atol=1e-4)
    np.testing.assert_allclose(pred.cov, [[3.0]], atol=1e-3)


def test_predict_nonlinear_transition() -> None:
    def f(x: Array) -> Array:
        return x + 0.1 * jnp.sin(x)

    model = NonlinearSSM(
        transition_fn=f,
        observation_fn=lambda x: x,
        transition_cov=jnp.array([[0.5]]),
        observation_cov=jnp.array([[1.0]]),
        state_dim=1,
        obs_dim=1,
    )
    state = GaussianState(mean=jnp.array([1.0]), cov=jnp.array([[0.001]]))
    w = compute_weights(n=1, alpha=1.0)
    pred = predict(state, model, w)
    expected = 1.0 + 0.1 * float(jnp.sin(1.0))
    np.testing.assert_allclose(pred.mean, [expected], atol=0.01)
    assert jnp.all(jnp.isfinite(pred.cov))


# ---------------------------------------------------------------------------
# Update step tests
# ---------------------------------------------------------------------------


def test_update_reduces_uncertainty() -> None:
    model = _linear_nonlinear_model(sigma_level=1.0, sigma_obs=1.0)
    predicted = GaussianState(mean=jnp.array([0.0]), cov=jnp.array([[10.0]]))
    obs = jnp.array([5.0])
    w = compute_weights(n=1)
    filtered, ll = update(predicted, obs, model, w)
    assert float(filtered.cov[0, 0]) < 10.0
    assert float(filtered.mean[0]) > 0.0
    assert jnp.isfinite(ll)


def test_update_nan_skips() -> None:
    model = _linear_nonlinear_model()
    predicted = GaussianState(mean=jnp.array([3.0]), cov=jnp.array([[2.0]]))
    obs = jnp.array([jnp.nan])
    w = compute_weights(n=1)
    filtered, ll = update(predicted, obs, model, w)
    np.testing.assert_allclose(filtered.mean, predicted.mean)
    np.testing.assert_allclose(filtered.cov, predicted.cov)
    assert float(ll) == 0.0


# ---------------------------------------------------------------------------
# UKF matches Kalman on linear models
# ---------------------------------------------------------------------------


def test_ukf_matches_kalman_on_linear_model() -> None:
    """When the model is linear, UKF should produce similar results to Kalman.

    The UKF with default alpha=1e-3 and a diffuse prior can diverge slightly
    due to the extreme scaling. With alpha=1 and a tighter prior, the match
    is close.
    """
    sigma_level, sigma_obs = 40.0, 120.0
    nl_model = _linear_nonlinear_model(sigma_level, sigma_obs)
    lin_model = _linear_ssm(sigma_level, sigma_obs)

    observations = NILE.reshape(-1, 1)

    # Use a tighter (non-diffuse) initial state for better UKF-Kalman agreement
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)

    ukf_result = ukf_filter(nl_model, observations, initial_state=init, alpha=1.0)
    kf_result = kalman_filter(lin_model, observations, initial_state=init)

    # After initial transient, filtered states should be very close
    np.testing.assert_allclose(
        ukf_result.filtered_states[10:], kf_result.filtered_states[10:], atol=0.5
    )
    np.testing.assert_allclose(
        ukf_result.log_likelihood, kf_result.log_likelihood, atol=5.0
    )


# ---------------------------------------------------------------------------
# Full filter scan tests
# ---------------------------------------------------------------------------


def test_ukf_filter_shapes() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = ukf_filter(model, observations)

    assert isinstance(result, FilterResult)
    assert result.filtered_states.shape == (100, 1)
    assert result.filtered_covariances.shape == (100, 1, 1)
    assert result.predicted_states.shape == (100, 1)
    assert result.predicted_covariances.shape == (100, 1, 1)
    assert result.log_likelihood.shape == ()


def test_ukf_filter_finite() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = ukf_filter(model, observations)

    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.all(jnp.isfinite(result.filtered_covariances))
    assert jnp.isfinite(result.log_likelihood)


def test_ukf_filter_negative_log_likelihood() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = ukf_filter(model, observations)
    assert float(result.log_likelihood) < 0.0


def test_ukf_filter_with_missing_obs() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    observations = observations.at[10, 0].set(jnp.nan)
    observations = observations.at[20, 0].set(jnp.nan)

    result = ukf_filter(model, observations)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)
    np.testing.assert_allclose(
        result.filtered_states[10], result.predicted_states[10], atol=1e-5
    )


# ---------------------------------------------------------------------------
# Nonlinear model tests
# ---------------------------------------------------------------------------


def test_ukf_nonlinear_tracking() -> None:
    """UKF should track a nonlinear state through noisy observations."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
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
    result = ukf_filter(model, observations, initial_state=init)

    filtered = result.filtered_states[:, 0]
    correlation = jnp.corrcoef(jnp.stack([filtered, true_states]))[0, 1]
    assert float(correlation) > 0.7, f"Correlation {correlation} too low"
    assert jnp.all(jnp.isfinite(result.filtered_states))


def test_ukf_2d_model() -> None:
    """Test UKF with a 2D state, 2D observation model."""

    def f(x: Array) -> Array:
        return x * 0.99

    def h(x: Array) -> Array:
        return x

    model = NonlinearSSM(
        transition_fn=f,
        observation_fn=h,
        transition_cov=jnp.eye(2) * 0.1,
        observation_cov=jnp.eye(2) * 1.0,
        state_dim=2,
        obs_dim=2,
    )

    key = jax.random.PRNGKey(7)
    observations = jax.random.normal(key, (50, 2))

    init = GaussianState(mean=jnp.array([0.0, 0.0]), cov=jnp.eye(2) * 10.0)
    result = ukf_filter(model, observations, initial_state=init, alpha=1.0)

    assert result.filtered_states.shape == (50, 2)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------


def test_ukf_custom_alpha_beta_kappa() -> None:
    """UKF should work with custom sigma-point parameters."""
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:20].reshape(-1, 1)

    result = ukf_filter(model, observations, alpha=0.5, beta=2.0, kappa=1.0)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_ukf_filter_jit() -> None:
    """Verify ukf_filter works with JIT compilation."""
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:20].reshape(-1, 1)
    r1 = ukf_filter(model, observations)
    r2 = ukf_filter(model, observations)
    np.testing.assert_allclose(r1.log_likelihood, r2.log_likelihood, atol=1e-5)


def test_grad_through_ukf() -> None:
    """Verify autodiff works through the UKF log-likelihood."""
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
        result = ukf_filter(model, observations)
        return -result.log_likelihood

    grad_fn = jax.grad(neg_ll, argnums=(0, 1))
    g1, g2 = grad_fn(jnp.log(jnp.array(1600.0)), jnp.log(jnp.array(15000.0)))
    assert jnp.isfinite(g1)
    assert jnp.isfinite(g2)


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------


def test_ukf_class_scan() -> None:
    ukf = UnscentedKalmanFilter()
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:10].reshape(-1, 1)
    result = ukf.scan(model, observations)
    assert isinstance(result, FilterResult)
    assert result.filtered_states.shape == (10, 1)


def test_ukf_class_custom_params() -> None:
    ukf = UnscentedKalmanFilter(alpha=0.5, beta=2.0, kappa=1.0)
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:10].reshape(-1, 1)
    result = ukf.scan(model, observations)
    assert isinstance(result, FilterResult)
    assert jnp.isfinite(result.log_likelihood)
