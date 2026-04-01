"""Tests for the Bootstrap Particle Filter / Sequential Monte Carlo."""

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
from dynaris.filters.particle import (
    ParticleFilter,
    _multinomial_resample,
    _stratified_resample,
    _systematic_resample,
    _weighted_covariance,
    _weighted_mean,
    compute_log_weights,
    effective_sample_size,
    particle_filter,
    predict_particles,
)

NILE = load_nile_jax()


# ---------------------------------------------------------------------------
# Helper: linear model as NonlinearSSM (for comparison with Kalman)
# ---------------------------------------------------------------------------


def _linear_nonlinear_model(sigma_level: float = 1.0, sigma_obs: float = 1.0) -> NonlinearSSM:
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
# Resampling strategy tests
# ---------------------------------------------------------------------------


def test_multinomial_resample_shape() -> None:
    key = jax.random.PRNGKey(0)
    weights = jnp.ones(100) / 100
    indices = _multinomial_resample(key, weights, 100)
    assert indices.shape == (100,)
    assert jnp.all(indices >= 0) and jnp.all(indices < 100)


def test_systematic_resample_shape() -> None:
    key = jax.random.PRNGKey(1)
    weights = jnp.ones(100) / 100
    indices = _systematic_resample(key, weights, 100)
    assert indices.shape == (100,)
    assert jnp.all(indices >= 0) and jnp.all(indices < 100)


def test_stratified_resample_shape() -> None:
    key = jax.random.PRNGKey(2)
    weights = jnp.ones(100) / 100
    indices = _stratified_resample(key, weights, 100)
    assert indices.shape == (100,)
    assert jnp.all(indices >= 0) and jnp.all(indices < 100)


def test_systematic_resample_degenerate_weights() -> None:
    """All weight on one particle: all indices should be that particle."""
    key = jax.random.PRNGKey(3)
    weights = jnp.zeros(50).at[7].set(1.0)
    indices = _systematic_resample(key, weights, 50)
    np.testing.assert_array_equal(indices, jnp.full(50, 7))


def test_stratified_resample_degenerate_weights() -> None:
    """All weight on one particle: all indices should be that particle."""
    key = jax.random.PRNGKey(4)
    weights = jnp.zeros(50).at[3].set(1.0)
    indices = _stratified_resample(key, weights, 50)
    np.testing.assert_array_equal(indices, jnp.full(50, 3))


# ---------------------------------------------------------------------------
# Weighted statistics tests
# ---------------------------------------------------------------------------


def test_weighted_mean() -> None:
    particles = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = jnp.array([0.5, 0.3, 0.2])
    mean = _weighted_mean(particles, weights)
    expected = (
        0.5 * jnp.array([1.0, 2.0]) + 0.3 * jnp.array([3.0, 4.0]) + 0.2 * jnp.array([5.0, 6.0])
    )
    np.testing.assert_allclose(mean, expected, atol=1e-6)


def test_weighted_covariance() -> None:
    particles = jnp.array([[1.0], [3.0], [5.0]])
    weights = jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    mean = _weighted_mean(particles, weights)
    cov = _weighted_covariance(particles, weights, mean)
    # Variance of {1, 3, 5} with equal weights = E[x^2] - E[x]^2
    expected_var = (1 + 9 + 25) / 3 - 9.0  # = 8/3
    np.testing.assert_allclose(cov[0, 0], expected_var, atol=1e-5)


def test_effective_sample_size_uniform() -> None:
    """ESS of uniform weights should equal N."""
    n = 200
    weights = jnp.ones(n) / n
    ess = effective_sample_size(weights)
    np.testing.assert_allclose(ess, float(n), atol=1e-3)


def test_effective_sample_size_degenerate() -> None:
    """ESS of one-hot weights should equal 1."""
    weights = jnp.zeros(100).at[0].set(1.0)
    ess = effective_sample_size(weights)
    np.testing.assert_allclose(ess, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Predict and log-weight tests
# ---------------------------------------------------------------------------


def test_predict_particles_shape() -> None:
    model = _linear_nonlinear_model()
    key = jax.random.PRNGKey(10)
    particles = jnp.ones((50, 1)) * 5.0
    predicted = predict_particles(particles, model, key)
    assert predicted.shape == (50, 1)
    assert jnp.all(jnp.isfinite(predicted))


def test_compute_log_weights_shape() -> None:
    model = _linear_nonlinear_model()
    particles = jnp.ones((50, 1)) * 5.0
    obs = jnp.array([5.0])
    log_w = compute_log_weights(particles, obs, model)
    assert log_w.shape == (50,)
    assert jnp.all(jnp.isfinite(log_w))


def test_compute_log_weights_closer_is_higher() -> None:
    """Particles closer to the observation should have higher log-weights."""
    model = _linear_nonlinear_model(sigma_obs=1.0)
    particles = jnp.array([[5.0], [0.0]])
    obs = jnp.array([5.0])
    log_w = compute_log_weights(particles, obs, model)
    assert float(log_w[0]) > float(log_w[1])


# ---------------------------------------------------------------------------
# PF matches Kalman on linear models
# ---------------------------------------------------------------------------


def test_pf_matches_kalman_on_linear_model() -> None:
    """On a linear model, PF with many particles should approximate Kalman."""
    sigma_level, sigma_obs = 40.0, 120.0
    nl_model = _linear_nonlinear_model(sigma_level, sigma_obs)
    lin_model = _linear_ssm(sigma_level, sigma_obs)

    observations = NILE.reshape(-1, 1)
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)

    pf_result = particle_filter(
        nl_model,
        observations,
        n_particles=5000,
        key=jax.random.PRNGKey(42),
        initial_state=init,
    )
    kf_result = kalman_filter(lin_model, observations, initial_state=init)

    # After transient, filtered states should be roughly close
    np.testing.assert_allclose(
        pf_result.filtered_states[20:], kf_result.filtered_states[20:], atol=30.0
    )


# ---------------------------------------------------------------------------
# Full filter scan tests
# ---------------------------------------------------------------------------


def test_particle_filter_shapes() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = particle_filter(model, observations, n_particles=100, key=jax.random.PRNGKey(0))

    assert isinstance(result, FilterResult)
    assert result.filtered_states.shape == (100, 1)
    assert result.filtered_covariances.shape == (100, 1, 1)
    assert result.predicted_states.shape == (100, 1)
    assert result.predicted_covariances.shape == (100, 1, 1)
    assert result.log_likelihood.shape == ()


def test_particle_filter_finite() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = particle_filter(model, observations, n_particles=200, key=jax.random.PRNGKey(1))

    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.all(jnp.isfinite(result.filtered_covariances))
    assert jnp.isfinite(result.log_likelihood)


def test_particle_filter_negative_log_likelihood() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = particle_filter(model, observations, n_particles=500, key=jax.random.PRNGKey(2))
    assert float(result.log_likelihood) < 0.0


def test_particle_filter_with_missing_obs() -> None:
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    observations = observations.at[10, 0].set(jnp.nan)
    observations = observations.at[20, 0].set(jnp.nan)

    result = particle_filter(model, observations, n_particles=200, key=jax.random.PRNGKey(3))
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)
    # At NaN indices, filtered should equal predicted
    np.testing.assert_allclose(result.filtered_states[10], result.predicted_states[10], atol=1e-5)
    np.testing.assert_allclose(result.filtered_states[20], result.predicted_states[20], atol=1e-5)


# ---------------------------------------------------------------------------
# Nonlinear model tests
# ---------------------------------------------------------------------------


def test_pf_nonlinear_tracking() -> None:
    """PF should track a nonlinear state through noisy observations."""
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
    result = particle_filter(
        model,
        observations,
        n_particles=1000,
        key=k_filter,
        initial_state=init,
    )

    filtered = result.filtered_states[:, 0]
    correlation = jnp.corrcoef(jnp.stack([filtered, true_states]))[0, 1]
    assert float(correlation) > 0.6, f"Correlation {correlation} too low"
    assert jnp.all(jnp.isfinite(result.filtered_states))


def test_pf_2d_model() -> None:
    """Test PF with a 2D state, 2D observation model."""

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
    k_obs, k_filter = jax.random.split(key)
    observations = jax.random.normal(k_obs, (50, 2))

    init = GaussianState(mean=jnp.array([0.0, 0.0]), cov=jnp.eye(2) * 10.0)
    result = particle_filter(
        model,
        observations,
        n_particles=500,
        key=k_filter,
        initial_state=init,
    )

    assert result.filtered_states.shape == (50, 2)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# Resampling strategy comparison
# ---------------------------------------------------------------------------


def test_all_resampling_strategies() -> None:
    """All three resampling strategies should produce finite results."""
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:20].reshape(-1, 1)

    for strategy in ("multinomial", "systematic", "stratified"):
        result = particle_filter(
            model,
            observations,
            n_particles=200,
            key=jax.random.PRNGKey(0),
            resampling=strategy,
        )
        assert jnp.all(jnp.isfinite(result.filtered_states)), f"Non-finite states with {strategy}"
        assert jnp.isfinite(result.log_likelihood), f"Non-finite LL with {strategy}"


# ---------------------------------------------------------------------------
# JIT compatibility and stochasticity
# ---------------------------------------------------------------------------


def test_particle_filter_jit_deterministic() -> None:
    """Same key should produce identical results."""
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:20].reshape(-1, 1)
    key = jax.random.PRNGKey(99)

    r1 = particle_filter(model, observations, n_particles=100, key=key)
    r2 = particle_filter(model, observations, n_particles=100, key=key)
    np.testing.assert_allclose(r1.filtered_states, r2.filtered_states, atol=1e-5)
    np.testing.assert_allclose(r1.log_likelihood, r2.log_likelihood, atol=1e-5)


def test_particle_filter_different_keys() -> None:
    """Different keys should produce different results."""
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:20].reshape(-1, 1)

    r1 = particle_filter(model, observations, n_particles=100, key=jax.random.PRNGKey(0))
    r2 = particle_filter(model, observations, n_particles=100, key=jax.random.PRNGKey(1))
    # Results should differ (not identical)
    assert not jnp.allclose(r1.filtered_states, r2.filtered_states)


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------


def test_pf_class_scan() -> None:
    pf = ParticleFilter(n_particles=100)
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:10].reshape(-1, 1)
    result = pf.scan(model, observations, key=jax.random.PRNGKey(0))
    assert isinstance(result, FilterResult)
    assert result.filtered_states.shape == (10, 1)


def test_pf_class_custom_params() -> None:
    pf = ParticleFilter(n_particles=200, resampling="stratified")
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:10].reshape(-1, 1)
    result = pf.scan(model, observations, key=jax.random.PRNGKey(1))
    assert isinstance(result, FilterResult)
    assert jnp.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_pf_small_particle_count() -> None:
    """PF should still produce finite results with very few particles."""
    model = _linear_nonlinear_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:10].reshape(-1, 1)
    result = particle_filter(model, observations, n_particles=10, key=jax.random.PRNGKey(5))
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)
