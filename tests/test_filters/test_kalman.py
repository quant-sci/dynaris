"""Tests for the Kalman filter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dynaris.core.protocols import FilterProtocol
from dynaris.core.results import FilterResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState
from dynaris.filters.kalman import KalmanFilter, kalman_filter, predict, update
from dynaris.datasets import load_nile_jax


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Nile river annual flow data (1871-1970), a classic DLM benchmark.
# Source: Durbin & Koopman (2012), "Time Series Analysis by State Space Methods"
NILE = load_nile_jax()


def _local_level_model(
    sigma_level: float = 1.0, sigma_obs: float = 1.0
) -> StateSpaceModel:
    """Simple local-level (random walk + noise) model."""
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
    model = _local_level_model()
    state = GaussianState(mean=jnp.array([5.0]), cov=jnp.array([[2.0]]))
    pred = predict(state, model)
    # F=I, so mean unchanged, cov = P + Q
    np.testing.assert_allclose(pred.mean, [5.0], atol=1e-6)
    np.testing.assert_allclose(pred.cov, [[3.0]], atol=1e-6)


def test_predict_with_control_input() -> None:
    model = StateSpaceModel(
        system_matrix=jnp.array([[1.0]]),
        observation_matrix=jnp.array([[1.0]]),
        evolution_cov=jnp.array([[0.1]]),
        obs_cov=jnp.array([[1.0]]),
        input_matrix=jnp.array([[2.0]]),
    )
    state = GaussianState(mean=jnp.array([0.0]), cov=jnp.array([[1.0]]))
    u = jnp.array([3.0])
    pred = predict(state, model, u)
    # mean = F @ x + B @ u = 0 + 2*3 = 6
    np.testing.assert_allclose(pred.mean, [6.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Update step tests
# ---------------------------------------------------------------------------


def test_update_reduces_uncertainty() -> None:
    model = _local_level_model(sigma_level=1.0, sigma_obs=1.0)
    predicted = GaussianState(
        mean=jnp.array([0.0]), cov=jnp.array([[10.0]])
    )
    obs = jnp.array([5.0])
    filtered, ll = update(predicted, obs, model)
    # After update, variance should decrease
    assert float(filtered.cov[0, 0]) < 10.0
    # Mean should move toward observation
    assert float(filtered.mean[0]) > 0.0
    # Log-likelihood should be finite
    assert jnp.isfinite(ll)


def test_update_nan_skips() -> None:
    model = _local_level_model()
    predicted = GaussianState(
        mean=jnp.array([3.0]), cov=jnp.array([[2.0]])
    )
    obs = jnp.array([jnp.nan])
    filtered, ll = update(predicted, obs, model)
    # NaN obs: filtered should equal predicted
    np.testing.assert_allclose(filtered.mean, predicted.mean)
    np.testing.assert_allclose(filtered.cov, predicted.cov)
    assert float(ll) == 0.0


# ---------------------------------------------------------------------------
# Full filter scan tests
# ---------------------------------------------------------------------------


def test_kalman_filter_shapes() -> None:
    model = _local_level_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = kalman_filter(model, observations)

    assert isinstance(result, FilterResult)
    assert result.filtered_states.shape == (100, 1)
    assert result.filtered_covariances.shape == (100, 1, 1)
    assert result.predicted_states.shape == (100, 1)
    assert result.predicted_covariances.shape == (100, 1, 1)
    assert result.log_likelihood.shape == ()
    assert result.observations.shape == (100, 1)


def test_kalman_filter_finite() -> None:
    model = _local_level_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = kalman_filter(model, observations)

    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.all(jnp.isfinite(result.filtered_covariances))
    assert jnp.isfinite(result.log_likelihood)


def test_kalman_filter_negative_log_likelihood() -> None:
    model = _local_level_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1)
    result = kalman_filter(model, observations)
    # Log-likelihood should be negative for real data
    assert float(result.log_likelihood) < 0.0


def test_kalman_filter_nile_known_values() -> None:
    """Test against known Nile local-level results.

    With sigma_level^2 = 1469.1 and sigma_obs^2 = 15099,
    the MLE log-likelihood is approximately -632.5 (Durbin & Koopman).
    We use approximate values and check the filter is in the right ballpark.
    """
    sigma_level = jnp.sqrt(1469.1)
    sigma_obs = jnp.sqrt(15099.0)
    model = _local_level_model(
        sigma_level=float(sigma_level), sigma_obs=float(sigma_obs)
    )
    observations = NILE.reshape(-1, 1)
    result = kalman_filter(model, observations)

    # Log-likelihood near -632.5 (MLE optimum); with diffuse prior
    # the value is slightly lower due to the initial transient.
    ll = float(result.log_likelihood)
    assert -645 < ll < -625, f"Log-likelihood {ll} not near -632.5"

    # Filtered state at end should be near mean of late observations
    last_filtered = float(result.filtered_states[-1, 0])
    late_mean = float(jnp.mean(NILE[-20:]))
    assert abs(last_filtered - late_mean) < 100


def test_kalman_filter_with_missing_obs() -> None:
    model = _local_level_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE.reshape(-1, 1).copy()
    # Insert NaNs at indices 10, 20, 30
    observations = observations.at[10, 0].set(jnp.nan)
    observations = observations.at[20, 0].set(jnp.nan)
    observations = observations.at[30, 0].set(jnp.nan)

    result = kalman_filter(model, observations)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)

    # At NaN points, predicted == filtered
    np.testing.assert_allclose(
        result.filtered_states[10], result.predicted_states[10], atol=1e-5
    )


def test_kalman_filter_with_inputs() -> None:
    model = StateSpaceModel(
        system_matrix=jnp.array([[1.0]]),
        observation_matrix=jnp.array([[1.0]]),
        evolution_cov=jnp.array([[0.1]]),
        obs_cov=jnp.array([[1.0]]),
        input_matrix=jnp.array([[1.0]]),
    )
    observations = jnp.ones((10, 1)) * 5.0
    inputs = jnp.ones((10, 1)) * 2.0
    result = kalman_filter(model, observations, inputs=inputs)
    assert result.filtered_states.shape == (10, 1)
    assert jnp.all(jnp.isfinite(result.filtered_states))


def test_kalman_filter_custom_initial_state() -> None:
    model = _local_level_model(sigma_level=1.0, sigma_obs=1.0)
    init = GaussianState(
        mean=jnp.array([1000.0]), cov=jnp.array([[1.0]])
    )
    observations = NILE[:10].reshape(-1, 1)
    result = kalman_filter(model, observations, initial_state=init)
    # First predicted state should be near our custom initial
    np.testing.assert_allclose(
        result.predicted_states[0, 0], 1000.0, atol=1e-4
    )


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_kalman_filter_jit() -> None:
    """Verify kalman_filter is JIT-compiled without errors."""
    model = _local_level_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:20].reshape(-1, 1)
    # kalman_filter is already @jax.jit, calling it twice checks tracing
    r1 = kalman_filter(model, observations)
    r2 = kalman_filter(model, observations)
    np.testing.assert_allclose(
        r1.log_likelihood, r2.log_likelihood, atol=1e-5
    )


def test_grad_through_filter() -> None:
    """Verify autodiff works through the Kalman filter log-likelihood."""
    observations = NILE[:20].reshape(-1, 1)

    def neg_ll(log_sigma_level: Array, log_sigma_obs: Array) -> Array:
        model = StateSpaceModel(
            system_matrix=jnp.array([[1.0]]),
            observation_matrix=jnp.array([[1.0]]),
            evolution_cov=jnp.exp(log_sigma_level) * jnp.eye(1),
            obs_cov=jnp.exp(log_sigma_obs) * jnp.eye(1),
        )
        result = kalman_filter(model, observations)
        return -result.log_likelihood

    grad_fn = jax.grad(neg_ll, argnums=(0, 1))
    g1, g2 = grad_fn(jnp.log(jnp.array(1600.0)), jnp.log(jnp.array(15000.0)))
    assert jnp.isfinite(g1)
    assert jnp.isfinite(g2)


# ---------------------------------------------------------------------------
# Multivariate test
# ---------------------------------------------------------------------------


def test_kalman_filter_multivariate() -> None:
    """Test with a 2D state, 2D observation model."""
    model = StateSpaceModel(
        system_matrix=jnp.eye(2) * 0.99,
        observation_matrix=jnp.eye(2),
        evolution_cov=jnp.eye(2) * 0.1,
        obs_cov=jnp.eye(2) * 1.0,
    )
    key = jax.random.PRNGKey(42)
    observations = jax.random.normal(key, (50, 2))
    result = kalman_filter(model, observations)

    assert result.filtered_states.shape == (50, 2)
    assert result.filtered_covariances.shape == (50, 2, 2)
    assert jnp.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_kalman_filter_class_satisfies_protocol() -> None:
    kf = KalmanFilter()
    assert isinstance(kf, FilterProtocol)


def test_kalman_filter_class_scan() -> None:
    kf = KalmanFilter()
    model = _local_level_model(sigma_level=40.0, sigma_obs=120.0)
    observations = NILE[:10].reshape(-1, 1)
    result = kf.scan(model, observations)
    assert isinstance(result, FilterResult)
    assert result.filtered_states.shape == (10, 1)


# ---------------------------------------------------------------------------
# Simulated data test
# ---------------------------------------------------------------------------


def test_filter_recovers_simulated_state() -> None:
    """Filter should track a known simulated state sequence."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    n_steps = 200

    # True state: random walk
    state_noise = jax.random.normal(k1, (n_steps,)) * 1.0
    true_states = jnp.cumsum(state_noise)

    # Observations: true state + noise
    obs_noise = jax.random.normal(k2, (n_steps,)) * 5.0
    observations = (true_states + obs_noise).reshape(-1, 1)

    model = _local_level_model(sigma_level=1.0, sigma_obs=5.0)
    result = kalman_filter(model, observations)

    # Filtered states should correlate well with true states
    filtered = result.filtered_states[:, 0]
    correlation = jnp.corrcoef(
        jnp.stack([filtered, true_states])
    )[0, 1]
    assert float(correlation) > 0.8
