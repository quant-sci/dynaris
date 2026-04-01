"""Tests for built-in nonlinear state-space models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.core.types import GaussianState
from dynaris.filters.ekf import ekf_filter
from dynaris.filters.particle import particle_filter
from dynaris.filters.ukf import ukf_filter
from dynaris.models.nonlinear import (
    BearingsTracking,
    LorenzAttractor,
    StochasticVolatility,
    transform_returns,
)

# ---------------------------------------------------------------------------
# Stochastic Volatility
# ---------------------------------------------------------------------------


def test_sv_shapes() -> None:
    model = StochasticVolatility()
    assert model.state_dim == 1
    assert model.obs_dim == 1
    assert model.Q.shape == (1, 1)
    assert model.R.shape == (1, 1)


def test_sv_default_obs_var() -> None:
    model = StochasticVolatility()
    np.testing.assert_allclose(model.R[0, 0], jnp.pi**2 / 2.0, atol=1e-6)


def test_sv_custom_params() -> None:
    model = StochasticVolatility(mu=1.0, phi=0.9, sigma=0.3, obs_var=5.0)
    np.testing.assert_allclose(model.Q[0, 0], 0.09, atol=1e-6)
    np.testing.assert_allclose(model.R[0, 0], 5.0, atol=1e-6)


def test_sv_transition_fn() -> None:
    model = StochasticVolatility(mu=1.0, phi=0.9)
    h = jnp.array([2.0])
    result = model.f(h)
    expected = 1.0 + 0.9 * (2.0 - 1.0)  # = 1.9
    np.testing.assert_allclose(result, [expected], atol=1e-6)


def test_sv_observation_fn() -> None:
    model = StochasticVolatility()
    h = jnp.array([3.0])
    result = model.h(h)
    np.testing.assert_allclose(result, [3.0], atol=1e-6)


def test_sv_ekf() -> None:
    model = StochasticVolatility()
    key = jax.random.PRNGKey(0)
    obs = jax.random.normal(key, (50,))
    y_star = transform_returns(obs).reshape(-1, 1)

    result = ekf_filter(model, y_star)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


def test_sv_ukf() -> None:
    model = StochasticVolatility()
    key = jax.random.PRNGKey(1)
    obs = jax.random.normal(key, (50,))
    y_star = transform_returns(obs).reshape(-1, 1)

    init = GaussianState(mean=jnp.array([0.0]), cov=jnp.array([[10.0]]))
    result = ukf_filter(model, y_star, initial_state=init, alpha=1.0)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


def test_sv_particle() -> None:
    model = StochasticVolatility()
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    obs = jax.random.normal(k1, (30,))
    y_star = transform_returns(obs).reshape(-1, 1)

    result = particle_filter(model, y_star, n_particles=200, key=k2)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


def test_transform_returns_shape() -> None:
    y = jnp.array([0.01, -0.02, 0.03])
    y_star = transform_returns(y)
    assert y_star.shape == (3,)
    assert jnp.all(jnp.isfinite(y_star))


def test_transform_returns_values() -> None:
    y = jnp.array([1.0])
    y_star = transform_returns(y)
    expected = jnp.log(1.0 + 1e-8) + 1.2704
    np.testing.assert_allclose(y_star, [expected], atol=1e-5)


# ---------------------------------------------------------------------------
# Bearings-Only Tracking
# ---------------------------------------------------------------------------


def test_bearings_shapes() -> None:
    model = BearingsTracking()
    assert model.state_dim == 4
    assert model.obs_dim == 1
    assert model.Q.shape == (4, 4)
    assert model.R.shape == (1, 1)


def test_bearings_transition() -> None:
    model = BearingsTracking(dt=1.0)
    state = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = model.f(state)
    # x = 1 + 1*2 = 3, vx = 2, y = 3 + 1*4 = 7, vy = 4
    np.testing.assert_allclose(result, [3.0, 2.0, 7.0, 4.0], atol=1e-6)


def test_bearings_observation() -> None:
    model = BearingsTracking(sensor_pos=(0.0, 0.0))
    # Target at (1, 0, 1, 0) -> bearing = atan2(1, 1) = pi/4
    state = jnp.array([1.0, 0.0, 1.0, 0.0])
    result = model.h(state)
    np.testing.assert_allclose(result, [jnp.pi / 4], atol=1e-6)


def test_bearings_observation_custom_sensor() -> None:
    model = BearingsTracking(sensor_pos=(1.0, 1.0))
    # Target at (2, 0, 2, 0) relative to sensor: dx=1, dy=1 -> pi/4
    state = jnp.array([2.0, 0.0, 2.0, 0.0])
    result = model.h(state)
    np.testing.assert_allclose(result, [jnp.pi / 4], atol=1e-6)


def test_bearings_q_symmetric() -> None:
    model = BearingsTracking()
    np.testing.assert_allclose(model.Q, model.Q.T, atol=1e-10)


def test_bearings_ekf() -> None:
    model = BearingsTracking(sigma_accel=0.5, sigma_bearing=0.1)
    key = jax.random.PRNGKey(10)

    # Simulate a target moving in a straight line
    n_steps = 50
    true_state = jnp.array([10.0, 1.0, 10.0, 0.5])
    obs_list = []
    for _ in range(n_steps):
        true_state = model.f(true_state)
        bearing = model.h(true_state)
        obs_list.append(bearing + jax.random.normal(key, (1,)) * 0.1)
        key, _ = jax.random.split(key)
    observations = jnp.stack(obs_list)

    init = GaussianState(mean=jnp.array([10.0, 1.0, 10.0, 0.5]), cov=jnp.eye(4) * 10.0)
    result = ekf_filter(model, observations, initial_state=init)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


def test_bearings_particle() -> None:
    model = BearingsTracking(sigma_accel=0.5, sigma_bearing=0.1)
    key = jax.random.PRNGKey(20)
    k_sim, k_filter = jax.random.split(key)

    # Simulate a target on a known trajectory
    true_state = jnp.array([10.0, 1.0, 10.0, 0.5])
    obs_list = []
    for _ in range(30):
        true_state = model.f(true_state)
        bearing = model.h(true_state)
        obs_list.append(bearing + jax.random.normal(k_sim, (1,)) * 0.1)
        k_sim, _ = jax.random.split(k_sim)
    observations = jnp.stack(obs_list)

    init = GaussianState(mean=jnp.array([10.0, 1.0, 10.0, 0.5]), cov=jnp.eye(4) * 10.0)
    result = particle_filter(model, observations, n_particles=500, key=k_filter, initial_state=init)
    assert jnp.all(jnp.isfinite(result.filtered_states))


# ---------------------------------------------------------------------------
# Lorenz Attractor
# ---------------------------------------------------------------------------


def test_lorenz_shapes_3d() -> None:
    model = LorenzAttractor()
    assert model.state_dim == 3
    assert model.obs_dim == 3


def test_lorenz_shapes_2d() -> None:
    model = LorenzAttractor(obs_dims=2)
    assert model.state_dim == 3
    assert model.obs_dim == 2


def test_lorenz_shapes_1d() -> None:
    model = LorenzAttractor(obs_dims=1)
    assert model.state_dim == 3
    assert model.obs_dim == 1


def test_lorenz_transition_euler_step() -> None:
    model = LorenzAttractor(lorenz_sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01)
    state = jnp.array([1.0, 1.0, 1.0])
    result = model.f(state)

    # Manual Euler step
    dx = 10.0 * (1.0 - 1.0)
    dy = 1.0 * (28.0 - 1.0) - 1.0
    dz = 1.0 * 1.0 - (8.0 / 3.0) * 1.0
    expected = jnp.array([1.0 + 0.01 * dx, 1.0 + 0.01 * dy, 1.0 + 0.01 * dz])
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_lorenz_observation_partial() -> None:
    model = LorenzAttractor(obs_dims=2)
    state = jnp.array([1.0, 2.0, 3.0])
    result = model.h(state)
    np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-6)


def test_lorenz_observation_full() -> None:
    model = LorenzAttractor(obs_dims=3)
    state = jnp.array([1.0, 2.0, 3.0])
    result = model.h(state)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-6)


def test_lorenz_ukf() -> None:
    model = LorenzAttractor(dt=0.01, process_noise=0.5, obs_noise=2.0)
    key = jax.random.PRNGKey(30)

    # Simulate a short trajectory
    state = jnp.array([1.0, 1.0, 1.0])
    obs_list = []
    for _ in range(100):
        state = model.f(state) + jax.random.normal(key, (3,)) * 0.5
        obs_list.append(model.h(state) + jax.random.normal(key, (3,)) * 2.0)
        key, _ = jax.random.split(key)
    observations = jnp.stack(obs_list)

    init = GaussianState(mean=jnp.array([1.0, 1.0, 1.0]), cov=jnp.eye(3) * 10.0)
    result = ukf_filter(model, observations, initial_state=init, alpha=1.0)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


def test_lorenz_particle() -> None:
    model = LorenzAttractor(dt=0.01, process_noise=0.5, obs_noise=2.0, obs_dims=3)
    key = jax.random.PRNGKey(40)
    k_sim, k_filter = jax.random.split(key)

    state = jnp.array([1.0, 1.0, 1.0])
    obs_list = []
    for _ in range(50):
        state = model.f(state) + jax.random.normal(k_sim, (3,)) * 0.5
        obs_list.append(model.h(state) + jax.random.normal(k_sim, (3,)) * 2.0)
        k_sim, _ = jax.random.split(k_sim)
    observations = jnp.stack(obs_list)

    init = GaussianState(mean=jnp.array([1.0, 1.0, 1.0]), cov=jnp.eye(3) * 10.0)
    result = particle_filter(model, observations, n_particles=200, key=k_filter, initial_state=init)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert jnp.isfinite(result.log_likelihood)


def test_lorenz_custom_params() -> None:
    model = LorenzAttractor(lorenz_sigma=5.0, rho=15.0, beta=1.0, dt=0.005)
    assert model.state_dim == 3
    state = jnp.array([1.0, 1.0, 1.0])
    result = model.f(state)
    assert jnp.all(jnp.isfinite(result))


# ---------------------------------------------------------------------------
# Pytree compatibility
# ---------------------------------------------------------------------------


def test_all_models_pytree_roundtrip() -> None:
    """Verify all models survive JAX pytree flatten/unflatten."""
    models = [
        StochasticVolatility(),
        BearingsTracking(),
        LorenzAttractor(),
    ]
    for model in models:
        leaves, treedef = jax.tree_util.tree_flatten(model)
        reconstructed = treedef.unflatten(leaves)
        assert reconstructed.state_dim == model.state_dim
        assert reconstructed.obs_dim == model.obs_dim
