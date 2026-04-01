"""Bearings-only tracking — EKF on a nonlinear observation model.

A target moves with near-constant velocity in 2D, observed only via
bearing (angle) from a fixed sensor. The observation function is
nonlinear (atan2), making this a classic EKF/UKF benchmark.

This example:
1. Simulates a target on a curved trajectory
2. Generates noisy bearing measurements from a sensor at the origin
3. Tracks the 4D state (x, vx, y, vy) using the EKF
4. Plots the reconstructed trajectory vs ground truth
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dynaris.core.types import GaussianState
from dynaris.filters.ekf import ekf_filter
from dynaris.models import BearingsTracking

# --- Model ---
model = BearingsTracking(sensor_pos=(0.0, 0.0), dt=1.0, sigma_accel=0.01, sigma_bearing=0.05)

# --- Simulate target trajectory (gentle curve) ---
key = jax.random.PRNGKey(7)
n_steps = 100

true_state = jnp.array([5.0, 0.3, 5.0, 0.2])
true_states = []
observations = []
for _ in range(n_steps):
    key, k_obs = jax.random.split(key)
    true_state = model.f(true_state)
    bearing = model.h(true_state)
    noisy_bearing = bearing + jax.random.normal(k_obs, (1,)) * 0.05
    true_states.append(true_state)
    observations.append(noisy_bearing)
true_states = jnp.stack(true_states)
observations = jnp.stack(observations)

# --- Track with EKF ---
init = GaussianState(
    mean=jnp.array([5.0, 0.3, 5.0, 0.2]),
    cov=jnp.diag(jnp.array([1.0, 0.5, 1.0, 0.5])),
)
result = ekf_filter(model, observations, initial_state=init)

# --- Plot trajectory ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2D trajectory
ax = axes[0]
ax.plot(
    np.asarray(true_states[:, 0]),
    np.asarray(true_states[:, 2]),
    "k-",
    linewidth=1.5,
    label="True path",
)
ax.plot(
    np.asarray(result.filtered_states[:, 0]),
    np.asarray(result.filtered_states[:, 2]),
    "o-",
    markersize=2,
    linewidth=0.8,
    label="EKF estimate",
)
ax.plot(0, 0, "r^", markersize=10, label="Sensor")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("2D Target Trajectory")
ax.legend(frameon=False)
ax.set_aspect("equal")

# Bearing observations
ax = axes[1]
ax.plot(np.asarray(observations[:, 0]), "k.", markersize=2, alpha=0.5, label="Observed bearings")
true_bearings = np.asarray(jax.vmap(model.h)(true_states)[:, 0])
ax.plot(true_bearings, "r-", linewidth=1, label="True bearing")
ax.set_xlabel("Time step")
ax.set_ylabel("Bearing (rad)")
ax.set_title("Bearing Observations")
ax.legend(frameon=False)

fig.suptitle("Bearings-Only Tracking with EKF")
fig.tight_layout()
plt.show()
