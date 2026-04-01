"""Lorenz attractor tracking — nonlinear filter comparison.

Demonstrates EKF, UKF, and Particle Filter on the chaotic Lorenz system.
The Lorenz attractor is a classic benchmark for nonlinear filtering,
with three coupled differential equations producing deterministic chaos.

This example:
1. Simulates a trajectory from the Lorenz system
2. Generates noisy partial observations (x and y only)
3. Tracks the full 3D state using EKF, UKF, and Particle Filter
4. Compares filter accuracy via correlation with the true state
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dynaris.core.types import GaussianState
from dynaris.filters.ekf import ekf_filter
from dynaris.filters.particle import particle_filter
from dynaris.filters.ukf import ukf_filter
from dynaris.models import LorenzAttractor

# --- Model ---
model = LorenzAttractor(dt=0.01, process_noise=0.5, obs_noise=2.0, obs_dims=2)

# --- Simulate true trajectory ---
key = jax.random.PRNGKey(42)
k1, k2, k_pf = jax.random.split(key, 3)
n_steps = 500

state = jnp.array([1.0, 1.0, 1.0])
true_states = []
observations = []
for _ in range(n_steps):
    k1, k_state, k_obs = jax.random.split(k1, 3)
    state = model.f(state) + jax.random.normal(k_state, (3,)) * 0.5
    obs = model.h(state) + jax.random.normal(k_obs, (2,)) * 2.0
    true_states.append(state)
    observations.append(obs)
true_states = jnp.stack(true_states)
observations = jnp.stack(observations)

# --- Initial state ---
init = GaussianState(mean=jnp.array([1.0, 1.0, 1.0]), cov=jnp.eye(3) * 10.0)

# --- Run filters ---
print("Running EKF...")
ekf_result = ekf_filter(model, observations, initial_state=init)

print("Running UKF...")
ukf_result = ukf_filter(model, observations, initial_state=init, alpha=1.0)

print("Running Particle Filter (1000 particles)...")
pf_result = particle_filter(model, observations, n_particles=1000, key=k_pf, initial_state=init)

# --- Compare accuracy ---
print("\nFilter accuracy (correlation with true x-component):")
for name, result in [("EKF", ekf_result), ("UKF", ukf_result), ("PF", pf_result)]:
    corr = float(jnp.corrcoef(jnp.stack([result.filtered_states[:, 0], true_states[:, 0]]))[0, 1])
    print(f"  {name}: r = {corr:.4f}")

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
labels = ["x", "y", "z"]
for i, ax in enumerate(axes):
    ax.plot(np.asarray(true_states[:, i]), "k-", alpha=0.4, linewidth=0.8, label="True")
    ax.plot(np.asarray(ekf_result.filtered_states[:, i]), label="EKF", linewidth=1.0)
    ax.plot(np.asarray(ukf_result.filtered_states[:, i]), label="UKF", linewidth=1.0)
    ax.plot(np.asarray(pf_result.filtered_states[:, i]), label="PF", linewidth=1.0)
    ax.set_ylabel(labels[i])
    if i == 0:
        ax.legend(frameon=False, ncol=4)
axes[-1].set_xlabel("Time step")
fig.suptitle("Lorenz Attractor Tracking — EKF vs UKF vs Particle Filter")
fig.tight_layout()
plt.show()
