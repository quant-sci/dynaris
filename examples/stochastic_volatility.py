"""Stochastic volatility — particle filter on financial returns.

Demonstrates the particle filter on a stochastic volatility model,
where the latent log-volatility follows an AR(1) process and is
observed through the variance of financial returns.

This example:
1. Simulates returns from a known SV model
2. Transforms returns via the KSC linearization
3. Filters with EKF, UKF, and Particle Filter
4. Recovers the latent volatility path
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dynaris.core.types import GaussianState
from dynaris.filters.ekf import ekf_filter
from dynaris.filters.particle import particle_filter
from dynaris.filters.ukf import ukf_filter
from dynaris.models import StochasticVolatility, transform_returns

# --- True parameters ---
mu, phi, sigma_eta = 0.0, 0.97, 0.15
model = StochasticVolatility(mu=mu, phi=phi, sigma=sigma_eta)

# --- Simulate ---
key = jax.random.PRNGKey(123)
k1, k2, k_pf = jax.random.split(key, 3)
n = 500

# Generate log-volatility and returns
h = jnp.zeros(n)
returns = jnp.zeros(n)
state = 0.0
for t in range(n):
    k1, k_h, k_r = jax.random.split(k1, 3)
    state = mu + phi * (state - mu) + sigma_eta * jax.random.normal(k_h)
    h = h.at[t].set(state)
    returns = returns.at[t].set(jnp.exp(state / 2) * jax.random.normal(k_r))

# --- Transform returns for the linearized model ---
y_star = transform_returns(returns).reshape(-1, 1)

# --- Initial state ---
init = GaussianState(mean=jnp.array([0.0]), cov=jnp.array([[10.0]]))

# --- Run filters ---
print("Running EKF...")
ekf_res = ekf_filter(model, y_star, initial_state=init)

print("Running UKF (alpha=1.0)...")
ukf_res = ukf_filter(model, y_star, initial_state=init, alpha=1.0)

print("Running Particle Filter (2000 particles)...")
pf_res = particle_filter(model, y_star, n_particles=2000, key=k_pf, initial_state=init)

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
time = np.arange(n)

# Returns
ax = axes[0]
ax.plot(time, np.asarray(returns), "k-", linewidth=0.5, alpha=0.7)
ax.set_ylabel("Returns")
ax.set_title("Simulated Financial Returns")

# True vs filtered log-volatility
ax = axes[1]
ax.plot(time, np.asarray(h), "k-", linewidth=1.5, alpha=0.5, label="True h(t)")
ax.plot(time, np.asarray(ekf_res.filtered_states[:, 0]), label="EKF", linewidth=0.8)
ax.plot(time, np.asarray(ukf_res.filtered_states[:, 0]), label="UKF", linewidth=0.8)
ax.plot(time, np.asarray(pf_res.filtered_states[:, 0]), label="PF", linewidth=0.8)
ax.set_ylabel("Log-volatility h(t)")
ax.legend(frameon=False, ncol=4)
ax.set_title("Filtered Log-Volatility")

# True vs filtered volatility (exp scale)
ax = axes[2]
ax.plot(time, np.exp(np.asarray(h) / 2), "k-", linewidth=1.5, alpha=0.5, label="True")
ax.plot(
    time, np.exp(np.asarray(pf_res.filtered_states[:, 0]) / 2), label="PF estimate", linewidth=1.0
)
ax.set_ylabel("Volatility exp(h/2)")
ax.set_xlabel("Time")
ax.legend(frameon=False)
ax.set_title("Filtered Volatility")

fig.suptitle("Stochastic Volatility — Particle Filter")
fig.tight_layout()
plt.show()
