"""Regime-switching model — detecting volatility regimes.

Demonstrates the Hamilton filter and Kim smoother on a Markov-switching
DLM with two regimes: a "calm" low-volatility regime and a "volatile"
high-volatility regime.

This example:
1. Simulates data from a 2-regime process with a known switch point
2. Fits the Hamilton filter to estimate regime probabilities
3. Runs the Kim smoother for retrospective regime inference
4. Plots the filtered and smoothed regime probabilities
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dynaris import LocalLevel, MarkovSwitchingSSM
from dynaris.core.types import GaussianState
from dynaris.filters.hamilton import hamilton_filter
from dynaris.smoothers.kim import kim_smooth

# --- Simulate a 2-regime process ---
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
n = 300

# Regime 0 (calm): t < 150, Regime 1 (volatile): 150 <= t < 250, back to 0
regimes = np.zeros(n, dtype=int)
regimes[150:250] = 1

level = 100.0
obs = np.zeros(n)
for t in range(n):
    k1, k_l, k_o = jax.random.split(k1, 3)
    sigma_l = 1.0 if regimes[t] == 0 else 5.0
    sigma_o = 5.0 if regimes[t] == 0 else 25.0
    level = level + float(jax.random.normal(k_l)) * sigma_l
    obs[t] = level + float(jax.random.normal(k_o)) * sigma_o

observations = jnp.array(obs).reshape(-1, 1)

# --- Build the switching model ---
calm = LocalLevel(sigma_level=1.0, sigma_obs=5.0)
volatile = LocalLevel(sigma_level=5.0, sigma_obs=25.0)

model = MarkovSwitchingSSM(
    models=(calm, volatile),
    transition_matrix=jnp.array([[0.98, 0.02], [0.05, 0.95]]),
    initial_probs=jnp.array([0.9, 0.1]),
)

# --- Hamilton filter ---
init = GaussianState(mean=jnp.array([100.0]), cov=jnp.eye(1) * 100.0)
filt = hamilton_filter(model, observations, initial_state=init)
print(f"Log-likelihood: {float(filt.log_likelihood):.2f}")

# --- Kim smoother ---
smoothed = kim_smooth(model, filt)

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Observations and filtered states
ax = axes[0]
ax.plot(obs, "k-", linewidth=0.5, alpha=0.5, label="Observed")
ax.plot(np.asarray(filt.filtered_states[:, 0]), "C0-", linewidth=1.2, label="Filtered")
ax.axvspan(150, 250, alpha=0.15, color="red", label="True volatile regime")
ax.set_ylabel("Level")
ax.legend(frameon=False)
ax.set_title("Observations and Filtered States")

# Filtered regime probabilities
ax = axes[1]
ax.plot(np.asarray(filt.regime_filtered_probs[:, 1]), "C1-", linewidth=1.0)
ax.axvspan(150, 250, alpha=0.15, color="red")
ax.set_ylabel("P(volatile)")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Filtered Regime Probability (Volatile)")

# Smoothed regime probabilities
ax = axes[2]
ax.plot(np.asarray(smoothed.regime_smoothed_probs[:, 1]), "C2-", linewidth=1.0)
ax.axvspan(150, 250, alpha=0.15, color="red")
ax.set_ylabel("P(volatile)")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Time")
ax.set_title("Smoothed Regime Probability (Kim Smoother)")

fig.suptitle("Markov-Switching DLM — Regime Detection")
fig.tight_layout()
plt.show()
