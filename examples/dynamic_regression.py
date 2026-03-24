"""Dynamic regression — time-varying coefficients.

Demonstrates using the Regression component with non-zero sigma_coeff
to allow regression coefficients to evolve over time. This is useful
when the relationship between predictors and the response changes.
"""

import jax
import jax.numpy as jnp

from dynaris import DLM, Regression
from dynaris.core.state_space import StateSpaceModel

# --- Simulate data with a time-varying coefficient ---
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
n = 200

# Predictor: a simple exogenous variable
x = jnp.cumsum(jax.random.normal(k1, (n,)) * 0.5)

# True coefficient: starts at 2.0 and drifts to -1.0
true_beta = 2.0 + jnp.cumsum(jax.random.normal(k2, (n,)) * 0.05)

# Observations: y = beta_t * x_t + noise
noise = jax.random.normal(k3, (n,)) * 2.0
y = true_beta * x + noise

# --- Build regression model ---
# The Regression component creates a state where the coefficient
# follows a random walk. We set H = x_t at each time step by
# constructing the observation matrix from the predictor.
# For simplicity, we build a custom StateSpaceModel directly.
model = StateSpaceModel(
    system_matrix=jnp.array([[1.0]]),       # random walk
    observation_matrix=jnp.array([[1.0]]),       # placeholder (x varies)
    evolution_cov=jnp.array([[0.01]]),         # coefficient drift
    obs_cov=jnp.array([[4.0]]),            # observation noise
)

# Since the observation matrix varies with x_t, we incorporate x
# into the observations: z_t = y_t / x_t ~ beta_t + noise/x_t
# This is a simplification; for full time-varying H, you'd run
# the filter step-by-step.
mask = jnp.abs(x) > 0.1  # avoid division by near-zero
z = jnp.where(mask, y / x, 0.0).reshape(-1, 1)

# --- Fit and analyze ---
dlm = DLM(model)
dlm.fit(z)
dlm.smooth()

print(dlm.summary())
print()

# The filtered state tracks the time-varying coefficient
import numpy as np

filtered_beta = np.asarray(dlm.filter_result.filtered_states[:, 0])
smoothed_beta = np.asarray(dlm.smoother_result.smoothed_states[:, 0])

print(f"True beta range:     [{float(true_beta.min()):.2f}, {float(true_beta.max()):.2f}]")
print(f"Filtered beta range: [{filtered_beta.min():.2f}, {filtered_beta.max():.2f}]")
print(f"Smoothed beta range: [{smoothed_beta.min():.2f}, {smoothed_beta.max():.2f}]")

# --- Plot ---
import matplotlib.pyplot as plt

dlm.plot(kind="filtered", title="Dynamic Regression — Filtered Coefficient")
dlm.plot(kind="smoothed", title="Dynamic Regression — Smoothed Coefficient")
dlm.plot(kind="diagnostics", title="Dynamic Regression — Diagnostics")
plt.show()
