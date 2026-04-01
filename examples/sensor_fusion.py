"""Multi-sensor fusion — DFM for combining noisy sensor readings.

Multiple noisy sensors observe the same underlying signal (e.g.,
temperature at a location). The DFM "fuses" these readings into a
single latent estimate with uncertainty quantification.

This example:
1. Simulates a slowly varying signal observed by 5 noisy sensors
2. Fits a 1-factor DFM to fuse the sensor readings
3. Plots the fused estimate vs individual sensors and ground truth
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dynaris.models.dfm_api import DFMModel

# --- Simulate ---
key = jax.random.PRNGKey(99)
k1, k2 = jax.random.split(key)
n_obs = 200
n_sensors = 5

# True slowly varying signal
true_signal = jnp.cumsum(jax.random.normal(k1, (n_obs,)) * 0.3) + 20.0

# Sensor readings with different noise levels and biases
noise_levels = jnp.array([0.5, 1.0, 1.5, 0.8, 2.0])
noise = jax.random.normal(k2, (n_obs, n_sensors)) * noise_levels[None, :]
sensor_data = true_signal[:, None] + noise

sensor_names = [f"Sensor {i + 1} (noise={float(noise_levels[i]):.1f})" for i in range(n_sensors)]

print("Sensor data shape:", sensor_data.shape)
print(f"True signal range: [{float(jnp.min(true_signal)):.1f}, {float(jnp.max(true_signal)):.1f}]")

# --- Fit 1-factor DFM ---
print("\nFitting 1-factor DFM...")
dfm = DFMModel(n_factors=1, max_iter=50)
dfm.fit(np.asarray(sensor_data))
print(dfm.summary())

# --- Extract fused signal ---
fused = np.asarray(dfm.result.factor_states[:, 0])
loadings = np.asarray(dfm.result.loadings[:, 0])

# Scale factor to match the signal scale (factor * mean_loading)
scale = np.mean(loadings)
fused_scaled = fused * scale

print("\nEstimated loadings (sensor weights):")
for i, name in enumerate(sensor_names):
    print(f"  {name}: {loadings[i]:.3f}")

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7))

# Individual sensors vs fused estimate
ax = axes[0]
for i in range(n_sensors):
    ax.plot(np.asarray(sensor_data[:, i]), alpha=0.3, linewidth=0.5, label=sensor_names[i])
ax.plot(np.asarray(true_signal), "k-", linewidth=2, label="True signal")
ax.plot(fused_scaled, "r-", linewidth=1.5, label="DFM fused estimate")
ax.set_ylabel("Temperature")
ax.set_title("Sensor Readings vs Fused Estimate")
ax.legend(frameon=False, fontsize=7, ncol=3)

# Estimation error
ax = axes[1]
for i in range(n_sensors):
    err = np.asarray(sensor_data[:, i]) - np.asarray(true_signal)
    ax.plot(err, alpha=0.3, linewidth=0.5)
fused_err = fused_scaled - np.asarray(true_signal)
ax.plot(fused_err, "r-", linewidth=1.5, label="DFM error")
ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
ax.set_ylabel("Error")
ax.set_xlabel("Time")
ax.set_title(f"Estimation Error (DFM RMSE: {float(np.sqrt(np.mean(fused_err**2))):.3f})")
ax.legend(frameon=False)

fig.suptitle("Multi-Sensor Fusion with Dynamic Factor Model")
fig.tight_layout()
plt.show()
