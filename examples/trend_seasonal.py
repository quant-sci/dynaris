"""Sales forecasting with trend + seasonality.

Demonstrates composing a local linear trend with a seasonal component
to model and forecast monthly data with an upward trend and yearly cycle.
"""

import jax
import jax.numpy as jnp
import pandas as pd

from dynaris import DLM, LocalLinearTrend, Seasonal

# --- Simulate monthly sales data ---
key = jax.random.PRNGKey(42)
n_months = 120  # 10 years of monthly data

t = jnp.arange(n_months, dtype=jnp.float32)
trend = 100.0 + 0.5 * t
seasonal = 15.0 * jnp.sin(2.0 * jnp.pi * t / 12.0)
noise = jax.random.normal(key, (n_months,)) * 5.0
sales = trend + seasonal + noise

y = pd.Series(
    sales,
    index=pd.date_range("2015-01", periods=n_months, freq="ME"),
    name="sales",
)

# --- Build composed model ---
model = (
    LocalLinearTrend(sigma_level=2.0, sigma_slope=0.1, sigma_obs=0.0)
    + Seasonal(period=12, sigma_seasonal=1.0, sigma_obs=5.0)
)

# --- Fit, smooth, forecast ---
dlm = DLM(model)
dlm.fit(y)
dlm.smooth()

print(dlm.summary())
print()

# Forecast 24 months
fc = dlm.forecast(steps=24)
print("24-month Forecast:")
print(fc)
print()

# --- Plots ---
import matplotlib.pyplot as plt

dlm.plot(kind="filtered", title="Sales — Filtered")
dlm.plot(kind="smoothed", title="Sales — Smoothed")
dlm.plot(kind="forecast", title="Sales — 24-month Forecast", n_history=36)
dlm.plot(kind="diagnostics", title="Sales — Diagnostics")

# Component decomposition
dlm.plot(
    kind="components",
    component_dims={"Level": 0, "Slope": 1, "Seasonal": 2},
    title="Sales — Component Decomposition",
)
plt.show()
