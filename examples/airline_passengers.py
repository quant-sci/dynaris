"""Airline passengers -- trend + seasonal decomposition (economics).

The Box-Jenkins airline dataset (1949-1960) shows an upward trend with
multiplicative seasonality. We apply a log-transform to make the
seasonality additive, then fit a trend + seasonal DLM.

Domain: economics / transportation engineering.
"""

import matplotlib.pyplot as plt
import numpy as np

from dynaris import DLM, LocalLinearTrend, Seasonal
from dynaris.datasets import load_airline

# --- Data ---
y = load_airline()
y_log = np.log(y)  # log-transform for additive DLM
y_log.name = "log_passengers"

# --- Model: trend + monthly seasonality ---
model = LocalLinearTrend(sigma_level=0.01, sigma_slope=0.001, sigma_obs=0.0) + Seasonal(
    period=12, sigma_seasonal=0.005, sigma_obs=0.01
)

dlm = DLM(model)
dlm.fit(y_log).smooth()

print(dlm.summary())
print()

# --- Forecast 24 months ahead ---
fc = dlm.forecast(steps=24)
print("24-month forecast (log scale):")
print(fc.head())
print()

# --- Plots ---
dlm.plot(kind="filtered", title="Airline Passengers (log) -- Filtered")
dlm.plot(kind="smoothed", title="Airline Passengers (log) -- Smoothed")
dlm.forecast(steps=24)
dlm.plot(kind="forecast", title="Airline Passengers -- 24-month Forecast", n_history=36)
dlm.plot(kind="diagnostics", title="Airline Passengers -- Diagnostics")
dlm.plot(
    kind="components",
    component_dims={"Level": 0, "Slope": 1, "Seasonal": 2},
    title="Airline Passengers -- Component Decomposition",
)
plt.show()
