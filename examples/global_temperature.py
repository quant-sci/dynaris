"""Global temperature anomaly -- trend detection (physics / climate).

Annual global mean temperature anomaly (1880-2023) relative to the
1951-1980 baseline. A local linear trend model detects and quantifies
the warming trend with uncertainty.

Domain: physics / climate science.
"""

import matplotlib.pyplot as plt

from dynaris import DLM, LocalLinearTrend
from dynaris.datasets import load_temperature

# --- Data ---
y = load_temperature()

# --- Model: local linear trend ---
model = LocalLinearTrend(sigma_level=0.05, sigma_slope=0.005, sigma_obs=0.1)

dlm = DLM(model)
dlm.fit(y).smooth()

print(dlm.summary())
print()

# --- Current warming rate ---
import numpy as np

smoothed = dlm.smoothed_states_df()
current_slope = float(smoothed.iloc[-1]["state_1"])
print(f"Estimated warming rate (2023): {current_slope:.4f} deg C/year")
print(f"Equivalent to {current_slope * 10:.2f} deg C/decade")
print()

# --- Forecast 30 years ahead ---
fc = dlm.forecast(steps=30)
print("30-year temperature forecast:")
print(fc.tail())
print()

# --- Plots ---
dlm.plot(kind="filtered", title="Global Temperature Anomaly -- Filtered")
dlm.plot(kind="smoothed", title="Global Temperature Anomaly -- Smoothed")
dlm.forecast(steps=30)
dlm.plot(kind="forecast", title="Temperature Anomaly -- 30-year Forecast", n_history=50)
dlm.plot(kind="diagnostics", title="Temperature Anomaly -- Diagnostics")
dlm.plot(
    kind="components",
    component_dims={"Level": 0, "Slope (warming rate)": 1},
    title="Temperature Anomaly -- Trend Decomposition",
)
plt.show()
