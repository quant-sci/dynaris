"""Panel overview -- a single figure summarizing the full DLM analysis.

Demonstrates the `plot(kind="panel")` method, which produces a clean
3x2 grid with filtered, smoothed, forecast, residuals, Q-Q plot, and
ACF in one view. Applied to the airline passengers dataset with a
trend + seasonal model.

Domain: economics / transportation.
"""

import matplotlib.pyplot as plt
import numpy as np

from dynaris import DLM, LocalLinearTrend, Seasonal
from dynaris.datasets import load_airline

# --- Data ---
y = load_airline()
y_log = np.log(y)
y_log.name = "log_passengers"

# --- Model: trend + monthly seasonality ---
model = (
    LocalLinearTrend(sigma_level=0.01, sigma_slope=0.001, sigma_obs=0.0)
    + Seasonal(period=12, sigma_seasonal=0.005, sigma_obs=0.01)
)

# --- Fit, smooth, forecast ---
dlm = DLM(model)
dlm.fit(y_log).smooth()
dlm.forecast(steps=24)

print(dlm.summary())
print()

# --- Single panel view ---
dlm.plot(
    kind="panel",
    title="Airline Passengers (log) -- Full Analysis",
    n_history=48,
)

plt.show()
