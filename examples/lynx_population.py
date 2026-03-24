"""Canadian lynx population cycles (biology / ecology).

The lynx trapping dataset (1821-1934) shows a prominent ~10-year
population cycle driven by predator-prey (lynx-hare) dynamics.
We model the log-transformed data with a cycle component.

Domain: biology / population ecology.
"""

import matplotlib.pyplot as plt
import numpy as np

from dynaris import DLM, Cycle, LocalLevel
from dynaris.datasets import load_lynx

# --- Data ---
y = load_lynx()
y_log = np.log(y)  # log-transform to stabilize variance
y_log.name = "log_lynx"

# --- Model: local level + ~10-year cycle ---
model = LocalLevel(sigma_level=0.1, sigma_obs=0.0) + Cycle(
    period=10.0, damping=0.95, sigma_cycle=0.3, sigma_obs=0.2
)

dlm = DLM(model)
dlm.fit(y_log).smooth()

print(dlm.summary())
print()

# --- Forecast 20 years ahead ---
fc = dlm.forecast(steps=20)
print("20-year forecast (log scale):")
print(fc.head())
print()

# --- Plots ---
dlm.plot(kind="filtered", title="Canadian Lynx (log) -- Filtered")
dlm.plot(kind="smoothed", title="Canadian Lynx (log) -- Smoothed")
dlm.forecast(steps=20)
dlm.plot(kind="forecast", title="Canadian Lynx -- 20-year Forecast", n_history=40)
dlm.plot(kind="diagnostics", title="Canadian Lynx -- Diagnostics")
dlm.plot(
    kind="components",
    component_dims={"Level": 0, "Cycle (cos)": 1, "Cycle (sin)": 2},
    title="Canadian Lynx -- Component Decomposition",
)
plt.show()
