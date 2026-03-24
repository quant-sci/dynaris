"""Sunspot number cycles (physics / astronomy).

Annual sunspot numbers (1700-1988) exhibit an approximately 11-year
solar cycle with varying amplitude. We model this with a damped cycle
component superimposed on a local level.

Domain: physics / solar astronomy.
"""

import matplotlib.pyplot as plt

from dynaris import DLM, Cycle, LocalLevel
from dynaris.datasets import load_sunspots

# --- Data ---
y = load_sunspots()

# --- Model: local level + ~11-year solar cycle ---
model = LocalLevel(sigma_level=5.0, sigma_obs=0.0) + Cycle(
    period=11.0, damping=0.98, sigma_cycle=10.0, sigma_obs=15.0
)

dlm = DLM(model)
dlm.fit(y).smooth()

print(dlm.summary())
print()

# --- Forecast 22 years (2 full cycles) ---
fc = dlm.forecast(steps=22)
print("22-year forecast:")
print(fc.head(10))
print()

# --- Plots ---
dlm.plot(kind="filtered", title="Sunspot Numbers -- Filtered")
dlm.plot(kind="smoothed", title="Sunspot Numbers -- Smoothed")
dlm.forecast(steps=22)
dlm.plot(kind="forecast", title="Sunspot Numbers -- 22-year Forecast", n_history=60)
dlm.plot(kind="diagnostics", title="Sunspot Numbers -- Diagnostics")
dlm.plot(
    kind="components",
    component_dims={"Level": 0, "Cycle (cos)": 1, "Cycle (sin)": 2},
    title="Sunspot Numbers -- Component Decomposition",
)
plt.show()
