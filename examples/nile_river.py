"""Nile river flow — classic DLM example.

Demonstrates the local-level model on the Nile river annual flow dataset
(1871-1970). This is the canonical introductory example from
Durbin & Koopman (2012), "Time Series Analysis by State Space Methods".
"""

from dynaris import DLM, LocalLevel
from dynaris.datasets import load_nile

# --- Data ---
y = load_nile()

# --- Build and fit ---
model = LocalLevel(sigma_level=38.33, sigma_obs=122.9)
dlm = DLM(model)
dlm.fit(y)
dlm.smooth()

# --- Summary ---
print(dlm.summary())
print()

# --- Forecast 10 years ahead ---
fc = dlm.forecast(steps=10)
print("Forecast:")
print(fc)
print()

# --- Plots ---
import matplotlib.pyplot as plt

dlm.plot(kind="filtered", title="Nile — Filtered")
dlm.plot(kind="smoothed", title="Nile — Smoothed")
dlm.forecast(steps=10)
dlm.plot(kind="forecast", title="Nile — 10-year Forecast", n_history=30)
dlm.plot(kind="diagnostics", title="Nile — Residual Diagnostics")
plt.show()
