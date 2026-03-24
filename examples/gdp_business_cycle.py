"""US GDP growth -- business cycle analysis (economics / finance).

Quarterly US real GDP annualized growth rates (1947-2023). A local
level model with an autoregressive component captures the persistent
business cycle fluctuations around a long-run mean.

Domain: economics / macrofinance.
"""

import matplotlib.pyplot as plt
import numpy as np

from dynaris import DLM, Autoregressive, LocalLevel
from dynaris.datasets import load_gdp

# --- Data ---
y = load_gdp()

# --- Model: local level + AR(2) for business cycle dynamics ---
model = LocalLevel(sigma_level=0.5, sigma_obs=0.0) + Autoregressive(
    order=2, sigma_ar=2.0, sigma_obs=3.0
)

dlm = DLM(model)
dlm.fit(y).smooth()

print(dlm.summary())
print()

# --- Recession detection: negative filtered level ---
filtered = dlm.filtered_states_df()
level = np.asarray(filtered["state_0"])
n_negative = np.sum(level < 0)
print(f"Quarters with negative filtered level: {n_negative} / {len(level)}")
print()

# --- Forecast 8 quarters (2 years) ---
fc = dlm.forecast(steps=8)
print("8-quarter GDP growth forecast:")
print(fc)
print()

# --- Plots ---
dlm.plot(kind="filtered", title="US GDP Growth -- Filtered")
dlm.plot(kind="smoothed", title="US GDP Growth -- Smoothed")
dlm.forecast(steps=8)
dlm.plot(kind="forecast", title="US GDP Growth -- 2-year Forecast", n_history=40)
dlm.plot(kind="diagnostics", title="US GDP Growth -- Diagnostics")
dlm.plot(
    kind="components",
    component_dims={"Level": 0, "AR(1)": 1, "AR(2)": 2},
    title="US GDP Growth -- Component Decomposition",
)
plt.show()
