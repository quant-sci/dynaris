"""Nile river flow — classic DLM example.

Demonstrates the local-level model on the Nile river annual flow dataset
(1871–1970). This is the canonical introductory example from
Durbin & Koopman (2012), "Time Series Analysis by State Space Methods".
"""

import jax.numpy as jnp
import pandas as pd

from dynaris import DLM, LocalLevel

# --- Data ---
nile = jnp.array([
    1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140,
    995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140,
    1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840,
    874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969,
    831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821,
    768, 845, 864, 862, 698, 845, 744, 796, 1040, 759,
    781, 865, 845, 944, 984, 897, 822, 1010, 771, 676,
    649, 846, 812, 742, 801, 1040, 860, 874, 848, 890,
    744, 749, 838, 1050, 918, 986, 797, 923, 975, 815,
    1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740,
], dtype=jnp.float32)

y = pd.Series(nile, index=pd.date_range("1871", periods=100, freq="YE"), name="flow")

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
