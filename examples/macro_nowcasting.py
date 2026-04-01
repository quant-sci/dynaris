"""Macroeconomic nowcasting — Dynamic Factor Model.

Demonstrates the Dynamic Factor Model (DFM) for extracting latent
economic factors from a panel of macroeconomic indicators. This is the
canonical application of DFMs: reducing many noisy economic series to
a small number of interpretable factors.

This example:
1. Simulates a panel of 10 macro indicators driven by 2 latent factors
2. Fits a DFM via EM to recover the factors and loading matrix
3. Plots extracted factors, loadings heatmap, and explained variance
4. Generates multivariate forecasts
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynaris.models.dfm_api import DFMModel

# --- Simulate a macroeconomic panel ---
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
n_obs, n_vars, n_factors = 200, 10, 2

# True factors: "real activity" (trending) and "inflation" (cyclical)
factor_noise = jax.random.normal(k1, (n_obs, n_factors)) * 0.3
true_factors = jnp.cumsum(factor_noise, axis=0)

# True loadings (structured: first 5 load on factor 1, next 5 on factor 2)
true_loadings = jnp.array(
    [
        [1.0, 0.2],
        [0.9, 0.1],
        [0.8, 0.3],
        [0.7, 0.1],
        [1.1, 0.2],
        [0.2, 1.0],
        [0.1, 0.9],
        [0.3, 0.8],
        [0.1, 1.1],
        [0.2, 0.7],
    ]
)

# Observed panel
obs_noise = jax.random.normal(k2, (n_obs, n_vars)) * 0.5
panel = true_factors @ true_loadings.T + obs_noise

# Wrap as DataFrame with indicator names
indicator_names = [
    "GDP",
    "Employment",
    "Industrial Prod",
    "Retail Sales",
    "Housing",
    "CPI",
    "PPI",
    "Wages",
    "Import Prices",
    "Commodities",
]
dates = pd.date_range("2005-01-01", periods=n_obs, freq="MS")
panel_df = pd.DataFrame(np.asarray(panel), index=dates, columns=indicator_names)

print("Panel shape:", panel_df.shape)
print(panel_df.head())
print()

# --- Fit DFM ---
print("Fitting Dynamic Factor Model (2 factors)...")
dfm = DFMModel(n_factors=2, max_iter=100, tol=1e-5)
dfm.fit(panel_df)
print(dfm.summary())
print()

# --- Extracted factors ---
factors_df = dfm.factor_states_df()
print("Extracted factors shape:", factors_df.shape)

# --- Factor correlation with truth ---
for j in range(n_factors):
    corrs = []
    for k in range(n_factors):
        c = abs(
            float(
                jnp.corrcoef(
                    jnp.stack([jnp.array(factors_df.iloc[:, j].values), true_factors[:, k]])
                )[0, 1]
            )
        )
        corrs.append(c)
    print(f"  Factor {j} best correlation with truth: {max(corrs):.3f}")

# --- Forecast ---
fc = dfm.forecast(steps=12)
print(f"\nForecast shape: {fc.shape}")

# --- Plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Factor time series
ax = axes[0, 0]
true_np = np.asarray(true_factors)
for j in range(n_factors):
    ax.plot(dates, factors_df.iloc[:, j], linewidth=1.2, label=f"Estimated factor {j}")
    ax.plot(dates, true_np[:, j], "--", linewidth=0.8, alpha=0.6, label=f"True factor {j}")
ax.set_title("Extracted vs True Factors")
ax.legend(frameon=False, fontsize=7)
ax.set_xlabel("Date")

# Loading matrix heatmap
ax = axes[0, 1]
loadings_np = np.asarray(dfm.result.loadings)
im = ax.imshow(loadings_np, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
ax.set_yticks(range(n_vars))
ax.set_yticklabels(indicator_names, fontsize=7)
ax.set_xticks(range(n_factors))
ax.set_xticklabels([f"Factor {j}" for j in range(n_factors)])
ax.set_title("Loading Matrix")
fig.colorbar(im, ax=ax, shrink=0.8)

# Explained variance
ax = axes[1, 0]
ev_df = dfm.explained_variance_df()
ax.bar(ev_df["factor"], ev_df["proportion"])
ax.set_ylabel("Proportion of variance")
ax.set_title("Explained Variance per Factor")

# Forecast for first 3 variables
ax = axes[1, 1]
for _i, name in enumerate(indicator_names[:3]):
    ax.plot(np.asarray(panel_df[name].values[-30:]), ".-", markersize=2, label=f"{name} (obs)")
    fc_vals = np.asarray(fc[name].values)
    fc_time = np.arange(n_obs - 1, n_obs - 1 + len(fc_vals))
    ax.plot(fc_time, fc_vals, "--", linewidth=1.5, label=f"{name} (forecast)")
ax.set_title("12-Month Forecast (First 3 Variables)")
ax.legend(frameon=False, fontsize=7)
ax.set_xlabel("Time index")

fig.suptitle("Macroeconomic Nowcasting with Dynamic Factor Model")
fig.tight_layout()
plt.show()
