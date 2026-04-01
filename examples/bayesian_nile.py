"""Bayesian estimation — airline passengers with trend + seasonality.

Demonstrates full Bayesian inference for a composed DLM (local linear
trend + seasonal) using NumPyro's NUTS sampler. The airline passenger
data is log-transformed to convert multiplicative seasonality to additive.

This example:
1. Fits a trend + seasonal model to log(airline passengers) via MLE
2. Fits the same model via Bayesian MCMC with a normal prior
3. Compares posterior mean with MLE point estimates
4. Generates posterior predictive forecasts with credible intervals
5. Runs prior predictive checks
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dynaris import LocalLinearTrend, Seasonal
from dynaris.core.state_space import StateSpaceModel
from dynaris.datasets import load_airline
from dynaris.estimation import fit_bayesian, fit_mle
from dynaris.estimation.predictive import posterior_predictive_forecast, prior_predictive
from dynaris.estimation.priors import normal_log_prior

# --- Data (log-transform for additive seasonality) ---
y = load_airline()
obs = jnp.log(jnp.array(y.values, dtype=jnp.float32)).reshape(-1, 1)
print(f"Airline passengers: {len(obs)} monthly observations (log-transformed)")


# --- Model factory: trend + seasonal (period=12) ---
# params: [log_sigma_level, log_sigma_slope, log_sigma_seasonal, log_sigma_obs]
def model_fn(params: jax.Array) -> StateSpaceModel:
    trend = LocalLinearTrend(
        sigma_level=jnp.exp(params[0]),
        sigma_slope=jnp.exp(params[1]),
        sigma_obs=0.0,
    )
    seasonal = Seasonal(
        period=12,
        sigma_seasonal=jnp.exp(params[2]),
        sigma_obs=jnp.exp(params[3]),
    )
    return trend + seasonal


# --- MLE fit (baseline) ---
print("\nFitting MLE...")
init_params = jnp.array([0.0, -2.0, -2.0, -1.0])
mle_result = fit_mle(model_fn, obs, init_params)
print(f"  MLE params: {np.round(np.asarray(mle_result.params), 3)}")
print(f"  MLE log-likelihood: {mle_result.log_likelihood:.2f}")

# --- Bayesian fit ---
print("\nRunning Bayesian MCMC (NUTS)...")
prior = normal_log_prior(loc=0.0, scale=5.0)
bayes_result = fit_bayesian(
    model_fn,
    obs,
    init_params=mle_result.params,
    log_prior_fn=prior,
    n_warmup=500,
    n_samples=1000,
    key=jax.random.PRNGKey(42),
    param_names=(
        "log_sigma_level",
        "log_sigma_slope",
        "log_sigma_seasonal",
        "log_sigma_obs",
    ),
)

samples = bayes_result.samples
names = bayes_result.param_names or ()
print("  Posterior means:")
for j, name in enumerate(names):
    post_mean = float(jnp.mean(samples[:, j]))
    mle_val = float(mle_result.params[j])
    print(f"    {name}: {post_mean:.3f} (MLE: {mle_val:.3f})")
mean_ll = float(jnp.mean(bayes_result.log_likelihood_samples))
print(f"  Posterior mean log-likelihood: {mean_ll:.2f}")
if bayes_result.info:
    print(f"  Divergences: {bayes_result.info.get('n_divergences', 'N/A')}")

# --- Posterior predictive forecast (24 months ahead) ---
print("\nGenerating posterior predictive forecast (24 months)...")
fc = posterior_predictive_forecast(
    bayes_result,
    model_fn,
    obs,
    steps=24,
    n_posterior_samples=200,
)

# --- Prior predictive ---
print("Running prior predictive check...")


def prior_sample_fn(key: jax.Array) -> jax.Array:
    return jax.random.normal(key, (4,)) * 1.0 + mle_result.params


prior_sims = prior_predictive(model_fn, prior_sample_fn, n_steps=144, n_samples=50)

# --- Plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Posterior parameter distributions
ax = axes[0, 0]
for j, name in enumerate(names):
    ax.hist(np.asarray(samples[:, j]), bins=25, alpha=0.6, label=name)
    ax.axvline(float(mle_result.params[j]), color=f"C{j}", linestyle="--", linewidth=1)
ax.set_xlabel("Parameter value (log scale)")
ax.set_ylabel("Count")
ax.set_title("Posterior Distributions (dashed = MLE)")
ax.legend(frameon=False, fontsize=7)

# Posterior predictive forecast (back-transformed to passenger scale)
ax = axes[0, 1]
n_obs = len(obs)
time_obs = np.arange(n_obs)
time_fc = np.arange(n_obs, n_obs + 24)
ax.plot(
    time_obs[-48:],
    np.exp(np.asarray(obs[-48:, 0])),
    "k.-",
    markersize=2,
    label="Observed",
)
ax.plot(
    time_fc,
    np.exp(np.asarray(fc["mean"][:, 0])),
    "C0-",
    linewidth=1.5,
    label="Forecast mean",
)
ax.fill_between(
    time_fc,
    np.exp(np.asarray(fc["lower"][:, 0])),
    np.exp(np.asarray(fc["upper"][:, 0])),
    alpha=0.3,
    color="C0",
    label="95% credible interval",
)
ax.set_xlabel("Month index")
ax.set_ylabel("Passengers")
ax.set_title("Posterior Predictive Forecast (24 months)")
ax.legend(frameon=False, fontsize=8)

# Prior predictive (log scale — simulations start from zero)
ax = axes[1, 0]
for i in range(min(20, prior_sims.shape[0])):
    ax.plot(np.asarray(prior_sims[i, :, 0]), alpha=0.4, linewidth=0.5, color="gray")
ax.axhline(float(jnp.mean(obs)), color="k", linestyle="--", linewidth=0.8, label="Data mean")
ax.set_xlabel("Time step")
ax.set_ylabel("log(passengers)")
ax.set_title("Prior Predictive Samples (log scale)")
ax.legend(frameon=False, fontsize=8)

# Trace plot
ax = axes[1, 1]
for j, name in enumerate(names):
    ax.plot(np.asarray(samples[:, j]), alpha=0.6, linewidth=0.4, label=name)
ax.set_xlabel("Sample index")
ax.set_ylabel("Value")
ax.set_title("MCMC Traces")
ax.legend(frameon=False, fontsize=7)

fig.suptitle("Bayesian Estimation — Airline Passengers (Trend + Seasonality)")
fig.tight_layout()
plt.show()
