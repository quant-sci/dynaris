
[![PyPI](https://img.shields.io/pypi/v/dynaris)](https://pypi.org/project/dynaris/)
[![GitHub](https://img.shields.io/github/license/quant-sci/dynaris)](https://github.com/quant-sci/dynaris/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/dynaris/badge/?version=latest)](https://dynaris.readthedocs.io/en/latest/?badge=latest)

**dynaris** is a JAX-powered Python library for state-space models -- from composable DLMs to nonlinear filters, switching systems, Bayesian estimation, and dynamic factor models, all with automatic differentiation and GPU acceleration.

## Installation

```bash
pip install dynaris

# With Bayesian estimation support
pip install dynaris[bayesian]
```

## Documentation

Full documentation at [dynaris.readthedocs.io](https://dynaris.readthedocs.io).

## Quickstart

### DLM: Trend + Seasonality

```python
from dynaris import LocalLinearTrend, Seasonal, DLM
from dynaris.datasets import load_airline

model = LocalLinearTrend() + Seasonal(period=12)
dlm = DLM(model)
dlm.fit(load_airline()).smooth()
dlm.forecast(steps=24)
dlm.plot(kind="panel")
```

### Nonlinear Filtering

```python
from dynaris import SSM, LorenzAttractor

model = LorenzAttractor(dt=0.01, obs_noise=2.0)
ssm = SSM(model, filter="ukf")  # auto-selects UKF for nonlinear models
ssm.fit(observations)
```

### Regime Switching

```python
from dynaris import LocalLevel, MarkovSwitchingSSM
from dynaris.filters import hamilton_filter
from dynaris.smoothers import kim_smooth
import jax.numpy as jnp

switching = MarkovSwitchingSSM(
    models=(LocalLevel(1, 5), LocalLevel(5, 20)),
    transition_matrix=jnp.array([[0.95, 0.05], [0.10, 0.90]]),
    initial_probs=jnp.array([0.5, 0.5]),
)
result = hamilton_filter(switching, observations)
smoothed = kim_smooth(switching, result)
```

### Bayesian Estimation

```python
from dynaris import LocalLevel, fit_bayesian
from dynaris.estimation.priors import inverse_gamma_log_prior

def model_fn(params):
    return LocalLevel(sigma_level=jnp.exp(params[0]), sigma_obs=jnp.exp(params[1]))

result = fit_bayesian(model_fn, observations, jnp.zeros(2),
                      log_prior_fn=inverse_gamma_log_prior(shape=2.0, scale=1.0))
# result.samples -> (n_samples, n_params) posterior draws
```

### Dynamic Factor Models

```python
from dynaris.models import DFMModel

dfm = DFMModel(n_factors=2)
dfm.fit(panel_data)  # (T, m) multivariate panel
print(dfm.loadings_df())
print(dfm.factor_states_df())
dfm.forecast(steps=12)
```

## Components

Build DLMs by combining components with `+`:

| Component | State dim | Description |
|-----------|-----------|-------------|
| `LocalLevel` | 1 | Random walk + noise |
| `LocalLinearTrend` | 2 | Level + slope |
| `Seasonal` | period - 1 | Dummy or Fourier form |
| `Regression` | n_regressors | Dynamic/static coefficients |
| `Autoregressive` | order | AR(p) in companion form |
| `Cycle` | 2 | Damped stochastic sinusoid |

## Filters & Smoothers

| Algorithm | Model type | Use case |
|-----------|-----------|----------|
| Kalman filter | Linear | Exact inference for DLMs |
| Extended KF (EKF) | Nonlinear | First-order linearization |
| Unscented KF (UKF) | Nonlinear | Sigma-point propagation |
| Particle filter (SMC) | Any | Non-Gaussian, multi-modal |
| Hamilton filter | Switching | Markov regime models |
| RTS smoother | Linear | Retrospective state estimation |
| Kim smoother | Switching | Retrospective regime inference |

## Parameter Estimation

| Method | Function | Description |
|--------|----------|-------------|
| MLE | `fit_mle()` | Gradient-based via `jax.grad` + scipy |
| EM | `fit_em()` | Expectation-Maximization for variances |
| Bayesian | `fit_bayesian()` | NUTS/HMC via NumPyro |
| DFM-EM | `fit_dfm_em()` | EM with loading matrix updates |

## Built-in Nonlinear Models

| Model | Description |
|-------|-------------|
| `StochasticVolatility` | AR(1) log-volatility (KSC linearization) |
| `BearingsTracking` | 2D constant-velocity target, bearing observations |
| `LorenzAttractor` | Chaotic 3D system (Euler discretization) |

## Datasets

| Dataset | Loader | N | Frequency | Domain |
|---------|--------|---|-----------|--------|
| Nile river flow | `load_nile()` | 100 | Annual | Hydrology |
| Airline passengers | `load_airline()` | 144 | Monthly | Transportation |
| Lynx population | `load_lynx()` | 114 | Annual | Ecology |
| Sunspot numbers | `load_sunspots()` | 288 | Annual | Astronomy |
| Global temperature | `load_temperature()` | 144 | Annual | Climate |
| US GDP growth | `load_gdp()` | 319 | Quarterly | Economics |

## Performance

All filters run inside `jax.lax.scan` with `@jax.jit` -- GPU/TPU acceleration is automatic. Additional features:

- **Batch processing** via `jax.vmap` for parallel multi-series inference
- **Memory-efficient** long series via `jax.checkpoint` (trade compute for memory)
- **Parallel MCMC** chains via NumPyro's `chain_method="parallel"`
- **Pure NumPy backend** for lightweight / no-GPU environments

## License

MIT License. See [LICENSE](LICENSE) for details.
