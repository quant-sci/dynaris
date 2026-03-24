<h1 align="left">
<img src="public/logo.png" width="100">
</h1>

[![PyPI](https://img.shields.io/pypi/v/dynaris)](https://pypi.org/project/dynaris/)
[![GitHub](https://img.shields.io/github/license/quant-sci/dynaris)](https://github.com/quant-sci/dynaris/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/dynaris/badge/?version=latest)](https://dynaris.readthedocs.io/en/latest/?badge=latest)

**dynaris** is a JAX-powered Python library for Dynamic Linear Models -- from composable DLM components to Kalman filtering, smoothing, forecasting, and parameter estimation, all with automatic differentiation.

## Installation

```bash
uv add dynaris
```

## Quickstart

```python
from dynaris import LocalLevel, Seasonal, DLM
from dynaris.datasets import load_nile

# Load data
y = load_nile()

# Build a model by composing components
model = LocalLevel(sigma_level=38.0, sigma_obs=123.0) + Seasonal(period=12)

# Fit, smooth, forecast
dlm = DLM(model)
dlm.fit(y).smooth()
fc = dlm.forecast(steps=12)

# Print summary
print(dlm.summary())

# Single-figure overview
dlm.plot(kind="panel")
```

## Components

Build models by combining components with `+`:

```python
from dynaris import LocalLinearTrend, Seasonal, Cycle, Autoregressive, Regression

model = (
    LocalLinearTrend(sigma_level=1.0, sigma_slope=0.1)
    + Seasonal(period=12, sigma_seasonal=0.5)
    + Cycle(period=40, damping=0.95)
)
```

| Component | State dim | Description |
|-----------|-----------|-------------|
| `LocalLevel` | 1 | Random walk + noise |
| `LocalLinearTrend` | 2 | Level + slope |
| `Seasonal` | period - 1 | Dummy or Fourier form |
| `Regression` | n_regressors | Dynamic/static coefficients |
| `Autoregressive` | order | AR(p) in companion form |
| `Cycle` | 2 | Damped stochastic sinusoid |

## Parameter Estimation

```python
import jax.numpy as jnp
from dynaris import LocalLevel
from dynaris.estimation import fit_mle

def model_fn(params):
    return LocalLevel(
        sigma_level=jnp.exp(params[0]),
        sigma_obs=jnp.exp(params[1]),
    )

result = fit_mle(model_fn, y, init_params=jnp.zeros(2))
print(f"Log-likelihood: {result.log_likelihood:.2f}")
```

## Datasets

```python
from dynaris.datasets import load_nile, load_airline, load_lynx, load_sunspots, load_temperature, load_gdp

y = load_airline()     # 144 monthly obs, 1949-1960
y = load_lynx()        # 114 annual obs, 1821-1934 (~10-year cycle)
y = load_sunspots()    # 288 annual obs, 1700-1987 (~11-year cycle)
y = load_temperature() # 144 annual obs, 1880-2023 (warming trend)
y = load_gdp()         # 319 quarterly obs, 1947-2026 (business cycle)
```

## Notation

Dynaris follows the West & Harrison (1997) notation:

| Symbol | Code | Meaning |
|--------|------|---------|
| **G** | `model.G` / `system_matrix` | System (evolution) matrix |
| **F** | `model.F` / `observation_matrix` | Observation (regression) matrix |
| **W** | `model.W` / `evolution_cov` | Evolution covariance |
| **V** | `model.V` / `obs_cov` | Observational variance |

## License

MIT License. See [LICENSE](LICENSE) for details.
