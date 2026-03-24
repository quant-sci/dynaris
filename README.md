<h1 align="left">
<img src="public/logo.png" width="100">
</h1>

[![PyPI](https://img.shields.io/pypi/v/dynaris)](https://pypi.org/project/dynaris/)
[![GitHub](https://img.shields.io/github/license/quant-sci/dynaris)](https://github.com/quant-sci/dynaris/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/dynaris/badge/?version=latest)](https://dynaris.readthedocs.io/en/latest/?badge=latest)

**dynaris** is a JAX-powered Python library for Dynamic Linear Models -- from composable DLM components to Kalman filtering, smoothing, forecasting, and parameter estimation, all with automatic differentiation.

## Installation

```bash
pip install dynaris
# or
uv add dynaris
```

## Documentation

Full documentation is available at [dynaris.readthedocs.io](https://dynaris.readthedocs.io).

## Quickstart

```python
from dynaris import LocalLevel, DLM
from dynaris.datasets import load_nile

# Load data
y = load_nile()

# Build a local-level model and fit
dlm = DLM(LocalLevel(sigma_level=38.0, sigma_obs=123.0))
dlm.fit(y).smooth()

# Forecast and plot
fc = dlm.forecast(steps=10)
print(dlm.summary())
dlm.plot(kind="panel")
```

## Components

Build models by combining components with `+`:

```python
from dynaris import LocalLinearTrend, Seasonal, Cycle

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

| Dataset | Loader | N | Frequency | Domain |
|---------|--------|---|-----------|--------|
| Nile river flow | `load_nile()` | 100 | Annual | Hydrology |
| Airline passengers | `load_airline()` | 144 | Monthly | Transportation |
| Lynx population | `load_lynx()` | 114 | Annual | Ecology |
| Sunspot numbers | `load_sunspots()` | 288 | Annual | Astronomy |
| Global temperature | `load_temperature()` | 144 | Annual | Climate |
| US GDP growth | `load_gdp()` | 319 | Quarterly | Economics |

## License

MIT License. See [LICENSE](LICENSE) for details.
