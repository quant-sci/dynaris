Examples
========

Ready-to-run scripts demonstrating dynaris on real-world datasets. Each
example lives in the ``examples/`` directory of the
`repository <https://github.com/quant-sci/dynaris/tree/main/examples>`_.

Classic / Introductory
----------------------

**Nile River Flow** --- ``nile_river.py``
   Local level model on annual Nile discharge (1871--1970). The simplest
   possible DLM: a random walk observed with noise. Demonstrates fitting,
   smoothing, and forecasting.

**Trend + Seasonality** --- ``trend_seasonal.py``
   Simulated sales data with a local linear trend and monthly seasonality.
   Shows how to compose components and extract individual contributions.

Economics & Finance
-------------------

**Airline Passengers** --- ``airline_passengers.py``
   Box-Jenkins airline data (1949--1960). Trend plus seasonal decomposition,
   forecasting, and component visualization.

**GDP Business Cycle** --- ``gdp_business_cycle.py``
   US quarterly GDP growth. Local level plus AR(2) to capture business
   cycle dynamics.

**Stochastic Volatility** --- ``stochastic_volatility.py``
   Particle filter on a stochastic volatility model for financial returns.
   Compares EKF, UKF, and particle filter on latent log-volatility recovery
   using the Kim-Shephard-Chib linearization.

**Regime Switching** --- ``regime_switching.py``
   Hamilton filter and Kim smoother detecting volatility regimes. Simulates
   a 2-regime process (calm vs volatile) and tracks filtered and smoothed
   regime probabilities over time.

Natural Sciences
----------------

**Lynx Population** --- ``lynx_population.py``
   Canadian lynx trappings (1821--1934). Uses an autoregressive component to
   model the ~10-year population cycle.

**Sunspot Cycles** --- ``sunspot_cycles.py``
   Annual sunspot numbers (1700--1987). Level plus cycle component to detect
   the ~11-year solar cycle.

**Global Temperature** --- ``global_temperature.py``
   Annual temperature anomaly (1880--2023). Linear trend detection in climate
   data.

Nonlinear Filtering
-------------------

**Lorenz Attractor Tracking** --- ``lorenz_tracking.py``
   Compares EKF, UKF, and particle filter on the chaotic Lorenz system.
   Simulates a 3D trajectory with partial (2D) noisy observations and
   tracks the full state. Plots per-component tracking accuracy.

**Bearings-Only Tracking** --- ``bearings_tracking.py``
   EKF on a 2D target tracking problem with nonlinear bearing-angle
   observations. Demonstrates the classic radar/sonar tracking scenario
   where the sensor measures only the angle to the target.

Bayesian Estimation
-------------------

**Bayesian Airline Passengers** --- ``bayesian_nile.py``
   Full Bayesian inference for a trend + seasonal model on log-transformed
   airline data using NumPyro's NUTS sampler. Compares posterior with MLE,
   generates posterior predictive forecasts with credible intervals, and
   runs prior predictive checks.

Dynamic Factor Models
---------------------

**Macroeconomic Nowcasting** --- ``macro_nowcasting.py``
   Extracts 2 latent factors from a panel of 10 simulated macroeconomic
   indicators via DFM-EM. Plots factor time series, loading matrix heatmap,
   explained variance, and multivariate forecasts.

**Multi-Sensor Fusion** --- ``sensor_fusion.py``
   Fuses 5 noisy temperature sensors into a single optimal estimate using
   a 1-factor DFM. Demonstrates how the DFM weights sensors by their
   reliability (inverse noise variance).

Advanced
--------

**Dynamic Regression** --- ``dynamic_regression.py``
   Time-varying regression coefficients. Shows how regression weights evolve
   over time when ``sigma_reg > 0``.

**Panel Overview** --- ``panel_overview.py``
   Demonstrates the multi-panel plot combining filtered, smoothed, forecast,
   and diagnostic views in a single figure.
