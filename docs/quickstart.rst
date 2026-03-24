Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install dynaris

Your first DLM
--------------

Dynaris models are built by composing components with the ``+`` operator,
then wrapped in a :class:`~dynaris.dlm.api.DLM` for a high-level interface.

.. code-block:: python

   from dynaris import LocalLevel, DLM

   # 1. Define a local-level model (random walk + noise)
   model = LocalLevel(sigma_level=38.0, sigma_obs=123.0)

   # 2. Wrap in DLM and fit
   dlm = DLM(model)
   dlm.fit(y)  # y can be a numpy array, JAX array, or pandas Series

   # 3. Inspect results
   print(dlm.summary())

Composing components
--------------------

The real power of dynaris is composition. Combine a trend with seasonality
in a single line:

.. code-block:: python

   from dynaris import LocalLinearTrend, Seasonal, DLM

   model = LocalLinearTrend(sigma_level=2.0, sigma_slope=0.1) + Seasonal(period=12)
   dlm = DLM(model)
   dlm.fit(y)

This produces a single state-space model with block-diagonal system matrices.
The state vector concatenates the trend states (level, slope) with the
seasonal states (period - 1 dimensions).

Smoothing
---------

The Kalman filter processes observations forward in time. The RTS smoother
uses future observations to refine past estimates:

.. code-block:: python

   dlm.fit(y).smooth()

   # Smoothed states have lower variance than filtered states
   df = dlm.smoothed_states_df()

Forecasting
-----------

Generate multi-step-ahead forecasts with uncertainty intervals:

.. code-block:: python

   forecast_df = dlm.forecast(steps=24)
   print(forecast_df)
   #            mean     lower_95    upper_95
   # 0     850.123     612.456    1087.790
   # 1     850.123     598.234    1102.012
   # ...

If you fit with a pandas Series that has a ``DatetimeIndex``, the forecast
DataFrame continues the date index automatically.

Plotting
--------

All plots use a clean, minimalist style with the cividis colormap:

.. code-block:: python

   # Filtered vs observed
   dlm.plot(kind="filtered")

   # Smoothed states
   dlm.plot(kind="smoothed")

   # Forecast fan chart
   dlm.forecast(steps=24)
   dlm.plot(kind="forecast", n_history=36)

   # Residual diagnostics (QQ-plot, ACF, histogram)
   dlm.plot(kind="diagnostics")

   # Component decomposition (requires smooth first)
   dlm.smooth()
   dlm.plot(kind="components", component_dims={
       "Level": 0,
       "Slope": 1,
       "Seasonal": 2,
   })

Parameter estimation
--------------------

Estimate unknown variance parameters via maximum likelihood:

.. code-block:: python

   import jax.numpy as jnp
   from dynaris import LocalLevel
   from dynaris.estimation import fit_mle

   def model_fn(params):
       return LocalLevel(
           sigma_level=jnp.exp(params[0]),
           sigma_obs=jnp.exp(params[1]),
       )

   result = fit_mle(model_fn, y, init_params=jnp.zeros(2))
   print(f"Optimal log-likelihood: {result.log_likelihood:.2f}")
   fitted_model = result.model

Or use the EM algorithm:

.. code-block:: python

   from dynaris.estimation import fit_em

   result = fit_em(y, initial_model, max_iter=100)
   print(f"Converged: {result.converged}")

Batch processing
----------------

Fit or forecast multiple series in parallel with ``jax.vmap``:

.. code-block:: python

   import jax.numpy as jnp

   # y_batch: shape (n_series, T, obs_dim)
   batch_result = dlm.fit_batch(y_batch)
   print(batch_result.log_likelihood)  # shape (n_series,)

All components
--------------

Dynaris provides six composable DLM building blocks:

+-------------------------+------------+--------------------------------------------+
| Component               | State dim  | Description                                |
+=========================+============+============================================+
| ``LocalLevel``          | 1          | Random walk + noise                        |
+-------------------------+------------+--------------------------------------------+
| ``LocalLinearTrend``    | 2          | Level + slope                              |
+-------------------------+------------+--------------------------------------------+
| ``Seasonal``            | period - 1 | Dummy or Fourier form                      |
+-------------------------+------------+--------------------------------------------+
| ``Regression``          | n_regressors | Dynamic/static coefficients              |
+-------------------------+------------+--------------------------------------------+
| ``Autoregressive``      | order      | AR(p) in companion form                    |
+-------------------------+------------+--------------------------------------------+
| ``Cycle``               | 2          | Damped stochastic sinusoid                 |
+-------------------------+------------+--------------------------------------------+

Combine any of these with ``+``:

.. code-block:: python

   model = (
       LocalLinearTrend()
       + Seasonal(period=12)
       + Cycle(period=40, damping=0.95)
   )
   # state_dim = 2 + 11 + 2 = 15
