Quickstart
==========

This page walks through the core dynaris workflow: build a model, fit it to
data, smooth, forecast, and plot.

Your first DLM
--------------

The simplest DLM is a **local level** model --- a random walk observed with
noise. Let's fit one to the classic Nile river dataset:

.. code-block:: python

   from dynaris import LocalLevel, DLM
   from dynaris.datasets import load_nile

   # 1. Load data (100 annual observations)
   y = load_nile()

   # 2. Define a local-level model
   model = LocalLevel(sigma_level=38.0, sigma_obs=123.0)

   # 3. Wrap in DLM and fit
   dlm = DLM(model)
   dlm.fit(y)

   # 4. Inspect results
   print(dlm.summary())

The ``fit`` method runs the Kalman filter forward through the observations,
computing filtered state estimates and the log-likelihood.

Composing components
--------------------

The real power of dynaris is composition. Combine a trend with seasonality
in a single line:

.. code-block:: python

   from dynaris import LocalLinearTrend, Seasonal, DLM
   from dynaris.datasets import load_airline

   y = load_airline()  # 144 monthly observations

   model = LocalLinearTrend(sigma_level=2.0, sigma_slope=0.1) + Seasonal(period=12)
   dlm = DLM(model)
   dlm.fit(y)

This produces a single state-space model with block-diagonal system matrices.
See :doc:`/user-guide/components` for all six available components.

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

If you fit with a pandas Series that has a ``DatetimeIndex``, the forecast
DataFrame continues the date index automatically.

See :doc:`/user-guide/forecasting` for advanced options.

Plotting
--------

View results with a single call:

.. code-block:: python

   # Single-figure overview (filtered, smoothed, forecast, diagnostics)
   dlm.plot(kind="panel")

   # Or individual plot types
   dlm.plot(kind="filtered")
   dlm.plot(kind="forecast", n_history=36)

See :doc:`/user-guide/plotting` for all available plot kinds.

Next steps
----------

- :doc:`concepts` --- understand the DLM framework and notation
- :doc:`/user-guide/components` --- explore all six building blocks
- :doc:`/user-guide/estimation` --- learn about parameter estimation (MLE and EM)
- :doc:`/math` --- full mathematical foundations
