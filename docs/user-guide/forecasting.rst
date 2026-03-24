Forecasting
===========

dynaris produces multi-step-ahead forecasts with uncertainty intervals by
iterating the state-space prior equations forward without new observations.

Basic forecasting
-----------------

After fitting (and optionally smoothing), call ``forecast``:

.. code-block:: python

   from dynaris import LocalLinearTrend, Seasonal, DLM
   from dynaris.datasets import load_airline

   y = load_airline()
   model = LocalLinearTrend() + Seasonal(period=12)
   dlm = DLM(model)
   dlm.fit(y).smooth()

   forecast_df = dlm.forecast(steps=24)
   print(forecast_df)
   #            mean     lower_95    upper_95
   # ...

The returned DataFrame contains the forecast mean and 95% confidence bands.

DatetimeIndex propagation
-------------------------

If you fit with a pandas Series that has a ``DatetimeIndex``, the forecast
DataFrame continues the date index:

.. code-block:: python

   # y has monthly DatetimeIndex 1949-01 to 1960-12
   forecast_df = dlm.forecast(steps=12)
   # Index continues: 1961-01, 1961-02, ...

Filtered vs smoothed initialization
------------------------------------

Forecasts can start from either the filtered or smoothed terminal state:

- **Filtered** (default after ``fit``): uses only past observations up to
  time :math:`T`
- **Smoothed** (after ``smooth``): uses the full dataset, giving a more
  refined starting point

The lower-level functions give explicit control:

.. code-block:: python

   from dynaris.forecast import forecast_from_filter, forecast_from_smoother

   fc_filt = forecast_from_filter(filter_result, model, steps=12)
   fc_smooth = forecast_from_smoother(smoother_result, model, steps=12)

Confidence bands
----------------

Confidence bands widen with the forecast horizon as evolution noise
accumulates:

.. code-block:: python

   from dynaris.forecast import confidence_bands

   lower, upper = confidence_bands(forecast_result, level=0.95)

Visualize with:

.. code-block:: python

   dlm.plot(kind="forecast", n_history=36)

See :doc:`/api/forecast` for the full API.
