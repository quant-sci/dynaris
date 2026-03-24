Forecasting
===========

Multi-step-ahead predictions with uncertainty quantification and batch
processing. Forecasts can be initialized from either the filtered or
smoothed terminal state:

- **From filtered state** (``forecast_from_filter``): uses observations up to
  time :math:`T` only
- **From smoothed state** (``forecast_from_smoother``): uses the full dataset
  for a more refined starting point

The high-level ``forecast`` function is called internally by
:meth:`DLM.forecast() <dynaris.dlm.api.DLM.forecast>`.

.. autofunction:: dynaris.forecast.forecast.forecast

.. autofunction:: dynaris.forecast.forecast.forecast_from_filter

.. autofunction:: dynaris.forecast.forecast.forecast_from_smoother

.. autofunction:: dynaris.forecast.forecast.confidence_bands

.. autofunction:: dynaris.forecast.forecast.forecast_batch

.. autofunction:: dynaris.forecast.forecast.fit_batch

.. autoclass:: dynaris.forecast.forecast.ForecastResult
   :members:
   :no-index:
