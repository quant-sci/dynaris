Batch Processing
================

dynaris supports fitting and forecasting multiple time series in parallel
using ``jax.vmap``.

Batch fitting
-------------

Pass a batch of series as a 3D array with shape ``(n_series, T, obs_dim)``:

.. code-block:: python

   import jax.numpy as jnp
   from dynaris import LocalLevel, DLM

   model = LocalLevel()
   dlm = DLM(model)

   # y_batch: (n_series, T, 1)
   batch_result = dlm.fit_batch(y_batch)
   print(batch_result.log_likelihood)  # shape (n_series,)

Each series is filtered independently, but all series run in parallel on the
same hardware (CPU cores or GPU).

Batch forecasting
-----------------

After batch fitting, generate forecasts for all series at once:

.. code-block:: python

   from dynaris.forecast import forecast_batch

   fc = forecast_batch(batch_result, model, steps=12)

Low-level API
-------------

The batch functions wrap ``jax.vmap`` over the single-series equivalents:

.. code-block:: python

   from dynaris.forecast import fit_batch, forecast_batch

   batch_filter = fit_batch(model, y_batch)
   batch_fc = forecast_batch(batch_filter, model, steps=12)

See :doc:`/api/forecast` for the full API.
