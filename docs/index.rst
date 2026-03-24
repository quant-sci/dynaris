Dynaris provides a fast, composable, JAX-native DLM library with autodiff
parameter estimation, multi-step forecasting, and clean visualization.

.. code-block:: python

   from dynaris import LocalLevel, Seasonal, DLM

   model = LocalLevel() + Seasonal(period=12)
   dlm = DLM(model)
   dlm.fit(y)
   dlm.forecast(steps=12)
   dlm.plot()

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   math
   api/index
