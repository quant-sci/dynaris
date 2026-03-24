Plotting
========

dynaris provides minimalist visualization functions. All plots are accessible
through the ``DLM.plot()`` method via the ``kind`` parameter.

Plot kinds
----------

filtered
~~~~~~~~

Overlay filtered state estimates on the observed data.

.. code-block:: python

   dlm.fit(y)
   dlm.plot(kind="filtered")

Shows the Kalman filter's one-step-ahead estimates with confidence intervals.

smoothed
~~~~~~~~

Display smoothed (retrospective) state estimates.

.. code-block:: python

   dlm.fit(y).smooth()
   dlm.plot(kind="smoothed")

Smoothed estimates have lower variance because they use the full dataset.

forecast
~~~~~~~~

Fan chart showing the forecast mean and confidence bands, with recent
historical observations for context.

.. code-block:: python

   dlm.forecast(steps=24)
   dlm.plot(kind="forecast", n_history=36)

The ``n_history`` parameter controls how many past observations appear.

diagnostics
~~~~~~~~~~~

Residual diagnostic panel with QQ-plot, ACF, and histogram.

.. code-block:: python

   dlm.plot(kind="diagnostics")

Use this to check whether the model's assumptions hold.

components
~~~~~~~~~~

Decompose the series into individual state components. Requires smoothing
first and a mapping of component names to state dimensions:

.. code-block:: python

   dlm.smooth()
   dlm.plot(kind="components", component_dims={
       "Level": 0,
       "Slope": 1,
       "Seasonal": 2,
   })

panel
~~~~~

A single-figure overview combining filtered, smoothed, forecast, and
diagnostics:

.. code-block:: python

   dlm.fit(y).smooth()
   dlm.forecast(steps=12)
   dlm.plot(kind="panel")

Customization
-------------

All plot methods accept an optional ``ax`` parameter to draw on an existing
Matplotlib axes, and a ``title`` parameter:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots()
   dlm.plot(kind="filtered", ax=ax, title="Nile River Flow")
   plt.show()

See :doc:`/api/plotting` for the full function signatures.
