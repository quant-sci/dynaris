Plotting
========

Visualization functions with a clean, minimalist style. All functions are
called internally by :meth:`DLM.plot() <dynaris.dlm.api.DLM.plot>` but can
also be used standalone.

- ``plot_filtered`` --- observed data with filtered state overlay
- ``plot_smoothed`` --- smoothed state estimates with confidence bands
- ``plot_components`` --- decomposition into individual state components
- ``plot_forecast`` --- forecast fan chart with historical context
- ``plot_diagnostics`` --- QQ-plot, ACF, and histogram of residuals

.. autofunction:: dynaris.plotting.plots.plot_filtered

.. autofunction:: dynaris.plotting.plots.plot_smoothed

.. autofunction:: dynaris.plotting.plots.plot_components

.. autofunction:: dynaris.plotting.plots.plot_forecast

.. autofunction:: dynaris.plotting.plots.plot_diagnostics
