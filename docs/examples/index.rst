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

Advanced
--------

**Dynamic Regression** --- ``dynamic_regression.py``
   Time-varying regression coefficients. Shows how regression weights evolve
   over time when ``sigma_reg > 0``.

**Panel Overview** --- ``panel_overview.py``
   Demonstrates the multi-panel plot combining filtered, smoothed, forecast,
   and diagnostic views in a single figure.
