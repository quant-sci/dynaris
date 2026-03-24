Key Concepts
============

What is a Dynamic Linear Model?
--------------------------------

A Dynamic Linear Model (DLM) is a state-space model where both the state
evolution and observation equations are linear and driven by Gaussian noise.
DLMs are a flexible framework for decomposing a time series into interpretable
components --- trend, seasonality, regression effects, cycles --- and
forecasting with uncertainty.

At each time step :math:`t`, a DLM is defined by:

- A **state** :math:`\boldsymbol{\theta}_t` that evolves over time (e.g., the
  current level, slope, and seasonal effects)
- An **observation** :math:`Y_t` that is a noisy linear function of the state

The Kalman filter estimates the state given observations, and the RTS smoother
refines those estimates using the full dataset.

Components and composition
--------------------------

dynaris provides six building blocks, each defining a specific structure
for the state-space matrices:

+-------------------------+------------+--------------------------------------------+
| Component               | State dim  | Description                                |
+=========================+============+============================================+
| ``LocalLevel``          | 1          | Random walk plus noise                     |
+-------------------------+------------+--------------------------------------------+
| ``LocalLinearTrend``    | 2          | Level plus slope                           |
+-------------------------+------------+--------------------------------------------+
| ``Seasonal``            | period - 1 | Dummy or Fourier seasonal effects          |
+-------------------------+------------+--------------------------------------------+
| ``Regression``          | n_regressors | Dynamic or static coefficients           |
+-------------------------+------------+--------------------------------------------+
| ``Autoregressive``      | order      | AR(p) in companion form                    |
+-------------------------+------------+--------------------------------------------+
| ``Cycle``               | 2          | Damped stochastic sinusoid                 |
+-------------------------+------------+--------------------------------------------+

Combine any of these with the ``+`` operator:

.. code-block:: python

   from dynaris import LocalLinearTrend, Seasonal, Cycle

   model = LocalLinearTrend() + Seasonal(period=12) + Cycle(period=40, damping=0.95)
   # Combined state_dim = 2 + 11 + 2 = 15

Under the hood, this builds a single ``StateSpaceModel`` with block-diagonal
system matrices (the **superposition principle** from West and Harrison, 1997).

The workflow
------------

A typical dynaris session follows four steps:

1. **Build** --- compose components into a model
2. **Fit** --- run the Kalman filter on observed data
3. **Smooth** --- run the RTS smoother for refined estimates
4. **Forecast** --- project forward with uncertainty

.. code-block:: python

   from dynaris import LocalLevel, DLM

   model = LocalLevel()
   dlm = DLM(model)
   dlm.fit(y)           # Kalman filter
   dlm.smooth()         # RTS smoother
   fc = dlm.forecast(steps=12)
   dlm.plot(kind="panel")

Notation
--------

dynaris follows the West and Harrison (1997) notation throughout:

+----------+-------------------------------+------------------------------------------+
| Symbol   | Code                          | Meaning                                  |
+==========+===============================+==========================================+
| **F**    | ``model.F`` / ``observation_matrix``  | Observation (regression) vector  |
+----------+-------------------------------+------------------------------------------+
| **G**    | ``model.G`` / ``system_matrix``       | System (evolution) matrix        |
+----------+-------------------------------+------------------------------------------+
| **V**    | ``model.V`` / ``obs_cov``             | Observational variance           |
+----------+-------------------------------+------------------------------------------+
| **W**    | ``model.W`` / ``evolution_cov``       | Evolution covariance             |
+----------+-------------------------------+------------------------------------------+

For the full mathematical treatment, see :doc:`/math`.

References
----------

- West, M. and Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models*,
  2nd edition. Springer.
