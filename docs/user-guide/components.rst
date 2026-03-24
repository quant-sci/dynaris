Components
==========

dynaris provides six composable DLM building blocks. Each function returns a
:class:`~dynaris.core.state_space.StateSpaceModel` that can be combined with
others using the ``+`` operator.

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

Trend components
----------------

LocalLevel
~~~~~~~~~~

The simplest DLM: a random walk observed with noise.

.. math::

   \mu_t = \mu_{t-1} + \omega_t, \quad
   Y_t = \mu_t + \nu_t

.. code-block:: python

   from dynaris import LocalLevel

   model = LocalLevel(sigma_level=1.0, sigma_obs=1.0)

**When to use:** stationary or slowly changing series without a clear trend
or seasonality. Classic example: Nile river annual flow.

LocalLinearTrend
~~~~~~~~~~~~~~~~

Extends ``LocalLevel`` with a slope (growth rate) component.

.. math::

   \mu_t = \mu_{t-1} + \beta_{t-1} + \omega_{\mu,t}, \quad
   \beta_t = \beta_{t-1} + \omega_{\beta,t}

.. code-block:: python

   from dynaris import LocalLinearTrend

   model = LocalLinearTrend(sigma_level=2.0, sigma_slope=0.1, sigma_obs=1.0)

**When to use:** series with a changing trend direction, such as GDP growth
or temperature anomalies.

Periodic components
-------------------

Seasonal
~~~~~~~~

Models repeating patterns at a fixed period. Supports both **dummy** (default)
and **Fourier** forms.

.. code-block:: python

   from dynaris import Seasonal

   # Dummy-form seasonal (default)
   model = Seasonal(period=12, sigma_seasonal=0.5)

   # Fourier-form seasonal
   model = Seasonal(period=12, sigma_seasonal=0.5, form="fourier")

**When to use:** monthly, quarterly, or weekly data with repeating patterns.
State dimension is ``period - 1``.

See :doc:`/math` for the mathematical formulation.

Cycle
~~~~~

A damped stochastic sinusoid for quasi-periodic behavior.

.. code-block:: python

   from dynaris import Cycle

   model = Cycle(period=40, damping=0.95, sigma_cycle=1.0)

**When to use:** series with approximate cycles of known period, such as
sunspot activity (~11 years) or business cycles. Setting ``damping=1.0``
gives an undamped cycle; values below 1.0 let the cycle decay.

Other components
----------------

Regression
~~~~~~~~~~

Dynamic (time-varying) or static regression coefficients.

.. code-block:: python

   from dynaris import Regression

   # Dynamic coefficients (random walk)
   model = Regression(n_regressors=2, sigma_reg=0.1)

   # Static coefficients (set sigma_reg=0)
   model = Regression(n_regressors=2, sigma_reg=0.0)

**When to use:** when external predictors (regressors) influence the series.
With ``sigma_reg > 0``, coefficients evolve over time; with ``sigma_reg = 0``,
they are constant.

Autoregressive
~~~~~~~~~~~~~~

An AR(p) process in companion-form state space.

.. code-block:: python

   from dynaris import Autoregressive

   model = Autoregressive(coefficients=[0.5, -0.3], sigma_ar=1.0)

**When to use:** series with serial correlation not captured by trend or
seasonal components. Common in population dynamics (e.g., lynx data).

Composing components
--------------------

Combine any components with ``+`` to build richer models:

.. code-block:: python

   from dynaris import LocalLinearTrend, Seasonal, Cycle, DLM

   model = (
       LocalLinearTrend(sigma_level=2.0, sigma_slope=0.1)
       + Seasonal(period=12, sigma_seasonal=0.5)
       + Cycle(period=40, damping=0.95)
   )
   # state_dim = 2 + 11 + 2 = 15

   dlm = DLM(model)
   dlm.fit(y)

This uses the DLM **superposition principle**: system matrices become
block-diagonal, and observation noise adds across components. See
:doc:`/math` for details.
