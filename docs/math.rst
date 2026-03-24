Mathematical Background
=======================

This section presents the mathematical foundations of Dynamic Linear Models
following the notation and framework of West and Harrison (1997),
*Bayesian Forecasting and Dynamic Models*.

The DLM Quadruple
-----------------

A Dynamic Linear Model is defined by the quadruple
:math:`\{\mathbf{F}_t, \mathbf{G}_t, V_t, \mathbf{W}_t\}` at each time
:math:`t`, together with initial information
:math:`(\boldsymbol{\theta}_0 \mid D_0) \sim \mathcal{N}(\mathbf{m}_0, \mathbf{C}_0)`.

**Observation equation:**

.. math::

   Y_t = \mathbf{F}_t^\prime\,\boldsymbol{\theta}_t + \nu_t, \qquad \nu_t \sim \mathcal{N}(0, V_t)

**System equation** (state evolution):

.. math::

   \boldsymbol{\theta}_t = \mathbf{G}_t\,\boldsymbol{\theta}_{t-1} + \boldsymbol{\omega}_t, \qquad \boldsymbol{\omega}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{W}_t)

where:

- :math:`Y_t \in \mathbb{R}` is the observation at time :math:`t`,
- :math:`\boldsymbol{\theta}_t \in \mathbb{R}^n` is the state (parameter) vector,
- :math:`\mathbf{F}_t \in \mathbb{R}^n` is the regression (observation) vector,
- :math:`\mathbf{G}_t \in \mathbb{R}^{n \times n}` is the system (evolution) matrix,
- :math:`V_t > 0` is the observational variance,
- :math:`\mathbf{W}_t \in \mathbb{R}^{n \times n}` is the evolution covariance matrix,
- :math:`D_t = \{Y_1, \ldots, Y_t\}` denotes all information available at time :math:`t`.

The error sequences :math:`\{\nu_t\}` and :math:`\{\boldsymbol{\omega}_t\}` are
mutually independent, independent over time, and independent of the initial
state :math:`\boldsymbol{\theta}_0`.

.. note::

   Dynaris uses West--Harrison notation directly:

   - ``observation_matrix`` / ``.F`` :math:`= \mathbf{F}_t^\prime`
   - ``system_matrix`` / ``.G`` :math:`= \mathbf{G}_t`
   - ``obs_cov`` / ``.V`` :math:`= V_t`
   - ``evolution_cov`` / ``.W`` :math:`= \mathbf{W}_t`

Kalman Filter (Sequential Updating)
------------------------------------

The Kalman filter computes the posterior
:math:`(\boldsymbol{\theta}_t \mid D_t) \sim \mathcal{N}(\mathbf{m}_t, \mathbf{C}_t)`
recursively. Given the posterior at :math:`t-1`:

.. math::

   (\boldsymbol{\theta}_{t-1} \mid D_{t-1}) \sim \mathcal{N}(\mathbf{m}_{t-1}, \mathbf{C}_{t-1})

Prior (one-step forecast of the state)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \mathbf{a}_t &= \mathbf{G}_t\,\mathbf{m}_{t-1} \\
   \mathbf{R}_t &= \mathbf{G}_t\,\mathbf{C}_{t-1}\,\mathbf{G}_t^\prime + \mathbf{W}_t

so that :math:`(\boldsymbol{\theta}_t \mid D_{t-1}) \sim \mathcal{N}(\mathbf{a}_t, \mathbf{R}_t)`.

One-step forecast of the observation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   f_t &= \mathbf{F}_t^\prime\,\mathbf{a}_t \\
   Q_t &= \mathbf{F}_t^\prime\,\mathbf{R}_t\,\mathbf{F}_t + V_t

so that :math:`(Y_t \mid D_{t-1}) \sim \mathcal{N}(f_t, Q_t)`.

Posterior (updating)
~~~~~~~~~~~~~~~~~~~~

Upon observing :math:`Y_t`:

.. math::

   e_t &= Y_t - f_t \qquad \text{(forecast error)} \\
   \mathbf{A}_t &= \mathbf{R}_t\,\mathbf{F}_t\,Q_t^{-1} \qquad \text{(adaptive coefficient / Kalman gain)} \\
   \mathbf{m}_t &= \mathbf{a}_t + \mathbf{A}_t\,e_t \\
   \mathbf{C}_t &= \mathbf{R}_t - \mathbf{A}_t\,Q_t\,\mathbf{A}_t^\prime

This gives :math:`(\boldsymbol{\theta}_t \mid D_t) \sim \mathcal{N}(\mathbf{m}_t, \mathbf{C}_t)`.

Log-likelihood
~~~~~~~~~~~~~~

The log-likelihood decomposes via the prediction error decomposition:

.. math::

   \log p(Y_{1:T}) = \sum_{t=1}^{T} \log p(Y_t \mid D_{t-1})
   = -\frac{1}{2} \sum_{t=1}^{T} \left[ \log(2\pi) + \log Q_t + \frac{e_t^2}{Q_t} \right]

This is computed inside ``jax.lax.scan`` and is fully differentiable via
JAX's autodiff, enabling gradient-based parameter estimation.

Missing observations
~~~~~~~~~~~~~~~~~~~~

When :math:`Y_t` is missing, the update step is skipped. The posterior
reverts to the prior: :math:`\mathbf{m}_t = \mathbf{a}_t`,
:math:`\mathbf{C}_t = \mathbf{R}_t`. Uncertainty grows without data to
constrain the state.

Retrospective Analysis (Smoothing)
----------------------------------

The Rauch--Tung--Striebel smoother computes the retrospective distribution
:math:`(\boldsymbol{\theta}_t \mid D_T)` for :math:`t < T`, using all
available data. This is the "what do we now believe about the past?"
question in West and Harrison's framework.

Starting from :math:`(\boldsymbol{\theta}_T \mid D_T) \sim \mathcal{N}(\mathbf{m}_T, \mathbf{C}_T)`,
the backward recursion for :math:`t = T-1, T-2, \ldots, 1` is:

.. math::

   \mathbf{B}_t &= \mathbf{C}_t\,\mathbf{G}_{t+1}^\prime\,\mathbf{R}_{t+1}^{-1} \qquad \text{(smoother gain)} \\
   \mathbf{m}_t^{(s)} &= \mathbf{m}_t + \mathbf{B}_t\,(\mathbf{m}_{t+1}^{(s)} - \mathbf{a}_{t+1}) \\
   \mathbf{C}_t^{(s)} &= \mathbf{C}_t + \mathbf{B}_t\,(\mathbf{C}_{t+1}^{(s)} - \mathbf{R}_{t+1})\,\mathbf{B}_t^\prime

where superscript :math:`(s)` denotes smoothed quantities. The smoother
uses future information to reduce uncertainty, so
:math:`\mathbf{C}_t^{(s)} \leq \mathbf{C}_t` in the positive-definite sense.

Implemented via ``jax.lax.scan(..., reverse=True)``.

DLM Superposition
-----------------

West and Harrison's superposition principle: if two independent DLMs
share the same observation, the combined model is obtained by stacking
their state vectors. Given component DLMs
:math:`\{\mathbf{F}_t^{(i)}, \mathbf{G}_t^{(i)}, V_t^{(i)}, \mathbf{W}_t^{(i)}\}`
for :math:`i = 1, 2`, the superposition yields:

.. math::

   \mathbf{G}_t &= \begin{pmatrix} \mathbf{G}_t^{(1)} & \mathbf{0} \\ \mathbf{0} & \mathbf{G}_t^{(2)} \end{pmatrix}, \qquad
   \mathbf{F}_t = \begin{pmatrix} \mathbf{F}_t^{(1)} \\ \mathbf{F}_t^{(2)} \end{pmatrix} \\
   \mathbf{W}_t &= \begin{pmatrix} \mathbf{W}_t^{(1)} & \mathbf{0} \\ \mathbf{0} & \mathbf{W}_t^{(2)} \end{pmatrix}, \qquad
   V_t = V_t^{(1)} + V_t^{(2)}

The observation is the sum of component contributions:

.. math::

   Y_t = \mathbf{F}_t^{(1)\prime}\,\boldsymbol{\theta}_t^{(1)} + \mathbf{F}_t^{(2)\prime}\,\boldsymbol{\theta}_t^{(2)} + \nu_t

In dynaris, this is the ``+`` operator:

.. code-block:: python

   model = LocalLinearTrend() + Seasonal(period=12) + Cycle(period=40)

Component Models
----------------

Polynomial (Trend) DLMs
~~~~~~~~~~~~~~~~~~~~~~~

**First-order polynomial** (local level). The simplest DLM: a random walk
observed with noise. The state is the current level :math:`\mu_t`:

.. math::

   \mu_t &= \mu_{t-1} + \omega_t, \quad \omega_t \sim \mathcal{N}(0, W) \\
   Y_t &= \mu_t + \nu_t, \quad \nu_t \sim \mathcal{N}(0, V)

Here :math:`G = [1]`, :math:`F = [1]`. State dimension: 1.

**Second-order polynomial** (local linear trend). The state is
:math:`\boldsymbol{\theta}_t = (\mu_t, \beta_t)^\prime`, where :math:`\mu_t`
is the level and :math:`\beta_t` the growth rate:

.. math::

   \begin{pmatrix} \mu_t \\ \beta_t \end{pmatrix}
   = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
   \begin{pmatrix} \mu_{t-1} \\ \beta_{t-1} \end{pmatrix}
   + \boldsymbol{\omega}_t, \qquad
   Y_t = \begin{pmatrix} 1 & 0 \end{pmatrix}
   \begin{pmatrix} \mu_t \\ \beta_t \end{pmatrix} + \nu_t

State dimension: 2.

Seasonal DLMs
~~~~~~~~~~~~~

**Free-form (dummy) seasonal.** The seasonal effect :math:`\gamma_t`
satisfies the sum-to-zero constraint across one full period :math:`s`:

.. math::

   \gamma_t = -\sum_{j=1}^{s-1} \gamma_{t-j} + \omega_t

The system matrix :math:`\mathbf{G}` has :math:`-1` in the first row and a
shift identity below. State dimension: :math:`s - 1`.

**Fourier-form seasonal.** Each harmonic :math:`j = 1, \ldots, \lfloor s/2 \rfloor`
contributes a rotation at frequency :math:`\omega_j = 2\pi j / s`:

.. math::

   \begin{pmatrix} \gamma_{j,t} \\ \gamma_{j,t}^* \end{pmatrix}
   = \begin{pmatrix} \cos\omega_j & \sin\omega_j \\ -\sin\omega_j & \cos\omega_j \end{pmatrix}
   \begin{pmatrix} \gamma_{j,t-1} \\ \gamma_{j,t-1}^* \end{pmatrix}
   + \boldsymbol{\omega}_{j,t}

The system matrix :math:`\mathbf{G}` is block-diagonal with rotation blocks.
State dimension: :math:`s - 1`.

Regression DLMs
~~~~~~~~~~~~~~~

The state vector holds regression coefficients
:math:`\boldsymbol{\theta}_t = \boldsymbol{\beta}_t`. With
:math:`\mathbf{G} = \mathbf{I}` (random walk coefficients):

.. math::

   \boldsymbol{\beta}_t &= \boldsymbol{\beta}_{t-1} + \boldsymbol{\omega}_t \\
   Y_t &= \mathbf{x}_t^\prime\,\boldsymbol{\beta}_t + \nu_t

where :math:`\mathbf{x}_t` is the vector of regressors at time :math:`t`
(playing the role of :math:`\mathbf{F}_t`). Setting
:math:`\mathbf{W} = \mathbf{0}` gives static (time-invariant) coefficients.

Cyclic DLMs
~~~~~~~~~~~

A damped stochastic cycle with period :math:`\lambda` and damping factor
:math:`\rho \in (0, 1]`:

.. math::

   \begin{pmatrix} \psi_t \\ \psi_t^* \end{pmatrix}
   = \rho \begin{pmatrix} \cos\omega & \sin\omega \\ -\sin\omega & \cos\omega \end{pmatrix}
   \begin{pmatrix} \psi_{t-1} \\ \psi_{t-1}^* \end{pmatrix}
   + \boldsymbol{\omega}_t, \qquad \omega = \frac{2\pi}{\lambda}

State dimension: 2. Setting :math:`\rho = 1` recovers an undamped cycle.
The eigenvalues of :math:`\mathbf{G}` have modulus :math:`\rho`, so the
cycle decays geometrically when :math:`\rho < 1`.

Autoregressive DLMs
~~~~~~~~~~~~~~~~~~~

An AR(p) process cast in companion-form state space. The state is
:math:`\boldsymbol{\theta}_t = (x_t, x_{t-1}, \ldots, x_{t-p+1})^\prime`
with system matrix:

.. math::

   \mathbf{G} = \begin{pmatrix}
   \phi_1 & \phi_2 & \cdots & \phi_p \\
   1 & 0 & \cdots & 0 \\
   & \ddots & & \vdots \\
   0 & \cdots & 1 & 0
   \end{pmatrix}

and :math:`\mathbf{F} = (1, 0, \ldots, 0)^\prime`. State dimension: :math:`p`.

Forecasting
-----------

The :math:`k`-step-ahead forecast distribution is obtained by iterating the
prior equations without updating:

.. math::

   \mathbf{a}_t(k) &= \mathbf{G}^k\,\mathbf{m}_t \\
   \mathbf{R}_t(k) &= \mathbf{G}\,\mathbf{R}_t(k-1)\,\mathbf{G}^\prime + \mathbf{W}

with :math:`\mathbf{R}_t(0) = \mathbf{C}_t`. The forecast distribution for
the observable is:

.. math::

   (Y_{t+k} \mid D_t) \sim \mathcal{N}\!\left(\mathbf{F}^\prime\,\mathbf{a}_t(k),\; \mathbf{F}^\prime\,\mathbf{R}_t(k)\,\mathbf{F} + V\right)

Forecast uncertainty grows with horizon :math:`k` due to the accumulated
evolution noise :math:`\mathbf{W}`.

Parameter Estimation
--------------------

Maximum Likelihood Estimation (MLE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The log-likelihood
:math:`\ell(\boldsymbol{\psi}) = \log p(Y_{1:T} \mid \boldsymbol{\psi})`
(where :math:`\boldsymbol{\psi}` collects the unknown hyperparameters
:math:`V` and :math:`\mathbf{W}`) is computed via the prediction error
decomposition from the Kalman filter. Since the entire computation is
implemented in JAX, gradients
:math:`\nabla_{\boldsymbol{\psi}} \ell` are obtained via automatic
differentiation.

Variance parameters must be positive. Dynaris provides log and softplus
transforms to map unconstrained parameters to valid ranges:

.. math::

   \sigma^2 = \exp(\psi) \qquad \text{or} \qquad \sigma^2 = \log(1 + \exp(\psi))

EM Algorithm
~~~~~~~~~~~~

The Expectation--Maximization algorithm alternates between:

**E-step:** Run the Kalman filter and smoother to obtain
:math:`\mathbf{m}_t^{(s)}` and :math:`\mathbf{C}_t^{(s)}` for all :math:`t`.

**M-step:** Update the variance components. For the observational variance:

.. math::

   \hat{V} = \frac{1}{T} \sum_{t=1}^{T} \left[
      (Y_t - \mathbf{F}^\prime\,\mathbf{m}_t^{(s)})^2
      + \mathbf{F}^\prime\,\mathbf{C}_t^{(s)}\,\mathbf{F}
   \right]

The EM algorithm guarantees non-decreasing log-likelihood at each iteration
for the exact M-step.

References
----------

- West, M. and Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models*,
  2nd edition. Springer.
- Durbin, J. and Koopman, S.J. (2012). *Time Series Analysis by State Space
  Methods*, 2nd edition. Oxford University Press.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge University Press.
- Petris, G., Petrone, S., and Campagnoli, P. (2009). *Dynamic Linear Models
  with R*. Springer.
