Mathematical Background
=======================

This section presents the mathematical foundations of Dynamic Linear Models
following the notation and framework of West and Harrison (1997),
*Bayesian Forecasting and Dynamic Models*.

For a plain-language introduction, see :doc:`getting-started/concepts`.

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

Nonlinear Filters
-----------------

When the state-space model is nonlinear, the Kalman filter no longer gives
exact inference. Dynaris provides three approximate filters.

Extended Kalman Filter (EKF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EKF linearizes the nonlinear functions at each time step via first-order
Taylor expansion, then applies the standard Kalman recursion.

Given the nonlinear state-space model:

.. math::

   \boldsymbol{\theta}_t &= f(\boldsymbol{\theta}_{t-1}) + \boldsymbol{\omega}_t, \quad \boldsymbol{\omega}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}) \\
   Y_t &= h(\boldsymbol{\theta}_t) + \boldsymbol{\nu}_t, \quad \boldsymbol{\nu}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})

**Predict:**

.. math::

   \mathbf{a}_t &= f(\mathbf{m}_{t-1}) \\
   \mathbf{F}_t &= \left.\frac{\partial f}{\partial \boldsymbol{\theta}}\right|_{\mathbf{m}_{t-1}} \qquad \text{(Jacobian via } \texttt{jax.jacfwd}\text{)} \\
   \mathbf{R}_t &= \mathbf{F}_t\,\mathbf{C}_{t-1}\,\mathbf{F}_t^\prime + \mathbf{Q}

**Update:**

.. math::

   \hat{Y}_t &= h(\mathbf{a}_t) \\
   \mathbf{H}_t &= \left.\frac{\partial h}{\partial \boldsymbol{\theta}}\right|_{\mathbf{a}_t} \\
   \mathbf{S}_t &= \mathbf{H}_t\,\mathbf{R}_t\,\mathbf{H}_t^\prime + \mathbf{R} \\
   \mathbf{K}_t &= \mathbf{R}_t\,\mathbf{H}_t^\prime\,\mathbf{S}_t^{-1} \\
   \mathbf{m}_t &= \mathbf{a}_t + \mathbf{K}_t\,(Y_t - \hat{Y}_t) \\
   \mathbf{C}_t &= (\mathbf{I} - \mathbf{K}_t\,\mathbf{H}_t)\,\mathbf{R}_t

In dynaris, the Jacobians are computed automatically via ``jax.jacfwd`` ---
no manual derivation required.

Unscented Kalman Filter (UKF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The UKF avoids linearization by propagating deterministically chosen
**sigma points** through the nonlinear functions, then recovering the
mean and covariance from the transformed points.

**Sigma points** for a state with mean :math:`\mathbf{m}` and covariance
:math:`\mathbf{C}` of dimension :math:`n`:

.. math::

   \boldsymbol{\chi}_0 &= \mathbf{m} \\
   \boldsymbol{\chi}_i &= \mathbf{m} + \left[\sqrt{(n + \lambda)\,\mathbf{C}}\right]_i, \quad i = 1, \ldots, n \\
   \boldsymbol{\chi}_{n+i} &= \mathbf{m} - \left[\sqrt{(n + \lambda)\,\mathbf{C}}\right]_i, \quad i = 1, \ldots, n

where :math:`\lambda = \alpha^2(n + \kappa) - n` and
:math:`[\sqrt{\mathbf{M}}]_i` is the :math:`i`-th row of the Cholesky
factor of :math:`\mathbf{M}`.

**Weights:**

.. math::

   w_0^{(m)} &= \frac{\lambda}{n + \lambda}, \quad w_0^{(c)} = w_0^{(m)} + 1 - \alpha^2 + \beta \\
   w_i^{(m)} &= w_i^{(c)} = \frac{1}{2(n + \lambda)}, \quad i = 1, \ldots, 2n

**Predict:** Propagate sigma points through :math:`f`, recover predicted
mean and covariance from the weighted transformed points.

**Update:** Propagate sigma points through :math:`h`, compute cross-covariance,
and apply the Kalman gain.

The UKF captures second-order nonlinear effects that the EKF misses, and
requires no Jacobian computation.

Particle Filter (Sequential Monte Carlo)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The particle filter represents the filtering distribution as a weighted set
of :math:`N` **particles** (samples), enabling inference in arbitrarily
nonlinear and non-Gaussian models.

**Algorithm** (bootstrap particle filter):

1. **Resample:** Draw ancestor indices from the categorical distribution
   defined by the current weights.

2. **Propagate:** For each particle :math:`i`:

   .. math::

      \boldsymbol{\theta}_t^{(i)} = f\!\left(\boldsymbol{\theta}_{t-1}^{(i)}\right) + \boldsymbol{\omega}_t^{(i)}, \quad \boldsymbol{\omega}_t^{(i)} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})

   Implemented via ``jax.vmap`` for vectorized propagation.

3. **Reweight:** Compute unnormalized weights from the observation likelihood:

   .. math::

      \tilde{w}_t^{(i)} = p\!\left(Y_t \mid \boldsymbol{\theta}_t^{(i)}\right)

   Normalize: :math:`w_t^{(i)} = \tilde{w}_t^{(i)} / \sum_j \tilde{w}_t^{(j)}`.

4. **Estimate:** The filtered mean and covariance are weighted statistics
   of the particle cloud.

**Resampling strategies:** Multinomial, systematic (default), and stratified.
The **effective sample size** :math:`\text{ESS} = 1 / \sum_i (w_t^{(i)})^2`
monitors particle degeneracy.

**Log-likelihood:** :math:`\log \hat{p}(Y_t \mid Y_{1:t-1}) = \log\!\left(\frac{1}{N}\sum_i \tilde{w}_t^{(i)}\right)`.

Markov-Switching Models
-----------------------

Hamilton Filter
~~~~~~~~~~~~~~~

The Hamilton filter extends the Kalman filter to models with :math:`K`
discrete regimes governed by a Markov chain with transition matrix
:math:`\mathbf{P}`, where :math:`P_{ij} = \Pr(S_t = j \mid S_{t-1} = i)`.

At each time step, the filter maintains :math:`K` Gaussian state beliefs
(one per regime) and a vector of regime probabilities.

**Algorithm** (with Kim's collapse approximation):

1. **Collapse:** For each target regime :math:`j`, compute the
   probability-weighted mixture of the :math:`K` source-regime states:

   .. math::

      w_{i|j} &= \frac{P_{ij}\,\pi_{t-1}^{(i)}}{\sum_k P_{kj}\,\pi_{t-1}^{(k)}} \\
      \mathbf{m}_{t|t-1}^{(j)} &= \sum_i w_{i|j}\,\mathbf{m}_{t-1}^{(i)} \\
      \mathbf{C}_{t|t-1}^{(j)} &= \sum_i w_{i|j}\!\left[\mathbf{C}_{t-1}^{(i)} + \boldsymbol{\delta}_i\,\boldsymbol{\delta}_i^\prime\right]

   where :math:`\boldsymbol{\delta}_i = \mathbf{m}_{t-1}^{(i)} - \mathbf{m}_{t|t-1}^{(j)}`.

2. **Predict/Update:** Run :math:`K` parallel Kalman predict+update steps
   (one per regime), each using regime-specific matrices
   :math:`\{\mathbf{G}_j, \mathbf{F}_j, \mathbf{W}_j, V_j\}`.
   Implemented via ``jax.vmap``.

3. **Regime probability update:**

   .. math::

      \pi_t^{(j)} = \frac{p(Y_t \mid S_t = j,\, D_{t-1})\,\bar{\pi}_t^{(j)}}{\sum_k p(Y_t \mid S_t = k,\, D_{t-1})\,\bar{\pi}_t^{(k)}}

   where :math:`\bar{\pi}_t^{(j)} = \sum_i P_{ij}\,\pi_{t-1}^{(i)}` are
   the predicted regime probabilities.

Kim Smoother
~~~~~~~~~~~~

The Kim smoother is the backward counterpart of the Hamilton filter,
producing smoothed regime probabilities and state estimates.

For :math:`t = T-1, T-2, \ldots, 1`:

1. **Smoothed regime probabilities:**

   .. math::

      \pi_t^{(i,s)} = \pi_t^{(i)} \sum_j \frac{P_{ij}\,\pi_{t+1}^{(j,s)}}{\bar{\pi}_{t+1}^{(j)}}

2. **Per-regime RTS step:** Apply the RTS smoother backward recursion
   independently for each regime :math:`j`, using regime-specific
   :math:`\mathbf{G}_j`.

3. **Collapse** the :math:`K` smoothed states using the smoothed
   regime probabilities.

Dynamic Factor Models
---------------------

A DFM with :math:`r` latent factors and :math:`m` observed variables:

.. math::

   \mathbf{f}_t &= \mathbf{G}_f\,\mathbf{f}_{t-1} + \boldsymbol{\omega}_t, \quad \boldsymbol{\omega}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_r) \\
   \mathbf{y}_t &= \boldsymbol{\Lambda}\,\mathbf{f}_t + \mathbf{e}_t, \quad \mathbf{e}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})

where :math:`\boldsymbol{\Lambda} \in \mathbb{R}^{m \times r}` is the
**loading matrix**, :math:`\mathbf{R}` is diagonal (idiosyncratic noise),
and :math:`\mathbf{Q} = \mathbf{I}_r` for identification.

**EM estimation** alternates:

- **E-step:** Kalman filter + RTS smoother on the factor state space.
- **M-step:** Update :math:`\boldsymbol{\Lambda}` and :math:`\mathbf{R}`:

  .. math::

     \hat{\boldsymbol{\Lambda}} &= \left[\sum_t \mathbf{y}_t\,\mathbf{m}_t^{(s)\prime}\right] \left[\sum_t \left(\mathbf{C}_t^{(s)} + \mathbf{m}_t^{(s)}\,\mathbf{m}_t^{(s)\prime}\right)\right]^{-1} \\
     \hat{R}_{ii} &= \frac{1}{T}\sum_t \left[(y_{t,i} - \hat{\boldsymbol{\Lambda}}_i\,\mathbf{m}_t^{(s)})^2 + \hat{\boldsymbol{\Lambda}}_i\,\mathbf{C}_t^{(s)}\,\hat{\boldsymbol{\Lambda}}_i^\prime\right]

**Varimax rotation** is applied post-estimation for interpretability,
maximizing the variance of squared loadings across factors.

Bayesian Estimation
-------------------

The log-posterior for a DLM with hyperparameters :math:`\boldsymbol{\psi}`
(variance parameters) is:

.. math::

   \log p(\boldsymbol{\psi} \mid Y_{1:T}) \propto \underbrace{\log p(Y_{1:T} \mid \boldsymbol{\psi})}_{\text{Kalman filter log-likelihood}} + \underbrace{\log p(\boldsymbol{\psi})}_{\text{prior}}

Dynaris uses NumPyro's NUTS (No-U-Turn Sampler) to draw posterior samples.
Since the entire Kalman filter log-likelihood is differentiable via JAX
autodiff, NUTS can efficiently explore the posterior using Hamiltonian
dynamics.

**Posterior predictive forecasting:** For each posterior sample
:math:`\boldsymbol{\psi}^{(s)}`, run the Kalman filter and forecast.
Aggregate across samples for credible intervals.

**Model comparison** via WAIC and LOO-CV requires **pointwise**
log-likelihoods :math:`\log p(Y_t \mid Y_{1:t-1}, \boldsymbol{\psi})`,
which are the individual terms of the prediction error decomposition.

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
- Julier, S.J. and Uhlmann, J.K. (2004). "Unscented Filtering and Nonlinear
  Estimation." *Proceedings of the IEEE*, 92(3), 401-422.
- Doucet, A., de Freitas, N. and Gordon, N. (2001). *Sequential Monte Carlo
  Methods in Practice*. Springer.
- Kim, C.-J. (1994). "Dynamic Linear Models with Markov-Switching."
  *Journal of Econometrics*, 60(1-2), 1-22.
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2).
- Stock, J.H. and Watson, M.W. (2002). "Forecasting Using Principal Components
  from a Large Number of Predictors." *JASA*, 97(460).
