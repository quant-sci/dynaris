dynaris
=======

**dynaris** is a JAX-powered Python library for Dynamic Linear Models (DLMs).
Build composable state-space models, estimate parameters with automatic
differentiation, and produce forecasts with uncertainty --- all in a few lines
of code.

.. code-block:: bash

   pip install dynaris

.. code-block:: python

   from dynaris import LocalLevel, DLM
   from dynaris.datasets import load_nile

   y = load_nile()
   dlm = DLM(LocalLevel(sigma_level=38.0, sigma_obs=123.0))
   dlm.fit(y).smooth()
   dlm.forecast(steps=10)
   dlm.plot(kind="panel")

Why State-Space Models?
-----------------------

State-space models are a unifying framework for time series analysis. Rather
than fitting a single equation to observed data, they maintain a latent
**state** --- the true level, trend, seasonal pattern, or regression
coefficient --- and update it recursively as new observations arrive. This
separation of *what we observe* from *what drives the process* gives
state-space models several structural advantages:

- **Decomposition.** A complex series is expressed as the sum of interpretable
  components (trend, seasonality, cycles, regression effects), each evolving
  according to its own dynamics. The components are estimated jointly, not
  sequentially stripped away.
- **Exact uncertainty quantification.** The Kalman filter propagates a full
  Gaussian posterior at every time step, so filtered estimates, smoothed
  retrospectives, and multi-step forecasts all carry principled confidence
  intervals --- no bootstrap or asymptotic approximation required.
- **Missing data and irregular spacing.** When an observation is absent, the
  filter simply skips the update step and lets the prior covariance grow.
  No imputation heuristics are needed.
- **Online and retrospective inference.** The same model supports real-time
  filtering (estimate the present), smoothing (revise the past given the
  future), and forecasting (project forward), all from a single set of
  sufficient statistics.

Dynamic Linear Models (DLMs) are the linear-Gaussian specialization of this
framework, where closed-form Kalman recursions replace approximate inference.
They are the workhorse behind structural time series decomposition, adaptive
forecasting, and dynamic regression in fields from econometrics and signal
processing to environmental monitoring.

About dynaris
-------------

dynaris implements the full DLM inference pipeline in JAX. Models are built by
composing six structural components --- ``LocalLevel``, ``LocalLinearTrend``,
``Seasonal``, ``Cycle``, ``Regression``, and ``Autoregressive`` --- using the
``+`` operator, which constructs block-diagonal state-space matrices via the
superposition principle of West and Harrison (1997).

Filtering and smoothing run inside ``jax.lax.scan``, making them JIT-compilable
and end-to-end differentiable. This means the full Kalman log-likelihood can
be optimized with gradient-based methods (``fit_mle``) or the EM algorithm
(``fit_em``), and batch inference over many series parallelizes naturally
through ``jax.vmap``.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/installation
   getting-started/quickstart
   getting-started/concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/index

.. toctree::
   :maxdepth: 1
   :caption: Theory

   math

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
