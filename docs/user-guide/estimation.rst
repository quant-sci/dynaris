Parameter Estimation
====================

dynaris provides two approaches for estimating unknown variance parameters:
**maximum likelihood estimation (MLE)** via automatic differentiation and the
**EM algorithm**.

Maximum Likelihood (MLE)
------------------------

The log-likelihood is computed by the Kalman filter's prediction error
decomposition. Since the entire computation runs in JAX, gradients are
obtained via ``jax.grad`` and passed to a gradient-based optimizer.

.. code-block:: python

   import jax.numpy as jnp
   from dynaris import LocalLevel
   from dynaris.estimation import fit_mle

   def model_fn(params):
       return LocalLevel(
           sigma_level=jnp.exp(params[0]),
           sigma_obs=jnp.exp(params[1]),
       )

   result = fit_mle(model_fn, y, init_params=jnp.zeros(2))
   print(f"Log-likelihood: {result.log_likelihood:.2f}")
   fitted_model = result.model

The ``model_fn`` maps unconstrained parameters to a ``StateSpaceModel``.
Use ``jnp.exp`` (log transform) or ``softplus`` to ensure variance parameters
stay positive.

The result is an :class:`~dynaris.estimation.mle.MLEResult` containing the
optimized model, final parameters, and log-likelihood.

EM Algorithm
------------

The Expectation-Maximization algorithm alternates between running the Kalman
smoother (E-step) and updating variance estimates (M-step):

.. code-block:: python

   from dynaris.estimation import fit_em

   result = fit_em(y, initial_model, max_iter=100)
   print(f"Converged: {result.converged}")
   print(f"Iterations: {result.n_iter}")
   fitted_model = result.model

The result is an :class:`~dynaris.estimation.em.EMResult` with the fitted
model, convergence status, and iteration count.

MLE vs EM
---------

+-------------------+----------------------------------+----------------------------------+
| Criterion         | MLE                              | EM                               |
+===================+==================================+==================================+
| Speed             | Fewer iterations (gradient info) | More iterations, but each is     |
|                   |                                  | cheap and closed-form            |
+-------------------+----------------------------------+----------------------------------+
| Flexibility       | Any differentiable               | Variance parameters only         |
|                   | parameterization                 |                                  |
+-------------------+----------------------------------+----------------------------------+
| Convergence       | Can find local minima            | Guaranteed non-decreasing        |
|                   |                                  | log-likelihood                   |
+-------------------+----------------------------------+----------------------------------+
| Setup             | Requires writing ``model_fn``    | Just pass the initial model      |
+-------------------+----------------------------------+----------------------------------+

**Rule of thumb:** use MLE for complex parameterizations; use EM when you only
need variance estimates and want a simple interface.

Parameter transforms
--------------------

Variance parameters must be positive. dynaris provides two transforms for
mapping unconstrained parameters to valid ranges:

- **Log transform:** :math:`\sigma^2 = \exp(\psi)` --- simple, widely used
- **Softplus transform:** :math:`\sigma^2 = \log(1 + \exp(\psi))` --- smoother
  gradient near zero

.. code-block:: python

   from dynaris.estimation import softplus, inverse_softplus

   # Map unconstrained -> positive
   sigma_sq = softplus(raw_param)

   # Map positive -> unconstrained
   raw_param = inverse_softplus(sigma_sq)

Diagnostics
-----------

After fitting, check model adequacy with residual diagnostics:

.. code-block:: python

   from dynaris.estimation import standardized_residuals, acf, ljung_box

   resid = standardized_residuals(filter_result, model)
   autocorr = acf(resid, max_lag=20)
   lb = ljung_box(resid, max_lag=10)
   print(f"Ljung-Box p-value: {lb.p_value:.4f}")

Or use the built-in diagnostic plot:

.. code-block:: python

   dlm.plot(kind="diagnostics")

This produces a QQ-plot, ACF plot, and histogram of standardized residuals.
See :doc:`/api/estimation` for the full API.
