Parameter Estimation
====================

Maximum likelihood estimation, EM algorithm, residual diagnostics, and
parameter transforms. See :doc:`/user-guide/estimation` for a guide on
choosing between MLE and EM.

MLE
---

Gradient-based optimization of the log-likelihood using JAX autodiff.
Flexible: supports any differentiable parameterization via a user-defined
``model_fn``.

.. autofunction:: dynaris.estimation.mle.fit_mle

.. autoclass:: dynaris.estimation.mle.MLEResult
   :members:

EM Algorithm
------------

Iterative variance estimation with guaranteed non-decreasing log-likelihood.
Simpler setup than MLE --- just pass an initial model.

.. autofunction:: dynaris.estimation.em.fit_em

.. autoclass:: dynaris.estimation.em.EMResult
   :members:

Diagnostics
-----------

Tools for checking model adequacy after fitting.

.. autofunction:: dynaris.estimation.diagnostics.standardized_residuals

.. autofunction:: dynaris.estimation.diagnostics.acf

.. autofunction:: dynaris.estimation.diagnostics.pacf

.. autofunction:: dynaris.estimation.diagnostics.ljung_box

Transforms
----------

Map unconstrained parameters to positive values for variance estimation.

.. autofunction:: dynaris.estimation.transforms.softplus

.. autofunction:: dynaris.estimation.transforms.inverse_softplus
