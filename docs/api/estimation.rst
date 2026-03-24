Parameter Estimation
====================

Maximum likelihood estimation, EM algorithm, and model diagnostics.

MLE
---

.. autofunction:: dynaris.estimation.mle.fit_mle

.. autoclass:: dynaris.estimation.mle.MLEResult
   :members:

EM Algorithm
------------

.. autofunction:: dynaris.estimation.em.fit_em

.. autoclass:: dynaris.estimation.em.EMResult
   :members:

Diagnostics
-----------

.. autofunction:: dynaris.estimation.diagnostics.standardized_residuals

.. autofunction:: dynaris.estimation.diagnostics.acf

.. autofunction:: dynaris.estimation.diagnostics.pacf

.. autofunction:: dynaris.estimation.diagnostics.ljung_box

Transforms
----------

.. autofunction:: dynaris.estimation.transforms.softplus

.. autofunction:: dynaris.estimation.transforms.inverse_softplus
