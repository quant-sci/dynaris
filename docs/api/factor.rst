Dynamic Factor Models
=====================

Dynamic Factor Models (DFMs) reduce high-dimensional multivariate time
series to a small number of latent factors. The observation matrix
(loading matrix) maps latent factors to observed variables.

High-Level API
--------------

.. autoclass:: dynaris.models.dfm_api.DFMModel
   :members:
   :show-inheritance:

Model Factory
-------------

.. autofunction:: dynaris.models.factor.DynamicFactorModel

Estimation
----------

.. autofunction:: dynaris.estimation.dfm.fit_dfm_em

.. autoclass:: dynaris.estimation.dfm.DFMResult
   :members:

Utilities
---------

.. autofunction:: dynaris.models.factor.initialize_loadings_pca

.. autofunction:: dynaris.models.factor.rotate_loadings

.. autofunction:: dynaris.models.factor.apply_identification_constraints
