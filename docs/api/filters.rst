Kalman Filter
=============

Forward filtering for linear-Gaussian state-space models. The Kalman filter
processes observations sequentially, computing the posterior state distribution
at each time step.

.. note::

   Most users do not need to call these functions directly ---
   :meth:`DLM.fit() <dynaris.dlm.api.DLM.fit>` wraps the Kalman filter
   internally. These are available for advanced use cases requiring direct
   access to intermediate filter quantities.

.. autoclass:: dynaris.filters.kalman.KalmanFilter
   :members:
   :show-inheritance:

.. autofunction:: dynaris.filters.kalman.kalman_filter

.. autofunction:: dynaris.filters.kalman.predict

.. autofunction:: dynaris.filters.kalman.update
