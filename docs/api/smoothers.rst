RTS Smoother
============

Rauch-Tung-Striebel backward smoother for linear-Gaussian models. Uses the
full dataset to refine state estimates, producing lower-variance posteriors
than the forward-only Kalman filter.

.. note::

   Most users do not need to call these functions directly ---
   :meth:`DLM.smooth() <dynaris.dlm.api.DLM.smooth>` wraps the RTS smoother
   internally.

.. autoclass:: dynaris.smoothers.rts.RTSSmoother
   :members:
   :show-inheritance:

.. autofunction:: dynaris.smoothers.rts.rts_smooth
