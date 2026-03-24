DLM --- High-Level Interface
============================

The :class:`~dynaris.dlm.api.DLM` class is the primary user-facing interface.
It wraps a :class:`~dynaris.core.state_space.StateSpaceModel` with convenient
methods for the full modeling workflow:

1. ``fit(y)`` --- run the Kalman filter
2. ``smooth()`` --- run the RTS smoother
3. ``forecast(steps)`` --- multi-step-ahead predictions
4. ``plot(kind)`` --- visualize results
5. ``summary()`` --- print model and fit information

Most users only need this class. The lower-level filter, smoother, and
forecast functions are available for advanced use cases.

.. autoclass:: dynaris.dlm.api.DLM
   :members:
   :show-inheritance:
