Core Types
==========

Fundamental data structures used throughout dynaris. These are the building
blocks that filters, smoothers, and the DLM class operate on.

StateSpaceModel
---------------

The central model representation. Holds the four system matrices (F, G, V, W)
following West and Harrison (1997) notation. Returned by all component
functions and composed via ``+``.

.. autoclass:: dynaris.core.state_space.StateSpaceModel
   :members:
   :show-inheritance:

GaussianState
-------------

Represents a Gaussian belief about the state: a mean vector and covariance
matrix. Used internally by the Kalman filter and smoother at each time step.

.. autoclass:: dynaris.core.types.GaussianState
   :members:
   :show-inheritance:

FilterResult
------------

Container returned by the Kalman filter. Holds filtered state means,
covariances, log-likelihood, and forecast errors for all time steps.

.. autoclass:: dynaris.core.results.FilterResult
   :members:
   :show-inheritance:
   :no-index:

SmootherResult
--------------

Container returned by the RTS smoother. Holds smoothed state means and
covariances for all time steps.

.. autoclass:: dynaris.core.results.SmootherResult
   :members:
   :show-inheritance:
   :no-index:

Protocols
---------

Interfaces that filter and smoother implementations must satisfy. Useful
for type checking and extending dynaris with custom algorithms.

.. autoclass:: dynaris.core.protocols.FilterProtocol
   :members:

.. autoclass:: dynaris.core.protocols.SmootherProtocol
   :members:
