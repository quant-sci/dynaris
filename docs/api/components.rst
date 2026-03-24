DLM Components
==============

Composable building blocks for Dynamic Linear Models. Each function returns
a :class:`~dynaris.core.state_space.StateSpaceModel` that can be composed
via the ``+`` operator. See :doc:`/user-guide/components` for usage guidance.

Trend
-----

.. autofunction:: dynaris.dlm.components.LocalLevel

.. autofunction:: dynaris.dlm.components.LocalLinearTrend

Periodic
--------

.. autofunction:: dynaris.dlm.components.Seasonal

.. autofunction:: dynaris.dlm.components.Cycle

Other
-----

.. autofunction:: dynaris.dlm.components.Regression

.. autofunction:: dynaris.dlm.components.Autoregressive
