Switching & Regime Models
=========================

Markov-switching state-space models with K discrete regimes, each with
its own linear-Gaussian dynamics. The Hamilton filter tracks filtered
regime probabilities, and the Kim smoother provides smoothed estimates.

Model
-----

.. autoclass:: dynaris.core.switching.MarkovSwitchingSSM
   :members:
   :show-inheritance:

Hamilton Filter
---------------

Forward filtering for Markov-switching models. Runs K parallel Kalman
filters with Kim's moment-matching collapse approximation.

.. autoclass:: dynaris.filters.hamilton.HamiltonFilter
   :members:
   :show-inheritance:

.. autofunction:: dynaris.filters.hamilton.hamilton_filter

Kim Smoother
------------

Backward smoothing for Markov-switching models. Produces smoothed
state estimates and regime probabilities.

.. autoclass:: dynaris.smoothers.kim.KimSmoother
   :members:
   :show-inheritance:

.. autofunction:: dynaris.smoothers.kim.kim_smooth

Model Selection
---------------

.. autofunction:: dynaris.estimation.model_selection.switching_aic

.. autofunction:: dynaris.estimation.model_selection.switching_bic
