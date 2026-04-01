Filters
=======

Forward filtering algorithms for state-space models. Dynaris provides four
filtering algorithms: the Kalman filter for linear-Gaussian models, and the
Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Particle
Filter for nonlinear models.

Kalman Filter
-------------

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


Extended Kalman Filter (EKF)
----------------------------

Linearizes nonlinear transition and observation functions at each time step
using automatic Jacobians via ``jax.jacfwd``, then applies the standard
Kalman predict/update equations to the linearized system.

.. autoclass:: dynaris.filters.ekf.ExtendedKalmanFilter
   :members:
   :show-inheritance:

.. autofunction:: dynaris.filters.ekf.ekf_filter

.. autofunction:: dynaris.filters.ekf.predict

.. autofunction:: dynaris.filters.ekf.update


Unscented Kalman Filter (UKF)
-----------------------------

Propagates sigma points through nonlinear transition and observation functions
to capture the posterior mean and covariance without linearization. Uses the
scaled unscented transform with configurable alpha, beta, kappa parameters.

.. autoclass:: dynaris.filters.ukf.UnscentedKalmanFilter
   :members:
   :show-inheritance:

.. autofunction:: dynaris.filters.ukf.ukf_filter

.. autofunction:: dynaris.filters.ukf.predict

.. autofunction:: dynaris.filters.ukf.update


Particle Filter (SMC)
---------------------

Bootstrap particle filter / Sequential Monte Carlo. Represents the filtering
posterior as a weighted set of particles, enabling inference in nonlinear,
non-Gaussian state-space models. Supports multinomial, systematic, and
stratified resampling strategies.

.. autoclass:: dynaris.filters.particle.ParticleFilter
   :members:
   :show-inheritance:

.. autofunction:: dynaris.filters.particle.particle_filter

.. autofunction:: dynaris.filters.particle.effective_sample_size
