Comparison Guide
================

How dynaris compares to other state-space / Kalman filter libraries.

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Feature
     - **dynaris**
     - dynamax
     - statsmodels
     - filterpy
     - pykalman
   * - Backend
     - JAX
     - JAX
     - NumPy/Cython
     - NumPy
     - NumPy
   * - GPU/TPU
     - Yes
     - Yes
     - No
     - No
     - No
   * - JIT compilation
     - Yes
     - Yes
     - No
     - No
     - No
   * - Autodiff (gradients)
     - Yes (full)
     - Yes
     - Limited
     - No
     - No
   * - Kalman filter
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - RTS smoother
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Extended KF
     - Yes
     - Yes
     - No
     - Yes
     - No
   * - Unscented KF
     - Yes
     - Yes
     - No
     - Yes
     - No
   * - Particle filter
     - Yes
     - Yes (via Blackjax)
     - No
     - Yes
     - No
   * - Hamilton filter
     - Yes
     - Yes
     - Yes
     - No
     - No
   * - DLM components
     - Yes (6 types)
     - No
     - Yes (via UnobservedComponents)
     - No
     - No
   * - Model composition (+)
     - Yes
     - No
     - No
     - No
     - No
   * - Dynamic Factor Models
     - Yes
     - No
     - Yes
     - No
     - No
   * - Bayesian (MCMC)
     - Yes (NumPyro)
     - No (manual)
     - No
     - No
     - No
   * - MLE estimation
     - Yes (JAX grad)
     - Yes
     - Yes (scipy)
     - No
     - Yes (EM only)
   * - EM estimation
     - Yes
     - Yes
     - No
     - No
     - Yes
   * - Batch (vmap)
     - Yes
     - Yes
     - No
     - No
     - No
   * - Missing data
     - Yes (NaN)
     - Yes
     - Yes
     - Manual
     - No
   * - Forecast
     - Yes
     - Yes
     - Yes
     - Manual
     - No
   * - Plotting
     - Yes (matplotlib)
     - No
     - Yes
     - No
     - No
   * - pandas integration
     - Yes
     - No
     - Yes
     - No
     - No

When to use dynaris
-------------------

- You want **composable DLM components** (trend + seasonal + regression via ``+``)
- You need **gradient-based optimization** through the filter (MLE, Bayesian)
- You want **GPU/TPU acceleration** for large-scale or batch inference
- You need **nonlinear filters** (EKF, UKF, particle) alongside linear DLMs
- You want **regime-switching models** with the Hamilton filter
- You need **Bayesian estimation** with NUTS/HMC via NumPyro

When to use alternatives
------------------------

- **dynamax**: If you want a general-purpose JAX state-space library with
  more model types (HMMs, LDS) and variational inference.
- **statsmodels**: If you need a mature, well-tested library with extensive
  econometric functionality and don't need GPU/autodiff.
- **filterpy**: If you need a lightweight, NumPy-only library for real-time
  tracking applications (robotics, navigation).
- **pykalman**: If you need a simple EM-based Kalman filter with minimal
  dependencies and don't need nonlinear filters or GPU.
