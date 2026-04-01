API Reference
=============

Complete reference for all public classes and functions in dynaris.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Description
   * - :doc:`ssm`
     - Unified ``SSM`` class with automatic filter selection
   * - :doc:`dlm`
     - High-level ``DLM`` class (fit, smooth, forecast, plot)
   * - :doc:`components`
     - Six composable building blocks (``LocalLevel``, etc.)
   * - :doc:`models`
     - Built-in nonlinear models (stochastic vol, tracking, etc.)
   * - :doc:`factor`
     - Dynamic Factor Models (DFMModel, loadings, rotation)
   * - :doc:`core`
     - ``StateSpaceModel``, ``GaussianState``, result containers
   * - :doc:`filters`
     - Kalman, EKF, UKF, and Particle filters
   * - :doc:`switching`
     - Markov-switching models, Hamilton filter, Kim smoother
   * - :doc:`smoothers`
     - Rauch-Tung-Striebel backward smoother
   * - :doc:`estimation`
     - MLE, EM, Bayesian, diagnostics, model selection
   * - :doc:`forecast`
     - Multi-step forecasting and batch processing
   * - :doc:`plotting`
     - Visualization functions for all plot kinds
   * - :doc:`datasets`
     - Built-in dataset loaders

.. toctree::
   :maxdepth: 2
   :hidden:

   ssm
   dlm
   components
   models
   factor
   core
   filters
   switching
   smoothers
   estimation
   forecast
   plotting
   datasets
