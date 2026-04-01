API Reference
=============

Complete reference for all public classes and functions in dynaris.

+------------------+-------------------------------------------------------------+
| Module           | Description                                                 |
+==================+=============================================================+
| :doc:`ssm`       | Unified ``SSM`` class with automatic filter selection       |
+------------------+-------------------------------------------------------------+
| :doc:`dlm`       | High-level ``DLM`` class (fit, smooth, forecast, plot)      |
+------------------+-------------------------------------------------------------+
| :doc:`components` | Six composable building blocks (``LocalLevel``, etc.)      |
+------------------+-------------------------------------------------------------+
| :doc:`models`    | Built-in nonlinear models (stochastic vol, tracking, etc.)  |
+------------------+-------------------------------------------------------------+
| :doc:`core`      | ``StateSpaceModel``, ``GaussianState``, result containers   |
+------------------+-------------------------------------------------------------+
| :doc:`filters`   | Kalman, EKF, UKF, and Particle filters                     |
+------------------+-------------------------------------------------------------+
| :doc:`switching` | Markov-switching models, Hamilton filter, Kim smoother       |
+------------------+-------------------------------------------------------------+
| :doc:`smoothers` | Rauch-Tung-Striebel backward smoother                       |
+------------------+-------------------------------------------------------------+
| :doc:`estimation` | MLE, EM algorithm, diagnostics, model selection             |
+------------------+-------------------------------------------------------------+
| :doc:`forecast`  | Multi-step forecasting and batch processing                 |
+------------------+-------------------------------------------------------------+
| :doc:`plotting`  | Visualization functions for all plot kinds                  |
+------------------+-------------------------------------------------------------+
| :doc:`datasets`  | Built-in dataset loaders                                    |
+------------------+-------------------------------------------------------------+

.. toctree::
   :maxdepth: 2
   :hidden:

   ssm
   dlm
   components
   models
   core
   filters
   switching
   smoothers
   estimation
   forecast
   plotting
   datasets
