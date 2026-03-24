API Reference
=============

Complete reference for all public classes and functions in dynaris.

+------------------+-------------------------------------------------------------+
| Module           | Description                                                 |
+==================+=============================================================+
| :doc:`dlm`       | High-level ``DLM`` class (fit, smooth, forecast, plot)      |
+------------------+-------------------------------------------------------------+
| :doc:`components` | Six composable building blocks (``LocalLevel``, etc.)      |
+------------------+-------------------------------------------------------------+
| :doc:`core`      | ``StateSpaceModel``, ``GaussianState``, result containers   |
+------------------+-------------------------------------------------------------+
| :doc:`filters`   | Kalman filter (predict, update, full forward pass)          |
+------------------+-------------------------------------------------------------+
| :doc:`smoothers` | Rauch-Tung-Striebel backward smoother                       |
+------------------+-------------------------------------------------------------+
| :doc:`estimation` | MLE, EM algorithm, diagnostics, transforms                 |
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

   dlm
   components
   core
   filters
   smoothers
   estimation
   forecast
   plotting
   datasets
