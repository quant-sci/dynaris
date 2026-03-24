Built-in Datasets
=================

dynaris ships with six classic time series datasets for examples and testing.
Each loader returns a pandas ``Series`` with an appropriate index.

+------------------+--------------------+------+-----------+-----------------+
| Dataset          | Loader             | N    | Frequency | Domain          |
+==================+====================+======+===========+=================+
| Nile river flow  | ``load_nile()``    | 100  | Annual    | Hydrology       |
+------------------+--------------------+------+-----------+-----------------+
| Airline          | ``load_airline()`` | 144  | Monthly   | Transportation  |
| passengers       |                    |      |           |                 |
+------------------+--------------------+------+-----------+-----------------+
| Lynx population  | ``load_lynx()``    | 114  | Annual    | Ecology         |
+------------------+--------------------+------+-----------+-----------------+
| Sunspot numbers  | ``load_sunspots()``| 288  | Annual    | Astronomy       |
+------------------+--------------------+------+-----------+-----------------+
| Global           | ``load_temperature | 144  | Annual    | Climate         |
| temperature      | ()``               |      |           |                 |
+------------------+--------------------+------+-----------+-----------------+
| US GDP growth    | ``load_gdp()``     | 319  | Quarterly | Economics       |
+------------------+--------------------+------+-----------+-----------------+

Usage
-----

.. code-block:: python

   from dynaris.datasets import load_airline

   y = load_airline()
   print(y.head())
   print(f"Shape: {y.shape}, Index: {y.index[0]} to {y.index[-1]}")

All loaders accept no arguments and return a pandas ``Series``. The index
is a ``DatetimeIndex`` for monthly/quarterly data or an integer index for
annual data.

See :doc:`/api/datasets` for the full API reference.
