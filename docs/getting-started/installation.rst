Installation
============

From PyPI
---------

.. code-block:: bash

   pip install dynaris

Or with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv add dynaris

From source
-----------

.. code-block:: bash

   git clone https://github.com/quant-sci/dynaris.git
   cd dynaris
   pip install -e .

Verify the installation
-----------------------

.. code-block:: python

   import dynaris
   print(dynaris.__version__)

Dependencies
------------

dynaris requires Python 3.12+ and depends on:

- `JAX <https://jax.readthedocs.io/>`_ (automatic differentiation and JIT)
- NumPy
- pandas
- SciPy
- Matplotlib

.. note::

   By default, JAX installs with CPU support. For GPU acceleration, see the
   `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.
