.. module:: arch.experimental.engine
   :noindex:
.. currentmodule:: arch.experimental.engine

Accelerated NumPy
=================

The feature is to allow users to choose alternative NumPy-like engine
to run on CPU and GPU. Currently, the following engine are supported in
CPU and GPU runtime


* `JAX <https://jax.readthedocs.io/en/latest/index.html#>`_

* `TensorFlow <https://www.tensorflow.org/guide/tf_numpy>`_

* `CuPy <https://docs.cupy.dev/en/stable/index.html>`_

There are two options users can switch the backend engine.

1. Context Manager

Users can use function ``use_backend`` in a ``with`` statement to temporarily
switch the NumPy engine.

In the below example, assume that ``data`` object is a timeseries in NumPy array.
The covariance estimation (NeweyWest) is computed in TensorFlow. Since the
output is in TensorFlow Tensor type, the last line convert the long term
covariance from Tensor to NumPy array type.

.. code-block:: python

    import numpy as np

    from arch.experimental import use_backend
    from arch.covariance.kernel import NeweyWest

    with use_backend("tensorflow"):
        cov = NeweyWest(data).cov

    long_term_cov = np.asarray(cov.long_term)

2. Global

Users can also configure the backend engine in global level with function
``set_backend``.

.. code-block:: python

    from arch.experimental import set_backend

    set_backend("tensorflow")

    # Output is already in TensorFlow Tensor type
    long_term_cov = NeweyWest(data).cov.long_term

For further examples, please refer to the example
`notebook <experimental_accelerated_numpy.ipynb>`_.

Configure
---------

.. autosummary::
   :toctree: generated/

    use_backend
    set_backend

