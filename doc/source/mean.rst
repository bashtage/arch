Mean Models
===========
All ARCH models start by specifying a mean model.

.. py:currentmodule:: arch.univariate

No Mean
~~~~~~~
.. autoclass:: ZeroMean
    :members: resids, simulate, fit

Constant Mean
~~~~~~~~~~~~~
.. autoclass:: ConstantMean
    :members: resids, simulate, fit

Autoregressions
~~~~~~~~~~~~~~~
.. autoclass:: ARX
    :members: resids, simulate, fit

Heterogeneous Autoregressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: HARX
    :members: resids, simulate, fit

Least Squares
~~~~~~~~~~~~~
.. autoclass:: LS
    :members: resids, simulate, fit


Writing New Mean Models
~~~~~~~~~~~~~~~~~~~~~~~
.. py:currentmodule:: arch.univariate.base

All mean models must inherit from :class:ARCHModel and provide all public
methods. There are two optional private methods that should be provided if
applicable.

.. autoclass:: ARCHModel
