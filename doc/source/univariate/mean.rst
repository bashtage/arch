Mean Models
===========
All ARCH models start by specifying a mean model.

.. py:currentmodule:: arch.univariate

No Mean
~~~~~~~
.. autoclass:: ZeroMean
    :members: resids, simulate, fit,  fix, forecast

Constant Mean
~~~~~~~~~~~~~
.. autoclass:: ConstantMean
    :members: resids, simulate, fit, forecast

Autoregressions
~~~~~~~~~~~~~~~
.. autoclass:: ARX
    :members: resids, simulate, fit, fix, forecast

Heterogeneous Autoregressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: HARX
    :members: resids, simulate, fit, fix, forecast

Least Squares
~~~~~~~~~~~~~
.. autoclass:: LS
    :members: resids, simulate, fit, fix


Writing New Mean Models
~~~~~~~~~~~~~~~~~~~~~~~
.. py:currentmodule:: arch.univariate.base

All mean models must inherit from :class:ARCHModel and provide all public
methods. There are two optional private methods that should be provided if
applicable.

.. autoclass:: ARCHModel
