Mean Models
===========
All ARCH models start by specifying a mean model.

.. py:currentmodule::arch.mean

.. automodule:: arch.mean

No Mean
~~~~~~~
.. autoclass:: ZeroMean
    :members: resids, simulate, fit
    :show-inheritance:

Constant Mean
~~~~~~~~~~~~~
.. autoclass:: ConstantMean
    :members: resids, simulate, fit
    :show-inheritance:

Autoregressions
~~~~~~~~~~~~~~~
.. autoclass:: ARX
    :members: resids, simulate, fit
    :show-inheritance:

Heterogeneous Autoregressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: HARX
    :members: resids, simulate, fit
    :show-inheritance:

Least Squares
~~~~~~~~~~~~~
.. autoclass:: LS
    :members: resids, simulate, fit
    :show-inheritance:


Writing New Mean Models
~~~~~~~~~~~~~~~~~~~~~~~
.. py:currentmodule::arch.base

All mean models must inherit from :class:ARCHModel and provide all public
methods. There are two optional private methods that should be provided if
applicable.

.. autoclass:: ARCHModel
