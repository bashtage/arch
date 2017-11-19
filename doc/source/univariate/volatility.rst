Volatility Processes
====================
A volatility process is added to a mean model to capture time-varying
volatility.

.. module:: arch.univariate.volatility
.. py:currentmodule:: arch.univariate

Constant Variance
-----------------

.. autoclass:: ConstantVariance
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate

GARCH
-----

.. autoclass:: GARCH
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate

EGARCH
------

.. autoclass:: EGARCH
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate

HARCH
-----

.. autoclass:: HARCH
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate

ARCH
----

.. autoclass:: ARCH
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate

CGARCH
------

.. autoclass:: CGARCH
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate

Parameterless Variance Processes
--------------------------------
Some volatility processes use fixed parameters and so have no parameters that
are estimable.

EWMA Variance
-------------

.. autoclass:: EWMAVariance
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate
    :show-inheritance:

RiskMetrics (2006)
------------------

.. autoclass:: RiskMetrics2006
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate
    :show-inheritance:

FixedVariance
-------------
The ``FixedVariance`` class is a special-purpose volatility process that allows
the so-called zig-zag algorithm to be used.  See the example for usage.

.. autoclass:: FixedVariance
    :show-inheritance:

Writing New Volatility Processes
--------------------------------
All volatility processes must inherit from :class:VolatilityProcess and provide
all public methods.

.. py:currentmodule:: arch.univariate.volatility
.. autoclass:: VolatilityProcess