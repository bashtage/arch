Volatility Processes
====================
A volatility process is added to a mean model to capture time-varying
volatility.

.. py:currentmodule::arch.volatility

.. automodule:: arch.volatility

Constant Variance
-----------------

.. autoclass:: ConstantVariance
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate
    :show-inheritance:

GARCH
-----

.. autoclass:: GARCH
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate
    :show-inheritance:

EGARCH
------

.. autoclass:: EGARCH
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate
    :show-inheritance:

HARCH
-----

.. autoclass:: HARCH
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate
    :show-inheritance:

ARCH
----

.. autoclass:: ARCH
    :members: starting_values, backcast, compute_variance, bounds, constraints, simulate
    :show-inheritance:

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

Writing New Volatility Processes
--------------------------------
All volatility processes must inherit from :class:VolatilityProcess and provide
all public methods.

.. autoclass:: VolatilityProcess