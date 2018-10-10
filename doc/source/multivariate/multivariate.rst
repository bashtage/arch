Multivariate ARCH Models
------------------------


Mean Models
===========
All multivariate ARCH models start by specifying a mean model.

.. module:: arch.multivariate.mean
.. py:currentmodule:: arch.multivariate

VAR with Optional Exogenous Regressors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: VARX
    :members: resids, simulate, fit

Constant Mean
~~~~~~~~~~~~~
.. autoclass:: ConstantMean
    :members: resids, simulate, fit

Zero Mean
~~~~~~~~~
.. autoclass:: ZeroMean
    :members: resids, simulate, fit

Volatility Processes
====================
A volatility process is added to a mean model to capture time-varying
covariance.

.. module:: arch.multivariate.volatility
.. py:currentmodule:: arch.multivariate

Constant Covariance
~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConstantCovariance
    :members: starting_values, backcast, compute_covariance, bounds, constraints, simulate

EWMA (RiskMetrics) Covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EWMACovariance
    :members: starting_values, backcast, compute_covariance, bounds, constraints, simulate

Distributions
=============
A distribution is the final component of a multivariate ARCH Model.

.. module:: arch.multivariate.distribution
.. py:currentmodule:: arch.multivariate

Multivariate Normal
~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultivariateNormal
    :members: starting_values, bounds, constraints, simulate, loglikelihood
