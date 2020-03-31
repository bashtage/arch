API Reference
=============

This page lists contains a list of the essential end-user API functions
and classes.

Volatility Modeling
-------------------

High-level
~~~~~~~~~~

.. autosummary::

   ~arch.univariate.arch_model

Mean Specification
~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~arch.univariate.ConstantMean
   ~arch.univariate.ZeroMean
   ~arch.univariate.HARX
   ~arch.univariate.ARX
   ~arch.univariate.LS

Volatility Process Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~arch.univariate.GARCH
   ~arch.univariate.EGARCH
   ~arch.univariate.HARCH
   ~arch.univariate.FIGARCH
   ~arch.univariate.MIDASHyperbolic
   ~arch.univariate.EWMAVariance
   ~arch.univariate.RiskMetrics2006
   ~arch.univariate.ConstantVariance
   ~arch.univariate.FixedVariance

Shock Distributions
~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~arch.univariate.Normal
   ~arch.univariate.StudentsT
   ~arch.univariate.SkewStudent
   ~arch.univariate.GeneralizedError

Unit Root Testing
-----------------
.. autosummary::

   ~arch.unitroot.ADF
   ~arch.unitroot.DFGLS
   ~arch.unitroot.PhillipsPerron
   ~arch.unitroot.ZivotAndrews
   ~arch.unitroot.VarianceRatio
   ~arch.unitroot.KPSS

Cointegration Testing
---------------------
.. autosummary::

   ~arch.unitroot.cointegration.engle_granger
   ~arch.unitroot.cointegration.phillips_ouliaris

Cointegrating Relationship Estimation
-------------------------------------
.. autosummary::

   ~arch.unitroot.cointegration.CanonicalCointegratingReg
   ~arch.unitroot.cointegration.DynamicOLS
   ~arch.unitroot.cointegration.FullyModifiedOLS

Bootstraps
----------

.. autosummary::
   ~arch.bootstrap.IIDBootstrap
   ~arch.bootstrap.IndependentSamplesBootstrap
   ~arch.bootstrap.StationaryBootstrap
   ~arch.bootstrap.CircularBlockBootstrap
   ~arch.bootstrap.MovingBlockBootstrap

Block-length Selection
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~arch.bootstrap.optimal_block_length

Testing with Multiple-Comparison
--------------------------------
.. autosummary::

   ~arch.bootstrap.SPA
   ~arch.bootstrap.MCS
   ~arch.bootstrap.StepM

Long-run Covariance (HAC) Estimation
------------------------------------
.. autosummary::

   ~arch.covariance.kernel.Bartlett
   ~arch.covariance.kernel.Parzen
   ~arch.covariance.kernel.ParzenCauchy
   ~arch.covariance.kernel.ParzenGeometric
   ~arch.covariance.kernel.ParzenRiesz
   ~arch.covariance.kernel.QuadraticSpectral
   ~arch.covariance.kernel.TukeyHamming
   ~arch.covariance.kernel.TukeyHanning
   ~arch.covariance.kernel.TukeyParzen
