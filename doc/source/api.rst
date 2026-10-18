API Reference
=============

This page contains a list of the essential end-user API functions
and classes. The data sets that are included in the package are described in
:doc:`data`, and the warnings and exceptions that the package raises are described
in :doc:`warnings`.

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
   ~arch.univariate.MAMean
   ~arch.univariate.ARCHInMean

Volatility Process Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~arch.univariate.ARCH
   ~arch.univariate.GARCH
   ~arch.univariate.EGARCH
   ~arch.univariate.APARCH
   ~arch.univariate.HARCH
   ~arch.univariate.FIGARCH
   ~arch.univariate.FIAPARCH
   ~arch.univariate.NGARCH
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

Residual Diagnostics
~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~arch.univariate.excess_kurtosis
   ~arch.univariate.hill_estimator
   ~arch.univariate.var_ratio

Model Results and Forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~arch.univariate.base.ARCHModelResult
   ~arch.univariate.base.ARCHModelFixedResult
   ~arch.univariate.base.ARCHModelForecast
   ~arch.univariate.base.ARCHModelForecastSimulation
   ~arch.univariate.volatility.VarianceForecast
   ~arch.utility.testing.WaldTestStatistic

Extending the Model Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~arch.univariate.base.ARCHModel
   ~arch.univariate.volatility.VolatilityProcess
   ~arch.univariate.recursions.VolatilityUpdater
   ~arch.univariate.distribution.Distribution

Unit Root Testing
-----------------
.. autosummary::

   ~arch.unitroot.ADF
   ~arch.unitroot.DFGLS
   ~arch.unitroot.PhillipsPerron
   ~arch.unitroot.ZivotAndrews
   ~arch.unitroot.VarianceRatio
   ~arch.unitroot.KPSS
   ~arch.unitroot.auto_bandwidth

Cointegration Testing
---------------------
.. autosummary::

   ~arch.unitroot.cointegration.engle_granger
   ~arch.unitroot.cointegration.phillips_ouliaris
   ~arch.unitroot.cointegration.EngleGrangerTestResults
   ~arch.unitroot.cointegration.PhillipsOuliarisTestResults

Cointegrating Relationship Estimation
-------------------------------------
.. autosummary::

   ~arch.unitroot.cointegration.CanonicalCointegratingReg
   ~arch.unitroot.cointegration.DynamicOLS
   ~arch.unitroot.cointegration.FullyModifiedOLS
   ~arch.unitroot.cointegration.CointegrationAnalysisResults
   ~arch.unitroot.cointegration.DynamicOLSResults

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

   ~arch.covariance.kernel.Andrews
   ~arch.covariance.kernel.Bartlett
   ~arch.covariance.kernel.Gallant
   ~arch.covariance.kernel.NeweyWest
   ~arch.covariance.kernel.Parzen
   ~arch.covariance.kernel.ParzenCauchy
   ~arch.covariance.kernel.ParzenGeometric
   ~arch.covariance.kernel.ParzenRiesz
   ~arch.covariance.kernel.QuadraticSpectral
   ~arch.covariance.kernel.TukeyHamming
   ~arch.covariance.kernel.TukeyHanning
   ~arch.covariance.kernel.TukeyParzen
   ~arch.covariance.kernel.ZeroLag
   ~arch.covariance.var.PreWhitenedRecolored
   ~arch.covariance.kernel.CovarianceEstimator
   ~arch.covariance.kernel.CovarianceEstimate

Data Sets
---------
.. autosummary::

   ~arch.data.sp500.load
   ~arch.data.nasdaq.load
   ~arch.data.vix.load
   ~arch.data.wti.load
   ~arch.data.crude.load
   ~arch.data.core_cpi.load
   ~arch.data.default.load
   ~arch.data.frenchdata.load
   ~arch.data.binary.load
   ~arch.data.utility.load_file

Utilities
---------
.. autosummary::
   :toctree: generated/

   ~arch.test
