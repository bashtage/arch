.. _volatility-processes:

Volatility Processes
====================
A volatility process is added to a mean model to capture time-varying
volatility.

.. module:: arch.univariate
   :noindex:

.. currentmodule:: arch.univariate

.. autosummary::
   :toctree: generated/

   ConstantVariance
   GARCH
   FIGARCH
   EGARCH
   HARCH
   MIDASHyperbolic
   ARCH
   APARCH

Parameterless Variance Processes
--------------------------------
Some volatility processes use fixed parameters and so have no parameters that
are estimable.

.. autosummary::
   :toctree: generated/

   EWMAVariance
   RiskMetrics2006

FixedVariance
-------------
The ``FixedVariance`` class is a special-purpose volatility process that allows
the so-called zig-zag algorithm to be used.  See the example for usage.

.. autosummary::
   :toctree: generated/

   FixedVariance

Writing New Volatility Processes
--------------------------------
All volatility processes must inherit from :class:`~arch.univariate.volatility.VolatilityProcess` and provide
all public methods.

.. currentmodule:: arch.univariate.volatility

.. autosummary::
   :toctree: generated/

   VolatilityProcess

They may optionally expose a
:class:`~arch.univariate.recursions_python.VolatilityUpdater` class
that can be used in :class:`~arch.univariate.ARCHInMean` estimation.

.. currentmodule:: arch.univariate.recursions_python

.. autosummary::
   :toctree: generated/

   VolatilityUpdater
