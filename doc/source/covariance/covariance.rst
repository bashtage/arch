Long-run Covariance Estimation
==============================

Long-run Covariance Estimators
------------------------------

Kernel-based Estimators
~~~~~~~~~~~~~~~~~~~~~~~

.. module:: arch.covariance.kernel
   :synopsis: Kernel-based long-run covariance estimation

.. currentmodule:: arch.covariance.kernel

.. autosummary::
   :toctree: generated/

   Andrews
   Bartlett
   Gallant
   NeweyWest
   Parzen
   ParzenCauchy
   ParzenGeometric
   ParzenRiesz
   QuadraticSpectral
   TukeyHamming
   TukeyHanning
   TukeyParzen
   ZeroLag


Vector AR-based Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: arch.covariance.var
   :synopsis: Vector-AR-based long-run covariance estimation

.. currentmodule:: arch.covariance.var

.. autosummary::
   :toctree: generated/

   PreWhitenedRecolored

Results
-------

All long-run covariance estimators return their results using the same type
of object.

.. currentmodule:: arch.covariance

.. autosummary::
   :toctree: generated/

   ~kernel.CovarianceEstimate
