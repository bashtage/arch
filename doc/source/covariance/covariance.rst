Long-run Covariance Estimation
==============================

Long-run Covariance Estimators
------------------------------

Kernel-based Estimators
~~~~~~~~~~~~~~~~~~~~~~~

.. module:: arch.covariance.kernel
   :synopsis: Kernel-based long-run covariance estimation

.. currentmodule:: arch.covariance.kernel

Estimators that accept a ``kernel`` select it by name. The names are the names
of the classes below in ``arch.covariance.kernel.KERNELS``, for example
``"Bartlett"`` or ``"QuadraticSpectral"``, and they are not case sensitive and
ignore hyphens and underscores, so that ``"quadratic-spectral"`` is the same as
``"QuadraticSpectral"``. ``ZeroLag`` is only used by
:class:`~arch.covariance.var.PreWhitenedRecolored` when ``kernel`` is None.

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


Base Class
----------
All long-run covariance estimators inherit from :class:`~arch.covariance.kernel.CovarianceEstimator`.

.. autosummary::
   :toctree: generated/

   ~kernel.CovarianceEstimator
