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
of the classes below in ``arch.covariance.kernel.KERNELS``, in lower case with
hyphens between the words, for example ``"bartlett"`` or ``"quadratic-spectral"``.
They are not case sensitive and ignore hyphens and underscores, so that
``"quadratic_spectral"`` is the same as ``"quadratic-spectral"``. ``ZeroLag`` is
only used by
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

The data are converted to ``float64``: integer, Boolean and ``float32`` data are
accepted and the calculation is done in ``float64``, and ``float16`` data raise a
``TypeError``. Complex data are also accepted, and are converted to ``complex128``. The
estimators are then Hermitian: the lag-:math:`j` autocovariance is
:math:`T^{-1}\sum_t x_t x_{t-j}^H`, where :math:`^H` is the conjugate transpose, and the
long-run covariance is a complex Hermitian matrix. For :math:`x_t=a_t+ib_t` it is
:math:`\Omega_{aa}+\Omega_{bb}+i(\Omega_{ba}-\Omega_{ab})`, where :math:`\Omega_{uv}` is
the long-run covariance of the real data :math:`u` and :math:`v` estimated with the same
kernel and bandwidth. :func:`~arch.utility.cov.cov_nw` follows the same rule.


Vector AR-based Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: arch.covariance.var
   :synopsis: Vector-AR-based long-run covariance estimation

.. currentmodule:: arch.covariance.var

.. autosummary::
   :toctree: generated/

   PreWhitenedRecolored

:class:`~arch.covariance.var.PreWhitenedRecolored` implements the pre-whitened
recolored (PWRC) estimator of Andrews and Monahan (1992) and the VAR-HAC estimator
of den Haan and Levin (2000). Kernel estimators can be poorly behaved when the data
are highly persistent, since the bandwidth required to capture the dependence is large
relative to the sample size. These estimators first remove most of the serial
dependence using a vector autoregression, estimate the long-run covariance of the
residuals, and then restore the dependence. They accept real data only, since there is
no single VAR for complex data (see the Notes of the class); use a kernel estimator, or
the VAR of the stacked real and imaginary parts. They are computed in three steps.

1. **Prewhitening.** A VAR is estimated by least squares,

   .. math::

      x_t = c + A_1 x_{t-1} + \ldots + A_P x_{t-P} + \epsilon_t

   where the constant is only included when ``center`` is ``True``. The order
   ``P`` is ``lags`` if it is provided. Otherwise it is chosen using the
   information criterion ``method``, which can be ``"aic"``, ``"hqc"`` or
   ``"bic"``, considering up to ``max_lag`` lags. When the data have more than
   one column and ``diagonal`` is ``True`` the search also considers models where
   the coefficients on the longest lags are diagonal, so that those lags only
   include the own lag of each series.
2. **Long-run covariance of the residuals.** The long-run covariance of the VAR
   residuals, :math:`\hat{\Omega}_\epsilon`, is computed using the ``kernel``
   (``"bartlett"`` by default), with a bandwidth that is chosen using the residuals
   if ``bandwidth`` is ``None``. If ``kernel`` is ``None`` the
   :class:`~arch.covariance.kernel.ZeroLag` kernel is used, which includes no
   autocovariances, so that :math:`\hat{\Omega}_\epsilon` is the covariance of the
   residuals. This is the VAR-HAC estimator.
3. **Recoloring.** The long-run covariance of :math:`x` is

   .. math::

      \hat{\Omega} = \hat{D}\hat{\Omega}_\epsilon\hat{D}^\prime,
      \qquad
      \hat{D} = \left(I - \sum_{i=1}^{P} \hat{A}_i\right)^{-1}

The estimate is available in ``cov``, as for all long-run covariance estimators. The
prewhitening step can be inspected. ``order`` is the order of the VAR, which is the
only way to find the order when it is selected using an information criterion. It is
a tuple, ``(p, q)``, where the VAR has unrestricted coefficients on the first ``p``
lags and diagonal coefficients on lags ``p + 1``, ..., ``q``. ``resid`` and
``resid_cov`` are the residuals of the VAR and their covariance.

The example compares the estimators using the absolute returns of the S&P 500 and
NASDAQ, which are highly persistent.

.. ipython:: python

    import pandas as pd

    import arch.data.nasdaq
    import arch.data.sp500
    from arch.covariance.kernel import Bartlett
    from arch.covariance.var import PreWhitenedRecolored

    sp500 = arch.data.sp500.load()["Adj Close"]
    nasdaq = arch.data.nasdaq.load()["Adj Close"]
    returns = 100 * pd.concat(
        [sp500.pct_change().abs(), nasdaq.pct_change().abs()],
        axis=1,
        keys=["SP500", "NASDAQ"],
    ).dropna()
    Bartlett(returns).cov.long_run
    pwrc = PreWhitenedRecolored(returns)
    pwrc.order
    pwrc.cov.long_run

The VAR-HAC estimator is computed by setting ``kernel`` to ``None``, and here the order
is selected using the BIC.

.. ipython:: python

    var_hac = PreWhitenedRecolored(returns, kernel=None, method="bic")
    var_hac.order
    var_hac.cov.long_run
    var_hac.resid_cov

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
