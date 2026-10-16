Theoretical Background
----------------------

The models in ``arch.univariate`` divide an observed series :math:`r_t`, usually a
return, into a mean, a conditional volatility and a standardized shock

.. math::
   :nowrap:

   \begin{eqnarray}
      r_t        & = & \mu_t + \epsilon_t \\
      \epsilon_t & = & \sigma_t e_t
   \end{eqnarray}

where :math:`\mu_t` is the mean, :math:`\sigma^2_t` is the conditional variance of
:math:`\epsilon_t` and :math:`e_t` is the standardized shock, which has mean 0 and
variance 1. An ARCH-type model specifies each of the three components: a mean model, a
volatility process and a distribution for :math:`e_t`. Model parameters are estimated
by maximum likelihood. The default estimator of the parameter covariance,
``cov_type="robust"``, corresponds in the ARCH literature to the estimator of
Bollerslev and Wooldridge [BollerslevWooldridge1992]_. The
:doc:`introduction <introduction>` shows how to build and estimate a model, and
:doc:`forecasting` and :doc:`results` describe what can be done with it.

Mean Models
~~~~~~~~~~~
The mean model determines :math:`\mu_t`, see :ref:`mean-models`.

* :class:`~arch.univariate.ZeroMean`, :math:`r_t = \epsilon_t`, and
  :class:`~arch.univariate.ConstantMean`, :math:`r_t = \mu + \epsilon_t`.
* :class:`~arch.univariate.LS`, :math:`r_t = \mu + \gamma' x_t + \epsilon_t`, where
  :math:`x_t` are exogenous regressors.
* :class:`~arch.univariate.ARX` and :class:`~arch.univariate.HARX`, which are an
  autoregression and a heterogeneous autoregression, both with optional exogenous
  regressors.
* :class:`~arch.univariate.MAMean`, a moving average,
  :math:`r_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \ldots + \theta_q \epsilon_{t-q}`.
* :class:`~arch.univariate.ARCHInMean`, in which a function :math:`f` of the
  conditional variance enters the mean,
  :math:`r_t = \mu + \kappa f(\sigma^2_t) + \sum_{i=1}^p \phi_{L_{i}} r_{t-L_{i}} + \gamma' x_t + \epsilon_t`.

Volatility Processes
~~~~~~~~~~~~~~~~~~~~
The volatility process determines :math:`\sigma^2_t`, see :ref:`volatility-processes`.
The :class:`~arch.univariate.GARCH` class covers the ARCH, GARCH, GJR-GARCH, AVARCH,
AVGARCH and TARCH specifications. Its variance dynamics, using the ``power``
:math:`\lambda`, are

.. math::

   \sigma_{t}^{\lambda}=\omega
   + \sum_{i=1}^{p}\alpha_{i}\left|\epsilon_{t-i}\right|^{\lambda}
   +\sum_{j=1}^{o}\gamma_{j}\left|\epsilon_{t-j}\right|^{\lambda}
   I\left[\epsilon_{t-j}<0\right]+\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{\lambda}

The GARCH(1,1) of the :doc:`introduction <introduction>` is the case :math:`p=q=1`,
:math:`o=0` and :math:`\lambda=2`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Process
     - Description
   * - :class:`~arch.univariate.ARCH`
     - :math:`\sigma_t^{2}=\omega+\sum_{i=1}^{p}\alpha_{i}\epsilon_{t-i}^{2}`
       [Engle1982]_
   * - :class:`~arch.univariate.GARCH`
     - ARCH, GARCH [Bollerslev1986]_, GJR-GARCH [GJR1993]_, AVARCH, AVGARCH and
       TARCH [Zakoian1994]_ specifications, using the equation above
   * - :class:`~arch.univariate.EGARCH`
     - :math:`\ln\sigma_{t}^{2}` is a function of the lagged absolute and signed
       standardized shocks :math:`e_{t-i}=\epsilon_{t-i}/\sigma_{t-i}` and of
       lagged :math:`\ln\sigma_{t}^{2}` [Nelson1991]_
   * - :class:`~arch.univariate.APARCH`
     - Asymmetric power ARCH, which models :math:`\sigma_t^{\delta}` using the
       terms :math:`(|\epsilon_{t-i}|-\gamma_i\epsilon_{t-i})^{\delta}`
       [DingGrangerEngle1993]_
   * - :class:`~arch.univariate.FIGARCH`
     - Fractionally integrated GARCH, which has a fractional differencing parameter
       :math:`d` and is estimated using its truncated ARCH(:math:`\infty`)
       representation [BaillieBollerslevMikkelsen1996]_
   * - :class:`~arch.univariate.FIAPARCH`
     - Fractionally integrated asymmetric power ARCH, which extends the FIGARCH with
       the asymmetry and the power of the APARCH [Tse1998]_
   * - :class:`~arch.univariate.HARCH`
     - Heterogeneous ARCH, in which the variance depends on averages of past squared
       shocks over several lag lengths
   * - :class:`~arch.univariate.MIDASHyperbolic`
     - Variance that depends on :math:`m` lagged squared shocks, with weights
       :math:`\phi_i(\theta)\propto\Gamma(i+\theta)/(\Gamma(i+1)\Gamma(\theta))`
       that sum to 1 [Foroni2013]_ [Sheppard2018]_
   * - :class:`~arch.univariate.EWMAVariance`
     - :math:`\sigma_t^{2}=\lambda\sigma_{t-1}^2 + (1-\lambda)\epsilon^2_{t-1}`, with
       :math:`\lambda=0.94` by default
   * - :class:`~arch.univariate.RiskMetrics2006`
     - A weighted average of ``kmax`` EWMA variance processes
   * - :class:`~arch.univariate.ConstantVariance`
     - The same variance in all periods
   * - :class:`~arch.univariate.FixedVariance`
     - A variance that is provided by the user

Distributions
~~~~~~~~~~~~~
The distribution is the distribution of :math:`e_t`, which is used to construct the
likelihood, see :ref:`distributions`. All of the distributions are standardized to
have mean 0 and variance 1.

* :class:`~arch.univariate.Normal`.
* :class:`~arch.univariate.StudentsT`, with one shape parameter :math:`\nu`
  [Bollerslev1987]_.
* :class:`~arch.univariate.SkewStudent`, a skewed Student's t with the tail shape
  :math:`\eta` and the skewness :math:`\lambda` [Hansen1994]_. When :math:`\lambda=0`
  it is the standardized Student's t.
* :class:`~arch.univariate.GeneralizedError`, the generalized error distribution, with
  one shape parameter :math:`\nu`.

References
~~~~~~~~~~
.. [BaillieBollerslevMikkelsen1996] Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O.
   (1996). Fractionally integrated generalized autoregressive conditional
   heteroskedasticity. *Journal of Econometrics*, 74(1), 3-30.

.. [Bollerslev1986] Bollerslev, T. (1986). Generalized autoregressive conditional
   heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

.. [Bollerslev1987] Bollerslev, T. (1987). A conditionally heteroskedastic time series
   model for speculative prices and rates of return. *The Review of Economics and
   Statistics*, 69(3), 542-547.

.. [BollerslevWooldridge1992] Bollerslev, T., & Wooldridge, J. M. (1992). Quasi-maximum
   likelihood estimation and inference in dynamic models with time-varying
   covariances. *Econometric Reviews*, 11(2), 143-172.

.. [DingGrangerEngle1993] Ding, Z., Granger, C. W. J., & Engle, R. F. (1993). A long
   memory property of stock market returns and a new model. *Journal of Empirical
   Finance*, 1(1), 83-106.

.. [Engle1982] Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with
   estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4),
   987-1007.

.. [Foroni2013] Foroni, Claudia, and Massimiliano Marcellino. "A survey of
   Econometric Methods for Mixed-Frequency Data". Norges Bank. (2013).

.. [GJR1993] Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation
   between the expected value and the volatility of the nominal excess return on
   stocks. *The Journal of Finance*, 48(5), 1779-1801.

.. [Hansen1994] Hansen, B. E. (1994). Autoregressive conditional density estimation.
   *International Economic Review*, 35(3), 705-730.

.. [Nelson1991] Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns:
   a new approach. *Econometrica*, 59(2), 347-370.

.. [Sheppard2018] Sheppard, Kevin. "Direct volatility modeling". Manuscript. (2018).

.. [Tse1998] Tse, Y. K. (1998). The conditional heteroscedasticity of the yen-dollar
   exchange rate. *Journal of Applied Econometrics*, 13(1), 49-55.

.. [Zakoian1994] Zakoian, J.-M. (1994). Threshold heteroskedastic models. *Journal of
   Economic Dynamics and Control*, 18(5), 931-955.
