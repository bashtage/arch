Utilities
=========
Utilities that do not fit well on other pages.

.. module:: arch.utility.testing
   :synopsis: Utilities that do not fit well in other modules
.. currentmodule:: arch.utility.testing

Test Results
------------

.. autoclass:: WaldTestStatistic
   :members:

Residual Diagnostics
--------------------

.. module:: arch.univariate
   :noindex:
.. currentmodule:: arch.univariate

If a model is correctly specified, the standardized residuals
:math:`\hat{e}_t = \hat{\epsilon}_t / \hat{\sigma}_t` behave like draws from the
distribution of the model, and so the tails of the residuals can be compared to the
tails of the distribution that was assumed. The functions below summarize the tails
of standardized residuals, and are also available for the results of a model, which
use the standardized residuals and the estimated distribution of the model.

* :func:`~arch.univariate.excess_kurtosis` is the sample excess kurtosis, the fourth
  central moment divided by the squared variance, less 3. It is 0 for a Normal, and
  is positive when the residuals have heavier tails than a Normal. Observations
  that are not finite are ignored.
* :func:`~arch.univariate.hill_estimator` is the Hill estimator of the tail index,
  using the ``k`` largest absolute residuals, and ``int(sqrt(n))`` of them if ``k``
  is not provided. The estimator is
  :math:`\hat{\nu} = \left(k^{-1}\sum_{i=1}^{k} \ln X_{(n-i+1)} - \ln X_{(n-k)}\right)^{-1}`,
  where :math:`X_{(j)}` is the :math:`j`-th smallest absolute residual. Smaller values
  indicate heavier tails. The tail index of a Student's t distribution is its degrees
  of freedom.
* :func:`~arch.univariate.var_ratio` is the ratio of the Value-at-Risk (VaR) of a
  Normal to the VaR of the distribution of the model, at a confidence level that is
  0.999 unless ``level`` is provided. A ratio below 1 means that the Normal
  understates the VaR of the model's distribution. It depends on the distribution
  and its parameters only, and not on residuals.

The results of a model have the same diagnostics. They use the standardized
residuals, ``std_resid``, and, for ``var_ratio``, the estimated distribution and
parameters.

.. ipython:: python

    import arch.data.sp500
    from arch.univariate import arch_model

    sp500 = arch.data.sp500.load()
    returns = 100 * sp500["Adj Close"].pct_change().dropna()
    res = arch_model(returns, dist="t").fit(disp="off")
    res.excess_kurtosis
    res.hill_estimator()
    res.var_ratio()

The functions can also be used with any standardized residuals, and with a
distribution that is not estimated.

.. ipython:: python

    from arch.univariate import (
        StudentsT,
        excess_kurtosis,
        hill_estimator,
        var_ratio,
    )

    excess_kurtosis(res.std_resid)
    hill_estimator(res.std_resid, k=50)
    var_ratio(0.99, StudentsT(), [6.0])

.. autosummary::
   :toctree: generated/

   excess_kurtosis
   hill_estimator
   var_ratio

The corresponding members of the results classes are
:attr:`~arch.univariate.base.ARCHModelFixedResult.excess_kurtosis`,
:meth:`~arch.univariate.base.ARCHModelFixedResult.hill_estimator` and
:meth:`~arch.univariate.base.ARCHModelFixedResult.var_ratio`.

Backtesting Value-at-Risk
-------------------------

.. module:: arch.univariate
   :noindex:
.. currentmodule:: arch.univariate

The functions below compare a series of Value-at-Risk (VaR) forecasts with the returns
that were realized. VaR is reported as a **positive loss** (a 99% VaR of 2.5 is the loss
that is exceeded with probability 1%), and an observation is an *exception* if the loss
exceeds the VaR, :math:`-r_t > \mathrm{VaR}_t`. VaR is named by its confidence level, and
``p`` is always the probability of an exception, the size of the tail: ``p=0.05`` for a
95% VaR and ``p=0.01`` for a 99% VaR. A quantile of the returns, which is
negative in the lower tail, must be negated first. The returns and the VaR must be finite.
If both are Series they are aligned on the index, and the observations that they have in
common are used. Otherwise they must have the same length.

* :func:`~arch.univariate.var_exceptions` is the bool array of exceptions.
* :func:`~arch.univariate.kupiec_pof` is the proportion-of-failures likelihood-ratio test
  of Kupiec (1995) that the probability of an exception is ``p``. It is
  :math:`\chi^2_1`, and it is two-sided, so that too few exceptions also reject.
* :func:`~arch.univariate.christoffersen_independence` is the likelihood-ratio test of
  Christoffersen (1998) that the exceptions are independent, against a first-order Markov
  chain. It is :math:`\chi^2_1`.
* :func:`~arch.univariate.christoffersen_conditional_coverage` is the sum of the two
  statistics, which tests the coverage and the independence jointly. It is
  :math:`\chi^2_2`.
* :func:`~arch.univariate.traffic_light` is the zone of the framework of the Basel
  Committee on Banking Supervision (1996) for the exceptions in the last ``window``
  observations (250 by default). The zone is green if the cumulative binomial probability
  of the number of exceptions is below 95%, yellow if it is at least 95% and below
  99.99%, and red otherwise. For 250 observations and ``p=0.01`` green is 0 to 4
  exceptions, yellow is 5 to 9 and red is 10 or more, as in Table 2 of the framework. It
  returns a :class:`~arch.univariate.TrafficLightResult` with the zone, the number of
  exceptions and the cumulative probability. A series that is shorter than ``window``
  raises an error.

The three tests return a :class:`~arch.utility.testing.WaldTestStatistic`. The
likelihood-ratio statistics are computed with :math:`0\ln 0=0`, so that a series with no
exception, or with only exceptions, has a statistic.

.. ipython:: python

    from arch.univariate import (
        christoffersen_conditional_coverage,
        christoffersen_independence,
        kupiec_pof,
        traffic_light,
        var_exceptions,
    )

    # A 99% VaR (p=0.01) from the model with Student's t errors, as a positive loss
    nu = res.params[["nu"]].to_numpy()
    quantile = float(res.model.distribution.ppf(0.01, nu))
    var = -(res.params["mu"] + res.conditional_volatility * quantile)
    exceptions = var_exceptions(returns, var)
    exceptions.sum()
    kupiec_pof(returns, var, 0.01)
    christoffersen_independence(returns, var).pval
    christoffersen_conditional_coverage(returns, var, 0.01).pval
    traffic_light(returns, var, p=0.01)

.. autosummary::
   :toctree: generated/

   var_exceptions
   kupiec_pof
   christoffersen_independence
   christoffersen_conditional_coverage
   traffic_light

.. autoclass:: TrafficLightResult
