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
