Confidence Intervals
--------------------
The confidence interval function allows three types of confidence intervals to
be constructed:

* Nonparametric, which only resample the data
* Semi-parametric, which use resampled residuals
* Parametric, which simulate residuals

Confidence intervals can then be computed using one of 5 methods:

* Basic (``basic``)
* Percentile (``percentile``)
* Studentized (``studentized``)
* Asymptotic using parameter covariance (``norm``, ``var`` or ``cov``)
* Bias-corrected (``bc``, ``bias-corrected`` or ``debiased``)
* Bias-corrected and accelerated (``bca``)

Finally, the studentized bootstrap can be conducted using either an analytic
parameter variance formula or using a nested-bootstrap.

Setup
=====

All examples will construct confidence intervals for the Sharpe ratio of the
S&P 500, which is the ratio of the assumalized mean to the annualized standard
deviation.  The parameters will be the annualized mean, the annualized standard
deviation and the Sharpe ratio.

The setup makes use of return data downloaded from Yahoo!

::

    import datetime as dt
    import pandas as pd
    import pandas.io.data as web
    start = dt.datetime(1951,1,1)
    end = dt.datetime(2014,1,1)
    sp500 = web.get_data_yahoo('^GSPC', start=start, end=end)
    start = sp500.index.min()
    end = sp500.index.max()
    monthly_dates = pd.date_range(start, end, freq='M')
    monthly = sp500.reindex(monthly_dates, method='ffill')
    returns = 100 * monthly['Adj Close'].pct_change().dropna()

The main function used will return a 3-element array containing the parameters.

::

    def sharpe_ratio(x):
        mu, sigma = 12 * x.mean(), np.sqrt(12 * x.var())
        return np.array([mu, sigma, mu / sigma])

Confidence Interval Types
=========================

Three types of confidence intervals can be computed.  The simplest are
non-parametric which only make use of parameter estimates from both the original
data as well as the bootstrap resamples.  Semi-parametric mix the original data
with a limited form or resamples, usually for residuals.  Finally, parametric
bootstrap confidence intervals make use of a parametric distribution to
construct "as-if" exact confidence intervals.

Nonparametric Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Non-parametric sampling is the simplest method to construct confidence
intervals.

This example makes use of the percentile bootstrap which is conceptually the
simplest method - it constructs many bootstrap replications and simply return
the required order statistics from these empirical distributions.

::

    from arch.bootstrap import IIDBootstrap
    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='percentile')

.. note::

    While returns have little serial correlation, squared returns are highly
    persistent.  The IID bootstrap is not a good choice here.  Instead a
    time-series bootstrap with an appropriately chosen block size should be
    used.

Semi-parametric Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`Semiparametric Bootstraps <semiparametric-parametric-bootstrap>`

Parametric Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`Parametric Bootstraps <semiparametric-parametric-bootstrap>`


Confidence Interval Methods
===========================

Basic (``basic``)
~~~~~~~~~~~~~~~~~

Basic confidence intervals construct many bootstrap replications
:math:`\hat{\theta}_b^\star` and then constructs the confidence interval as

.. math::

    \left[\hat{\theta} + \left(\hat{\theta} - \hat{\theta}^{\star}_{u} \right),
    \hat{\theta} + \left(\hat{\theta} - \hat{\theta}^{\star}_{l} \right) \right]

where :math:`\hat{\theta}^{\star}_{l}` and :math:`\hat{\theta}^{\star}_{u}` are
the :math:`\alpha/2` and :math:`1-\alpha/2` empirical quantiles of the bootstrap
distribution.  When :math:`\theta` is a vector, the empirical quantiles are
computed element-by-element.

::

    from arch.bootstrap import IIDBootstrap
    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='basic')


Percentile (``percentile``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic confidence intervals construct many bootstrap replications
:math:`\hat{\theta}_b^\star` and then constructs the confidence interval as

.. math::

    \left[\hat{\theta}^{\star}_{l}, \hat{\theta}^{\star}_{u} \right]

where :math:`\hat{\theta}^{\star}_{l}` and :math:`\hat{\theta}^{\star}_{u}` are
the :math:`\alpha/2` and :math:`1-\alpha/2` empirical quantiles of the bootstrap
distribution.

::

    from arch.bootstrap import IIDBootstrap
    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='percentile')

Asymptotic Normal Approximation (``norm``, ``cov`` or ``var``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The asymptotic normal approximation methos simply estimates the covairance of
the parameters and then uses this with the usual quantiles from a normal
distribution.  The confidence interval is then

.. math::

    \left[\hat{\theta} + \hat{\sigma}\Phi^{-1}\left(\alpha/2\right),
    \hat{\theta} - \hat{\sigma}\Phi^{-1}\left(\alpha/2\right), \right]

where :math:`\hat{\sigma}` is the bootstrap estimate of the parameter standard
error.

::

    from arch.bootstrap import IIDBootstrap
    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='norm')

Studentized (``studentized``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The studentized bootstrap may be more accurate than some of the other methods.
The studentized bootstrap makes use of either a standard error function, when
parameter standard errors can be analytically computed, or a nested bootstrap,
to bootstrap studentized versions of the original statistic.  This can produce
higher-order refinements in some circumstances.

The confidence interval is then

.. math::

    \left[\hat{\theta} + \hat{\sigma}\hat{G}^{-1}\left(\alpha/2\right),
    \hat{\theta} + \hat{\sigma}\hat{G}^{-1}\left(1-\alpha/2\right), \right]

where :math:`\hat{G}` is the estimated quantile function for the studentized
data.

The version that uses a nested bootstrap is simple to implement although it can
be slow since it requires :math:`B` inner bootstraps of each of the :math:`B`
outer bootstraps.

::

    from arch.bootstrap import IIDBootstrap
    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='studentized')

Demonstrating the use of the standard error function is simpler in the CAP-M
example. Assuming the data are homoskedastic, the parameter standard errors
can be computed by

::

    def ols_se(params, y, x):
        e = y - x.dot(params)
        nobs = y.shape[0]
        sigma2 = e.dot(e) / (nobs - x.shape[1])
        xpx = x.T.dot(x) / nobs
        vcv = sigma2 * np.inv(xpx)
        return np.sqrt(np.diag(vcv))

.. note::

    Standard error functions must return a 1-d array with the same number
    of element as params

.. note::

    Standard error functions must match the patters
    ``std_err_func(params, *args, **kwargs)`` where ``params`` is an array
    of estimated parameters constructed using ``*args`` and ``**kwargs``.

Bias-corrected (``bc``, ``bias-corrected`` or ``debiased``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bias corrected bootstrap makes use of a bootstrap estimate of the bias to
improve confidence intervals.


::

    from arch.bootstrap import IIDBootstrap
    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='bc')


Bias-corrected and accelerated (``bca``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bias-corrected and accelerated confidence intervals make use of both
a bootstrap bias estimate and a jackknife acceleration term.  BCa intervals may
offer higher-order accuracy if some conditions are satisfied. Bias-corrected
confidence intervals are a special case of BCa intervals where the acceleration
parameter is set to 0.

::

    from arch.bootstrap import IIDBootstrap
    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='bca')

