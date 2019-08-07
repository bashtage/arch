Confidence Intervals
--------------------
The confidence interval function allows three types of confidence intervals to
be constructed:

* Nonparametric, which only resamples the data
* Semi-parametric, which use resampled residuals
* Parametric, which simulate residuals

Confidence intervals can then be computed using one of 6 methods:

* Basic (``basic``)
* Percentile (``percentile``)
* Studentized (``studentized``)
* Asymptotic using parameter covariance (``norm``, ``var`` or ``cov``)
* Bias-corrected (``bc``, ``bias-corrected`` or ``debiased``)
* Bias-corrected and accelerated (``bca``)

.. contents::
    :local:

Setup
=====

All examples will construct confidence intervals for the Sharpe ratio of the
S&P 500, which is the ratio of the annualized mean to the annualized standard
deviation.  The parameters will be the annualized mean, the annualized standard
deviation and the Sharpe ratio.

The setup makes use of return data downloaded from Yahoo!

.. code-block:: python

    import datetime as dt

    import pandas as pd
    import pandas_datareader.data as web

    start = dt.datetime(1951, 1, 1)
    end = dt.datetime(2014, 1, 1)
    sp500 = web.DataReader('^GSPC', 'yahoo', start=start, end=end)
    low = sp500.index.min()
    high = sp500.index.max()
    monthly_dates = pd.date_range(low, high, freq='M')
    monthly = sp500.reindex(monthly_dates, method='ffill')
    returns = 100 * monthly['Adj Close'].pct_change().dropna()

The main function used will return a 3-element array containing the parameters.

.. code-block:: python

    def sharpe_ratio(x):
        mu, sigma = 12 * x.mean(), np.sqrt(12 * x.var())
        return np.array([mu, sigma, mu / sigma])

.. note::

    Functions must return 1-d NumPy arrays or Pandas Series.


Confidence Interval Types
=========================

Three types of confidence intervals can be computed.  The simplest are
non-parametric; these only make use of parameter estimates from both the
original data as well as the resampled data.  Semi-parametric mix the original
data with a limited form of resampling, usually for residuals.  Finally,
parametric bootstrap confidence intervals make use of a parametric distribution
to construct "as-if" exact confidence intervals.

Nonparametric Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Non-parametric sampling is the simplest method to construct confidence
intervals.

This example makes use of the percentile bootstrap which is conceptually the
simplest method - it constructs many bootstrap replications and returns
order statistics from these empirical distributions.

.. code-block:: python

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

.. note::

    ``conf_int`` can construct two-sided, upper or lower (one-sided) confidence
    intervals.  All examples use two-sided, 95% confidence intervals (the
    default).  This can be modified using the keyword inputs ``type``
    (``'upper'``, ``'lower'`` or ``'two-sided'``) and ``size``.

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

.. code-block:: python

    from arch.bootstrap import IIDBootstrap

    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='basic')


Percentile (``percentile``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The percentile method directly constructs confidence intervals from the empirical
CDF of the bootstrap parameter estimates, :math:`\hat{\theta}_b^\star`.
The confidence interval is then defined.

.. math::

    \left[\hat{\theta}^{\star}_{l}, \hat{\theta}^{\star}_{u} \right]

where :math:`\hat{\theta}^{\star}_{l}` and :math:`\hat{\theta}^{\star}_{u}` are
the :math:`\alpha/2` and :math:`1-\alpha/2` empirical quantiles of the bootstrap
distribution.

.. code-block:: python

    from arch.bootstrap import IIDBootstrap

    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='percentile')

Asymptotic Normal Approximation (``norm``, ``cov`` or ``var``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The asymptotic normal approximation method estimates the covariance of
the parameters and then combines this with the usual quantiles from a normal
distribution.  The confidence interval is then

.. math::

    \left[\hat{\theta} + \hat{\sigma}\Phi^{-1}\left(\alpha/2\right),
    \hat{\theta} - \hat{\sigma}\Phi^{-1}\left(\alpha/2\right), \right]

where :math:`\hat{\sigma}` is the bootstrap estimate of the parameter standard
error.

.. code-block:: python

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
data and where :math:`\hat{\sigma}` is a bootstrap estimate of the parameter
standard error.

The version that uses a nested bootstrap is simple to implement although it can
be slow since it requires :math:`B` inner bootstraps of each of the :math:`B`
outer bootstraps.

.. code-block:: python

    from arch.bootstrap import IIDBootstrap

    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='studentized')

In order to use the standard error function, it is necessary to estimate the
standard error of the parameters.  In this example, this can be done using a
method-of-moments argument and the delta-method. A detailed description of
the mathematical formula is beyond the intent of this document.

.. code-block:: python

    def sharpe_ratio_se(params, x):
        mu, sigma, sr = params
        y = 12 * x
        e1 = y - mu
        e2 = y ** 2.0 - sigma ** 2.0
        errors = np.vstack((e1, e2)).T
        t = errors.shape[0]
        vcv = errors.T.dot(errors) / t
        D = np.array([[1, 0],
                      [0, 0.5 * 1 / sigma],
                      [1.0 / sigma, - mu / (2.0 * sigma**3)]
                      ])
        avar = D.dot(vcv /t).dot(D.T)
        return np.sqrt(np.diag(avar))

The studentized bootstrap can then be implemented using the standard error
function.

.. code-block:: python

    from arch.bootstrap import IIDBootstrap
    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='studentized',
                     std_err_func=sharpe_ratio_se)

.. note::

    Standard error functions must return a 1-d array with the same number
    of element as params.

.. note::

    Standard error functions must match the patters
    ``std_err_func(params, *args, **kwargs)`` where ``params`` is an array
    of estimated parameters constructed using ``*args`` and ``**kwargs``.

Bias-corrected (``bc``, ``bias-corrected`` or ``debiased``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bias corrected bootstrap makes use of a bootstrap estimate of the bias to
improve confidence intervals.

.. code-block:: python

    from arch.bootstrap import IIDBootstrap
    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='bc')

The bias-corrected confidence interval is identical to the bias-corrected and
accelerated where :math:`a=0`.

Bias-corrected and accelerated (``bca``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bias-corrected and accelerated confidence intervals make use of both
a bootstrap bias estimate and a jackknife acceleration term.  BCa intervals may
offer higher-order accuracy if some conditions are satisfied. Bias-corrected
confidence intervals are a special case of BCa intervals where the acceleration
parameter is set to 0.

.. code-block:: python

    from arch.bootstrap import IIDBootstrap

    bs = IIDBootstrap(returns)
    ci = bs.conf_int(sharpe_ratio, 1000, method='bca')

The confidence interval is based on the empirical  distribution of the bootstrap
parameter estimates, :math:`\hat{\theta}_b^\star`, where the percentiles used
are

.. math::

    \Phi\left(
    \Phi^{-1}\left(\hat{b}\right)+\frac{\Phi^{-1}\left(\hat{b}\right)
    +z_{\alpha}}{1-\hat{a}\left(\Phi^{-1}\left(\hat{b}\right)+z_{\alpha}\right)}
    \right)



where :math:`z_{\alpha}` is the usual quantile from the normal distribution and
:math:`b` is the empirical bias estimate,

.. math::

    \hat{b}=\#\left\{ \hat{\theta}_{b}^{\star}<\hat{\theta}\right\} / B

:math:`a` is a skewness-like estimator using a leave-one-out jackknife.
