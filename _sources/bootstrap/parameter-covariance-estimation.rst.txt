Covariance Estimation
=====================
The bootstrap can be used to estimate parameter covariances in applications where
analytical computation is challenging, or simply as an alternative to
traditional estimators.

This example estimates the covariance of the mean, standard deviation and
Sharpe ratio of the S&P 500 using monthly returns computed from the daily
prices distributed with ``arch``.

.. code-block:: python

    import numpy as np
    import pandas as pd

    import arch.data.sp500

    sp500 = arch.data.sp500.load()
    low = sp500.index.min()
    high = sp500.index.max()
    monthly_dates = pd.date_range(low, high, freq='ME')
    monthly = sp500.reindex(monthly_dates, method='ffill')
    returns = 100 * monthly['Adj Close'].pct_change().dropna()

The function that returns the parameters.

.. code-block:: python

    def sharpe_ratio(r):
        mu = 12 * r.mean(axis=0)
        sigma = np.sqrt(12 * r.var(axis=0))
        sr = mu / sigma
        return np.array([mu, sigma, sr])

Like all applications of the bootstrap, it is important to choose a bootstrap
that captures the dependence in the data.  This example uses the stationary
bootstrap with an average block size of 12.

.. code-block:: python

    import pandas as pd
    from arch.bootstrap import StationaryBootstrap

    bs = StationaryBootstrap(12, returns)
    param_cov = bs.cov(sharpe_ratio)
    index = ['mu', 'sigma', 'SR']
    params = sharpe_ratio(returns)
    params = pd.Series(params, index=index)
    param_cov = pd.DataFrame(param_cov, index=index, columns=index)

The output is

.. code-block:: python

    >>> params
    mu        4.439391
    sigma    14.468318
    SR        0.306835
    dtype: float64

    >>> param_cov
                  mu     sigma        SR
    mu     12.521785 -3.272164  0.941381
    sigma  -3.272164  2.137770 -0.280233
    SR      0.941381 -0.280233  0.072791

.. note::

    The covariance estimator is centered using the average of the bootstrapped
    estimators. The original sample estimator can be used to center using the
    keyword argument ``recenter=False``.
