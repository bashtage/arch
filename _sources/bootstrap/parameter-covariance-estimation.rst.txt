Covariance Estimation
=====================
The bootstrap can be used to estimate parameter covariances in applications where
analytical computation is challenging, or simply as an alternative to
traditional estimators.

This example estimates the covariance of the mean, standard deviation and
Sharpe ratio of the S&P 500 using Yahoo! Finance data.

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

The function that returns the parameters.

.. code-block:: python

    def sharpe_ratio(r):
        mu = 12 * r.mean(0)
        sigma = np.sqrt(12 * r.var(0))
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
    mu        8.148534
    sigma    14.508540
    SR        0.561637
    dtype: float64

    >>> param_cov
                 mu     sigma        SR
    mu     3.729435 -0.442891  0.273945
    sigma -0.442891  0.495087 -0.049454
    SR     0.273945 -0.049454  0.020830

.. note::

    The covariance estimator is centered using the average of the bootstrapped
    estimators. The original sample estimator can be used to center using the
    keyword argument ``recenter=False``.
