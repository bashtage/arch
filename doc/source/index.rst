ARCH
----
The ARCH toolbox currently contains routines for univariate volatility models
and bootstrapping.  Current plans are to continue to expand this toolbox to
include additional routines to analyze financial data.

Contents
========

.. toctree::
    :maxdepth: 1

    Univariate Volatility Models <univariate/univariate>
    Bootstrapping <bootstrap/bootstrap>
    Change Log <changes>

Introduction to ARCH Models
===========================
ARCH models are a popular class of volatility models that use observed values
of returns or residuals as volatility shocks.  A basic GARCH model is specified
as

.. math::
   :nowrap:

   \begin{eqnarray}
      r_t    & = & \mu + \epsilon_t \\
      \epsilon_t & = & \sigma_t e_t \\
      \sigma^2_t & = & \omega + \alpha \epsilon_t^2 + \beta \sigma^2_{t-1}
   \end{eqnarray}

An complete ARCH model is divided into three components:


..
.. Theoretical Background <background>
..

However, the simplest method to construct this model is to use the constructor
function :py:meth:`~arch.arch_model`

::

    from arch import arch_model
    import datetime as dt
    import pandas.io.data as web
    start = dt.datetime(2000,1,1)
    end = dt.datetime(2014,1,1)
    sp500 = web.get_data_yahoo('^GSPC', start=start, end=end)
    returns = 100 * sp500['Adj Close'].pct_change().dropna()
    am = arch_model(returns)

Alternatively, the same model can be manually assembled from the building
blocks of an ARCH model

::

    from arch import ConstantMean, GARCH, Normal
    am = ConstantMean(returns)
    am.volatility = GARCH(1,0,1)
    am.distribution = Normal()

In either case, model parameters are estimated using

::

    res = am.fit()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

