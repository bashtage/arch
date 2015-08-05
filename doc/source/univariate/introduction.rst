Introduction to ARCH Models
---------------------------
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

A complete ARCH model is divided into three components:


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



Core Model Constructor
======================
While models can be carefully specified using the individual components, most common specifications can be specified
using a simple model constructor.

.. py:currentmodule:: arch
.. autofunction:: arch_model

Model Results
=============
All model return the same object, a results class (:py:class:`ARCHModelResult`)

.. py:currentmodule:: arch.univariate.base
.. autoclass:: ARCHModelResult
    :members: summary, forecast, conf_int, plot, hedgehog_plot

When using the ``fix`` method, a (:py:class:`ARCHModelFixedResult`) is produced
that lacks some properties of a (:py:class:`ARCHModelResult`) that are not
relevant when parameters are not estimated.

.. autoclass:: ARCHModelFixedResult
    :members: summary, forecast, plot, hedgehog_plot
