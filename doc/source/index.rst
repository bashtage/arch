ARCH Models
===========
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
function :py:meth:`~arch.mean.arch`

::

    from arch.mean import arch
    import pandas.io.data as web
    sp500 = web.get_data_yahoo('^GSPC', start=start, end=end)
    returns = 100 * sp500['Adj Close'].pct_change().dropna()
    am = arch(returns)

Alternatively, the same model can be manually assembled from the building
blocks of an ARCH model

::

    from arch.mean import ConstantMean
    from arch.volatiltiy import GARCH
    from arch.distribution import Normal
    am = ConstantMean(returns)
    am.volatility = GARCH(1,0,1)
    am.distribution = Normal()

In either case, model parameters are estimated using

::

    res = am.fit()


.. toctree::
    :maxdepth: 1

    Examples <examples>
    Mean Models <mean>
    Volatility Processes <volatility>
    Distributions <distribution>


Model Constructor
-----------------
While models can be carefully specified using the individual components, most common specifications can be specified
using a simple model constructor.

.. py:currentmodule:: arch
.. autofunction:: arch_model

Model Results
-------------
All model return the same object, a results class (:py:class:`ARCHModelResult`)

.. py:currentmodule:: arch.base
.. autoclass:: ARCHModelResult
    :members: summary, plot, conf_int

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

