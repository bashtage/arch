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
      \sigma^2_t & = & \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma^2_{t-1}
   \end{eqnarray}

A complete ARCH model is divided into three components:

* a :ref:`mean model<mean-models>`, e.g., a constant mean or an :class:`~arch.univariate.mean.ARX`;
* a :ref:`volatility process<volatility-processes>`, e.g., a :class:`~arch.univariate.volatility.GARCH` or an :class:`~arch.univariate.volatility.EGARCH` process; and
* a :ref:`distribution<distributions>` for the standardized residuals.

In most applications, the simplest method to construct this model is to use the constructor
function :meth:`~arch.univariate.arch_model`

.. code-block:: python

    import datetime as dt

    import pandas_datareader.data as web

    from arch import arch_model

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2014, 1, 1)
    sp500 = web.DataReader('^GSPC', 'yahoo', start=start, end=end)
    returns = 100 * sp500['Adj Close'].pct_change().dropna()
    am = arch_model(returns)

Alternatively, the same model can be manually assembled from the building
blocks of an ARCH model

.. code-block:: python

    from arch import ConstantMean, GARCH, Normal

    am = ConstantMean(returns)
    am.volatility = GARCH(1, 0, 1)
    am.distribution = Normal()

In either case, model parameters are estimated using

.. code-block:: python

    res = am.fit()


with the following output

::

    Iteration:      1,   Func. Count:      6,   Neg. LLF: 5159.58323938
    Iteration:      2,   Func. Count:     16,   Neg. LLF: 5156.09760149
    Iteration:      3,   Func. Count:     24,   Neg. LLF: 5152.29989336
    Iteration:      4,   Func. Count:     31,   Neg. LLF: 5146.47531817
    Iteration:      5,   Func. Count:     38,   Neg. LLF: 5143.86337547
    Iteration:      6,   Func. Count:     45,   Neg. LLF: 5143.02096168
    Iteration:      7,   Func. Count:     52,   Neg. LLF: 5142.24105141
    Iteration:      8,   Func. Count:     60,   Neg. LLF: 5142.07138907
    Iteration:      9,   Func. Count:     67,   Neg. LLF: 5141.416653
    Iteration:     10,   Func. Count:     73,   Neg. LLF: 5141.39212288
    Iteration:     11,   Func. Count:     79,   Neg. LLF: 5141.39023885
    Iteration:     12,   Func. Count:     85,   Neg. LLF: 5141.39023359
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 5141.39023359
                Iterations: 12
                Function evaluations: 85
                Gradient evaluations: 12

.. code-block:: python

    print(res.summary())

yields

::

                         Constant Mean - GARCH Model Results
    ==============================================================================
    Dep. Variable:              Adj Close   R-squared:                      -0.001
    Mean Model:             Constant Mean   Adj. R-squared:                 -0.001
    Vol Model:                      GARCH   Log-Likelihood:               -5141.39
    Distribution:                  Normal   AIC:                           10290.8
    Method:            Maximum Likelihood   BIC:                           10315.4
                                            No. Observations:                 3520
    Date:                Fri, Dec 02 2016   Df Residuals:                     3516
    Time:                        22:22:28   Df Model:                            4
                                      Mean Model
    ==============================================================================
                     coef    std err          t      P>|t|        95.0% Conf. Int.
    ------------------------------------------------------------------------------
    mu             0.0531  1.487e-02      3.569  3.581e-04   [2.392e-02,8.220e-02]
                                   Volatility Model
    ==============================================================================
                     coef    std err          t      P>|t|        95.0% Conf. Int.
    ------------------------------------------------------------------------------
    omega          0.0156  4.932e-03      3.155  1.606e-03   [5.892e-03,2.523e-02]
    alpha[1]       0.0879  1.140e-02      7.710  1.260e-14     [6.554e-02,  0.110]
    beta[1]        0.9014  1.183e-02     76.163      0.000       [  0.878,  0.925]
    ==============================================================================

    Covariance estimator: robust


Model Constructor
=================
While models can be carefully specified using the individual components, most common specifications can be specified
using a simple model constructor.

.. currentmodule:: arch.univariate
.. autofunction:: arch_model
