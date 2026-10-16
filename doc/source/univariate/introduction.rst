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

    import arch.data.sp500

    from arch import arch_model

    sp500 = arch.data.sp500.load()
    sp500 = sp500.loc['2000-01-01':'2014-01-01']
    returns = 100 * sp500['Adj Close'].pct_change().dropna()
    am = arch_model(returns)

Alternatively, the same model can be manually assembled from the building
blocks of an ARCH model

.. code-block:: python

    from arch.univariate import ConstantMean, GARCH, Normal

    am = ConstantMean(returns)
    am.volatility = GARCH(1, 0, 1)
    am.distribution = Normal()

In either case, model parameters are estimated using

.. code-block:: python

    res = am.fit()


with the following output

::

    Iteration:      1,   Func. Count:      6,   Neg. LLF: 68805886954919.59
    Iteration:      2,   Func. Count:     15,   Neg. LLF: 1575359856.4496076
    Iteration:      3,   Func. Count:     22,   Neg. LLF: 5194.449409971215
    Iteration:      4,   Func. Count:     28,   Neg. LLF: 5190.7422748202325
    Iteration:      5,   Func. Count:     34,   Neg. LLF: 5191.570045895966
    Iteration:      6,   Func. Count:     40,   Neg. LLF: 5150.829260399659
    Iteration:      7,   Func. Count:     46,   Neg. LLF: 6474.487216207122
    Iteration:      8,   Func. Count:     53,   Neg. LLF: 5143.161859778254
    Iteration:      9,   Func. Count:     59,   Neg. LLF: 5141.392123039943
    Iteration:     10,   Func. Count:     64,   Neg. LLF: 5141.390238848524
    Iteration:     11,   Func. Count:     69,   Neg. LLF: 5141.390233587619
    Iteration:     12,   Func. Count:     73,   Neg. LLF: 5141.390233588856
    Optimization terminated successfully    (Exit mode 0)
                Current function value: 5141.390233587619
                Iterations: 12
                Function evaluations: 73
                Gradient evaluations: 12

.. code-block:: python

    print(res.summary())

yields

::

                         Constant Mean - GARCH Model Results
    ==============================================================================
    Dep. Variable:              Adj Close   R-squared:                       0.000
    Mean Model:             Constant Mean   Adj. R-squared:                  0.000
    Vol Model:                      GARCH   Log-Likelihood:               -5141.39
    Distribution:                  Normal   AIC:                           10290.8
    Method:            Maximum Likelihood   BIC:                           10315.4
                                            No. Observations:                 3520
    Date:                Fri, Oct 16 2026   Df Residuals:                     3519
    Time:                        11:57:57   Df Model:                            1
                                     Mean Model
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0531  1.487e-02      3.569  3.581e-04 [2.392e-02,8.220e-02]
                                  Volatility Model
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    omega          0.0156  4.932e-03      3.155  1.606e-03 [5.892e-03,2.523e-02]
    alpha[1]       0.0879  1.140e-02      7.710  1.260e-14   [6.554e-02,  0.110]
    beta[1]        0.9014  1.183e-02     76.163      0.000     [  0.878,  0.925]
    ============================================================================

    Covariance estimator: robust


Model Constructor
=================
While models can be carefully specified using the individual components, most common specifications can be specified
using a simple model constructor.

.. currentmodule:: arch.univariate
.. autofunction:: arch_model
