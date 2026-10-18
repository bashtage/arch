.. _mean-models:

Mean Models
===========
All ARCH models start by specifying a mean model.

.. module:: arch.univariate
   :noindex:

.. currentmodule:: arch.univariate

.. autosummary::
   :toctree: generated/

   ZeroMean
   ConstantMean
   ARX
   HARX
   LS
   MAMean

Moving Average Models
---------------------

:class:`~arch.univariate.MAMean` is a moving average model of order :math:`q`
with an optional constant, which can be used with any volatility process and
distribution. The model is

.. math::
   :nowrap:

   \begin{eqnarray}
      y_t        & = & \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \ldots + \theta_q \epsilon_{t-q} \
      \epsilon_t & = & \sigma_t e_t
   \end{eqnarray}

where :math:`\sigma^2_t` is the conditional variance from the volatility process
and :math:`e_t` has the distribution of the model. The order is set using ``lags``,
and the constant :math:`\mu` is omitted when ``constant=False``. The parameters
are ordered :math:`[\mu, \theta_1, \ldots, \theta_q]`.

The innovations :math:`\epsilon_t` are not observed. They are computed recursively
from the data, with the innovations before the first observation set to 0, so that
the parameters are estimated by maximizing the conditional likelihood. The effect
of the initial values vanishes when the model is invertible. Invertibility is
imposed when :math:`q` is 1 or 2, but not for larger values of :math:`q`, and the
starting values are always invertible. Exogenous regressors are not supported,
and :class:`~arch.univariate.ARX` or :class:`~arch.univariate.HARX` should be used
when they are needed.

.. ipython:: python

    import arch.data.sp500
    from arch.univariate import GARCH, MAMean, StudentsT

    sp500 = arch.data.sp500.load()
    returns = 100 * sp500["Adj Close"].pct_change().dropna()
    ma = MAMean(returns, lags=2, volatility=GARCH(1, 0, 1), distribution=StudentsT())
    res = ma.fit(disp="off")
    res.params

The forecast of the mean depends on the most recent innovations for the first
:math:`q` steps, and is the constant :math:`\mu` after that.

.. ipython:: python

    res.forecast(horizon=4, reindex=False).mean

``MAMean`` is a class only. It is not available through
:func:`~arch.univariate.arch_model`, so that ``arch_model(y, mean="ma")`` raises a
``ValueError``.

Adding Observations to a Model
------------------------------

Once a model has been estimated, observations that arrive later can be added using
:meth:`~arch.univariate.base.ARCHModel.append`, which is available for all mean
models. It modifies the model in-place and does not change its parameters, so that
forecasts and results that are computed using parameters estimated using earlier
data include the new observations without estimating the model again. See
:ref:`forecasting-append` for details and an example.

(G)ARCH-in-mean Models
----------------------

(G)ARCH-in-mean models allow the conditional variance (or a transformation of it)
to enter the conditional mean.

.. autosummary::
   :toctree: generated/

   ARCHInMean

Special Requirements
~~~~~~~~~~~~~~~~~~~~
A volatility process can only be used in AIM modeling if it supports updating.
Specifically, the property ``updateable`` must be ``True``. All of the volatility
processes in the package support updating. A process that is written by the
user must provide a volatility updater, see :ref:`volatility-processes`.

.. ipython::

   In [1]: from arch.univariate import GARCH, EGARCH

   In [2]: GARCH().updateable

   In [3]: EGARCH().updateable

Writing New Mean Models
-----------------------
.. currentmodule:: arch.univariate.base

All mean models must inherit from :class:`~arch.univariate.base.ARCHModel` and provide all public
methods. There are two optional private methods that should be provided if
applicable.

.. autosummary::
   :toctree: generated/

   ARCHModel
