Forecasting
-----------

Multi-period forecasts can be easily produced for ARCH-type models using
forward recursion, with some caveats.  In particular, models that are
non-linear in the sense that they do not evolve using squares or residuals
do not normally have analytically tractable multi-period forecasts available.

All models support three methods of forecasting:

* Analytical: analytical forecasts are always available for the 1-step ahead
  forecast due to the structure of ARCH-type models.  Multi-step analytical
  forecasts are only available for model which are linear in the square of
  the residual, such as GARCH or HARCH.
* Simulation: simulation-based forecasts are always available for any
  horizon, although they are only useful for horizons larger than 1 since
  the first out-of-sample forecast from an ARCH-type model is always fixed.
  Simulation-based forecasts make use of the structure of an ARCH-type model
  to forward simulate using the assumed distribution of residuals, e.g., a
  Normal or Student's t.
* Bootstrap: bootstrap-based forecasts are similar to simulation based
  forecasts except that they make use of the standardized residuals
  from the actual data used in the estimation rather than assuming a
  specific distribution. Like simulation-base forecasts, bootstrap-based
  forecasts are only useful for horizons larger than 1. Additionally,
  the bootstrap forecasting method requires a minimal amount of in-sample
  data to use prior to producing the forecasts.

This document will use a standard GARCH(1,1) with a constant mean to explain
the choices available for forecasting.  The model can be described as

.. math::
   :nowrap:

   \begin{eqnarray}
      r_t        & =    & \mu + \epsilon_t \\
      \epsilon_t & =    & \sigma_t e_t \\
      \sigma^2_t & =    & \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma^2_{t-1} \\
      e_t        & \sim & N(0,1)
   \end{eqnarray}

In code this model can be constructed using data from the S&P 500 using

.. code-block:: python

    from arch import arch_model
    import datetime as dt
    import pandas_datareader.data as web
    start = dt.datetime(2000,1,1)
    end = dt.datetime(2014,1,1)
    sp500 = web.get_data_yahoo('^GSPC', start=start, end=end)
    returns = 100 * sp500['Adj Close'].pct_change().dropna()
    am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')

The model will be estimated using the first 10 years to estimate parameters
and then forecasts will be produced for the final 5.

.. code-block:: python

    split_date = dt.datetime(2010,1,1)
    res = am.fit(last_obs=split_date)


Analytical Forecasts
~~~~~~~~~~~~~~~~~~~~
Analytical forecasts are available for most models that evolve in terms of
the squares of the model residuals, e.g., GARCH, HARCH, etc. These forecasts
exploit the relationship :math:`E_t[\epsilon_{t+1}^2] = \sigma_{t+1}^2` to
recursively compute forecasts.

Variance forecasts are constructed for the conditional variances as

.. math::
   :nowrap:

   \begin{eqnarray}
      \sigma^2_{t+1} & = & \omega + \alpha \epsilon_t^2 + \beta \sigma^2_t \\
      \sigma^2_{t+h} & = & \omega + \alpha  E_{t}[\epsilon_{t+h-1}^2] + \beta E_{t}[\sigma^2_{t+h-1}] \, h \geq 2 \\
                     & = & \omega + \left(\alpha  + \beta\right) E_{t}[\sigma^2_{t+h-1}] \, h \geq 2
   \end{eqnarray}

.. code-block:: python

   forecasts = res.forecast(horizon=5, start=split_date)
   forecasts.variance[split_date:].plot()


Simulation Forecasts
~~~~~~~~~~~~~~~~~~~~
Simulation-based forecasts use the model random number generator to simulate
draws of the standardized residuals, :math:`e_{t+h}`.  These are used to
generate a pre-specified number of paths of the variances which are then
averaged to produce the forecasts.  In models like GARCH which evolve in the
squares of the residuals, there are few advantages to simulation-based
forecasting. These methods are more valuable when producing multi-step
forecasts from models that do not have closed form multi-step forecasts
such as EGARCH models.

Assume there are :math:`B` simulated paths.  A single simulated path is
generated using

.. math::
   :nowrap:

   \begin{eqnarray}
      \sigma^2_{t+h, b} & = & \omega + \alpha \epsilon_{t+h-1, b}^2 + \beta \sigma^2_{t+h-1, b} \\
      \epsilon_{t+h, b} & = & e_{t+h, b} \sqrt{\sigma^2_{t+h, b}}
   \end{eqnarray}

where the simulated shocks are
:math:`e_{t+1, b}, e_{t+2, b},\ldots, e_{t+h, b}` where :math:`b` is included
to indicate that the simulations are independent across paths. Note that the
first residual, :math:`\epsilon_{t}`, is in-sample and so is not simulated.

The final variance forecasts are then computed using the :math:`B` simulations

.. math::
   :nowrap:

   \begin{equation}
        E_t[\epsilon^2_{t+h}] = \sigma^2_{t+h} = B^{-1}\sum_{b=1}^B \sigma^2_{t+h,b}.
   \end{equation}

.. code-block:: python

   forecasts = res.forecast(horizon=5, start=split_date, method='simulation')


Bootstrap Forecasts
~~~~~~~~~~~~~~~~~~~
Bootstrap-based forecasts are virtually identical to simulation-based forecasts
except that the standardized residuals are generated by the model.  These
standardized residuals are generated using the observed data and the
estimated parameters as

.. math::
   :nowrap:

   \begin{equation}
        \hat{e}_t = \frac{r_t-\hat{\mu}}{\hat{\sigma}_t}
   \end{equation}

The generation scheme is identical to the simulation-based method except that
the simulated shocks are drawn (i.i.d., with replacement) from
:math:`\hat{e}_{1}, \hat{e}_{2},\ldots, \hat{e}_{t}`.  so that only
data available at time :math:`t` are used to simulate the paths.

Forecasting Options
~~~~~~~~~~~~~~~~~~~
The :meth:`~arch.univariate.base.ARCHModelResult.forecast`  method
is attached to a model fit result.`

* ``params`` - The model parameters used to forecast the mean and variance.
  If not specified, the parameters estimated during the call to ``fit``
  the produced the result are used.
* ``horizon`` - A positive integer value indicating the maximum horizon to
  produce forecasts.
* ``start`` - A positive integer or, if the input to the mode is a DataFrame,
  a date (string, datetime, datetime64 or Timestamp). Forecasts are produced
  from ``start`` until the end of the sample.  If not provided, ``start`` is
  set to the length of the input data minus 1 so that only 1 forecast is
  produced.
* ``align`` - One of 'origin' (default) or 'target' that describes how the
  forecasts aligned in the output. Origin aligns forecasts to the last
  observation used in producing the forecast, while target aligns forecasts
  to the observation index that is being forecast.
* ``method`` - One of 'analytic' (default), 'simulation' or 'bootstrap' that
  describes the method used to produce the forecasts.  Not all methods are
  available for all horizons.
* ``simulations`` - A non-negative integer indicating the number of
  simulation to use when ``method`` is 'simulation' or 'bootstrap'


Understanding Forecast Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Any call to :meth:`~arch.univariate.base.ARCHModelResult.forecast` returns a
:class:`~arch.univariate.base.ARCHModelForecast` object with has 3 core
attributes and 1 which may be useful when using simulation- or bootstrap-based
forecasts.

The three core attributes are

* ``mean`` - The forecast conditional mean.
* ``variance`` - The forecast conditional variance.
* ``residual_variance`` - The forecast conditional variance of residuals.
  This will differ from ``variance`` whenever the model has dynamics
  (e.g. an AR model) for horizons larger than 1.

Each attribute contains a ``DataFrame`` with a common structure.

.. code-block:: python

   print(forecasts.variance.tail())

which returns

::

                    h.1       h.2       h.3       h.4       h.5
   Date
   2013-12-24  0.489534  0.495875  0.501122  0.509194  0.518614
   2013-12-26  0.474691  0.480416  0.483664  0.491932  0.502419
   2013-12-27  0.447054  0.454875  0.462167  0.467515  0.475632
   2013-12-30  0.421528  0.430024  0.439856  0.448282  0.457368
   2013-12-31  0.407544  0.415616  0.422848  0.430246  0.439451

The values in the columns ``h.1`` are one-step ahead forecast, while values in
``h.2``, ..., ``h.5`` are 2, ..., 5-observation ahead forecasts.  The output
is aligned so that the Date column is the final data used to generate the
forecast, so that ``h.1`` in row ``2013-12-31`` is the one-step ahead forecast
made using data **up to and including** December 31, 2013.

By default forecasts are only produced for observations after the final
observation used to estimate the model.

.. code-block:: python

   day = dt.timedelta(1)
   print(forecasts.variance[split_date - 5 * day:split_date + 5 * day])

which produces

::

                   h.1       h.2       h.3       h.4       h.5
   Date
   2009-12-28       NaN       NaN       NaN       NaN       NaN
   2009-12-29       NaN       NaN       NaN       NaN       NaN
   2009-12-30       NaN       NaN       NaN       NaN       NaN
   2009-12-31       NaN       NaN       NaN       NaN       NaN
   2010-01-04  0.739303  0.741100  0.744529  0.746940  0.752688
   2010-01-05  0.695349  0.702488  0.706812  0.713342  0.721629
   2010-01-06  0.649343  0.654048  0.664055  0.672742  0.681263

The output will always have as many rows as the data input.  Values
that are not forecast are ``nan`` filled.

Output Classes
~~~~~~~~~~~~~~
.. currentmodule:: arch.univariate.base

.. autosummary::
   :toctree: generated/

   ARCHModelForecast
   ARCHModelForecastSimulation
