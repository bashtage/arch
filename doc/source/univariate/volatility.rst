.. _volatility-processes:

Volatility Processes
====================
A volatility process is added to a mean model to capture time-varying
volatility.

.. module:: arch.univariate
   :noindex:

.. currentmodule:: arch.univariate

.. autosummary::
   :toctree: generated/

   ConstantVariance
   GARCH
   FIGARCH
   EGARCH
   HARCH
   MIDASHyperbolic
   ARCH
   APARCH
   FIAPARCH
   NGARCH

Fractionally Integrated Asymmetric Power ARCH
---------------------------------------------

:class:`~arch.univariate.FIAPARCH` is a fractionally integrated asymmetric power
ARCH process. It extends :class:`~arch.univariate.FIGARCH`, which has long memory in
the volatility, with the asymmetry and the power of :class:`~arch.univariate.APARCH`.
The conditional volatility evolves according to

.. math::

   \sigma_t^{\delta} = \omega
   + \left[1-\beta L - (1-\phi L)(1-L)^d\right]
   (|\epsilon_t| - \gamma \epsilon_t)^{\delta}
   + \beta \sigma_{t-1}^{\delta}

where :math:`L` is the lag operator and :math:`d` is the fractional differencing
parameter. The shocks enter through :math:`|\epsilon_t| - \gamma \epsilon_t`, so that
:math:`\gamma>0` produces a larger response to negative shocks than to positive
shocks of the same size, and the power :math:`\delta` determines the scale on
which the volatility is modeled. As in the FIGARCH, the process is estimated using
its ARCH(:math:`\infty`) representation

.. math::

   \sigma_t^{\delta} = (1-\beta)^{-1}\omega + \sum_{i=1}^{\infty} \lambda_i
   (|\epsilon_{t-i}| - \gamma \epsilon_{t-i})^{\delta}

with weights :math:`\lambda_i` that are the same as in the FIGARCH, truncated after
``truncation`` terms (1000 by default).

The orders ``p``, ``o`` and ``q`` can be 0 or 1. They select, respectively, the
parameters :math:`\phi`, :math:`\gamma` and :math:`\beta`, and the parameters are
ordered ``omega``, ``phi``, ``d``, ``beta``, ``gamma`` and ``delta``, omitting the
ones that are not in the model. The power :math:`\delta` is estimated, with values
between 0.05 and 4, unless ``delta`` is provided, which fixes it to a value in the
same range. Fixing :math:`\delta` makes estimation considerably faster. The process
is named ``"FI Power ARCH"`` when ``o=0``, since it then has no asymmetry.

.. ipython:: python

    import arch.data.nasdaq
    from arch.univariate import ConstantMean, FIAPARCH

    nasdaq = arch.data.nasdaq.load()
    returns = 100 * nasdaq["Adj Close"].pct_change().dropna()
    am = ConstantMean(returns, volatility=FIAPARCH(delta=1.0, truncation=250))
    res = am.fit(disp="off")
    res.params

The process can also be selected using ``vol="fiaparch"`` in
:func:`~arch.univariate.arch_model`, which estimates :math:`\delta` and uses the
default truncation. Analytic forecasts are only available for a horizon of 1.
Longer horizons require ``method="simulation"`` or ``method="bootstrap"``.

.. ipython:: python

    import numpy as np

    forecasts = res.forecast(
        horizon=3,
        method="simulation",
        simulations=500,
        random_state=np.random.RandomState(0),
    )
    forecasts.variance

Nonlinear Asymmetric GARCH
--------------------------

:class:`~arch.univariate.NGARCH` is the nonlinear asymmetric GARCH of Engle and Ng
[EngleNg1993]_, which the rugarch package for R calls NAGARCH. The shock is shifted by
a multiple of the lagged volatility before it is squared,

.. math::

   \sigma_t^{2} = \omega
   + \sum_{i=1}^{p}\alpha_i\left(\epsilon_{t-i} - \theta_i I_{[o\geq i]}
   \sigma_{t-i}\right)^{2}
   + \sum_{j=1}^{q}\beta_j \sigma_{t-j}^{2}.

In terms of the standardized shock :math:`e_t=\epsilon_t/\sigma_t` the news impact is
:math:`\sigma_{t-1}^2(e_{t-1}-\theta)^2`, a parabola with a minimum at
:math:`e_{t-1}=\theta`. With :math:`\theta>0`, the leverage effect, a negative shock
increases the variance by more than a positive shock of the same size. Engle and Ng
write the shift as :math:`(\epsilon_{t-1}+\gamma\sigma_{t-1})^2` and expect
:math:`\gamma<0`, so that :math:`\theta=-\gamma`. The sign convention of
:class:`~arch.univariate.NGARCH` is that of rugarch. The parameters are named
``omega``, ``alpha[i]``, ``theta[i]`` and ``beta[j]``, and the orders must satisfy
:math:`o\leq p`, as for :class:`~arch.univariate.APARCH`. Unlike the GJR-GARCH, the
expected value of the shock term does not depend on the distribution of :math:`e_t`
beyond its mean and variance, since :math:`E[(e_t-\theta)^2]=1+\theta^2` for any
standardized distribution.

.. note::

   Another model, the nonlinear ARCH of Higgins and Bera [HigginsBera1992]_, is also
   called NGARCH. It is not available in arch.

The process is covariance stationary when the persistence
:math:`\phi=\sum_i\alpha_i(1+\theta_i^2)+\sum_j\beta_j` is smaller than 1. Estimation
imposes the linear constraints :math:`\sum_i\alpha_i+\sum_j\beta_j<1` and
non-negativity, as for :class:`~arch.univariate.APARCH`, and not
:math:`\phi<1`, which is not linear in the parameters. The persistence of the
estimated model should be checked.

.. ipython:: python

    import arch.data.sp500
    from arch.univariate import ConstantMean, NGARCH

    sp500 = arch.data.sp500.load()
    returns = 100 * sp500["Adj Close"].pct_change().dropna()
    am = ConstantMean(returns, volatility=NGARCH(p=1, o=1, q=1))
    res = am.fit(disp="off")
    res.params
    params = res.params
    params["alpha[1]"] * (1 + params["theta[1]"] ** 2) + params["beta[1]"]

The process can also be selected using ``vol="ngarch"`` in
:func:`~arch.univariate.arch_model`, where ``o=1`` adds the shift. Analytic forecasts
are exact at all horizons. For ``p=q=1`` they satisfy
:math:`f_h=\omega+\phi f_{h-1}`.

.. ipython:: python

    res.forecast(horizon=5).variance.iloc[-1]

Parameterless Variance Processes
--------------------------------
Some volatility processes use fixed parameters and so have no parameters that
are estimable.

.. autosummary::
   :toctree: generated/

   EWMAVariance
   RiskMetrics2006

FixedVariance
-------------
The ``FixedVariance`` class is a special-purpose volatility process that allows
the so-called zig-zag algorithm to be used.  See the example for usage.

.. autosummary::
   :toctree: generated/

   FixedVariance

Writing New Volatility Processes
--------------------------------
All volatility processes must inherit from :class:`~arch.univariate.volatility.VolatilityProcess` and provide
all public methods.

.. currentmodule:: arch.univariate.volatility

.. autosummary::
   :toctree: generated/

   VolatilityProcess

They may optionally expose a
:class:`~arch.univariate.recursions.VolatilityUpdater` class
that can be used in :class:`~arch.univariate.ARCHInMean` estimation.

.. currentmodule:: arch.univariate.recursions

.. autosummary::
   :toctree: generated/

   VolatilityUpdater
