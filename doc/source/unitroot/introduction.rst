Introduction
------------

All tests expect a 1-d series as the first input.  The input can be any array that
can `squeeze` into a 1-d array, a pandas `Series` or a pandas `DataFrame` that
contains a single variable.

All tests share a common structure.  The key elements are:

- `stat` - Returns the test statistic
- `pvalue` - Returns the p-value of the test statistic
- `lags` - Sets or gets the number of lags used in the model.  In most test, can be ``None`` to trigger automatic selection.
- `trend` - Sets or gets the trend used in the model.  Supported trends vary by model, but include:

   - `'nc'`: No constant
   - `'c'`: Constant
   - `'ct'`: Constant and time trend
   - `'ctt'`: Constant, time trend and quadratic time trend

- `summary()` - Returns a summary object that can be printed to get a formatted table


Basic Example
=============

This basic example show the use of the Augmented-Dickey fuller to test whether the default premium,
defined as the difference between the yields of large portfolios of BAA and AAA bonds.  This example
uses a constant and time trend.


.. code-block:: python

    import arch.data.default
    from arch.unitroot import ADF

    default_data = arch.data.default.load()
    default = default_data['BAA'] - default_data['AAA']
    default = default.loc['1919-01-01':'2014-01-01']
    adf = ADF(default, trend='ct')

    print(adf.summary())

which yields

::

       Augmented Dickey-Fuller Results   
    =====================================
    Test Statistic                 -3.448
    P-value                         0.045
    Lags                               21
    -------------------------------------

    Trend: Constant and Linear Time Trend
    Critical Values: -3.97 (1%), -3.41 (5%), -3.13 (10%)
    Null Hypothesis: The process contains a unit root.
    Alternative Hypothesis: The process is weakly stationary.


Lag Length Selection
====================

:class:`~arch.unitroot.ADF`, :class:`~arch.unitroot.DFGLS` and
:class:`~arch.unitroot.ZivotAndrews` include lagged differences of the series to
account for serial correlation in the errors. When ``lags`` is ``None``, the number
of lags is selected from 0, 1, ..., ``max_lags`` lags using the ``method``. The
choices, which are not case sensitive, are:

- ``"aic"``: minimize the Akaike information criterion (the default);
- ``"bic"``: minimize the Schwarz/Bayesian information criterion;
- ``"t-stat"``: use the longest lag with a t-statistic that is significant, an
  absolute value of at least 1.645 (``"t"`` and ``"tstat"`` are accepted as aliases); and
- ``"maic"``: minimize the modified AIC of Ng and Perron (2001).

Ng and Perron showed that the information criteria tend to select too few lags in
the presence of a large negative moving average root in the errors, which produces
tests that reject a true unit root too often. Their modified AIC adds a term to the
penalty that depends on the estimated coefficient on the lagged level of the series,
:math:`\hat{\beta}_0`, which is larger in absolute value when the data look stationary,

.. math::

   MAIC(k) = \ln(\hat{\sigma}_k^2) + \frac{2(\tau_T(k) + k)}{T - k_{max}},
   \qquad
   \tau_T(k) = \hat{\sigma}_k^{-2}\hat{\beta}_0^2
   \sum_{t=k_{max}+1}^{T}\tilde{y}_{t-1}^2.

:math:`\hat{\sigma}_k^2` is the estimated variance of the errors using :math:`k`
lags, and :math:`\tilde{y}_{t-1}` is the lagged level after the deterministic terms
in the regression have been removed. All lag lengths use the same
:math:`T - k_{max}` observations. The modified AIC can select a different number of
lags than the AIC and BIC.

.. ipython:: python

    import numpy as np

    import arch.data.crude
    from arch.unitroot import ADF, DFGLS

    brent = np.log(arch.data.crude.load()["Brent"])
    for method in ("aic", "bic", "maic"):
        adf = ADF(brent, trend="c", method=method)
        print(f"{method:>4}: lags={adf.lags:2d}  stat={adf.stat:0.3f}  p={adf.pvalue:0.3f}")
    dfgls = DFGLS(brent, method="maic")
    print(f"DFGLS maic: lags={dfgls.lags}  stat={dfgls.stat:0.3f}")

Ng, S., & Perron, P. (2001). Lag length selection and the construction of unit
root tests with good size and power. *Econometrica*, 69(6), 1519-1554.
