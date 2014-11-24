Unit Root Testing
-----------------

Many time series are highly persistent, and determining whether the data appear
to be stationary or contains a unit root is the first step in many analyses.
This module contains a number of routines:

  * Augmented Dickey-Fuller (:py:class:`~arch.unitroot.ADF`)
  * Dickey-Fuller GLS (:py:class:`~arch.unitroot.DFGLS`)
  * Phillips-Perron (:py:class:`~arch.unitroot.PhillipsPerron`)
  * Variance Ratio (:py:class:`~arch.unitroot.VarianceRatio`)
  * KPSS (:py:class:`~arch.unitroot.KPSS`)


The first four all start with the null of a unit root and have an alternative
of a stationary process. The final test, KPSS, has a null of a stationary
process with an alternative of a unit root.

.. toctree::
    :maxdepth: 2

    Examples <unitroot_examples>
    Unit Root Tests <tests>


Introduction
============

All tests expect a 1-d series as the first input.  The input can be any array that can be `squeeze`d to 1-d, a pandas
`Series` or a pandas `DataFrame` that contains a single variable.

All tests share a common structure.  The key elements are:

  * `stat` - Returns the test statistic
  * `pvalue` - Returns the p-value of the test statistic
  * `lags` - Sets or gets the number of lags used in the model. In most test, can be `None` to trigger automatic selection.
  * `trend` - Sets of gets the trend used in the model.  Supported trends vary by model, but include:
     - `'nc'`: No constant
     - `'c'`: Constant
     - `'ct'`: Constant and time trend
     - `'ctt'`: Constant, time trend and quadratic time trend
  * `summary()` - Returns a summary object that can be printed to get a formatted table

Basic Example
=============

This basic example show the use of the Augmented-Dickey fuller to test whether the default premium, defined as the
difference between the yields of large portfolios of BAA and AAA bonds.  This example uses a constant and time trend.


::

    import pandas.io.data as web
    import datetime as dt
    aaa = web.DataReader("AAA", "fred", dt.datetime(1919,1,1), dt.datetime(2014,1,1))
    baa = web.DataReader("BAA", "fred", dt.datetime(1919,1,1), dt.datetime(2014,1,1))
    baa.columns = aaa.columns = ['default']
    default = baa - aaa
    from arch.unitroot import ADF
    adf = ADF(default)
    adf.trend = 'ct'
    print('Statistic: ' + str(adf.stat))
    print('P-value: ' + str(adf.pvalue))
    print('Summary \n')
    print(adf.summary())

