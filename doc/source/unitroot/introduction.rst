Introduction
------------

All tests expect a 1-d series as the first input.  The input can be any array that can `squeeze` into a 1-d array, a pandas
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


