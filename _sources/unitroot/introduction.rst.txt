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

    import datetime as dt

    import pandas_datareader.data as web
    from arch.unitroot import ADF

    start = dt.datetime(1919, 1, 1)
    end = dt.datetime(2014, 1, 1)

    df = web.DataReader(["AAA", "BAA"], "fred", start, end)
    df['diff'] = df['BAA'] - df['AAA']
    adf = ADF(df['diff'])
    adf.trend = 'ct'

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

