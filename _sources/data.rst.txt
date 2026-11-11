Data Sets
=========

The package includes the data sets that are used in the examples and in the
documentation, so that they can be run without a network connection. Each data
set is in its own module, ``arch.data.<name>``, and the function ``load`` returns
the data as a :class:`pandas.DataFrame`. When the data have a date, it is used
as the index.

.. ipython:: python

    import arch.data.sp500

    sp500 = arch.data.sp500.load()
    sp500.loc["2018-12-24":, ["Close", "Adj Close"]]

The available data sets are

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Module
     - Contents
     - Columns
   * - ``arch.data.sp500``
     - Daily S&P 500 index, January 1999 to December 2018.
     - ``Open``, ``High``, ``Low``, ``Close``, ``Adj Close``, ``Volume``
   * - ``arch.data.nasdaq``
     - Daily NASDAQ Composite index, January 1999 to December 2018.
     - ``Open``, ``High``, ``Low``, ``Close``, ``Adj Close``, ``Volume``
   * - ``arch.data.vix``
     - Daily VIX index, January 2014 to January 2019.
     - ``vix``
   * - ``arch.data.wti``
     - Daily price of West Texas Intermediate crude oil from FRED, January 1986 to
       January 2019.
     - ``DCOILWTICO``
   * - ``arch.data.crude``
     - Monthly prices of Brent and West Texas Intermediate crude oil, May 1987
       to January 2020.
     - ``Brent``, ``WTI``
   * - ``arch.data.core_cpi``
     - Monthly consumer price index for all urban consumers less food and energy
       from FRED, January 1957 to November 2018.
     - ``CPILFESL``
   * - ``arch.data.default``
     - Monthly yields on AAA and BAA rated corporate bonds, January 1919 to
       December 2018.
     - ``AAA``, ``BAA``
   * - ``arch.data.frenchdata``
     - Monthly excess return on the market, the size (SMB) and value (HML)
       factors and the risk-free rate from Ken French's data library,
       July 1926 to November 2018.
     - ``Mkt-RF``, ``SMB``, ``HML``, ``RF``
   * - ``arch.data.binary``
     - Graduate school admissions data (400 applicants) with the GRE score,
       GPA and class rank.
     - ``admit``, ``gre``, ``gpa``, ``rank``

Loading Data
------------

.. autosummary::
   :toctree: generated/

   arch.data.sp500.load
   arch.data.nasdaq.load
   arch.data.vix.load
   arch.data.wti.load
   arch.data.crude.load
   arch.data.core_cpi.load
   arch.data.default.load
   arch.data.frenchdata.load
   arch.data.binary.load

All of the loaders read a compressed csv file that is included with the package
using the function below. It can also be used to read other files that are stored
in a directory next to a module.

.. autosummary::
   :toctree: generated/

   arch.data.utility.load_file
