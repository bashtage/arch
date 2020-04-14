Unit Root Testing
-----------------

.. module::arch.unitroot

Many time series are highly persistent, and determining whether the data appear
to be stationary or contains a unit root is the first step in many analyses.
This module contains a number of routines:

* Augmented Dickey-Fuller (:class:`~arch.unitroot.ADF`)
* Dickey-Fuller GLS (:class:`~arch.unitroot.DFGLS`)
* Phillips-Perron (:class:`~arch.unitroot.PhillipsPerron`)
* KPSS (:class:`~arch.unitroot.KPSS`)
* Zivot-Andrews (:class:`~arch.unitroot.ZivotAndrews`)
* Variance Ratio (:class:`~arch.unitroot.VarianceRatio`)
* Automatic Bandwidth Selection (:func:`~arch.unitroot.auto_bandwidth`)

The first four all start with the null of a unit root and have an alternative
of a stationary process. The final test, KPSS, has a null of a stationary
process with an alternative of a unit root.

.. toctree::
    :maxdepth: 1

    Introduction <introduction>
    Unit Root Testing Examples <unitroot_examples.ipynb>
    Unit Root Testing <tests>

Cointegration Analysis
----------------------
The module extended the single-series unit root testing to multiple
series and cointegration testing and cointegrating vector estimation.

* Cointegrating Testing

  * Engle-Granger Test (:class:`~arch.unitroot.cointegration.engle_granger`)
  * Phillips-Ouliaris Tests (:class:`~arch.unitroot.cointegration.phillips_ouliaris`)

* Cointegrating Vector Estimation

  * Dynamic OLS (:class:`~arch.unitroot.cointegration.DynamicOLS`)
  * Fully Modified OLS (:class:`~arch.unitroot.cointegration.FullyModifiedOLS`)
  * Canonical Cointegrating Regression (:class:`~arch.unitroot.cointegration.CanonicalCointegratingReg`)


.. toctree::
    :maxdepth: 1

    Cointegration Testing Examples <unitroot_cointegration_examples.ipynb>
    Cointegration Testing and Estimation <cointegration>


