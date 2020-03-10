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
* Automatic Bandwidth Selection (:func:`arch.unitroot.auto_bandwidth`)

The first four all start with the null of a unit root and have an alternative
of a stationary process. The final test, KPSS, has a null of a stationary
process with an alternative of a unit root.

Cointegration Testing
---------------------
The module extended the single-series unit root testing to multiple
series and cointegration testing.

* Engle-Granger Cointegration Test (:class:`~arch.unitroot.engle_granger`)


.. toctree::
    :maxdepth: 1

    Introduction <introduction>
    Examples <unitroot_examples.ipynb>
    Unit Root Tests <tests>
    Cointegration Testing <unitroot_cointegration_examples.ipynb>
    Cointegration <cointegration>


