Unit Root Testing
-----------------

.. py:module::arch.unitroot

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
    :maxdepth: 1

    Introduction <introduction>
    Examples <unitroot_examples>
    Unit Root Tests <tests>


