The Unit Root Tests
===================

.. py:currentmodule:: arch.unitroot


Augmented-Dickey Fuller Testing
-------------------------------

.. autoclass:: ADF
    :members: stat, pvalue, critical_values, summary, regression, trend, lags, null_hypothesis, alternative_hypothesis, valid_trends


Dickey-Fuller GLS Testing
-------------------------

.. autoclass:: DFGLS
    :members: summary


Phillips-Perron Testing
-----------------------

.. autoclass:: PhillipsPerron
    :members: summary


Variance Ratios
---------------

.. autoclass:: VarianceRatio
    :members: summary


KPSS Testing
------------

.. autoclass:: KPSS
    :members: summary
