Cointegration Tests
===================

.. module:: arch.unitroot.cointegration
   :synopsis: Cointegration tests and estimation

.. currentmodule:: arch.unitroot.cointegration

When the trend includes a constant, the cointegrating regressions of
:func:`engle_granger` and :func:`phillips_ouliaris`, the first stage of
:class:`FullyModifiedOLS` and :class:`CanonicalCointegratingReg`, and the regression
of :class:`DynamicOLS` are estimated with the dependent variable and the other
regressors centered on their means. The estimates, including the constant and the
cointegrating vector, are reported for the original variables and do not depend on
the centering. See :ref:`centered-regressors`.

.. autosummary::
   :toctree: generated/

   engle_granger
   phillips_ouliaris

Cointegrating Vector Estimation
===============================
.. autosummary::
   :toctree: generated/

   DynamicOLS
   FullyModifiedOLS
   CanonicalCointegratingReg

Results Classes
---------------

.. autosummary::
   :toctree: generated/

   CointegrationAnalysisResults
   DynamicOLSResults
   EngleGrangerTestResults
   PhillipsOuliarisTestResults
