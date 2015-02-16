.. py:currentmodule:: arch.bootstrap

Multiple Comparison Procedures
------------------------------

Bootstrap-based multiple comparison procedures.

Test of Superior Predictive Ability (SPA), Reality Check
========================================================
**Note**: Also callable using :py:class:`~arch.bootstrap.RealityCheck`

.. autoclass:: SPA
    :members: compute, pvalues, critical_values

Stepwise Multiple Testing (StepM)
=================================

.. autoclass:: StepM
    :members: compute, superior_models

