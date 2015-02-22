Module Reference
----------------

.. py:currentmodule:: arch.bootstrap

Test of Superior Predictive Ability (SPA), Reality Check
========================================================
The test of Superior Predictive Ability (Hansen 2005), or SPA, is an improved
version of the Reality Check (White 2000).  It tests whether the best
forecasting performance from a set of models is better than that of the
forecasts from a benchmark model.  A model is "better" if its losses are
smaller than those from the benchmark.  Formally, it tests the null

.. math::

    H_0: \max_i E[L_i] \geq E[L_{bm}]

where :math:`L_i` is the loss from model *i* and :math:`L_{bm}` is the loss
from the benchmark model.  The alternative is

.. math::

    H_1: \max_i E[L_i] > E[L_{bm}]

This procedure accounts for dependence between the losses and the fact that
there are potentially alternative models being considered.

**Note**: Also callable using :py:class:`~arch.bootstrap.RealityCheck`

.. autoclass:: SPA
    :members: compute, pvalues, critical_values

Stepwise Multiple Testing (StepM)
=================================

The Stepwise Multiple Testing procedure (Romano & Wolf (2005)) is closely
related to the SPA, except that it returns a set of models that are superior
to the benchmark model, rather than the p-value from the null.  They are
so closely related that :py:class:`~arch.bootstrap.StepM` is essentially
a wrapper around :py:class:`~arch.bootstrap.SPA` with some small modifications
to allow multiple calls.

.. autoclass:: StepM
    :members: compute, superior_models

Model Confidence Set (MCS)
==========================

The Model Confidence Set (Hansen, Lunde & Nason (2011)) differs from other
multiple comparison procedures in that there is no benchmark.  The MCS attempts
to identify the set of models which produce the same expected loss, while
controlling the probability that a model that is worse than the best model is
in the model confidence set.  Like the other MCPs, it controls the
Familywise Error Rate rather than the usual test size.

.. autoclass:: MCS
    :members: compute
