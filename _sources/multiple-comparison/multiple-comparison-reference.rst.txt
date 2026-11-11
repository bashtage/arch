Module Reference
----------------

.. module:: arch.bootstrap.multiple_comparison
   :synopsis: Multiple comparison procedures

.. currentmodule:: arch.bootstrap

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

    H_1: \min_i E[L_i] < E[L_{bm}]

This procedure accounts for dependence between the losses and the fact that
there are potentially alternative models being considered.

**Note**: Also callable using ``arch.bootstrap.RealityCheck``


.. autosummary::
   :toctree: generated/

   SPA

.. _spa-studentization:

Studentization
~~~~~~~~~~~~~~

The loss differentials, :math:`d_{i,t} = L_{bm,t} - L_{i,t}`, are studentized by
default (``studentize=True``). The statistic of model *i* is the average loss
differential divided by an estimate of its standard error, so that the statistics
of models with different amounts of variation in their losses are comparable. The
standard error is estimated in the original data and again inside each bootstrap
replication, where it is used to studentize the re-centered average of the bootstrap
sample. As a result the bootstrap distribution reflects the sampling variation of
the estimated standard errors that is present in the statistic from the data.

.. versionchanged:: 9.0

   Previously ``studentize=True`` had no effect, and the statistics were always the
   average loss differentials. Studentization is now applied, and so p-values,
   critical values and the superior models obtained using the default arguments
   differ from previous versions. The critical values apply to the studentized
   statistics. Use ``studentize=False`` to obtain the previous statistics.

The standard errors are the square root of a kernel estimator of the long-run
variance of the loss differentials, divided by the number of observations. The
kernel is chosen to match the bootstrap, so that the estimator is the variance of
the sample mean that the bootstrap implies, with the average block size as the
bandwidth. The weights on the autocovariances are those of the stationary bootstrap
for the stationary bootstrap, and those of the Bartlett kernel for the circular and
moving block bootstraps. Setting ``nested=True`` replaces the kernel estimator with a nested
bootstrap, which estimates the variance of the average in the original data and in
every bootstrap replication. This uses a number of resamples that is the square of
``reps``, and so can be very slow. ``nested=True`` requires ``studentize=True`` and
raises a ``ValueError`` otherwise.

The same variances are used by the consistent p-value to select the models that are
re-centered, even when ``studentize=False``.

.. ipython:: python

    import numpy as np
    import pandas as pd

    from arch.bootstrap import SPA, StepM

    rng = np.random.default_rng(4)
    t = 250
    # The third and fourth models are more accurate than the benchmark
    scale = np.array([1.0, 1.0, 0.85, 0.8, 1.05])
    benchmark = pd.Series(rng.standard_normal(t) ** 2, name="benchmark")
    models = pd.DataFrame(
        (rng.standard_normal((t, 5)) * scale) ** 2,
        columns=[f"model_{i}" for i in range(5)],
    )

    spa = SPA(benchmark, models, seed=0)
    spa.compute()
    spa.pvalues
    spa.critical_values(0.05)

The critical values are for the studentized statistics. Without studentization they
are for the average loss differentials, and are much smaller.

.. ipython:: python

    spa_raw = SPA(benchmark, models, studentize=False, seed=0)
    spa_raw.compute()
    spa_raw.pvalues
    spa_raw.critical_values(0.05)

:class:`~arch.bootstrap.StepM` has the same options, and uses studentized loss
differentials by default. Here, studentization identifies both models that are better
than the benchmark, and not only the one with the largest improvement.

.. ipython:: python

    stepm = StepM(benchmark, models, size=0.1, seed=0)
    stepm.compute()
    stepm.superior_models
    stepm_raw = StepM(benchmark, models, size=0.1, studentize=False, seed=0)
    stepm_raw.compute()
    stepm_raw.superior_models

The bootstrap that is used is available from the read-only property ``bootstrap`` of
:class:`~arch.bootstrap.SPA` and :class:`~arch.bootstrap.StepM`. A new instance must be
created to use a different bootstrap.


Stepwise Multiple Testing (StepM)
=================================

The Stepwise Multiple Testing procedure (Romano & Wolf (2005)) is closely
related to the SPA, except that it returns a set of models that are superior
to the benchmark model, rather than the p-value from the null.  They are
so closely related that :class:`~arch.bootstrap.StepM` is essentially
a wrapper around :class:`~arch.bootstrap.SPA` with some small modifications
to allow multiple calls.

.. autosummary::
   :toctree: generated/

   StepM

Model Confidence Set (MCS)
==========================

The Model Confidence Set (Hansen, Lunde & Nason (2011)) differs from other
multiple comparison procedures in that there is no benchmark.  The MCS attempts
to identify the set of models which produce the same expected loss, while
controlling the probability that a model that is worse than the best model is
in the model confidence set.  Like the other MCPs, it controls the
Familywise Error Rate rather than the usual test size.

.. autosummary::
   :toctree: generated/

   MCS
