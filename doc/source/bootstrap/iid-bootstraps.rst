.. py:currentmodule:: arch.bootstrap

Independent, Identical Distributed Data (i.i.d.)
------------------------------------------------
:py:class:`~arch.bootstrap.IIDBootstrap` is the standard bootstrap that is appropriate for data that is
either i.i.d. or at least not serially dependant.

.. autoclass:: IIDBootstrap
   :members: conf_int, var, cov, apply, bootstrap, reset, seed, set_state, get_state

Independent Samples
-------------------
:py:class:`~arch.bootstrap.IndependentSamplesBootstrap` is a bootstrap that is appropriate for
data is totally independent, and where each variable may have a different sample size. This
type of data arises naturally in experimental settings, e.g., website A/B testing.

.. autoclass:: IndependentSamplesBootstrap
   :members: index, conf_int, var, cov, apply, bootstrap, reset, seed, set_state, get_state
