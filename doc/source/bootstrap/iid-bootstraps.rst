.. module:: arch.bootstrap
   :noindex:
.. currentmodule:: arch.bootstrap

Independent, Identical Distributed Data (i.i.d.)
------------------------------------------------
:class:`~arch.bootstrap.IIDBootstrap` is the standard bootstrap that is appropriate for data that is
either i.i.d. or at least not serially dependant.

.. autosummary::
   :toctree: generated/

   IIDBootstrap

Independent Samples
-------------------
:class:`~arch.bootstrap.IndependentSamplesBootstrap` is a bootstrap that is appropriate for
data is totally independent, and where each variable may have a different sample size. This
type of data arises naturally in experimental settings, e.g., website A/B testing.

.. autosummary::
   :toctree: generated/

   ~arch.bootstrap.IndependentSamplesBootstrap
