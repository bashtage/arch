.. _mean-models:

Mean Models
===========
All ARCH models start by specifying a mean model.

.. module:: arch.univariate
   :noindex:

.. currentmodule:: arch.univariate

.. autosummary::
   :toctree: generated/

   ZeroMean
   ConstantMean
   ARX
   HARX
   LS

Writing New Mean Models
-----------------------
.. currentmodule:: arch.univariate.base

All mean models must inherit from :class:ARCHModel and provide all public
methods. There are two optional private methods that should be provided if
applicable.

.. autosummary::
   :toctree: generated/

   ARCHModel
