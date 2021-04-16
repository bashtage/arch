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

(G)ARCH-in-mean Models
----------------------

(G)ARCH-in-mean models allow the conditional variance (or a transformation of it)
to enter the conditional mean.

.. autosummary::
   :toctree: generated/

   ARCHInMean

Special Requirements
~~~~~~~~~~~~~~~~~~~~
Not all volatility processes support application to AIM modeling.
Specifically, the property ``updateable`` must be ``True``.

.. ipython::

   In [1]: from arch.univariate import GARCH, EGARCH

   In [2]: GARCH().updateable

   In [3]: EGARCH().updateable

Writing New Mean Models
-----------------------
.. currentmodule:: arch.univariate.base

All mean models must inherit from :class:ARCHModel and provide all public
methods. There are two optional private methods that should be provided if
applicable.

.. autosummary::
   :toctree: generated/

   ARCHModel
