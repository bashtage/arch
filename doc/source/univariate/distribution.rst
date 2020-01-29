.. _distributions:

Distributions
=============
A distribution is the final component of an ARCH Model.

.. module:: arch.univariate
   :noindex:
.. currentmodule:: arch.univariate

.. autosummary::
   :toctree: generated/

   Normal
   StudentsT
   SkewStudent
   GeneralizedError

Writing New Distributions
-------------------------
All distributions must inherit from :class:Distribution and provide all public
methods.

.. currentmodule:: arch.univariate.distribution

.. autosummary::
   :toctree: generated/

   Distribution

