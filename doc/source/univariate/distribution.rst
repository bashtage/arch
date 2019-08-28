.. _distributions:

Distributions
=============
A distribution is the final component of an ARCH Model.

.. module:: arch.univariate.distribution
.. py:currentmodule:: arch.univariate

Normal
------

.. autoclass:: Normal
   :members: starting_values, bounds, constraints, simulate, loglikelihood

Student's t
-----------

.. autoclass:: StudentsT
   :members: starting_values, bounds, constraints, simulate, loglikelihood

Skew Student's t
----------------

.. autoclass:: SkewStudent
   :members: starting_values, bounds, constraints, simulate, loglikelihood

Generalized Error (GED)
-----------------------

.. autoclass:: GeneralizedError
   :members: starting_values, bounds, constraints, simulate, loglikelihood

Writing New Distributions
-------------------------
All distributions must inherit from :class:Distribution and provide all public
methods.

.. autoclass:: Distribution

