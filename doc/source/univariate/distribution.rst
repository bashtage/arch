Distributions
=============
A distribution is the final component of an ARCH Model.

.. py:currentmodule:: arch.univariate

Normal
------

.. autoclass:: Normal
    :members: starting_values, bounds, constraints, simulate, loglikelihoood

Student's t
-----------

.. autoclass:: StudentsT
    :members: starting_values, bounds, constraints, simulate, loglikelihoood

Skew Student's t
----------------

.. autoclass:: SkewStudent
    :members: starting_values, bounds, constraints, simulate, loglikelihoood

Writing New Distributions
-------------------------
All distributions must inherit from :class:Distribution and provide all public
methods.

.. autoclass:: Distribution

