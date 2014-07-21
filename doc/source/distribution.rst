Distributions
=============
A distribution is the final component of an ARCH Model.

.. py:currentmodule::arch.distribution

.. automodule:: arch.distribution

Normal
------

.. autoclass:: Normal
    :members: starting_values, bounds, constraints, simulate, loglikelihoood
    :show-inheritance:

Student's t
-----------

.. autoclass:: StudentsT
    :members: starting_values, bounds, constraints, simulate, loglikelihoood
    :show-inheritance:

Writing New Distributions
-------------------------
All distributions must inherit from :class:Distribution and provide all public
methods.

.. autoclass:: Distribution

