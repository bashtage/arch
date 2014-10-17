Univariate Volatility Models
----------------------------

.. toctree::
    :maxdepth: 1

    Examples <univariate_volatility_modeling>
    Mean Models <mean>
    Volatility Processes <volatility>
    Distributions <distribution>
    Background and References <background>


Core Model Constructor
======================
While models can be carefully specified using the individual components, most common specifications can be specified
using a simple model constructor.

.. py:currentmodule:: arch
.. autofunction:: arch_model

Model Results
=============
All model return the same object, a results class (:py:class:`ARCHModelResult`)

.. py:currentmodule:: arch.univariate.base
.. autoclass:: ARCHModelResult
    :members: summary, plot, conf_int
