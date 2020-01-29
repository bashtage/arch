Bootstrapping
-------------

.. module:: arch.bootstrap
   :synopsis: Bootstrap methods for simulation and parameter inference
.. currentmodule::arch.bootstrap

The bootstrap module provides both high- and low-level interfaces for
bootstrapping data contained in NumPy arrays or pandas Series or DataFrames.

All bootstraps have the same interfaces and only differ in their name, setup
parameters and the (internally generated) sampling scheme.

.. toctree::
    :maxdepth: 1

    Examples <bootstrap_examples.ipynb>
    Confidence Interval Construction <confidence-intervals>
    Parameter Covariance Estimation <parameter-covariance-estimation>
    Low-level Interface <low-level-interface>
    Semiparametric and Parametric Bootstraps <semiparametric-parametric-bootstrap>
    Bootstraps for IID data <iid-bootstraps>
    Bootstraps for Time-series Data <timeseries-bootstraps>
    Background and References <background>


