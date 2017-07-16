.. py:currentmodule:: arch.bootstrap

Time-series Bootstraps
----------------------
Bootstraps for time-series data come in a variety of forms.  The three contained
in this package are the stationary bootstrap
(:py:class:`~arch.bootstrap.StationaryBootstrap`), which uses blocks with an
exponentially distributed lengths, the circular block bootstrap
(:py:class:`~arch.bootstrap.CircularBlockBootstrap`), which uses
fixed length blocks, and the moving block bootstrap which also uses fixed
length blocks (:py:class:`~arch.bootstrap.MovingBlockBootstrap`).  The moving
block bootstrap does *not* wrap around and so observations near the start or
end of the series will be systematically under-sampled.  It is not recommended
for this reason.

The Stationary Bootstrap
========================

.. autoclass:: StationaryBootstrap
    :members: conf_int, var, cov, apply, bootstrap, reset, seed, set_state, get_state

The Circular Block Bootstrap
============================

.. autoclass:: CircularBlockBootstrap
    :members: conf_int, var, cov, apply, bootstrap, reset, seed, set_state, get_state

The Moving Block Bootstrap
==========================

.. autoclass:: MovingBlockBootstrap
    :members: conf_int, var, cov, apply, bootstrap, reset, seed, set_state, get_state
