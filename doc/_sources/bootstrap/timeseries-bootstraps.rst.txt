.. module:: arch.bootstrap
   :noindex:
.. currentmodule:: arch.bootstrap

Time-series Bootstraps
----------------------
Bootstraps for time-series data come in a variety of forms.  The three contained
in this package are the stationary bootstrap
(:class:`~arch.bootstrap.StationaryBootstrap`), which uses blocks with an
exponentially distributed lengths, the circular block bootstrap
(:class:`~arch.bootstrap.CircularBlockBootstrap`), which uses
fixed length blocks, and the moving block bootstrap which also uses fixed
length blocks (:class:`~arch.bootstrap.MovingBlockBootstrap`).  The moving
block bootstrap does *not* wrap around and so observations near the start or
end of the series will be systematically under-sampled.  It is not recommended
for this reason.

.. autosummary::
   :toctree: generated/

   StationaryBootstrap
   CircularBlockBootstrap
   MovingBlockBootstrap
   optimal_block_length