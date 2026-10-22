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

These bootstraps are designed for stationary data.  The local block bootstrap
(:class:`~arch.bootstrap.LocalBlockBootstrap`) is for series whose properties
change slowly, for example a series with a trend or with a variance that moves
over time.  It uses fixed length blocks like the moving block bootstrap, but
replaces a block only with a block that starts close to it, within a window of
``locality`` observations of the start of the block in the data
[PaparoditisPolitis]_.  A ``locality`` of 0 keeps each block in place, and a
``locality`` that is at least the number of observations gives the moving block
bootstrap.

.. autosummary::
   :toctree: generated/

   StationaryBootstrap
   CircularBlockBootstrap
   MovingBlockBootstrap
   LocalBlockBootstrap
   optimal_block_length

Local Block Bootstrap
~~~~~~~~~~~~~~~~~~~~~

The block size and the locality are both chosen by the user.  They should satisfy
``1 << block_size << locality << n``: the blocks are long enough to capture the
dependence, the window contains many blocks and the window is a small part of
the sample.  :func:`~arch.bootstrap.optimal_block_length` assumes stationary data
and does not give values for the block size or the locality.  The locality is a
number of observations, and not a fraction of the sample, so that
:meth:`~arch.bootstrap.LocalBlockBootstrap.clone` and the studentized interval
can use it with data of other lengths.

.. code-block:: python

    import numpy as np
    from arch.bootstrap import LocalBlockBootstrap

    rng = np.random.default_rng(0)
    y = 3 * np.arange(600) / 600 + rng.standard_normal(600)
    bs = LocalBlockBootstrap(10, 40, y, seed=rng)
    ci = bs.conf_int(np.mean, 1000, method="percentile")

The mean of this series has a standard deviation of :math:`1/\sqrt{600} = 0.041`.
The local block bootstrap estimates it as about 0.04, and the circular block bootstrap,
which treats the trend as if it was noise, as about 0.11.

Some features need care when the data are not stationary:

* The BCa interval (``method="bca"``) uses the leave-one-out jackknife to estimate
  the acceleration. This ignores the dependence and the changes in the data, so
  it is a heuristic.
* The studentized interval (``method="studentized"``) estimates the standard
  error of each resampled statistic using a local block bootstrap of the resampled
  data with the same block size and locality.  Use ``std_err_func`` to provide
  another standard error.
* The local block bootstrap is not an option of the multiple-comparison
  procedures :class:`~arch.bootstrap.SPA`, :class:`~arch.bootstrap.StepM` and
  :class:`~arch.bootstrap.MCS`, which test hypotheses about stationary losses.
