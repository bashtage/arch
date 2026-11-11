Warnings and Exceptions
=======================

The warnings that the package issues are subclasses of ``Warning`` that can be
used to filter them using the :mod:`warnings` module. Most are defined in
``arch.utility.exceptions``.

.. code-block:: python

    import warnings

    from arch.utility.exceptions import DataScaleWarning

    # Do not warn about the scale of the data
    warnings.filterwarnings("ignore", category=DataScaleWarning)

The warning about convergence can also be turned off in a single call using
``show_warning=False`` in :meth:`~arch.univariate.base.ARCHModel.fit`. The warning
about the scale of the data is not issued when ``rescale`` is set when the model is
created.

.. py:currentmodule:: arch.utility.exceptions

Univariate Volatility Models
----------------------------

.. py:exception:: ConvergenceWarning

   Issued by :meth:`~arch.univariate.base.ARCHModel.fit` when the optimizer does not
   report that it has converged, for example because the maximum number of
   iterations was reached. The message contains the return code of the optimizer and
   its message. The estimates should not be used without checking them, and it
   may help to rescale the data, to provide different ``starting_values`` or to
   increase the number of iterations using ``options``.

.. py:exception:: DataScaleWarning

   Issued when a model is created using data with a variance that is outside of the
   range 0.1 to 10,000, which can make the optimization difficult, and ``rescale``
   is not set. The message gives the recommended scale. Set ``rescale=True`` to
   rescale the data automatically, which is reported in the results, or
   ``rescale=False`` to estimate the model using the data as given.

.. py:exception:: StartingValueWarning

   Issued by :meth:`~arch.univariate.base.ARCHModel.fit` when the
   ``starting_values`` are not valid for the model, for example because they violate
   the bounds or the constraints. They are ignored, and the default starting values
   are used.

.. py:exception:: InitialValueWarning

   Issued when simulating data, for example using the ``simulate`` method of a
   model, with volatility parameters that are not consistent with a stationary
   process, so that the unconditional variance is not defined. The initial value of
   the variance is set to the intercept of the volatility process.

.. py:exception:: ValueWarning

   Issued when simulating data using :class:`~arch.univariate.FIGARCH` or
   :class:`~arch.univariate.FIAPARCH` with a value of ``beta`` that is 1 or larger,
   so that the long-run variance is not defined. The intercept is used in its place.
   Subclass of ``UserWarning``.

Unit Root Tests and Cointegration
---------------------------------

.. py:exception:: InvalidLengthWarning

   Issued by :class:`~arch.unitroot.VarianceRatio` when ``overlap`` is ``False``
   and the number of observations does not allow the data to be divided into
   an exact number of blocks of ``lags``. The final observations are dropped, and
   the message says how many.

.. py:currentmodule:: arch.unitroot.cointegration

.. py:exception:: CriticalValueWarning

   Issued by :func:`~arch.unitroot.cointegration.phillips_ouliaris` when the
   sample size is smaller than the smallest sample size that was used to
   construct the tables of critical values, which are approximations. Interpret the
   results with caution. Subclass of ``RuntimeWarning``.

.. py:currentmodule:: arch.utility.exceptions

.. py:exception:: InfeasibleTestException

   Raised by the unit root tests when the regression of the test cannot be
   estimated, for example because the maximum lag length is too large for
   the number of observations or the series is constant. Reduce ``max_lags`` or
   provide ``lags``. Subclass of ``RuntimeError``.

Bootstrapping
-------------

.. py:exception:: StudentizationError

   Raised by ``conf_int`` with ``method="studentized"`` when the standard error
   that is estimated by the nested bootstrap is numerically 0 in a replication, for
   example because the statistic has no variability when the data are resampled.
   Subclass of ``RuntimeError``.

Other Warnings
--------------

.. py:exception:: PerformanceWarning

   Issued when a function that is compiled using Numba is run without compilation
   because Numba is not installed, which is much slower. The compiled extensions
   that are included in the binary releases of the package are used when they are
   available, and in that case the warning is not issued. Subclass of
   ``UserWarning``.

.. py:currentmodule:: arch.utility.timeseries

.. py:exception:: ColumnNameConflict

   Issued by ``add_trend`` when a DataFrame already has a column with the name of a
   column that is added. The name of the new column is changed, and the message
   lists the old and new names.
