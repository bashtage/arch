class InvalidLengthWarning(Warning):
    """Warning issued if observations are dropped to form complete blocks."""


invalid_length_doc: str = """
The length of {var} is not an exact multiple of {block}, and so the final
{drop} observations have been dropped.
"""


class ConvergenceWarning(Warning):
    """Warning issued if the optimizer does not report that it has converged."""


convergence_warning: str = """\
The optimizer returned code {code}. The message is:
{string_message}
See scipy.optimize.fmin_slsqp for code meaning.
"""


class StartingValueWarning(Warning):
    """Warning issued if the starting values are not valid for the model."""


starting_value_warning: str = """\
Starting values do not satisfy the parameter constraints in the model.  The
provided starting values will be ignored.
"""


class InitialValueWarning(Warning):
    """Warning issued if the parameters are not consistent with stationarity."""


initial_value_warning: str = """\
Parameters are not consistent with a stationary model. Using the intercept
to initialize the model.
"""


class DataScaleWarning(Warning):
    """Warning issued if the data are poorly scaled."""


data_scale_warning: str = """\
y is poorly scaled, which may affect convergence of the optimizer when
estimating the model parameters. The scale of y is {0:0.4g}. Parameter
estimation work better when this value is between 1 and 1000. The recommended
rescaling is {1:0.4g} * y.

This warning can be disabled by either rescaling y before initializing the
model or by setting rescale=False.
"""


arg_type_error: str = """\
Only NumPy arrays and pandas DataFrames and Series are supported in positional
arguments. Positional input {i} has type {arg_type}.
"""

kwarg_type_error: str = """\
Only NumPy arrays and pandas DataFrames and Series are supported in keyword
arguments. Input `{key}` has type {arg_type}.
"""


class StudentizationError(RuntimeError):
    """Error raised if the covariance used to studentize is numerically 0."""


studentization_error: str = """
The estimated covariance computed in the studentication is numerically 0.
This might occur if your statistic has no variation. It is not possible to
apply the studentized bootstrap if any of the variances the values returned
by func have not variability when resampled. The estimated covariance
is:\n\n {cov}
"""


class InfeasibleTestException(RuntimeError, ValueError):
    """
    Error raised if a test cannot be computed using the data provided.

    This is a ``ValueError``, since the data or the specification are not valid for
    the test, and also a ``RuntimeError``, which it was before ``ValueError`` was
    added, so that code that catches either keeps working.
    """


class PerformanceWarning(UserWarning):
    """Warning issued if recursions are run in CPython."""


class ValueWarning(UserWarning):
    """Warning issued if value is problematic but no fatal."""
