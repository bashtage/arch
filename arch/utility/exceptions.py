class InvalidLengthWarning(Warning):
    pass


invalid_length_doc: str = """
The length of {var} is not an exact multiple of {block}, and so the final
{drop} observations have been dropped.
"""

deprecation_doc: str = """
{old_func} has been deprecated.  Please use {new_func}.
"""


class ConvergenceWarning(Warning):
    pass


convergence_warning: str = """\
The optimizer returned code {code}. The message is:
{string_message}
See scipy.optimize.fmin_slsqp for code meaning.
"""


class StartingValueWarning(Warning):
    pass


starting_value_warning: str = """\
Starting values do not satisfy the parameter constraints in the model.  The
provided starting values will be ignored.
"""


class InitialValueWarning(Warning):
    pass


initial_value_warning: str = """\
Parameters are not consistent with a stationary model. Using the intercept
to initialize the model.
"""


class DataScaleWarning(Warning):
    pass


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
    pass


studentization_error: str = """
The estimated covariance computed in the studentication is numerically 0.
This might occur if your statistic has no variation. It is not possible to
apply the studentized bootstrap if any of the variances the values returned
by func have not variability when resampled. The estimated covariance
is:\n\n {cov}
"""


class InfeasibleTestException(RuntimeError):
    pass


class PerformanceWarning(UserWarning):
    """Warning issued if recursions are run in CPython"""


class ValueWarning(UserWarning):
    """Warning issued if value is problematic but no fatal."""
