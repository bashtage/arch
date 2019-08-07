class InvalidLengthWarning(Warning):
    pass


invalid_length_doc = """
The length of {var} is not an exact multiple of {block}, and so the final
{drop} observations have been dropped.
"""

deprecation_doc = """
{old_func} has been deprecated.  Please use {new_func}.
"""


class ConvergenceWarning(Warning):
    pass


convergence_warning = """\
The optimizer returned code {code}. The message is:
{string_message}
See scipy.optimize.fmin_slsqp for code meaning.
"""


class StartingValueWarning(Warning):
    pass


starting_value_warning = """\
Starting values do not satisfy the parameter constraints in the model.  The
provided starting values will be ignored.
"""


class InitialValueWarning(Warning):
    pass


initial_value_warning = """\
Parameters are not consistent with a stationary model. Using the intercept
to initialize the model.
"""


class DataScaleWarning(Warning):
    pass


data_scale_warning = """\
y is poorly scaled, which may affect convergence of the optimizer when
estimating the model parameters. The scale of y is {0:0.4g}. Parameter
estimation work better when this value is between 1 and 1000. The recommended
rescaling is {1:0.4g} * y.

This warning can be disabled by either rescaling y before initializing the
model or by setting rescale=False.
"""
