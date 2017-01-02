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


convergence_warning = """
The optimizer returned code {code}. The message is:
{string_message}
See scipy.optimize.fmin_slsqp for code meaning.
"""


class StartingValueWarning(Warning):
    pass


starting_value_warning = """
Starting values do not satisfy the parameter constraints in the model.  The
provided starting values will be ignored.
"""
