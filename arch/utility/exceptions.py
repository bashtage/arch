class InvalidLengthWarning(Warning):
    pass

invalid_length_doc = """
The length of {var} is not an exact multiple of {block}, and so the final
{drop} observations have been dropped.
"""

deprecation_doc = """
{old_func} has been deprecated.  Please use {new_func}.
"""
