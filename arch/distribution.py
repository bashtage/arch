"""
Temporary shim until functions finally moved.
Remove after March 1, 2015.
"""
from __future__ import absolute_import

import arch.univariate.distribution as dist
from .utils import deprecation_doc

class Normal(dist.Normal):
    def __new__(cls, *args, **kwargs):
        import warnings
        warnings.warn(deprecation_doc.format(func='Normal',
                                             new_location='arch.univariate'),
                      FutureWarning)
        return dist.Normal(*args, **kwargs)

class StudentsT(dist.StudentsT):
    def __new__(cls, *args, **kwargs):
        import warnings
        warnings.warn(deprecation_doc.format(func='StudentsT',
                                             new_location='arch.univariate'),
                      FutureWarning)
        return dist.StudentsT(*args, **kwargs)

