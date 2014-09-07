from __future__ import absolute_import

import arch.univariate.mean as mean
from .utils import deprecation_doc


def arch_model(*args, **kwargs):
    import warnings

    warnings.warn(deprecation_doc.format(func='arch_mean',
                                         new_location='arch.univariate'),
                  FutureWarning)

    return mean.arch_model(*args, **kwargs)


arch_model.__doc__ = mean.arch_model.__doc__


class HARX(mean.HARX):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='HARX',
                                             new_location='arch'),
                      FutureWarning)
        return mean.HARX(*args, **kwargs)


class ARX(mean.ARX):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='ARX',
                                             new_location='arch'),
                      FutureWarning)
        return mean.ARX(*args, **kwargs)


class ConstantMean(mean.ConstantMean):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='ConstantMean',
                                             new_location='arch'),
                      FutureWarning)
        return mean.ConstantMean(*args, **kwargs)


class ZeroMean(mean.ZeroMean):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='ZeroMean',
                                             new_location='arch'),
                      FutureWarning)
        return mean.ZeroMean(*args, **kwargs)


class LS(mean.LS):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='LS',
                                             new_location='arch'),
                      FutureWarning)
        return mean.LS(*args, **kwargs)
