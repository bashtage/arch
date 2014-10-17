from __future__ import absolute_import
import arch.univariate.volatility as vol
from .utils import deprecation_doc

class ARCH(vol.ARCH):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='ARCH',
                                             new_location='arch.univariate'),
                      FutureWarning)
        return vol.ARCH(*args, **kwargs)


class EWMAVariance(vol.EWMAVariance):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='EWMAVariance',
                                             new_location='arch.univariate'),
                      FutureWarning)
        return vol.EWMAVariance(*args, **kwargs)


class RiskMetrics2006(vol.RiskMetrics2006):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='RiskMetrics2006',
                                             new_location='arch.univariate'),
                      FutureWarning)
        return vol.RiskMetrics2006(*args, **kwargs)


class GARCH(vol.GARCH):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='GARCH',
                                             new_location='arch.univariate'),
                      FutureWarning)
        return vol.GARCH(*args, **kwargs)

class EGARCH(vol.EGARCH):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='EGARCH',
                                             new_location='arch.univariate'),
                      FutureWarning)
        return vol.EGARCH(*args, **kwargs)

class HARCH(vol.HARCH):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='HARCH',
                                             new_location='arch.univariate'),
                      FutureWarning)
        return vol.HARCH(*args, **kwargs)

class ConstantVariance(vol.ConstantVariance):
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(deprecation_doc.format(func='ConstantVariance',
                                             new_location='arch.univariate'),
                      FutureWarning)
        return vol.ConstantVariance(*args, **kwargs)

