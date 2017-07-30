from arch.bootstrap.multiple_comparison import StepM, SPA, RealityCheck
import warnings

# Remove after October 2017
warnings.warn('Misspelled module deprecated.  Use '
              'arch.bootstrap.multiple_comparison instead', DeprecationWarning,
              stacklevel=2)

__all__ = ['StepM', 'SPA', 'RealityCheck']
