from .univariate.mean import arch_model
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions


def doc():
    import webbrowser
    webbrowser.open('http://arch.readthedocs.org/en/latest/')


__all__ = ['arch_model', '__version__', 'doc']
