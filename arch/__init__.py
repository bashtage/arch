from .univariate.mean import arch_model
from ._version import __version__


def doc():
    import webbrowser
    webbrowser.open('http://arch.readthedocs.org/en/latest/')


__all__ = ['arch_model', '__version__', 'doc']
