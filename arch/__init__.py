import os

from numpy.testing import Tester

from .univariate.mean import arch_model
from ._version import __version__

test = Tester().test

def doc():
    import webbrowser
    webbrowser.open('http://arch.readthedocs.org/en/latest/')
