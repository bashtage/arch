"""
Script to run performance tests to show speed.
"""

import sys
from arch.univariate.tests.test_recursions import TestRecursions

if __name__ == '__main__':
    try:
        import numba
    except ImportError:
        sys.exit(0)


    t = TestRecursions()
    t.setup_class()
    t.test_garch_performance()
    t.test_harch_performance()
    t.test_egarch_performance()
