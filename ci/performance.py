"""
Script to run performance tests to show speed.
"""

import sys

from arch.tests.univariate.test_recursions import TestRecursions

if __name__ == "__main__":
    try:
        import numba  # noqa: F401
    except ImportError:
        print("numba not available -- skipping performance tests")
        sys.exit(0)

    t = TestRecursions()
    t.setup_class()

    t.test_garch_performance()
    t.test_harch_performance()
    t.test_egarch_performance()
    t.test_midas_performance()
    t.test_figarch_performance()
