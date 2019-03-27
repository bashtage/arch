import os
import sys

from arch.utility.cov import cov_nw

PKG = os.path.dirname(os.path.dirname(__file__))


def test(extra_args=None):
    try:
        import pytest
    except ImportError:
        raise ImportError("Need pytest to run tests")
    cmd = []
    if extra_args:
        if not isinstance(extra_args, list):
            extra_args = [extra_args]
        cmd = extra_args
    cmd += [PKG]
    print("running: pytest {}".format(' '.join(cmd)))
    sys.exit(pytest.main(cmd))


__all__ = ['cov_nw', 'test']
