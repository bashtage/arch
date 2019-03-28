import os
import sys

from arch.utility.cov import cov_nw

PKG = os.path.dirname(os.path.dirname(__file__))


def test(extra_args=None, exit=True):
    try:
        import pytest
    except ImportError:
        raise ImportError("Need pytest to run tests")
    cmd = []
    extra_args = ['--tb=short', '--disable-pytest-warnings'] if extra_args is None else extra_args
    if extra_args:
        if not isinstance(extra_args, list):
            extra_args = [extra_args]
        cmd = extra_args[:]
    cmd = [PKG] + cmd
    print("running: pytest {}".format(' '.join(cmd)))
    status = pytest.main(cmd)
    if exit:
        sys.exit(status)


__all__ = ['cov_nw', 'test']
