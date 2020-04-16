import os
import sys
from typing import List, Optional, Sequence, Union

from arch.utility.cov import cov_nw

PKG = os.path.dirname(os.path.dirname(__file__))


def test(
    extra_args: Optional[Union[str, Sequence[str]]] = None, exit: bool = True
) -> None:
    """
    Test runner that allows testing of installed package.

    Exists with test status code upon completion.

    Parameters
    ----------
    extra_args : {str, list[str]}, default None
        Extra arguments to pass to pytest. Default options are --tb=short
        and --disable-pytest-warnings. Providing extra_args overwrites the
        defaults with the user-provided arguments.
    """
    try:
        import pytest
    except ImportError:
        raise ImportError("Need pytest to run tests")
    cmd: List[str] = []
    default_args = ["--tb=short", "--disable-pytest-warnings"]
    if extra_args is None:
        extra_args_lst = default_args
    elif isinstance(extra_args, str):
        extra_args_lst = [extra_args]
    else:  # extra_args is not None
        extra_args_lst = list(extra_args)

    if extra_args_lst:
        cmd = extra_args_lst[:]
    cmd = [PKG] + cmd
    print("running: pytest {}".format(" ".join(cmd)))
    status = pytest.main(cmd)
    if exit:
        sys.exit(status)


__all__ = ["cov_nw", "test"]
