import os
import sys

PKG = os.path.dirname(os.path.dirname(__file__))


def test(
    extra_args: str | list[str] | None = None,
    exit: bool = True,
    append: bool = True,
    location: str = "",
) -> int:
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
        import pytest  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Need pytest to run tests") from exc

    cmd = ["--tb=auto", "--disable-pytest-warnings"]
    if extra_args:
        if not isinstance(extra_args, list):
            pytest_args = [extra_args]
        else:
            pytest_args = extra_args
        if append:
            cmd += pytest_args[:]
        else:
            cmd = pytest_args
    pkg_loc = PKG
    if location:
        pkg_loc = os.path.abspath(os.path.join(PKG, location))
    if not os.path.exists(pkg_loc):
        raise RuntimeError(f"{pkg_loc} was not found. Unable to run tests")
    cmd = [pkg_loc] + cmd
    cmd_str = " ".join(cmd)
    print(f"running: pytest {cmd_str}")
    status = pytest.main(cmd)
    if exit:  # pragma: no cover
        sys.exit(status)
    return status


__all__ = ["test"]
