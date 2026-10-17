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
    Run the tests of the installed package.

    Exit with the status code of the test run upon completion if ``exit`` is
    True.

    Parameters
    ----------
    extra_args : str or list of str, optional
        Extra arguments to pass to pytest. The default options are --tb=auto
        and --disable-pytest-warnings. The extra arguments are added to the
        default options if ``append`` is True and replace them otherwise.
    exit : bool, optional
        If True, exit the Python interpreter with the status code of the test
        run. If False, return the status code.
    append : bool, optional
        If True, add ``extra_args`` to the default options. If False, use only
        ``extra_args``.
    location : str, optional
        The location of the tests to run, as a path relative to the directory
        of the installed ``arch`` package. The default runs all of the tests of
        the package.

    Returns
    -------
    int
        The pytest exit status. Only returned if ``exit`` is False.

    Raises
    ------
    TypeError
        If ``exit`` or ``append`` is not a bool.
    ImportError
        If pytest is not installed.
    RuntimeError
        If the location of the tests does not exist.
    """

    # Imported here to keep importing arch.utility light
    from arch.utility._validation import check_bool  # noqa: PLC0415

    exit = check_bool("exit", exit)
    append = check_bool("append", append)
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
