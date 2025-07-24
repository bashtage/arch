from typing import Literal

import pytest

from arch.vendor._decorators import deprecate_kwarg, indent


@deprecate_kwarg("old", "new", {"yes": True, "no": False}, stacklevel=2)
def f(x: int, *, old: Literal["yes", "no"] = "yes", new: bool = True) -> int:
    """
    Function with keyword-only arguments.

    Parameters
    ----------
    x : int
        An integer value.
    y : str, optional
        An optional integer value.
    y_new : bool, optional
        Another optional integer value.
    """
    if new:
        return x + 1
    else:
        return x - 1


@deprecate_kwarg("old", None, stacklevel=2)
def g(x: int, *, old: bool = True) -> int:
    """
    Function with keyword-only arguments.

    Parameters
    ----------
    x : int
        An integer value.
    """
    return x + 1


@deprecate_kwarg("old", "new", stacklevel=2)
def bar(old=False, new=False):
    return new


def test_deprecate_kwarg():
    """
    Test the deprecation of the `y` keyword argument in the function `f`.
    """
    with pytest.warns(FutureWarning, match="the old"):
        f(1, old="yes")
    with pytest.warns(FutureWarning, match="the old"):
        f(1, old="no")
    f(2, new=True)
    f(2, new=False)


def test_deprecate_kwarg_no_alt():
    """
    Test the deprecation of the `y` keyword argument in the function `f`.
    """
    with pytest.warns(FutureWarning, match="the 'old'"):
        g(1, old=True)


def test_bad_deprecate_kwarg():
    def constructor():
        @deprecate_kwarg("old", None, [("yes", True), ("no", False)])
        def h(x: int, *, old: bool = True) -> int:
            """
            Function with keyword-only arguments.

            Parameters
            ----------
            x : int
                An integer value.
            """
            return x + 1

        return h

    with pytest.raises(TypeError):
        constructor()


def test_simple_depr():
    with pytest.warns(FutureWarning, match="the 'old'"):
        bar(old=True)


def test_indent():
    res = indent(
        """
This is a test
""",
        1,
    )
    assert res[:5] == "\n" + " " * 4
    assert res[5:] == "This is a test\n    "
