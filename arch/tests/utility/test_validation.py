from fractions import Fraction
import re

import numpy as np
from numpy.random import Generator, RandomState
import pandas as pd
import pytest

from arch.tests.validation_table import (
    BOOL_BAD,
    BOOL_GOOD,
    FLOAT_BAD,
    FLOAT_GOOD,
    INT_BAD,
    INT_GOOD,
    SEED_BAD,
    SEED_GOOD,
    Row,
    bad_value_rows,
    good_value_rows,
    ids_for,
    run_row,
    to_params,
)
from arch.utility import _validation as v
from arch.utility._validation import (
    check_array,
    check_bool,
    check_callable,
    check_float,
    check_index,
    check_instance,
    check_int,
    check_ndarray,
    check_option,
    check_seed,
    check_str,
)

FORM = r"^{name} must be .+; got .+ \(\w+\)$"

BOOL_BAD_VALUES = [value for value, _ in BOOL_BAD if value is not None]
INT_BAD_VALUES = [value for value, _ in INT_BAD]
FLOAT_BAD_TYPE = [value for value, exc in FLOAT_BAD if exc is TypeError]
FLOAT_BAD_VALUE = [value for value, exc in FLOAT_BAD if exc is ValueError]


def assert_form(exc_info, name):
    assert re.match(FORM.format(name=name), str(exc_info.value), re.DOTALL)


# check_option


def test_lower_case_value_is_returned():
    for value in ("aic", "AIC", "Aic", "aIc"):
        assert check_option("method", value, ("aic", "bic")) == "aic"


def test_mapping_returns_canonical_value():
    allowed = {"mean": "mean", "avg": "mean", "max": "max", "maximum": "max"}
    assert check_option("method", "avg", allowed) == "mean"
    assert check_option("method", "AVG", allowed) == "mean"
    assert check_option("method", "MAXIMUM", allowed) == "max"


def test_normalize_is_applied_after_lower():
    allowed = ("moving_block", "mbb")
    out = check_option(
        "bootstrap", "Moving Block", allowed, normalize=lambda s: s.replace(" ", "_")
    )
    assert out == "moving_block"


def test_value_error_message():
    with pytest.raises(ValueError, match=r"^method must") as exc:
        check_option("method", "xyz", ("aic", "bic", "t-stat", "maic"))
    assert str(exc.value) == (
        "method must be one of 'aic', 'bic', 't-stat' or 'maic'; got 'xyz' (str)"
    )


def test_two_and_one_choices():
    with pytest.raises(
        ValueError, match=r"x must be one of 'a' or 'b'; got 'c' \(str\)"
    ):
        check_option("x", "c", ("a", "b"))
    with pytest.raises(ValueError, match=r"x must be one of 'a'; got 'c' \(str\)"):
        check_option("x", "c", ("a",))


def test_display_overrides_allowed():
    with pytest.raises(ValueError, match=r"x must be one of 'r' or 'max'; got 'c'"):
        check_option("x", "c", {"r": "r", "maximum": "max"}, display=("r", "max"))


@pytest.mark.parametrize(
    "value",
    [None, 1, 1.5, b"aic", ("aic",)],
    ids=["none", "int", "float", "bytes", "tuple"],
)
def test_non_string_is_type_error(value):
    with pytest.raises(
        TypeError, match=r"method must be one of 'aic' or 'bic'; got "
    ) as e:
        check_option("method", value, ("aic", "bic"))
    assert_form(e, "method")
    assert f"({type(value).__name__})" in str(e.value)


def test_long_value_is_shortened():
    with pytest.raises(ValueError, match=r"^method must") as exc:
        check_option("method", "x" * 200, ("aic", "bic"))
    message = str(exc.value)
    assert message.endswith("... (str)")
    assert len(message) < 100


# check_bool


@pytest.mark.parametrize("value", BOOL_GOOD, ids=ids_for(BOOL_GOOD))
def test_bool_accepts(value):
    out = check_bool("flag", value)
    assert out is bool(value)
    assert type(out) is bool


@pytest.mark.parametrize("value", BOOL_BAD_VALUES, ids=ids_for(BOOL_BAD_VALUES))
def test_bool_rejects(value):
    with pytest.raises(TypeError, match=r"^flag must be True or False; got ") as e:
        check_bool("flag", value)
    assert_form(e, "flag")


def test_bool_none():
    assert check_bool("flag", None, allow_none=True) is None
    with pytest.raises(
        TypeError, match=r"flag must be True or False; got None \(NoneType\)"
    ):
        check_bool("flag", None)
    with pytest.raises(
        TypeError, match=r"flag must be True or False or None; got 'no'"
    ):
        check_bool("flag", "no", allow_none=True)


# check_int


@pytest.mark.parametrize("value", INT_GOOD, ids=ids_for(INT_GOOD))
def test_int_accepts_integers(value):
    out = check_int("lags", value)
    assert out == 3
    assert type(out) is int


def test_int_accepts_big_and_zero_dim_arrays():
    assert check_int("n", 10**30) == 10**30
    assert check_int("n", np.array(4)) == 4
    assert check_int("n", np.uint64(2**63)) == 2**63


@pytest.mark.parametrize("value", INT_BAD_VALUES, ids=ids_for(INT_BAD_VALUES))
def test_int_rejects_non_integers(value):
    with pytest.raises(TypeError, match=r"^lags must be an integer; got ") as e:
        check_int("lags", value)
    assert_form(e, "lags")


@pytest.mark.parametrize(
    "value",
    [np.array([1]), np.array(True), np.array(2.0), Fraction(3), "3", b"3", 3 + 0j],
    ids=["1d", "bool0d", "float0d", "fraction", "str", "bytes", "complex"],
)
def test_int_rejects_other_types(value):
    with pytest.raises(TypeError, match=r"^lags must be an integer"):
        check_int("lags", value)


@pytest.mark.parametrize(
    ("kwargs", "rule"),
    [
        ({"lower": 1}, r"an integer >= 1"),
        ({"upper": 5}, r"an integer <= 5"),
        ({"lower": 1, "upper": 5}, r"an integer in \[1, 5\]"),
    ],
    ids=["lower", "upper", "both"],
)
def test_int_bounds(kwargs, rule):
    lower, upper = kwargs.get("lower", -(10**9)), kwargs.get("upper", 10**9)
    assert check_int("n", lower, **kwargs) == lower
    assert check_int("n", upper, **kwargs) == upper
    outside = []
    if "lower" in kwargs:
        outside.append(kwargs["lower"] - 1)
    if "upper" in kwargs:
        outside.append(kwargs["upper"] + 1)
    for value in outside:
        for candidate in (value, np.int64(value)):
            with pytest.raises(
                ValueError, match=rf"^n must be {rule}; got {value}"
            ) as e:
                check_int("n", candidate, **kwargs)
            assert_form(e, "n")


def test_int_wrong_type_before_range():
    with pytest.raises(
        TypeError, match=r"n must be an integer >= 1; got 0.5 \(float\)"
    ):
        check_int("n", 0.5, lower=1)


def test_int_none():
    assert check_int("n", None, allow_none=True, lower=1) is None
    with pytest.raises(TypeError, match=r"n must be an integer >= 1 or None; got 2.5"):
        check_int("n", 2.5, lower=1, allow_none=True)
    with pytest.raises(ValueError, match=r"n must be an integer >= 1 or None; got 0"):
        check_int("n", 0, lower=1, allow_none=True)
    with pytest.raises(TypeError, match=r"n must be an integer >= 1; got None"):
        check_int("n", None, lower=1)


# check_float


@pytest.mark.parametrize("value", FLOAT_GOOD, ids=ids_for(FLOAT_GOOD))
def test_float_accepts_reals(value):
    out = check_float("size", value)
    assert out == 0.5
    assert type(out) is float


def test_float_accepts_integers_and_zero_dim_arrays():
    for value in (1, np.int8(1), np.uint32(1), np.array(1), np.array(1.0)):
        out = check_float("x", value)
        assert out == 1.0
        assert type(out) is float
    assert check_float("x", np.float32(0.1)) == float(np.float32(0.1))
    assert check_float("x", np.longdouble(2)) == 2.0


@pytest.mark.parametrize("value", FLOAT_BAD_TYPE, ids=ids_for(FLOAT_BAD_TYPE))
def test_float_rejects_non_numbers(value):
    with pytest.raises(TypeError, match=r"^x must be a finite float; got ") as e:
        check_float("x", value)
    assert_form(e, "x")


@pytest.mark.parametrize(
    "value", [np.array([0.5]), np.array("a"), np.array(True), "nan", b"1"]
)
def test_float_rejects_other_types(value):
    with pytest.raises(TypeError, match=r"^x must be a finite float"):
        check_float("x", value)


@pytest.mark.parametrize("value", FLOAT_BAD_VALUE, ids=ids_for(FLOAT_BAD_VALUE))
def test_float_rejects_non_finite(value):
    with pytest.raises(ValueError, match=r"^x must be a finite float; got ") as e:
        check_float("x", value)
    assert_form(e, "x")


def test_float_huge_int_is_value_error():
    with pytest.raises(ValueError, match=r"^x must be a finite float; got 1000"):
        check_float("x", 10**400)


def test_float_allow_nan_and_inf():
    assert np.isnan(check_float("x", np.nan, allow_nan=True))
    assert check_float("x", np.inf, allow_inf=True) == np.inf
    assert check_float("x", -np.inf, allow_inf=True) == -np.inf
    with pytest.raises(ValueError, match=r"x must be a float \(inf allowed\); got nan"):
        check_float("x", np.nan, allow_inf=True)
    with pytest.raises(ValueError, match=r"x must be a float \(NaN allowed\); got inf"):
        check_float("x", np.inf, allow_nan=True)
    out = check_float("x", np.nan, allow_nan=True, allow_inf=True, lower=0)
    assert np.isnan(out)
    with pytest.raises(
        ValueError, match=r"x must be a float >= 0 \(NaN and inf allowed\)"
    ):
        check_float("x", -np.inf, allow_nan=True, allow_inf=True, lower=0)
    with pytest.raises(ValueError, match=r"x must be a float <= 1 \(inf allowed\)"):
        check_float("x", np.inf, allow_inf=True, upper=1)


@pytest.mark.parametrize(
    ("inclusive", "lower_ok", "upper_ok", "rule"),
    [
        ("both", True, True, r"a finite float in \[0, 1\]"),
        ("neither", False, False, r"a finite float in \(0, 1\)"),
        ("left", True, False, r"a finite float in \[0, 1\)"),
        ("right", False, True, r"a finite float in \(0, 1\]"),
    ],
)
def test_float_inclusive(inclusive, lower_ok, upper_ok, rule):
    kw = {"lower": 0, "upper": 1, "inclusive": inclusive}
    assert check_float("size", 0.5, **kw) == 0.5
    for end, ok in ((0, lower_ok), (1, upper_ok)):
        if ok:
            assert check_float("size", end, **kw) == end
        else:
            with pytest.raises(ValueError, match=rf"^size must be {rule}; got {end}"):
                check_float("size", end, **kw)
    for outside in (-0.1, 1.1, np.float32(2)):
        with pytest.raises(ValueError, match=rf"^size must be {rule}; got "):
            check_float("size", outside, **kw)


@pytest.mark.parametrize(
    ("kwargs", "rule"),
    [
        ({"lower": 0}, r"a finite float >= 0"),
        ({"lower": 0, "inclusive": "neither"}, r"a finite float > 0"),
        ({"upper": 2.5}, r"a finite float <= 2.5"),
        ({"upper": 2.5, "inclusive": "neither"}, r"a finite float < 2.5"),
    ],
)
def test_float_one_sided_rules(kwargs, rule):
    bad = -1.0 if "lower" in kwargs else 10.0
    with pytest.raises(ValueError, match=rf"^x must be {rule}; got "):
        check_float("x", bad, **kwargs)


def test_float_none():
    assert check_float("x", None, allow_none=True) is None
    with pytest.raises(TypeError, match=r"x must be a finite float or None; got 'a'"):
        check_float("x", "a", allow_none=True)
    with pytest.raises(TypeError, match=r"x must be a finite float; got None"):
        check_float("x", None)


# check_seed


@pytest.mark.parametrize("value", SEED_GOOD, ids=ids_for(SEED_GOOD))
def test_seed_accepts(value):
    out = check_seed("seed", value)
    if isinstance(value, (Generator, RandomState)):
        assert out is value
    else:
        assert out == int(value)
        assert type(out) is int


def test_seed_none():
    assert check_seed("seed", None) is None
    with pytest.raises(
        TypeError, match=r"^seed must be a non-negative integer, a NumPy"
    ):
        check_seed("seed", None, allow_none=False)


@pytest.mark.parametrize(("value", "exc"), SEED_BAD, ids=ids_for(SEED_BAD))
def test_seed_rejects(value, exc):
    with pytest.raises(exc, match=r"^seed must be None, a non-negative integer, ") as e:
        check_seed("seed", value)
    assert_form(e, "seed")


# check_str, check_callable, check_instance


def test_str():
    assert check_str("title", "abc") == "abc"
    out = check_str("title", np.str_("abc"))
    assert type(out) is str
    assert check_str("title", None, allow_none=True) is None
    for bad in (1, b"abc", None, ["a"]):
        with pytest.raises(TypeError, match=r"^title must be a string; got ") as e:
            check_str("title", bad)
        assert_form(e, "title")
    with pytest.raises(TypeError, match=r"title must be a string or None; got 1"):
        check_str("title", 1, allow_none=True)


def test_callable():
    assert check_callable("func", len) is len
    assert check_callable("func", None, allow_none=True) is None
    for bad in (1, "len", None):
        with pytest.raises(TypeError, match=r"^func must be callable; got ") as e:
            check_callable("func", bad)
        assert_form(e, "func")
    with pytest.raises(TypeError, match=r"func must be callable or None; got 1"):
        check_callable("func", 1, allow_none=True)


def test_instance():
    value = [1]
    assert check_instance("x", value, list) is value
    assert check_instance("x", value, (tuple, list)) is value
    assert check_instance("x", None, list, allow_none=True) is None
    with pytest.raises(
        TypeError, match=r"^x must be an instance of list; got 1 \(int\)"
    ):
        check_instance("x", 1, list)
    with pytest.raises(
        TypeError, match=r"x must be an instance of list, tuple or dict; got "
    ):
        check_instance("x", 1, (list, tuple, dict))
    with pytest.raises(
        TypeError, match=r"x must be an instance of list or None; got 1"
    ):
        check_instance("x", 1, list, allow_none=True)
    with pytest.raises(TypeError, match=r"x must be an instance of list; got None"):
        check_instance("x", None, list)


# check_array


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        (1, 2, 3),
        np.array([1, 2, 3]),
        np.array([1, 2, 3], dtype=np.int8),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([1, 2, 3], dtype=object),
        pd.Series([1, 2, 3]),
        pd.DataFrame({"a": [1, 2, 3]}),
        np.array([[1], [2], [3]]),
        np.array([[1, 2, 3]]),
    ],
    ids=[
        "list",
        "tuple",
        "int64",
        "int8",
        "float32",
        "object",
        "series",
        "frame",
        "column",
        "row",
    ],
)
def test_array_1d_conversion(value):
    out = check_array("y", value, ndim=1)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_array_returns_the_same_float64_array():
    x = np.arange(5.0)
    assert check_array("y", x, ndim=1) is x
    assert check_array("y", x) is x
    assert check_array("y", x, shape=(5,)) is x
    x2 = np.arange(6.0).reshape(3, 2)
    assert check_array("y", x2, ndim=2) is x2


def test_array_scalar_and_squeeze():
    assert check_array("y", 2, ndim=1).shape == (1,)
    assert check_array("y", 2, ndim=2).shape == (1, 1)
    assert check_array("y", np.arange(3), ndim=2).shape == (3, 1)
    with pytest.raises(
        ValueError, match=r"^y must be a 1-d array of finite numbers; got shape \(\)"
    ):
        check_array("y", 2, ndim=1, squeeze=False)
    with pytest.raises(
        ValueError, match=r"^y must be a 2-d array of finite numbers; got shape \(3,\)"
    ):
        check_array("y", np.arange(3), ndim=2, squeeze=False)
    with pytest.raises(
        ValueError,
        match=r"^y must be a 1-d array of finite numbers; got shape \(3, 1\)",
    ):
        check_array("y", np.ones((3, 1)), ndim=1, squeeze=False)


def test_array_wrong_ndim():
    with pytest.raises(
        ValueError,
        match=r"^y must be a 1-d array of finite numbers; got shape \(5, 2\) \(ndarray\)",
    ) as e:
        check_array("y", np.ones((5, 2)), ndim=1)
    assert_form(e, "y")
    with pytest.raises(
        ValueError,
        match=r"^y must be a 1-d array of finite numbers; got shape \(5, 2\) \(DataFrame\)",
    ):
        check_array("y", pd.DataFrame(np.ones((5, 2))), ndim=1)
    with pytest.raises(
        ValueError,
        match=r"^y must be a 2-d array of finite numbers; got shape \(2, 2, 2\)",
    ):
        check_array("y", np.ones((2, 2, 2)), ndim=2)


@pytest.mark.parametrize(
    "value",
    [
        ["a", "b"],
        [1, "b"],
        ["2020-01-01"],
        None,
        [None, 1.0],
        pd.Series(["a"]),
    ],
    ids=["str", "mixed", "date-str", "none", "none-in-list", "series-str"],
)
def test_array_rejects_unconvertible(value):
    with pytest.raises(
        TypeError, match=r"^y must be a 1-d array of finite numbers; got "
    ) as e:
        check_array("y", value, ndim=1)
    assert_form(e, "y")


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (np.array([1 + 1j]), "complex128"),
        (np.array([1 + 1j], dtype=np.complex64), "complex64"),
        (np.ones(3, dtype=np.float16), "float16"),
        (pd.Series(np.ones(3, dtype=np.float16)), "float16"),
        (
            pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": np.ones(3, dtype=np.float16)}),
            "float16",
        ),
    ],
    ids=["complex128", "complex64", "float16", "series-float16", "frame-float16"],
)
def test_array_rejects_float16_and_complex(value, text):
    # The message gives the data type that was received and the data types accepted
    with pytest.raises(
        TypeError,
        match=r"^y must be an array of finite real numbers \(integer, Boolean, "
        rf"float32 or float64 data\); got {text} data with shape \(\d+(, \d+)?,?\) "
        r"\(\w+\)$",
    ) as e:
        check_array("y", value)
    assert_form(e, "y")


def test_array_complex_is_allowed_by_the_flag():
    value = np.array([1 + 1j, 2.0, np.nan])
    with pytest.raises(
        ValueError, match=r"^y must be a 1-d array of finite numbers; got 1 NaN"
    ):
        check_array("y", value, ndim=1, allow_complex=True)
    out = check_array("y", value, ndim=1, allow_complex=True, allow_nan=True)
    assert out.dtype == np.complex128
    with pytest.raises(
        TypeError, match=r"^y must be a 1-d array of finite real or complex"
    ):
        check_array("y", np.ones(3, dtype=np.float16), ndim=1, allow_complex=True)
    # NaN or inf in the imaginary part are found
    with pytest.raises(ValueError, match=r"got 1 inf \(ndarray\)"):
        check_array(
            "y", np.array([1 + 1j, complex(2, np.inf)]), ndim=1, allow_complex=True
        )


def test_array_rejects_dates():
    with pytest.raises(
        TypeError, match=r"^y must be an array of finite numbers; got shape \(2,\)"
    ):
        check_array("y", pd.date_range("2020-01-01", periods=2))


def test_array_non_finite():
    with pytest.raises(
        ValueError,
        match=r"^y must be a 1-d array of finite numbers; got 1 NaN \(list\)",
    ) as e:
        check_array("y", [1.0, np.nan], ndim=1)
    assert_form(e, "y")
    with pytest.raises(ValueError, match=r"got 1 NaN and 2 inf \(ndarray\)"):
        check_array("y", np.array([np.nan, np.inf, -np.inf]))
    with pytest.raises(
        ValueError, match=r"^y must be an array of numbers without NaN; got 1 NaN"
    ):
        check_array("y", np.array([np.nan, np.inf]), allow_inf=True)
    with pytest.raises(
        ValueError, match=r"^y must be an array of numbers without inf; got 1 inf"
    ):
        check_array("y", np.array([np.nan, np.inf]), allow_nan=True)
    out = check_array("y", np.array([np.nan, np.inf]), allow_nan=True, allow_inf=True)
    assert out.shape == (2,)
    with pytest.raises(ValueError, match="got 1 NaN"):
        check_array("y", pd.Series([1.0, np.nan]), ndim=1)
    with pytest.raises(ValueError, match="got 1 inf"):
        check_array("y", pd.DataFrame({"a": [1.0, np.inf]}), ndim=2)


def test_array_shape_and_length():
    x = np.ones((4, 2))
    assert check_array("x", x, shape=(4, 2)) is x
    assert check_array("x", x, shape=(None, 2)) is x
    assert check_array("x", x, shape=(4, None)) is x
    with pytest.raises(
        ValueError,
        match=r"^x must be a 2-d array of finite numbers with shape \(4, 3\); "
        r"got shape \(4, 2\) \(ndarray\)",
    ) as e:
        check_array("x", x, shape=(4, 3))
    assert_form(e, "x")
    with pytest.raises(ValueError, match=r"with shape \(n, 3\); got shape \(4, 2\)"):
        check_array("x", x, shape=(None, 3))
    with pytest.raises(
        ValueError,
        match=r"^y must be a 1-d array of finite numbers with shape \(6,\); got shape \(5,\)",
    ):
        check_array("y", np.ones(5), shape=(6,))
    # the shape replaces ndim
    assert check_array("y", np.ones((5, 1)), shape=(5,)).shape == (5,)
    assert check_array("x", np.ones(3), shape=(3, 1)).shape == (3, 1)


def test_array_min_length():
    assert check_array("y", np.ones(3), ndim=1, min_length=3).shape == (3,)
    with pytest.raises(
        ValueError,
        match=r"^y must be a 1-d array of finite numbers with at least 4 elements; "
        r"got shape \(3,\)",
    ):
        check_array("y", np.ones(3), ndim=1, min_length=4)
    with pytest.raises(
        ValueError,
        match=r"^y must be a 1-d array of finite numbers with at least 1 element; "
        r"got shape \(0,\)",
    ):
        check_array("y", [], ndim=1, min_length=1)
    with pytest.raises(
        ValueError,
        match=r"^x must be a 2-d array of finite numbers with at least 5 rows; got shape \(3, 2\)",
    ):
        check_array("x", np.ones((3, 2)), ndim=2, min_length=5)
    assert check_array("y", [], ndim=1).shape == (0,)
    with pytest.raises(ValueError, match="at least 1 row"):
        check_array("y", np.array(1.0), min_length=1)


def test_array_none():
    assert check_array("y", None, allow_none=True) is None
    with pytest.raises(
        TypeError, match=r"^y must be a 1-d array of finite numbers; got None"
    ):
        check_array("y", None, ndim=1)
    with pytest.raises(
        TypeError, match=r"y must be a 1-d array of finite numbers or None; got 'a'"
    ):
        check_array("y", "a", ndim=1, allow_none=True)


def test_array_does_not_modify_input():
    x = np.arange(4, dtype=np.int64)
    s = pd.Series(x)
    check_array("x", x, ndim=1)
    check_array("x", s, ndim=1)
    assert x.dtype == np.int64
    assert s.dtype == np.int64


# check_ndarray


def test_ndarray_accepts_and_does_not_copy():
    x = np.zeros(5)
    assert check_ndarray("sigma2", x, ndim=1) is x
    assert check_ndarray("sigma2", x, ndim=1, shape=(5,), writeable=True) is x
    assert check_ndarray("v", np.zeros((3, 2)), ndim=2, shape=(3, None)) is not None
    assert check_ndarray("s", None, ndim=1, allow_none=True) is None
    # a C contiguous slice is fine
    assert check_ndarray("s", np.zeros(10)[2:7], ndim=1) is not None


@pytest.mark.parametrize(
    "value",
    [None, [0.0, 1.0], (0.0, 1.0), 1.0, "abc", pd.Series([1.0, 2.0])],
    ids=["none", "list", "tuple", "float", "str", "series"],
)
def test_ndarray_rejects_non_arrays(value):
    with pytest.raises(
        TypeError, match=r"^sigma2 must be a 1-d C contiguous float64 ndarray; got "
    ) as e:
        check_ndarray("sigma2", value, ndim=1)
    assert_form(e, "sigma2")


@pytest.mark.parametrize(
    "dtype", [np.float32, np.int64, np.complex128, np.bool_, object]
)
def test_ndarray_rejects_other_dtypes(dtype):
    with pytest.raises(
        TypeError, match=rf"^sigma2 must be .*; got {np.dtype(dtype)} data \(ndarray\)"
    ):
        check_ndarray("sigma2", np.zeros(4, dtype=dtype), ndim=1)


def test_ndarray_rejects_shape_and_layout():
    with pytest.raises(
        ValueError,
        match=r"^s must be a 1-d C contiguous float64 ndarray; got shape \(3, 1\)",
    ):
        check_ndarray("s", np.zeros((3, 1)), ndim=1)
    with pytest.raises(
        ValueError,
        match=r"^s must be a 1-d C contiguous float64 ndarray with shape \(4,\); got shape \(5,\)",
    ):
        check_ndarray("s", np.zeros(5), ndim=1, shape=(4,))
    with pytest.raises(
        ValueError,
        match=r"^s must be a 2-d C contiguous float64 ndarray with shape \(n, 2\); "
        r"got shape \(4, 3\)",
    ):
        check_ndarray("s", np.zeros((4, 3)), ndim=2, shape=(None, 2))
    with pytest.raises(
        ValueError, match=r"got an array that is not C contiguous \(ndarray\)"
    ):
        check_ndarray("s", np.zeros(10)[::2], ndim=1)
    with pytest.raises(ValueError, match=r"got an array that is not C contiguous"):
        check_ndarray("s", np.zeros((3, 2)).T, ndim=2)
    ro = np.zeros(3)
    ro.flags.writeable = False
    assert check_ndarray("s", ro, ndim=1) is ro
    with pytest.raises(ValueError, match=r"got a read-only array \(ndarray\)"):
        check_ndarray("s", ro, ndim=1, writeable=True)
    with pytest.raises(TypeError, match="or None; got 'a'"):
        check_ndarray("s", "a", ndim=1, allow_none=True)


# check_index


@pytest.fixture(scope="module")
def dates():
    return pd.date_range("2020-01-01", periods=10, freq="D")


@pytest.mark.parametrize("value", [0, 4, 10, np.int64(10), np.uint8(3)])
def test_index_positions(value):
    out = check_index("first_obs", value, 10)
    assert out == int(value)
    assert type(out) is int


def test_index_none_and_default():
    assert check_index("first_obs", None, 10) is None
    assert check_index("first_obs", None, 10, default=0) == 0
    assert check_index("last_obs", None, 10, default=10) == 10
    with pytest.raises(
        TypeError,
        match=r"^first_obs must be an integer position in \[0, 10\]; got None",
    ):
        check_index("first_obs", None, 10, allow_none=False)


@pytest.mark.parametrize("value", [-1, -1000, 11, 10**6, np.int64(-3)])
def test_index_out_of_range(value):
    with pytest.raises(
        ValueError,
        match=r"^first_obs must be an integer position in \[0, 10\] or None; got ",
    ) as e:
        check_index("first_obs", value, 10)
    assert_form(e, "first_obs")


def test_index_upper():
    assert check_index("start", 9, 10, upper=9) == 9
    with pytest.raises(
        ValueError,
        match=r"start must be an integer position in \[0, 9\] or None; got 10",
    ):
        check_index("start", 10, 10, upper=9)


@pytest.mark.parametrize(
    "value",
    [2.0, 2.5, True, np.True_, "x", [1], np.nan],
    ids=["2.0", "2.5", "True", "np.True_", "str", "list", "nan"],
)
def test_index_wrong_types(value):
    # "x" is a string and a date-like value, without a date index it is a ValueError
    exc = ValueError if isinstance(value, str) else TypeError
    with pytest.raises(
        exc, match=r"^first_obs must be an integer position in \[0, 10\] or None; got "
    ):
        check_index("first_obs", value, 10)


def test_index_dates(dates):
    kw = {"index": dates}
    assert check_index("first_obs", "2020-01-05", 10, **kw) == 4
    assert check_index("first_obs", "2020-01-01", 10, **kw) == 0
    assert check_index("last_obs", "2020-01-10", 10, **kw) == 9
    assert check_index("first_obs", pd.Timestamp("2020-01-03"), 10, **kw) == 2
    assert check_index("first_obs", pd.Timestamp("2020-01-03 12:00"), 10, **kw) == 3
    assert check_index("first_obs", np.datetime64("2020-01-03"), 10, **kw) == 2
    assert (
        check_index("first_obs", pd.Timestamp("2020-01-03").to_pydatetime(), 10, **kw)
        == 2
    )
    assert check_index("first_obs", pd.Timestamp("2020-01-03").date(), 10, **kw) == 2
    assert check_index("first_obs", 3, 10, **kw) == 3
    assert check_index("first_obs", None, 10, default=7, **kw) == 7


def test_index_dates_outside(dates):
    for value in ("2019-12-31", "2020-01-11", "1999-01-01", pd.Timestamp("2021-01-01")):
        with pytest.raises(
            ValueError,
            match=r"^first_obs must be an integer position in \[0, 10\], "
            r"a date within the index or None; got ",
        ) as e:
            check_index("first_obs", value, 10, index=dates)
        assert_form(e, "first_obs")


def test_index_dates_unparseable_and_wrong_type(dates):
    with pytest.raises(
        ValueError,
        match=r"first_obs must be .* a date within the index.*; got 'zzz' \(str\)",
    ):
        check_index("first_obs", "zzz", 10, index=dates)
    with pytest.raises(
        TypeError,
        match=r"first_obs must be .* a date within the index.*; got 2.5 \(float\)",
    ):
        check_index("first_obs", 2.5, 10, index=dates)
    with pytest.raises(ValueError, match="a date within the index"):
        check_index("first_obs", np.datetime64("NaT", "ns"), 10, index=dates)
    with pytest.raises(ValueError, match="a date within the index"):
        check_index("first_obs", -1, 10, index=dates)


def test_index_dates_need_a_date_index():
    for index in (None, pd.RangeIndex(10), pd.Index(list("abcdefghij"))):
        with pytest.raises(
            ValueError,
            match=r"^first_obs must be an integer position in \[0, 10\] or None; "
            r"got '2020-01-05' \(str\)",
        ):
            check_index("first_obs", "2020-01-05", 10, index=index)


def test_index_period_and_timezone(dates):
    periods = dates.to_period("D")
    assert check_index("first_obs", pd.Period("2020-01-05"), 10, index=periods) == 4
    assert check_index("first_obs", "2020-01-05", 10, index=periods) == 4
    with pytest.raises(ValueError, match="a date within the index"):
        check_index("first_obs", pd.Period("2021-01-05"), 10, index=periods)
    aware = dates.tz_localize("UTC")
    assert (
        check_index("first_obs", pd.Timestamp("2020-01-05", tz="UTC"), 10, index=aware)
        == 4
    )
    # 2020-01-05 00:00 in US/Eastern is 05:00 UTC, which is between two observations
    eastern = pd.Timestamp("2020-01-05", tz="US/Eastern")
    assert check_index("first_obs", eastern, 10, index=aware) == 5
    with pytest.raises(ValueError, match="a date within the index"):
        check_index("first_obs", pd.Timestamp("2020-01-11", tz="UTC"), 10, index=aware)


def test_index_non_monotonic_index_is_a_value_error():
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-01", "2020-01-03"])
    with pytest.raises(ValueError, match="a date within the index"):
        check_index("first_obs", "2020-01-02", 3, index=index)


# The policy is made of these small functions


def test_policy_functions():
    assert v._is_bool(True)
    assert v._is_bool(np.False_)
    assert not v._is_bool(1)
    assert v._is_integer(np.int16(1))
    assert not v._is_integer(True)
    assert v._is_real(Fraction(1, 2))
    assert not v._is_real(np.True_)
    assert v._float_in_range(0.5, 0, 1, "both", allow_nan=False, allow_inf=False)


def test_message_is_shortened_and_has_type():
    with pytest.raises(TypeError) as e:
        check_int("lags", "9" * 500)
    message = str(e.value)
    assert message.startswith("lags must be an integer; got '99")
    assert message.endswith("... (str)")
    assert len(message) < 90


def test_numpy_scalars_are_shown_without_repr():
    with pytest.raises(
        TypeError, match=r"^lags must be an integer; got 2.5 \(float64\)"
    ):
        check_int("lags", np.float64(2.5))
    with pytest.raises(
        ValueError, match=r"^lags must be an integer >= 1; got 0 \(int8\)"
    ):
        check_int("lags", np.int8(0), lower=1)


# The shared table-driven helper


def make_func(**kwargs):
    check_int("lags", kwargs["lags"], lower=0)
    check_float("size", kwargs["size"], lower=0, upper=1, inclusive="neither")
    check_bool("flag", kwargs["flag"])
    check_seed("seed", kwargs["seed"])
    return kwargs


BASE = {"lags": 2, "size": 0.5, "flag": True, "seed": None}
ROWS = [
    *bad_value_rows("make", make_func, BASE, "lags", INT_BAD),
    *good_value_rows("make", make_func, BASE, "lags", INT_GOOD),
    *bad_value_rows("make", make_func, BASE, "size", [(b, e) for b, e in FLOAT_BAD]),
    *good_value_rows("make", make_func, BASE, "size", FLOAT_GOOD),
    *bad_value_rows("make", make_func, BASE, "flag", BOOL_BAD),
    *good_value_rows("make", make_func, BASE, "flag", BOOL_GOOD),
    *bad_value_rows(
        "make", make_func, BASE, "seed", SEED_BAD, rule=r"None, a non-negative .+"
    ),
    *good_value_rows("make", make_func, BASE, "seed", SEED_GOOD),
    Row(
        "make",
        make_func,
        {**BASE, "lags": -1},
        ValueError,
        r"lags must be an integer >= 0; got -1 \(int\)",
        id="make-lags-range",
        name="lags",
    ),
]


@pytest.mark.parametrize("row", to_params(ROWS))
def test_table_driven_rows(row):
    run_row(row)
