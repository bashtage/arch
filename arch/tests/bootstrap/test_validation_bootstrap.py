"""
Table-driven tests of the input validation of the bootstraps and optimal_block_length.

The data passed to a bootstrap are not converted and not checked for NaN: they go to the
function of the user as given.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import warnings

import numpy as np
from numpy.random import Generator, RandomState
import pandas as pd
import pytest

from arch.bootstrap import (
    CircularBlockBootstrap,
    IIDBootstrap,
    IndependentSamplesBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
    optimal_block_length,
)
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
    run_row,
    to_params,
)

NOBS = 100
Y = np.random.default_rng(20261017).standard_normal(NOBS)
Y2 = np.random.default_rng(1).standard_normal((NOBS, 2))


def mean_fn(y: Any) -> float:
    return float(np.mean(y))


def mean_kw(y: Any, **kwargs: Any) -> float:
    return float(np.mean(y))


def std_err_fn(params: Any, y: Any, **kwargs: Any) -> Any:
    return np.atleast_1d(np.std(y) / np.sqrt(len(y)))


SEED_RULE = r"None, a non-negative integer, a NumPy Generator or a NumPy RandomState"
REPS_RULE = r"an integer >= 1"
SEED_BAD_NO_NONE = list(SEED_BAD)
SEED_GOOD_ALL = [*SEED_GOOD, None]

CTORS: dict[str, Callable[..., Any]] = {
    "IID": lambda **kw: IIDBootstrap(Y, **kw),
    "Indep": lambda **kw: IndependentSamplesBootstrap(Y, Y[:60], **kw),
    "Circular": lambda **kw: CircularBlockBootstrap(5, Y, **kw),
    "Moving": lambda **kw: MovingBlockBootstrap(5, Y, **kw),
    "Stationary": lambda **kw: StationaryBootstrap(5, Y, **kw),
}

ROWS: list[Row] = []

# ---- seed: the constructors and clone
for _name, _ctor in CTORS.items():
    ROWS += bad_value_rows(
        f"{_name}-ctor", _ctor, {}, "seed", SEED_BAD_NO_NONE, rule=SEED_RULE
    )
    ROWS += good_value_rows(f"{_name}-ctor", _ctor, {}, "seed", SEED_GOOD_ALL)
ROWS += bad_value_rows(
    "clone",
    lambda **kw: IIDBootstrap(Y, seed=0).clone(Y, **kw),
    {},
    "seed",
    SEED_BAD_NO_NONE,
    rule=SEED_RULE,
)
ROWS += good_value_rows(
    "clone",
    lambda **kw: IIDBootstrap(Y, seed=0).clone(Y, **kw),
    {},
    "seed",
    SEED_GOOD_ALL,
)
ROWS += bad_value_rows(
    "clone-block",
    lambda **kw: StationaryBootstrap(5, Y, seed=0).clone(Y, **kw),
    {},
    "seed",
    SEED_BAD_NO_NONE,
    rule=SEED_RULE,
)

# ---- the data: positional and keyword
DATA_BAD: list[tuple[str, Any, type[Exception], str]] = [
    ("list", [1.0, 2.0, 3.0], TypeError, r"must be an instance of ndarray"),
    ("tuple", (1.0, 2.0, 3.0), TypeError, r"must be an instance of ndarray"),
    ("None", None, TypeError, r"must be an instance of ndarray"),
    ("str", "abc", TypeError, r"must be an instance of ndarray"),
    ("scalar", 5.0, TypeError, r"must be an instance of ndarray"),
    ("empty-array", np.array([]), ValueError, r"at least 1 row"),
    ("empty-2d", np.empty((0, 2)), ValueError, r"at least 1 row"),
    ("0-d-array", np.array(3.0), ValueError, r"at least 1 row"),
    ("empty-series", pd.Series([], dtype=float), ValueError, r"at least 1 row"),
    ("empty-frame", pd.DataFrame({"a": []}), ValueError, r"at least 1 row"),
]
for _name, _ctor in {
    "IID": IIDBootstrap,
    "Indep": IndependentSamplesBootstrap,
    "Circular": lambda **kw: CircularBlockBootstrap(5, **kw),
    "Moving": lambda **kw: MovingBlockBootstrap(5, **kw),
    "Stationary": lambda **kw: StationaryBootstrap(5, **kw),
}.items():
    for _label, _value, _exc, _rule in DATA_BAD:
        ROWS.append(
            Row(
                f"{_name}-data",
                _ctor,
                {"data": _value},
                _exc,
                rf"^data {_rule}" if _exc is TypeError else rf"^data must be .*{_rule}",
                id=f"{_name}-data-kw-{_label}",
                name="data",
            )
        )
for _label, _value, _exc, _rule in DATA_BAD:
    ROWS.append(
        Row(
            "IID-pos",
            lambda value: IIDBootstrap(Y, value),
            {"value": _value},
            _exc,
            (
                rf"^args\[1\] {_rule}"
                if _exc is TypeError
                else rf"^args\[1\] must be .*{_rule}"
            ),
            id=f"IID-data-pos1-{_label}",
            name="args[1]",
        )
    )
ROWS += [
    Row(
        "IID-nodata",
        IIDBootstrap,
        {},
        ValueError,
        r"^args must be at least one NumPy array, Series or DataFrame",
        id="IID-no-data",
        name="args",
    ),
    Row(
        "Circular-nodata",
        lambda: CircularBlockBootstrap(5),
        {},
        ValueError,
        r"^args must be at least one NumPy array, Series or DataFrame",
        id="Circular-no-data",
        name="args",
    ),
    Row(
        "IID-unequal",
        lambda: IIDBootstrap(Y, Y[:50]),
        {},
        ValueError,
        r"^args\[1\] must be an array with 100 rows, as args\[0\]; got shape \(50,\)",
        id="IID-unequal-pos",
        name="args[1]",
    ),
    Row(
        "IID-unequal-kw",
        lambda: IIDBootstrap(Y, z=Y[:50]),
        {},
        ValueError,
        r"^z must be an array with 100 rows, as args\[0\]; got shape \(50,\)",
        id="IID-unequal-kw",
        name="z",
    ),
    Row(
        "Circular-unequal",
        lambda: CircularBlockBootstrap(5, Y, Y[:50]),
        {},
        ValueError,
        r"^args\[1\] must be an array with 100 rows",
        id="Circular-unequal",
        name="args[1]",
    ),
    Row(
        "IID-reserved",
        lambda: IIDBootstrap(index=Y),
        {},
        ValueError,
        r"^kwargs must be keywords that are not attributes or methods of the "
        r"bootstrap; got 'index' \(str\)",
        id="IID-reserved-index",
        name="kwargs",
    ),
    Row(
        "IID-reserved",
        lambda: IIDBootstrap(apply=Y),
        {},
        ValueError,
        r"^kwargs must be keywords that are not attributes or methods of the "
        r"bootstrap; got 'apply' \(str\)",
        id="IID-reserved-apply",
        name="kwargs",
    ),
]
# Valid data are accepted, including data that the bootstrap does not convert
GOOD_DATA = {
    "series": pd.Series(Y),
    "frame": pd.DataFrame(Y2),
    "2d": Y2,
    "3d": np.zeros((NOBS, 2, 2)),
    "int": np.arange(NOBS),
    "bool": np.arange(NOBS) > 50,
    "nan": np.r_[np.nan, Y[1:]],
    "inf": np.r_[np.inf, Y[1:]],
    "object": np.array(["a", "b"] * 50, dtype=object),
    "one-row": np.zeros(1),
}
for _label, _value in GOOD_DATA.items():
    ROWS.append(
        Row(
            "IID-data-good",
            IIDBootstrap,
            {"v": _value},
            None,
            id=f"IID-good-{_label}",
        )
    )
    ROWS.append(
        Row(
            "Stationary-data-good",
            lambda v: StationaryBootstrap(1, v),
            {"v": _value},
            None,
            id=f"Stationary-good-{_label}",
        )
    )
ROWS.append(
    Row(
        "Indep-data-good",
        lambda: IndependentSamplesBootstrap(Y, Y2[:7], z=pd.Series(Y[:3])),
        {},
        None,
        id="Indep-different-lengths",
    )
)

# ---- block_size
INT_BLOCK_BAD = [
    *INT_BAD,
    (0, ValueError),
    (-1, ValueError),
    (np.int64(-3), ValueError),
]
for _name, _cls in {
    "Circular": CircularBlockBootstrap,
    "Moving": MovingBlockBootstrap,
}.items():
    ROWS += bad_value_rows(
        f"{_name}-block_size",
        lambda block_size, _cls=_cls: _cls(block_size, Y, seed=0),
        {},
        "block_size",
        INT_BLOCK_BAD,
        rule=r"an integer >= 1",
    )
    ROWS += good_value_rows(
        f"{_name}-block_size",
        lambda block_size, _cls=_cls: _cls(block_size, Y, seed=0),
        {},
        "block_size",
        [*INT_GOOD, 1, NOBS],
    )
FLOAT_BLOCK_BAD = [*FLOAT_BAD, (0, ValueError), (0.5, ValueError), (-1.0, ValueError)]
ROWS += bad_value_rows(
    "Stationary-block_size",
    lambda block_size: StationaryBootstrap(block_size, Y, seed=0),
    {},
    "block_size",
    FLOAT_BLOCK_BAD,
    rule=r"a finite float >= 1",
)
ROWS += good_value_rows(
    "Stationary-block_size",
    lambda block_size: StationaryBootstrap(block_size, Y, seed=0),
    {},
    "block_size",
    [*INT_GOOD, 1, 1.0, 2.5, 5.0, np.float32(2.5), np.float64(1.5), 1000],
)

# ---- bootstrap, apply, cov, var and conf_int: reps
REPS_BAD = [*INT_BAD, (0, ValueError), (-1, ValueError), (np.int64(-2), ValueError)]
METHODS: dict[str, Callable[..., Any]] = {
    "bootstrap": lambda **kw: list(IIDBootstrap(Y, seed=0).bootstrap(**kw)),
    "apply": lambda **kw: IIDBootstrap(Y, seed=0).apply(**kw),
    "cov": lambda **kw: IIDBootstrap(Y, seed=0).cov(**kw),
    "var": lambda **kw: IIDBootstrap(Y, seed=0).var(**kw),
    "conf_int": lambda **kw: IIDBootstrap(Y, seed=0).conf_int(**kw),
}
BASE = {"func": mean_fn, "reps": 3}
for _name, _call in METHODS.items():
    base = {"reps": 3} if _name == "bootstrap" else BASE
    ROWS += bad_value_rows(
        f"{_name}-reps", _call, base, "reps", REPS_BAD, rule=REPS_RULE
    )
    ROWS += good_value_rows(f"{_name}-reps", _call, base, "reps", [*INT_GOOD, 1])
    if _name != "bootstrap":
        ROWS += bad_value_rows(
            f"{_name}-func",
            _call,
            base,
            "func",
            [
                (None, TypeError),
                (5, TypeError),
                ("mean", TypeError),
                ([mean_fn], TypeError),
            ],
            rule=r"callable",
        )
        ROWS += bad_value_rows(
            f"{_name}-extra_kwargs",
            _call,
            {**base, "func": mean_kw},
            "extra_kwargs",
            [
                (5, TypeError),
                ([("a", 1)], TypeError),
                ("abc", TypeError),
                (1.5, TypeError),
                ({1: 2}, TypeError),
                ({"a": 1, 2: 2}, TypeError),
            ],
            rule=r"(an instance of Mapping or None|a dict with str keys that are not names"
            r" of the bootstrapped data)",
        )
        ROWS.append(
            Row(
                f"{_name}-extra_kwargs-clash",
                lambda _name=_name, **kw: getattr(IIDBootstrap(y=Y, seed=0), _name)(
                    **kw
                ),
                {**base, "func": mean_kw, "extra_kwargs": {"y": 1}},
                ValueError,
                r"^extra_kwargs must be a dict with str keys that are not names of the "
                r"bootstrapped data; got the key 'y' \(dict\)",
                id=f"{_name}-extra_kwargs-clash",
                name="extra_kwargs",
            )
        )
        ROWS += good_value_rows(
            f"{_name}-extra_kwargs",
            _call,
            {**base, "func": mean_kw},
            "extra_kwargs",
            [None, {}, {"a": 1}],
        )
for _name in ("cov", "var"):
    ROWS += bad_value_rows(
        f"{_name}-recenter",
        METHODS[_name],
        BASE,
        "recenter",
        BOOL_BAD,
        rule=r"True or False",
    )
    ROWS += good_value_rows(
        f"{_name}-recenter", METHODS[_name], BASE, "recenter", BOOL_GOOD
    )
ROWS.append(
    Row(
        "conf_int-func-first",
        lambda: IIDBootstrap(Y).conf_int(None),
        {},
        TypeError,
        r"^func must be callable; got None \(NoneType\)",
        id="conf_int-func-None-positional",
        name="func",
    )
)

# ---- conf_int: size, reuse, std_err_func, studentize_reps
CI_BASE = {"func": mean_fn, "reps": 20}
CI = METHODS["conf_int"]
SIZE_BAD = [
    *FLOAT_BAD,
    (0, ValueError),
    (1, ValueError),
    (0.0, ValueError),
    (1.0, ValueError),
    (-0.1, ValueError),
    (95, ValueError),
]
ROWS += bad_value_rows(
    "conf_int-size", CI, CI_BASE, "size", SIZE_BAD, rule=r"a finite float in \(0, 1\)"
)
ROWS += good_value_rows(
    "conf_int-size", CI, CI_BASE, "size", [*FLOAT_GOOD, 0.001, 0.999]
)
ROWS += bad_value_rows(
    "conf_int-reuse", CI, CI_BASE, "reuse", BOOL_BAD, rule=r"True or False"
)
ROWS += good_value_rows("conf_int-reuse", CI, CI_BASE, "reuse", BOOL_GOOD)
ROWS += bad_value_rows(
    "conf_int-std_err_func",
    CI,
    {**CI_BASE, "method": "studentized"},
    "std_err_func",
    [(5, TypeError), ("abc", TypeError), ([std_err_fn], TypeError)],
    rule=r"callable or None",
)
ROWS += bad_value_rows(
    "conf_int-std_err_func-unused",
    CI,
    CI_BASE,
    "std_err_func",
    [(5, TypeError), ("abc", TypeError)],
    rule=r"callable or None",
)
ROWS += good_value_rows(
    "conf_int-std_err_func",
    CI,
    {**CI_BASE, "method": "studentized"},
    "std_err_func",
    [std_err_fn],
)
ROWS += good_value_rows(
    "conf_int-std_err_func-none", CI, CI_BASE, "std_err_func", [None]
)
SR_BAD_TYPE = list(INT_BAD)
SR_BAD_VALUE = [(0, ValueError), (-1, ValueError), (np.int64(-4), ValueError)]
ROWS += bad_value_rows(
    "conf_int-studentize_reps-used",
    CI,
    {**CI_BASE, "method": "studentized"},
    "studentize_reps",
    [*SR_BAD_TYPE, *SR_BAD_VALUE],
    rule=r"an integer >= 1",
)
ROWS += bad_value_rows(
    "conf_int-studentize_reps-ignored",
    CI,
    {**CI_BASE, "method": "percentile"},
    "studentize_reps",
    SR_BAD_TYPE,
    rule=r"an integer",
)
ROWS += bad_value_rows(
    "conf_int-studentize_reps-std-err",
    CI,
    {**CI_BASE, "method": "studentized", "std_err_func": std_err_fn},
    "studentize_reps",
    SR_BAD_TYPE,
    rule=r"an integer",
)
ROWS += good_value_rows(
    "conf_int-studentize_reps-used",
    CI,
    {**CI_BASE, "method": "studentized"},
    "studentize_reps",
    [*INT_GOOD, 2],
)
# A value that is not used is not checked, as documented
ROWS += good_value_rows(
    "conf_int-studentize_reps-ignored",
    CI,
    {**CI_BASE, "method": "percentile"},
    "studentize_reps",
    [0, -3, 5, np.int64(7)],
)
ROWS += good_value_rows(
    "conf_int-studentize_reps-std-err",
    CI,
    {**CI_BASE, "method": "studentized", "std_err_func": std_err_fn},
    "studentize_reps",
    [0, -3],
)

# ---- reset
for _name, _cls in {
    "IID": lambda: IIDBootstrap(Y, seed=0),
    "Indep": lambda: IndependentSamplesBootstrap(Y, Y[:20], seed=0),
}.items():
    ROWS += bad_value_rows(
        f"{_name}-reset",
        lambda use_seed, _cls=_cls: _cls().reset(use_seed),
        {},
        "use_seed",
        BOOL_BAD,
        rule=r"True or False",
    )
    ROWS += good_value_rows(
        f"{_name}-reset",
        lambda use_seed, _cls=_cls: _cls().reset(use_seed),
        {},
        "use_seed",
        BOOL_GOOD,
    )

# ---- the generator and state properties


def _set_generator(value: Any) -> None:
    IIDBootstrap(Y, seed=0).generator = value


def _set_state(value: Any, seed: Any = 0) -> None:
    IIDBootstrap(Y, seed=seed).state = value


for _i, _value in enumerate([3, None, "abc", 2.5, [1]]):
    ROWS.append(
        Row(
            "generator",
            _set_generator,
            {"value": _value},
            TypeError,
            r"^generator must be an instance of Generator or RandomState; got ",
            id=f"generator-bad-{_i}",
            name="generator",
        )
    )
    ROWS.append(
        Row(
            "state",
            _set_state,
            {"value": _value},
            TypeError,
            r"^state must be an instance of Mapping; got ",
            id=f"state-bad-{_i}",
            name="state",
        )
    )
    ROWS.append(
        Row(
            "state-randomstate",
            lambda value: _set_state(value, RandomState(1)),
            {"value": _value},
            TypeError,
            r"^state must be an instance of tuple or Mapping; got ",
            id=f"state-rs-bad-{_i}",
            name="state",
        )
    )
ROWS += [
    Row(
        "generator-good",
        _set_generator,
        {"value": value},
        None,
        id=f"generator-good-{i}",
    )
    for i, value in enumerate([np.random.default_rng(3), RandomState(3)])
]

# ---- optimal_block_length
OBL_BASE = {"x": Y}
OBL_DATA_BAD: list[tuple[str, Any, type[Exception], str]] = [
    ("nan", np.r_[np.nan, Y[1:]], ValueError, r"1 NaN"),
    ("all-nan", np.full(NOBS, np.nan), ValueError, r"100 NaN"),
    ("inf", np.r_[np.inf, Y[1:]], ValueError, r"1 inf"),
    ("-inf", np.r_[-np.inf, Y[1:]], ValueError, r"1 inf"),
    ("nan-frame", pd.DataFrame(np.c_[Y, np.r_[np.nan, Y[1:]]]), ValueError, r"1 NaN"),
    ("nan-series", pd.Series(np.r_[Y[:-1], np.nan]), ValueError, r"1 NaN"),
    ("nan-list", [np.nan, *Y[1:].tolist()], ValueError, r"1 NaN"),
    ("empty", np.array([]), ValueError, r"shape \(0,\)"),
    ("scalar", 5.0, ValueError, r"shape \(\)"),
    ("short-5", Y[:5], ValueError, r"shape \(5,\)"),
    ("short-10", Y[:10], ValueError, r"shape \(10,\)"),
    ("short-frame", pd.DataFrame(Y2[:10]), ValueError, r"shape \(10, 2\)"),
    ("one-row", Y2[:1], ValueError, r"shape \(1, 2\)"),
    ("no-columns", np.zeros((NOBS, 0)), ValueError, r"shape \(100, 0\)"),
    ("3d", np.zeros((NOBS, 2, 2)), ValueError, r"shape \(100, 2, 2\)"),
    ("None", None, TypeError, r"None \(NoneType\)"),
    ("str", "abc", TypeError, r"'abc' \(str\)"),
    ("str-list", ["a"] * 20, TypeError, r"\(list\)"),
    ("complex", np.ones(20, dtype=complex), TypeError, r"\(ndarray\)"),
    ("dates", pd.date_range("2000-01-01", periods=20), TypeError, r"\(DatetimeIndex\)"),
    ("none-object", np.array([None] * 20, dtype=object), TypeError, r"\(ndarray\)"),
]
for _label, _value, _exc, _got in OBL_DATA_BAD:
    ROWS.append(
        Row(
            "optimal_block_length",
            optimal_block_length,
            {"x": _value},
            _exc,
            rf"^x must be an? .*; got .*{_got}",
            id=f"optimal_block_length-x-{_label}",
            name="x",
        )
    )
OBL_GOOD = {
    "array": Y,
    "11-obs": Y[:11],
    "list": Y[:20].tolist(),
    "series": pd.Series(Y, name="s"),
    "frame": pd.DataFrame(Y2, columns=["a", "b"]),
    "2d": Y2,
    "column": Y[:, None],
    "int": np.arange(NOBS) % 7,
    "float32": Y.astype(np.float32),
    "bool": np.arange(NOBS) % 3 == 0,
    "duplicate-columns": np.c_[Y, Y],
}
for _label, _value in OBL_GOOD.items():
    ROWS.append(
        Row(
            "optimal_block_length",
            optimal_block_length,
            {"x": _value},
            None,
            id=f"optimal_block_length-x-good-{_label}",
        )
    )


@pytest.mark.filterwarnings("ignore:The conditions to reuse:RuntimeWarning")
@pytest.mark.parametrize("row", to_params(ROWS))
def test_validation(row: Row) -> None:
    run_row(row)


# ---- checks that a row cannot make


def test_bootstrap_validates_when_called() -> None:
    # bootstrap is lazy, but reps is checked when it is called
    bs = IIDBootstrap(Y, seed=0)
    with pytest.raises(
        ValueError, match=r"^reps must be an integer >= 1; got 0 \(int\)"
    ):
        bs.bootstrap(0)
    with pytest.raises(TypeError, match=r"^reps must be an integer >= 1; got 2.5"):
        bs.bootstrap(2.5)
    assert len(list(bs.bootstrap(np.int64(4)))) == 4


def test_moving_block_larger_than_data() -> None:
    # The blocks do not wrap, so there is no valid block when the block is longer
    # than the data. Creating the bootstrap is allowed (the kernel of the
    # multiple comparisons uses it), sampling is not
    bs = MovingBlockBootstrap(NOBS + 1, Y, seed=0)
    msg = rf"^block_size must be an integer in \[1, {NOBS}\]; got {NOBS + 1} \(int\)"
    with pytest.raises(ValueError, match=msg):
        bs.bootstrap(3)
    with pytest.raises(ValueError, match=msg):
        bs.apply(mean_fn, 3)
    with pytest.raises(ValueError, match=msg):
        bs.conf_int(mean_fn, 20)
    # The circular and stationary bootstraps wrap around
    assert CircularBlockBootstrap(NOBS + 1, Y, seed=0).apply(mean_fn, 3).shape == (3, 1)
    assert StationaryBootstrap(NOBS + 1, Y, seed=0).apply(mean_fn, 3).shape == (3, 1)
    assert MovingBlockBootstrap(NOBS, Y, seed=0).apply(mean_fn, 3).shape == (3, 1)


def test_invalid_values_do_not_change_the_bootstrap() -> None:
    bs = IIDBootstrap(Y, seed=7)
    state = bs.state
    for call in (
        lambda: bs.apply(mean_fn, 0),
        lambda: bs.conf_int(mean_fn, 20, size=1.5),
        lambda: bs.cov(mean_fn, 20, recenter="yes"),
        lambda: bs.var(None),
    ):
        with pytest.raises((TypeError, ValueError)):
            call()
    assert bs.state == state


def test_stored_values_are_python_types() -> None:
    bs = CircularBlockBootstrap(np.int64(5), Y, seed=np.int64(3))
    assert type(bs.block_size) is int
    sb = StationaryBootstrap(np.int32(4), Y)
    assert type(sb.block_size) is int
    assert StationaryBootstrap(2.5, Y).block_size == 2.5
    assert type(StationaryBootstrap(np.float32(2.5), Y).block_size) is float


def test_data_are_used_as_given() -> None:
    # The data are neither converted nor checked for NaN: the function of the user
    # decides
    data = np.r_[np.nan, np.arange(1, NOBS, dtype=float)]
    seen = []

    def func(x: Any) -> float:
        seen.append(x)
        return float(np.nanmean(x))

    bs = IIDBootstrap(data, seed=0)
    assert bs.pos_data[0] is data
    out = bs.apply(func, 3)
    assert np.isfinite(out).all()
    assert np.isnan(seen[0]).sum() == 1
    ints = np.arange(NOBS)
    seen.clear()
    IIDBootstrap(ints, seed=0).apply(func, 2)
    assert all(arr.dtype == ints.dtype for arr in seen)
    objects = np.array(["a", "b"] * 50, dtype=object)
    result = IIDBootstrap(objects, seed=0).apply(lambda x: float((x == "a").mean()), 3)
    assert result.shape == (3, 1)


def test_studentize_reps_ignored_unless_used() -> None:
    bs = IIDBootstrap(Y, seed=1)
    base = bs.conf_int(mean_fn, 30, method="percentile", studentize_reps=0)
    again = IIDBootstrap(Y, seed=1).conf_int(mean_fn, 30, method="percentile")
    np.testing.assert_allclose(base, again)
    with pytest.raises(
        ValueError, match=r"^studentize_reps must be an integer >= 1; got 0 \(int\)"
    ):
        bs.conf_int(mean_fn, 30, method="studentized", studentize_reps=0)


def test_extra_kwargs_checked_when_results_are_reused() -> None:
    bs = IIDBootstrap(y=Y, seed=1)
    assert bs.conf_int(mean_kw, 20).shape == (2, 1)
    assert bs.conf_int(mean_kw, 20, reuse=True).shape == (2, 1)
    with pytest.raises(ValueError, match=r"^extra_kwargs must be a dict with str keys"):
        bs.conf_int(mean_kw, 20, reuse=True, extra_kwargs={"y": 1})


def test_seed_variants_are_equivalent() -> None:
    first = IIDBootstrap(Y, seed=np.int64(5)).apply(mean_fn, 5)
    second = IIDBootstrap(Y, seed=5).apply(mean_fn, 5)
    np.testing.assert_array_equal(first, second)
    gen = np.random.default_rng(5)
    assert isinstance(IIDBootstrap(Y, seed=gen).generator, Generator)
    assert IIDBootstrap(Y, seed=gen).generator is gen


def test_optimal_block_length_minimum_and_names() -> None:
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = optimal_block_length(rng.standard_normal(11))
    assert np.isfinite(out.to_numpy()).all()
    with pytest.raises(ValueError, match=r"at least 11 rows"):
        optimal_block_length(rng.standard_normal(10))
    frame = pd.DataFrame(rng.standard_normal((30, 2)), columns=["a", "b"])
    assert list(optimal_block_length(frame).index) == ["a", "b"]
    assert list(optimal_block_length(pd.Series(Y, name="s")).index) == ["s"]
    assert list(optimal_block_length(Y2).index) == [0, 1]
    assert list(optimal_block_length(Y).columns) == ["stationary", "circular"]
