"""
Table-driven tests of the input validation of MCS, SPA, StepM and RealityCheck.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from arch.bootstrap.multiple_comparison import MCS, SPA, RealityCheck, StepM
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

T = 30
K = 3
_RNG = np.random.default_rng(20241017)
LOSSES = _RNG.standard_normal((T, K)) + np.array([0.0, 0.1, 0.2])
BENCHMARK = LOSSES[:, 0].copy()
MODELS = LOSSES[:, 1:].copy()

MCS_BASE: dict[str, Any] = {"losses": LOSSES, "size": 0.1, "reps": 10, "seed": 0}
SPA_BASE: dict[str, Any] = {
    "benchmark": BENCHMARK,
    "models": MODELS,
    "reps": 10,
    "seed": 0,
}
STEPM_BASE: dict[str, Any] = {**SPA_BASE, "size": 0.1}


def make_mcs(**kwargs: Any) -> MCS:
    return MCS(**kwargs)


def make_spa(**kwargs: Any) -> SPA:
    return SPA(**kwargs)


def make_reality_check(**kwargs: Any) -> RealityCheck:
    return RealityCheck(**kwargs)


def make_stepm(**kwargs: Any) -> StepM:
    return StepM(**kwargs)


def computed_spa() -> SPA:
    spa = SPA(BENCHMARK, MODELS, reps=20, seed=0)
    spa.compute()
    return spa


def critical_values(**kwargs: Any) -> Any:
    return computed_spa().critical_values(**kwargs)


def better_models(**kwargs: Any) -> Any:
    return computed_spa().better_models(**kwargs)


def subset(**kwargs: Any) -> Any:
    return SPA(BENCHMARK, MODELS, reps=20, seed=0).subset(**kwargs)


def compute_mcs(**kwargs: Any) -> Any:
    mcs = MCS(**kwargs)
    mcs.compute()
    return mcs.pvalues


def compute_spa(**kwargs: Any) -> Any:
    spa = SPA(**kwargs)
    spa.compute()
    return spa.pvalues


def compute_stepm(**kwargs: Any) -> Any:
    stepm = StepM(**kwargs)
    stepm.compute()
    return stepm.superior_models


PROCEDURES: list[tuple[str, Callable[..., Any], dict[str, Any]]] = [
    ("MCS", make_mcs, MCS_BASE),
    ("SPA", make_spa, SPA_BASE),
    ("RealityCheck", make_reality_check, SPA_BASE),
    ("StepM", make_stepm, STEPM_BASE),
]
SPA_PROCEDURES = PROCEDURES[1:]

# Values outside (0, 1) for the size and the pvalue, which are open at both ends
OPEN_UNIT_BAD: list[tuple[Any, type[Exception]]] = [
    (0.0, ValueError),
    (1.0, ValueError),
    (-0.1, ValueError),
    (1.5, ValueError),
    (np.float32(2.0), ValueError),
    (0, ValueError),
    (1, ValueError),
]
OPEN_UNIT_GOOD: list[Any] = [*FLOAT_GOOD, 0.001, 0.999, np.float64(0.05)]

# Integers that are in the wrong range, for reps and for block_size
REPS_BAD: list[tuple[Any, type[Exception]]] = [
    *INT_BAD,
    (0, ValueError),
    (-1, ValueError),
    (np.int64(0), ValueError),
]
BLOCK_SIZE_BAD: list[tuple[Any, type[Exception]]] = [
    *[(v, e) for v, e in INT_BAD if v is not None],
    (0, ValueError),
    (-1, ValueError),
    (T + 1, ValueError),
    (np.int64(T + 1), ValueError),
]

ROWS: list[Row] = []

for _name, _func, _base in PROCEDURES:
    _label = _name
    ROWS += bad_value_rows(_label, _func, _base, "reps", REPS_BAD)
    ROWS += bad_value_rows(_label, _func, _base, "block_size", BLOCK_SIZE_BAD)
    ROWS += bad_value_rows(_label, _func, _base, "seed", SEED_BAD)
    ROWS += good_value_rows(_label, _func, _base, "reps", [*INT_GOOD, 1])
    ROWS += good_value_rows(
        _label, _func, _base, "block_size", [*INT_GOOD, None, 1, T, np.int64(T)]
    )
    ROWS += good_value_rows(_label, _func, _base, "seed", SEED_GOOD)

# size: MCS and StepM
for _name, _func, _base in (PROCEDURES[0], PROCEDURES[3]):
    ROWS += bad_value_rows(_name, _func, _base, "size", [*FLOAT_BAD, *OPEN_UNIT_BAD])
    ROWS += good_value_rows(_name, _func, _base, "size", OPEN_UNIT_GOOD)

# studentize and nested: SPA, RealityCheck and StepM
for _name, _func, _base in SPA_PROCEDURES:
    for _flag in ("studentize", "nested"):
        ROWS += bad_value_rows(_name, _func, _base, _flag, BOOL_BAD)
    ROWS += good_value_rows(_name, _func, _base, "studentize", BOOL_GOOD)
    ROWS += good_value_rows(
        _name, _func, {**_base, "studentize": True}, "nested", BOOL_GOOD
    )
    ROWS.append(
        Row(
            _name,
            _func,
            {**_base, "studentize": False, "nested": True},
            ValueError,
            r"^nested can only be True when studentize is True$",
            id=f"{_name}-nested-without-studentize",
            form=False,
        )
    )
    ROWS.append(
        Row(
            _name,
            _func,
            {**_base, "studentize": "no", "nested": True},
            TypeError,
            r"^studentize must be True or False; got 'no' \(str\)$",
            id=f"{_name}-studentize-checked-first",
            name="studentize",
        )
    )

# The data
for _name, _func, _base in PROCEDURES[1:]:
    ROWS += [
        Row(
            _name,
            _func,
            {**_base, "benchmark": np.ones((T, 2))},
            ValueError,
            r"^benchmark must be a 1-d array.*; got shape \(30, 2\) \(ndarray\)$",
            id=f"{_name}-benchmark-two-columns",
            name="benchmark",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": np.ones((T, 2, 2))},
            ValueError,
            r"^benchmark must be a 1-d array",
            id=f"{_name}-benchmark-3d",
            name="benchmark",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": BENCHMARK[:2], "models": MODELS[:2]},
            ValueError,
            r"^benchmark must be a 1-d array.*at least 3 elements; got shape \(2,\)",
            id=f"{_name}-benchmark-too-short",
            name="benchmark",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": BENCHMARK[:1], "models": MODELS[:1]},
            ValueError,
            r"^benchmark must be a 1-d array.*at least 3 elements",
            id=f"{_name}-benchmark-one-obs",
            name="benchmark",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": np.empty(0), "models": np.empty((0, 2))},
            ValueError,
            r"^benchmark must be a 1-d array.*at least 3 elements",
            id=f"{_name}-benchmark-empty",
            name="benchmark",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": [str(v) for v in BENCHMARK]},
            TypeError,
            r"^benchmark must .*float64",
            id=f"{_name}-benchmark-strings",
            form=False,
            name="benchmark",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": None},
            TypeError,
            r"^benchmark must be .*; got None \(NoneType\)",
            id=f"{_name}-benchmark-none",
            name="benchmark",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": np.r_[np.nan, BENCHMARK[1:]]},
            ValueError,
            r"^benchmark must be finite",
            id=f"{_name}-benchmark-nan",
            form=False,
        ),
        Row(
            _name,
            _func,
            {**_base, "models": MODELS[:-1]},
            ValueError,
            r"^models must be a 2-d array.*shape \(30, n\); got shape \(29, 2\)",
            id=f"{_name}-models-too-few-rows",
            name="models",
        ),
        Row(
            _name,
            _func,
            {**_base, "models": np.r_[MODELS, MODELS[:1]]},
            ValueError,
            r"^models must be a 2-d array.*shape \(30, n\); got shape \(31, 2\)",
            id=f"{_name}-models-too-many-rows",
            name="models",
        ),
        Row(
            _name,
            _func,
            {**_base, "models": np.empty((T, 0))},
            ValueError,
            r"^models must be a 2-d array with at least 1 column; got shape \(30, 0\)",
            id=f"{_name}-models-no-columns",
            name="models",
        ),
        Row(
            _name,
            _func,
            {**_base, "models": np.ones((T, 2, 2))},
            ValueError,
            r"^models must be a 2-d array",
            id=f"{_name}-models-3d",
            name="models",
        ),
        Row(
            _name,
            _func,
            {**_base, "models": [[str(v)] for v in BENCHMARK]},
            TypeError,
            r"^models must .*float64",
            id=f"{_name}-models-strings",
            form=False,
            name="models",
        ),
        Row(
            _name,
            _func,
            {**_base, "models": None},
            TypeError,
            r"^models must be .*; got None \(NoneType\)",
            id=f"{_name}-models-none",
            name="models",
        ),
        Row(
            _name,
            _func,
            {**_base, "models": np.where(MODELS > 10, MODELS, np.inf)},
            ValueError,
            r"^models must be finite",
            id=f"{_name}-models-inf",
            form=False,
        ),
    ]
    # Valid shapes
    ROWS += [
        Row(
            _name,
            _func,
            {**_base, "benchmark": BENCHMARK[:, None]},
            None,
            id=f"{_name}-bm-column",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": BENCHMARK[None, :]},
            None,
            id=f"{_name}-bm-row",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": pd.Series(BENCHMARK)},
            None,
            id=f"{_name}-bm-series",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": BENCHMARK.tolist()},
            None,
            id=f"{_name}-bm-list",
        ),
        Row(
            _name,
            _func,
            {**_base, "benchmark": BENCHMARK[:3], "models": MODELS[:3]},
            None,
            id=f"{_name}-three-obs",
        ),
        Row(
            _name,
            _func,
            {**_base, "models": MODELS[:, 0]},
            None,
            id=f"{_name}-models-1d",
        ),
        Row(
            _name,
            _func,
            {**_base, "models": MODELS[:, :1]},
            None,
            id=f"{_name}-models-one-column",
        ),
        Row(
            _name,
            _func,
            {**_base, "models": pd.DataFrame(MODELS)},
            None,
            id=f"{_name}-models-frame",
        ),
        Row(
            _name,
            _func,
            {
                **_base,
                "benchmark": (BENCHMARK * 10).astype(np.int64),
                "models": (MODELS * 10).astype(np.int32),
            },
            None,
            id=f"{_name}-integer-data",
        ),
    ]

ROWS += [
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": np.ones((T, K, 2))},
        ValueError,
        r"^losses must be a 2-d array.*; got shape \(30, 3, 2\) \(ndarray\)$",
        id="MCS-losses-3d",
        name="losses",
    ),
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": LOSSES[:, 0]},
        ValueError,
        r"^losses must be a 2-d array with at least 2 columns; got shape \(30, 1\) "
        r"\(ndarray\)$",
        id="MCS-losses-one-column",
        name="losses",
    ),
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": np.empty((0, K))},
        ValueError,
        r"^losses must be a 2-d array.*at least 1 row; got shape \(0, 3\)",
        id="MCS-losses-no-rows",
        name="losses",
    ),
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": [["a", "b"], ["c", "d"]]},
        TypeError,
        r"^losses must .*float64",
        id="MCS-losses-strings",
        form=False,
        name="losses",
    ),
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": None},
        TypeError,
        r"^losses must be .*; got None \(NoneType\)",
        id="MCS-losses-none",
        name="losses",
    ),
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": np.where(LOSSES > 2, np.nan, LOSSES) + np.nan},
        ValueError,
        r"^losses must be finite",
        id="MCS-losses-nan",
        form=False,
    ),
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": LOSSES[:1]},
        None,
        id="MCS-losses-one-row",
    ),
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": pd.DataFrame(LOSSES)},
        None,
        id="MCS-losses-frame",
    ),
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": LOSSES.tolist()},
        None,
        id="MCS-losses-list",
    ),
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": (LOSSES * 10).astype(np.int64)},
        None,
        id="MCS-losses-integer",
    ),
    Row(
        "MCS",
        make_mcs,
        {**MCS_BASE, "losses": LOSSES[:, :2]},
        None,
        id="MCS-losses-two-columns",
    ),
]

# Wrong values are found when the object is created, not in compute
ROWS += [
    Row(
        "MCS",
        compute_mcs,
        {**MCS_BASE, "size": float("nan")},
        ValueError,
        r"^size must be a finite float in \(0, 1\); got nan \(float\)$",
        id="MCS-compute-size-nan",
        name="size",
    ),
    Row(
        "StepM",
        compute_stepm,
        {**STEPM_BASE, "size": 2},
        ValueError,
        r"^size must be a finite float in \(0, 1\); got 2 \(int\)$",
        id="StepM-compute-size-2",
        name="size",
    ),
    Row(
        "SPA",
        compute_spa,
        {**SPA_BASE, "reps": 0},
        ValueError,
        r"^reps must be an integer >= 1; got 0 \(int\)$",
        id="SPA-compute-reps-0",
        name="reps",
    ),
    Row(
        "MCS",
        compute_mcs,
        {**MCS_BASE, "size": np.float32(0.25)},
        None,
        id="MCS-compute-ok",
    ),
    Row("SPA", compute_spa, SPA_BASE, None, id="SPA-compute-ok"),
    Row("StepM", compute_stepm, STEPM_BASE, None, id="StepM-compute-ok"),
]

# Methods of SPA
for _arg in ("pvalue",):
    for _name, _func in (
        ("critical_values", critical_values),
        ("better_models", better_models),
    ):
        ROWS += bad_value_rows(_name, _func, {}, _arg, [*FLOAT_BAD, *OPEN_UNIT_BAD])
        ROWS += good_value_rows(_name, _func, {}, _arg, OPEN_UNIT_GOOD)
        ROWS.append(
            Row(
                _name,
                lambda _f=_func, **kw: _f(**kw),
                {"pvalue": 0.05},
                None,
                id=f"{_name}-pvalue-default-value",
            )
        )
ROWS += [
    Row(
        "better_models",
        better_models,
        {"pvalue_type": value},
        exc,
        r"^pvalue_type must be one of 'lower', 'consistent' or 'upper'; got ",
        id=f"better_models-pvalue_type-{i}",
        name="pvalue_type",
    )
    for i, (value, exc) in enumerate(
        [(5, TypeError), (None, TypeError), ("zzz", ValueError), ([], TypeError)]
    )
]
ROWS += [
    Row(
        "better_models",
        better_models,
        {"pvalue_type": value},
        None,
        id=f"better_models-pvalue_type-ok-{i}-{value}",
    )
    for i, value in enumerate(["lower", "Consistent", "UPPER", np.str_("lower")])
]
# Arguments are checked before the call order
ROWS += [
    Row(
        "critical_values",
        lambda **kw: SPA(BENCHMARK, MODELS, reps=10, seed=0).critical_values(**kw),
        {"pvalue": 2.0},
        ValueError,
        r"^pvalue must be a finite float in \(0, 1\); got 2.0 \(float\)$",
        id="critical_values-pvalue-before-compute",
        name="pvalue",
    ),
    Row(
        "critical_values",
        lambda **kw: SPA(BENCHMARK, MODELS, reps=10, seed=0).critical_values(**kw),
        {},
        RuntimeError,
        r"compute must be called before pvalues are available",
        id="critical_values-before-compute",
        form=False,
    ),
]

# subset
ROWS += [
    Row(
        "subset",
        subset,
        {"selector": value},
        exc,
        r"^selector must be .*; got ",
        id=f"subset-{i}",
        name="selector",
    )
    for i, (value, exc) in enumerate(
        [
            ("abc", TypeError),
            (None, TypeError),
            (np.array([1, 0]), TypeError),
            (np.array([1.0, 0.0]), TypeError),
            (np.ones(3, dtype=bool), ValueError),
            (np.ones(1, dtype=bool), ValueError),
            (np.ones((2, 2), dtype=bool), ValueError),
            (np.zeros(2, dtype=bool), ValueError),
            ([False, False], ValueError),
            (np.ones(0, dtype=bool), ValueError),
        ]
    )
]
ROWS += [
    Row("subset", subset, {"selector": value}, None, id=f"subset-ok-{i}")
    for i, value in enumerate(
        [
            np.array([True, False]),
            [True, True],
            np.array([False, True]),
            pd.Series([True, False]),
            np.array([True, False])[:, None],
            np.array([np.True_, np.False_]),
        ]
    )
]

ROW_PARAMS = to_params(ROWS)


@pytest.mark.parametrize("row", ROW_PARAMS)
def test_validation(row: Row) -> None:
    run_row(row)


def test_validated_values_are_python_types() -> None:
    spa = SPA(
        BENCHMARK,
        MODELS,
        reps=np.int64(7),
        block_size=np.int32(3),
        studentize=np.True_,
        nested=np.False_,
        seed=np.int64(1),
    )
    assert type(spa.reps) is int
    assert type(spa.block_size) is int
    assert type(spa.studentize) is bool
    assert type(spa.nested) is bool
    stepm = StepM(BENCHMARK, MODELS, size=np.float32(0.5), reps=np.int16(5), seed=2)
    assert type(stepm.size) is float
    assert type(stepm.reps) is int
    mcs = MCS(LOSSES, np.float32(0.25), reps=np.uint8(5), block_size=np.int8(2))
    assert type(mcs.size) is float
    assert type(mcs.reps) is int
    assert type(mcs.block_size) is int


def test_row_benchmark_is_a_column() -> None:
    spa = SPA(BENCHMARK[None, :], MODELS, reps=10)
    assert spa.benchmark.shape == (T, 1)
    assert np.array_equal(np.asarray(spa.benchmark)[:, 0], BENCHMARK)


def test_stepm_shares_the_spa_data() -> None:
    stepm = StepM(
        pd.Series(BENCHMARK), pd.DataFrame(MODELS, columns=["a", "b"]), reps=10
    )
    assert stepm.benchmark is stepm.spa.benchmark
    assert stepm.models is stepm.spa.models
    assert list(stepm.models.columns) == ["a", "b"]


def test_subset_stores_the_selector() -> None:
    spa = SPA(BENCHMARK, MODELS, reps=20, seed=0)
    spa.subset([True, False])
    spa.compute()
    assert spa.better_models().ndim == 1
    spa.subset(np.array([True, True]))
    assert spa._selector.dtype == np.bool_
