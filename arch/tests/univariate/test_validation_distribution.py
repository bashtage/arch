"""Input validation of the distributions in arch.univariate.distribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.tests.validation_table import (
    BOOL_BAD,
    BOOL_GOOD,
    FLOAT_BAD,
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
from arch.univariate.distribution import (
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
)

# The class and valid shape parameters of each distribution
DISTRIBUTIONS: list[tuple[type, tuple[float, ...]]] = [
    (Normal, ()),
    (StudentsT, (8.0,)),
    (GeneralizedError, (1.5,)),
    (SkewStudent, (8.0, 0.1)),
]
RESIDS = np.linspace(-2.0, 2.0, 25)
SIGMA2 = np.linspace(0.5, 1.5, 25)
NAN_FREE_DATA_BAD: list[tuple[Any, type[Exception]]] = [
    ("abc", TypeError),
    (["a", "b"], TypeError),
    (None, TypeError),
    ([1.0, None], TypeError),
    (1j, TypeError),
    (np.ones((3, 2)), ValueError),
    (np.ones((2, 2, 2)), ValueError),
    (pd.DataFrame(np.ones((3, 2))), ValueError),
]
DATA_GOOD: list[Any] = [
    np.linspace(-1, 1, 7),
    [0.1, 0.2, 0.3],
    (0.1, 0.2),
    np.arange(4),
    pd.Series([0.1, 0.2, 0.3]),
    np.linspace(-1, 1, 7)[:, None],
]
SIZE_BAD: list[tuple[Any, type[Exception]]] = [
    (-1, ValueError),
    (np.int64(-5), ValueError),
    ((3, -1), ValueError),
    ([2, -2], ValueError),
    (2.5, TypeError),
    (5.0, TypeError),
    (True, TypeError),
    (np.True_, TypeError),
    ("3", TypeError),
    (None, TypeError),
    ((2, 2.5), TypeError),
    ((True,), TypeError),
    (np.ones(3), TypeError),
]
SIZE_GOOD: list[Any] = [
    0,
    5,
    np.int64(5),
    np.uint8(5),
    (2, 3),
    [2, 3],
    (np.int32(2), 3),
    (0, 3),
]
Z_BAD = [(v, e) for v, e in FLOAT_BAD if not (isinstance(v, float) and np.isinf(v))]


def make(cls: type, shape: tuple[float, ...], method: str) -> Callable[..., Any]:
    """Call a method of a new distribution, with valid parameters by default."""

    def func(**kwargs: Any) -> Any:
        dist = cls(seed=0)
        if method != "simulate" and method not in ("bounds", "starting_values"):
            kwargs.setdefault("parameters", np.array(shape))
        return getattr(dist, method)(**kwargs)

    return func


def make_simulator(cls: type, shape: tuple[float, ...]) -> Callable[..., Any]:
    """Call the simulator of a distribution with valid parameters"""

    def func(**kwargs: Any) -> Any:
        return cls(seed=0).simulate(np.array(shape))(**kwargs)

    return func


def bad_parameters(shape: tuple[float, ...]) -> list[tuple[Any, type[Exception]]]:
    """Parameters that no method accepts"""
    n = len(shape)
    bad: list[tuple[Any, type[Exception]]] = [
        ("abc", TypeError),
        ([1.0, "a"], TypeError),
        ([np.nan] * max(n, 1), ValueError),
        ([np.inf] * max(n, 1), ValueError),
        ([8.0] * (n + 1), ValueError),
        (np.ones((n + 1, 2)), ValueError),
    ]
    if n:
        bad += [(None, ValueError), ([], ValueError)]
    else:
        bad += [(1.0, ValueError)]
    return bad


def good_parameters(shape: tuple[float, ...]) -> list[Any]:
    """Parameters that every method accepts"""
    good: list[Any] = [shape, list(shape), np.array(shape)]
    good.append(np.array(shape, dtype=np.float32))
    if not shape:
        good += [None, []]
    if len(shape) == 1:
        good += [shape[0], np.float64(shape[0])]
    return good


def build_rows(cls: type, shape: tuple[float, ...]) -> list[Row]:
    """Build the rows of one distribution"""
    name = cls.__name__
    rows: list[Row] = []

    # seed
    def construct(**kwargs: Any) -> Any:
        return cls(**kwargs)

    rows += bad_value_rows(f"{name}.seed", construct, {}, "seed", SEED_BAD)
    rows += good_value_rows(f"{name}.seed", construct, {}, "seed", [None, *SEED_GOOD])

    # bounds and starting_values: NaN and inf in the residuals are not checked
    # by bounds, since they are not used
    bounds = make(cls, shape, "bounds")
    rows += bad_value_rows(
        f"{name}.bounds", bounds, {"resids": RESIDS}, "resids", NAN_FREE_DATA_BAD
    )
    rows += good_value_rows(
        f"{name}.bounds",
        bounds,
        {"resids": RESIDS},
        "resids",
        [*DATA_GOOD, np.array([1.0, np.nan]), np.array([np.inf, 1.0]), np.array([])],
    )
    start = make(cls, shape, "starting_values")
    rows += bad_value_rows(
        f"{name}.starting_values",
        start,
        {"std_resid": RESIDS},
        "std_resid",
        [
            *NAN_FREE_DATA_BAD,
            ([1.0, np.nan], ValueError),
            ([np.inf, 1.0], ValueError),
            (np.array([]), ValueError),
        ],
    )
    rows += good_value_rows(
        f"{name}.starting_values", start, {"std_resid": RESIDS}, "std_resid", DATA_GOOD
    )

    # loglikelihood: only individual and the number of parameters are checked
    ll = make(cls, shape, "loglikelihood")
    base = {"resids": RESIDS, "sigma2": SIGMA2, "individual": False}
    rows += bad_value_rows(f"{name}.loglikelihood", ll, base, "individual", BOOL_BAD)
    rows += good_value_rows(f"{name}.loglikelihood", ll, base, "individual", BOOL_GOOD)
    wrong_lengths = [[0.5] * (len(shape) + 1), [0.5] * (len(shape) + 2)]
    if shape:
        wrong_lengths.append([])
    for i, value in enumerate(wrong_lengths):
        rows.append(
            Row(
                f"{name}.loglikelihood",
                ll,
                {**base, "parameters": value},
                ValueError,
                rf"^parameters must have {len(shape)} elements; got ",
                id=f"{name}.loglikelihood-parameters-length-{i}",
                name="parameters",
                form=False,
            )
        )
    if shape:
        for i, value in enumerate(["abc", [1.0, "a"], None]):
            rows.append(
                Row(
                    f"{name}.loglikelihood",
                    ll,
                    {**base, "parameters": value},
                    TypeError,
                    r"^parameters must",
                    id=f"{name}.loglikelihood-parameters-type-{i}",
                    name="parameters",
                    form=False,
                )
            )
    rows += good_value_rows(
        f"{name}.loglikelihood", ll, base, "parameters", good_parameters(shape)
    )

    # simulate and the simulator
    sim = make(cls, shape, "simulate")
    for i, (value, exc) in enumerate(bad_parameters(shape)):
        if value is None:
            # None is not converted to NaN: a TypeError, unless there are no
            # shape parameters and nothing is read
            exc = TypeError
            if not shape:
                continue
        rows.append(
            Row(
                f"{name}.simulate",
                sim,
                {"parameters": value},
                exc,
                r"^parameters must",
                id=f"{name}.simulate-parameters-{i}",
                name="parameters",
                form=False,
            )
        )
    rows += good_value_rows(
        f"{name}.simulate",
        sim,
        {},
        "parameters",
        [shape, list(shape), np.array(shape), np.array(shape, dtype=np.float32)],
    )
    simulator = make_simulator(cls, shape)
    rows += bad_value_rows(
        f"{name}.simulator",
        simulator,
        {},
        "size",
        [(v, e) for v, e in SIZE_BAD if not isinstance(v, (tuple, list))],
    )
    for i, (value, exc) in enumerate(SIZE_BAD):
        if isinstance(value, (tuple, list)):
            rows.append(
                Row(
                    f"{name}.simulator",
                    simulator,
                    {"size": value},
                    exc,
                    r"^size\[\d\] must be an integer >= 0; got ",
                    id=f"{name}.simulator-size-element-{i}",
                    form=False,
                )
            )
    rows += good_value_rows(f"{name}.simulator", simulator, {}, "size", SIZE_GOOD)

    # cdf and ppf: the values
    cdf = make(cls, shape, "cdf")
    ppf = make(cls, shape, "ppf")
    value_bad: list[tuple[Any, type[Exception]]] = [
        ("abc", TypeError),
        (["a", "b"], TypeError),
        (None, TypeError),
        (1j, TypeError),
        ([0.5, np.nan], ValueError),
        (np.nan, ValueError),
    ]
    value_good: list[Any] = [0.5, np.float32(0.5), np.array([0.25, 0.75]), [0.1, 0.9]]
    value_good.append(np.array(0.4))
    cdf_bad = [*value_bad]
    cdf_good = [*value_good, np.inf, -np.inf, [-np.inf, 0.0, np.inf], 3, -3]
    if cls in (SkewStudent, GeneralizedError):
        cdf_bad.append((np.ones((3, 2)), ValueError))
    else:
        cdf_good.append(np.array([[0.2, 0.3], [0.4, 0.5]]))
    rows += bad_value_rows(f"{name}.cdf", cdf, {"resids": 0.5}, "resids", cdf_bad)
    rows += good_value_rows(f"{name}.cdf", cdf, {"resids": 0.5}, "resids", cdf_good)
    ppf_bad = [
        *value_bad,
        ([0.5, np.inf], ValueError),
        (-0.1, ValueError),
        (1.01, ValueError),
        (np.array([0.2, -1e-9]), ValueError),
        (np.array([0.2, 3.0]), ValueError),
    ]
    ppf_good = [
        *value_good,
        0,
        1,
        [0.0, 1.0],
        np.array([[0.2, 0.3], [0.4, 0.5]]),
    ]
    rows += bad_value_rows(f"{name}.ppf", ppf, {"pits": 0.5}, "pits", ppf_bad)
    rows += good_value_rows(f"{name}.ppf", ppf, {"pits": 0.5}, "pits", ppf_good)

    # parameters of cdf, ppf, moment and partial_moment
    bases = {
        "cdf": {"resids": 0.5},
        "ppf": {"pits": 0.5},
        "moment": {"n": 2},
        "partial_moment": {"n": 2, "z": 0.5},
    }
    for method, base in bases.items():
        func = make(cls, shape, method)
        for i, (value, exc) in enumerate(bad_parameters(shape)):
            rows.append(
                Row(
                    f"{name}.{method}",
                    func,
                    {**base, "parameters": value},
                    exc,
                    r"^parameters must",
                    id=f"{name}.{method}-parameters-{i}",
                    name="parameters",
                    form=False,
                )
            )
        if shape:
            for i, value in enumerate(
                [[-5.0] * len(shape), [1e6] * len(shape), [1.0] * len(shape)]
            ):
                rows.append(
                    Row(
                        f"{name}.{method}",
                        func,
                        {**base, "parameters": value},
                        ValueError,
                        r"^parameters\[\d\] \(\w+\) must be a finite float in \[",
                        id=f"{name}.{method}-parameters-bounds-{i}",
                        form=False,
                    )
                )
        rows += good_value_rows(
            f"{name}.{method}", func, base, "parameters", good_parameters(shape)
        )

    # n and z
    for method in ("moment", "partial_moment"):
        func = make(cls, shape, method)
        base = {"n": 2} if method == "moment" else {"n": 2, "z": 0.5}
        rows += bad_value_rows(f"{name}.{method}", func, base, "n", INT_BAD)
        rows += good_value_rows(
            f"{name}.{method}", func, base, "n", [0, 1, *INT_GOOD[:3]]
        )
    func = make(cls, shape, "partial_moment")
    base = {"n": 2, "z": 0.5}
    rows += bad_value_rows(f"{name}.partial_moment", func, base, "z", Z_BAD)
    rows += good_value_rows(
        f"{name}.partial_moment",
        func,
        base,
        "z",
        [0, 1, -2.5, np.float32(0.5), np.int64(1)],
    )
    # inf is a valid bound, but the moments of order 2 and more are 0 * inf
    rows += good_value_rows(
        f"{name}.partial_moment-inf",
        func,
        {"n": 0, "z": 0.5},
        "z",
        [np.inf, -np.inf],
    )
    return rows


ROWS: list[Row] = []
for _cls, _shape in DISTRIBUTIONS:
    ROWS += build_rows(_cls, _shape)


def _range_row(
    cls: type, value: Any, label: str, rule: str, i: int, exc: type[Exception] | None
) -> Row:
    return Row(
        f"{cls.__name__}.simulate-range",
        lambda **kw: cls().simulate(**kw),
        {"parameters": value},
        exc,
        rf"^The {label} parameter must be {rule}; got " if exc else None,
        id=f"{cls.__name__}.simulate-range-{i}",
        name=f"The {label} parameter",
    )


# The ranges that simulate checks, which are wider than the bounds of the
# estimation (for example 2.05 for the degrees of freedom of a t)
RANGE_CASES: list[tuple[type, Any, str, str, type[Exception] | None]] = [
    (StudentsT, [2.0], "shape", "larger than 2", ValueError),
    (StudentsT, [1.5], "shape", "larger than 2", ValueError),
    (StudentsT, [0.0], "shape", "larger than 2", ValueError),
    (StudentsT, 2, "shape", "larger than 2", ValueError),
    (StudentsT, [2.01], "shape", "larger than 2", None),
    (GeneralizedError, [1.0], "shape", "larger than 1", ValueError),
    (GeneralizedError, [0.5], "shape", "larger than 1", ValueError),
    (GeneralizedError, [-1.0], "shape", "larger than 1", ValueError),
    (GeneralizedError, [1.01], "shape", "larger than 1", None),
    (SkewStudent, [2.0, 0.0], "shape", "larger than 2", ValueError),
    (SkewStudent, [1.0, 0.0], "shape", "larger than 2", ValueError),
    (SkewStudent, [8.0, 1.5], "skew", "smaller than 1 in absolute value", ValueError),
    (SkewStudent, [8.0, -1.01], "skew", "smaller than 1 in absolute value", ValueError),
    (SkewStudent, [2.01, 0.0], "shape", "larger than 2", None),
    (SkewStudent, [8.0, 1.0], "skew", "", None),
    (SkewStudent, [8.0, -1], "skew", "", None),
]
ROWS += [
    _range_row(c, v, la, r, i, e) for i, (c, v, la, r, e) in enumerate(RANGE_CASES)
]


@pytest.mark.parametrize("row", to_params(ROWS))
def test_validation(row):
    run_row(row)


def test_bounds_message_names_the_parameter():
    # The message used the first letter of the name of the distribution
    with pytest.raises(ValueError, match=r"^parameters\[0\] \(nu\) must be"):
        StudentsT().ppf([0.5], [1.0])
    with pytest.raises(ValueError, match=r"^parameters\[1\] \(lambda\) must be"):
        SkewStudent().ppf([0.5], [8.0, 2.0])
    with pytest.raises(ValueError, match=r"^parameters\[0\] \(eta\) must be"):
        SkewStudent().ppf([0.5], [1.0, 0.0])
    with pytest.raises(ValueError, match=r"^parameters\[0\] \(nu\) must be"):
        GeneralizedError().cdf([0.5], [1.0])


def test_negative_order_is_nan():
    # The documented result for a negative order is kept
    for cls, shape in DISTRIBUTIONS:
        dist = cls()
        assert np.isnan(dist.moment(-1, shape))
        assert np.isnan(dist.partial_moment(-1, 0.0, shape))


def test_values_unchanged_by_the_checks():
    # The standard normal and the standardized t, GED and skew t all have zero
    # mean and unit variance
    for cls, shape in DISTRIBUTIONS:
        dist = cls()
        assert_allclose(dist.moment(0, shape), 1.0, atol=1e-12)
        assert_allclose(dist.moment(1, shape), 0.0, atol=1e-10)
        assert_allclose(dist.moment(2, shape), 1.0, rtol=1e-10)
        assert_allclose(dist.partial_moment(0, np.inf, shape), 1.0, atol=1e-10)
        assert_allclose(dist.partial_moment(2, 1e3, shape), 1.0, rtol=1e-7)
    assert_allclose(Normal().partial_moment(0, 0.0), 0.5)
    assert_allclose(Normal().cdf(0.0), 0.5)
    assert_allclose(Normal().ppf(0.5), 0.0, atol=1e-15)
    assert isinstance(Normal().ppf(0.5), float)
    assert_allclose(StudentsT().ppf([0, 1], [8.0]), [-np.inf, np.inf])
    assert_allclose(StudentsT().cdf([-np.inf, np.inf], [8.0]), [0.0, 1.0])


def test_simulator_shape_and_seed():
    for cls, shape in DISTRIBUTIONS:
        draws = cls(seed=np.int64(4)).simulate(shape)((np.int32(5), 2))
        assert draws.shape == (5, 2)
        assert cls(seed=4).simulate(shape)(0).shape == (0,)
        first = cls(seed=7).simulate(shape)(10)
        second = cls(seed=np.uint8(7)).simulate(shape)(10)
        assert_allclose(first, second)
