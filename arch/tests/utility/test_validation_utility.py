"""Input validation of arch.utility: array, timeseries, io and testing."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest

from arch.utility.array import as_float64, ensure1d, ensure2d
from arch.utility.timeseries import add_trend


@pytest.mark.parametrize(
    ("trend", "expected_columns"),
    [
        ("c", ["a", "b"]),
        ("ct", ["a", "b", "trend"]),
        ("ctt", ["a", "b", "trend", "quadratic_trend"]),
        ("t", ["a", "b", "trend"]),
    ],
)
def test_add_trend_frame_skip_constant(trend, expected_columns):
    # x has a constant column, so "skip" drops the constant of the trend. The
    # DataFrame branch used to name the dropped constant and fail on the shape.
    x = pd.DataFrame({"a": np.ones(4), "b": [1.0, 2.0, 4.0, 3.0]})
    result = add_trend(x, trend, has_constant="skip")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == expected_columns
    # Hand computed: the trend columns are 1, ..., 4 and their squares
    columns = {
        "trend": np.arange(1.0, 5.0),
        "quadratic_trend": np.arange(1.0, 5.0) ** 2,
    }
    for name in expected_columns[2:]:
        assert_array_equal(result[name].to_numpy(), columns[name])
    assert_array_equal(result.to_numpy(), add_trend(x.to_numpy(), trend))
    assert result.index.equals(x.index)


@pytest.mark.parametrize("func", [as_float64, ensure1d, ensure2d])
@pytest.mark.parametrize(
    "value",
    [None, [1.0, None], (None,), np.array([1.0, None], dtype=object), [[None]]],
    ids=["None", "list-with-None", "tuple-None", "object-with-None", "nested-None"],
)
def test_none_is_not_nan(func, value):
    # np.asarray(None) is an object array and astype(float) turns None into NaN
    with pytest.raises(TypeError, match=r"^x must .*None"):
        func(value, "x")


@pytest.mark.parametrize("func", [as_float64, ensure1d, ensure2d])
def test_nan_is_still_data(func):
    # NaN is a number and missing values in float data are NaN
    out = func(np.array([1.0, np.nan]), "x")
    assert np.isnan(out).sum() == 1
    out = func(np.array([1.0, None], dtype=float), "x")
    assert np.isnan(out).sum() == 1
