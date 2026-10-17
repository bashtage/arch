"""
The data types of the input data: float32 is promoted, float16 and complex are rejected.

* ``test_float32_*``: float32 data are converted to float64 on entry and every
  calculation is done in float64. The result for float32 data is the result for the same
  values stored as float64, to rounding, which a calculation in float32 is not (it
  differs by about 1e-7 for well scaled data and by much more for data with a large mean,
  or fails: the rank check of the ADF regression with a trend fails for 5000
  observations in float32).
* ``test_float16_*`` and ``test_complex_*``: every case in ``test_integer_inputs`` is
  run with float16 and complex data and must raise a ``TypeError``, except the Hermitian
  functions that accept complex data.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.tests import test_integer_inputs as integer_inputs
from arch.tests.test_integer_inputs import CASES, Case
from arch.tests.test_validation_dtypes import TARGETS, Target
from arch.tests.validation_table import ids_for

CASE_IDS = [case_.name for case_ in CASES]
# The cases with integers near 2**32, which a float32 cannot hold
NOT_FLOAT32 = {"large_values_covariance", "large_values_unit_root"}
# The cases of the functions that are Hermitian and accept complex data
ACCEPT_COMPLEX = {
    "newey_west_function",
    "large_values_covariance",
    "add_trend_function",
}


def _cast(arrays: dict[str, np.ndarray], dtype: type) -> dict[str, Any]:
    # Integers near 2**32 overflow float16, which is still float16 data
    with np.errstate(over="ignore"):
        return {key: value.astype(dtype) for key, value in arrays.items()}


@pytest.mark.parametrize("case_", CASES, ids=CASE_IDS)
def test_float32_matches_float64(case_: Case) -> None:
    if case_.name in NOT_FLOAT32:
        pytest.skip("integers near 2**32 are not exactly representable in float32")
    arrays = case_.arrays()
    desired = case_.run(dict(arrays))
    actual = case_.run(_cast(arrays, np.float32))
    integer_inputs.assert_same(actual, desired, case_.name)


@pytest.mark.parametrize("case_", CASES, ids=CASE_IDS)
def test_float16_raises(case_: Case) -> None:
    with pytest.raises(TypeError, match=r"float16 data with shape"):
        case_.run(_cast(case_.arrays(), np.float16))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128], ids=["c64", "c128"])
@pytest.mark.parametrize("case_", CASES, ids=CASE_IDS)
def test_complex_raises_or_is_hermitian(case_: Case, dtype: type) -> None:
    data = _cast(case_.arrays(), dtype)
    if case_.name in ACCEPT_COMPLEX:
        case_.run(data)
        return
    with pytest.raises(TypeError, match=r"complex(64|128) data with shape"):
        case_.run(data)


# ---------------------------------------------------------------------------------------
# float32 data that are not exactly representable: the data are rounded to float32 first,
# and the result must be the result of the float64 calculation with the rounded values
# ---------------------------------------------------------------------------------------
WALKS = {
    "ADF",
    "DFGLS",
    "PhillipsPerron",
    "KPSS",
    "VarianceRatio",
    "ZivotAndrews",
    "auto_bandwidth",
    "engle_granger",
    "phillips_ouliaris",
    "DynamicOLS",
    "FullyModifiedOLS",
    "CanonicalCointegratingReg",
}


def _float_arrays(target: Target) -> dict[str, np.ndarray]:
    """Random data with the shapes of the arrays of the target"""
    rng = np.random.default_rng(sum(map(ord, target.name)))
    out = {}
    for key, arr in target.arrays.items():
        values = rng.standard_normal(arr.shape)
        if target.name in WALKS:
            values = np.cumsum(values, axis=0)
        elif key == "sigma2":
            values = rng.uniform(0.5, 2.0, size=arr.shape)
        elif key == "x" and target.name == "add_trend":
            values = values + 10.0
        out[key] = values
    return out


# add_trend returns the data and the trend, the others a statistic or the parameters
FLOAT_TARGETS = [
    dataclasses.replace(target, arrays=_float_arrays(target)) for target in TARGETS
]


@pytest.mark.parametrize(
    "target", FLOAT_TARGETS, ids=ids_for([t.name for t in FLOAT_TARGETS])
)
def test_float32_is_computed_in_float64(target: Target) -> None:
    rounded = {k: v.astype(np.float32) for k, v in target.arrays.items()}
    as_float64 = {k: v.astype(np.float64) for k, v in rounded.items()}
    actual = target.func(**rounded, **target.other)
    desired = target.func(**as_float64, **target.other)
    # The same values go through the same calculation. A calculation in float32 would
    # differ by 1e-7 or more
    assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(desired, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
