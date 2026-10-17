from fractions import Fraction

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from arch.covariance.kernel import Bartlett
from arch.utility.cov import cov_nw


@pytest.mark.parametrize("dtype", [np.int16, np.uint8, np.int64, np.bool_])
@pytest.mark.parametrize("lags", [0, 2])
def test_uncentered_discrete_covariance(dtype, lags):
    data = np.arange(1, 13).reshape(6, 2) * 20
    if dtype == np.bool_:
        data = data % 60 == 0
    data = data.astype(dtype)
    original = data.copy()
    nobs, nvar = data.shape
    short_run = np.array(
        [
            [sum(int(row[i]) * int(row[j]) for row in data) / nobs for j in range(nvar)]
            for i in range(nvar)
        ]
    )
    strict = np.zeros((nvar, nvar))
    for lag in range(1, lags + 1):
        weight = Fraction(lags + 1 - lag, lags + 1)
        for i in range(nvar):
            for j in range(nvar):
                products = sum(
                    int(data[t, i]) * int(data[t - lag, j]) for t in range(lag, nobs)
                )
                strict[i, j] += float(weight * products / nobs)
    expected = short_run + strict + strict.T

    cov = Bartlett(data, bandwidth=lags, center=False).cov

    assert_allclose(cov.short_run, short_run)
    assert_allclose(cov.one_sided_strict, strict)
    assert_allclose(cov.long_run, expected)
    assert_allclose(cov_nw(data, lags=lags, demean=False), expected)
    assert_array_equal(data, original)


@pytest.mark.parametrize("lags", [0, 2])
def test_uncentered_integer_newey_west(lags):
    data = np.array([2**32 + 1, 2**32 + 2, 2**32 + 3], dtype=np.int64)
    nobs = len(data)
    expected = Fraction(sum(int(x) ** 2 for x in data), nobs)
    for lag in range(1, lags + 1):
        products = sum(int(data[t]) * int(data[t - lag]) for t in range(lag, nobs))
        expected += Fraction(2 * (lags + 1 - lag) * products, (lags + 1) * nobs)

    assert_allclose(cov_nw(data, lags=lags, demean=False), float(expected))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_preserve_floating_kernel_dtype(dtype):
    data = np.arange(12).reshape(6, 2).astype(dtype) / 10
    result = Bartlett(data, bandwidth=2, center=False).cov
    expected = data.T @ data / len(data)
    assert result.short_run.dtype == expected.dtype
    assert_array_equal(result.short_run, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("lags", [0, 2])
def test_preserve_inexact_newey_west(dtype, lags):
    data = np.arange(12).reshape(6, 2).astype(dtype) / 10
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * data[::-1]
    expected = data.T @ data
    for lag in range(1, lags + 1):
        gamma = data[lag:].T @ data[:-lag]
        expected += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    expected /= len(data)
    result = cov_nw(data, lags=lags, demean=False)
    assert result.dtype == expected.dtype
    assert_array_equal(result, expected)
