import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.utility.cov import cov_nw


@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("lags", [0, 2])
def test_negative_covariance_axis(axis, demean, lags):
    data = np.arange(30.0).reshape(10, 3)
    centered = data - data.mean(axis=0) if demean else data
    expected = centered.T @ centered
    for lag in range(1, lags + 1):
        cross = centered[lag:].T @ centered[:-lag]
        expected += (1 - lag / (lags + 1)) * (cross + cross.T)
    expected /= data.shape[0]
    if axis == -1:
        data = data.T

    result = cov_nw(data, lags=lags, demean=demean, axis=axis)

    assert result.shape == (3, 3)
    assert_allclose(result, expected)


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("offset", [0, 1])
def test_covariance_axis_bounds(ndim, offset):
    data = np.ones((4,) * ndim)
    for axis in [ndim + offset, -ndim - 1 - offset]:
        with pytest.raises(ValueError, match="axis must be an integer in"):
            cov_nw(data, axis=axis)


def test_negative_axis_one_dimensional():
    data = np.arange(10.0)
    assert_allclose(cov_nw(data, axis=-1), np.var(data))
