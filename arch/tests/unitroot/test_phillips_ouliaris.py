from typing import Tuple

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.iolib.summary import Summary

from arch.typing import NDArray
from arch.unitroot._phillips_ouliaris import (
    CriticalValueWarning,
    phillips_ouliaris_cv,
    phillips_ouliaris_pval,
)
from arch.unitroot.cointegration import phillips_ouliaris
from arch.utility.timeseries import add_trend


def z_tests(z: NDArray, lag: int, trend: str) -> Tuple[float, float]:
    z = add_trend(z, trend=trend)
    u = z
    if z.shape[1] > 1:
        delta = np.linalg.lstsq(z[:, 1:], z[:, 0], rcond=None)[0]
        u = z[:, 0] - z[:, 1:] @ delta
    alpha = (u[:-1].T @ u[1:]) / (u[:-1].T @ u[:-1])
    k = u[1:] - alpha * u[:-1]
    nobs = u.shape[0]
    one_sided_strict = 0.0
    for i in range(1, lag + 1):
        w = 1 - i / (lag + 1)
        one_sided_strict += 1 / nobs * w * k[i:].T @ k[:-i]
    u2 = u[:-1].T @ u[:-1]
    z = (alpha - 1) - nobs * one_sided_strict / u2
    z_a = nobs * z
    long_run = k.T @ k / nobs + 2 * one_sided_strict
    se = np.sqrt(long_run / u2)
    z_t = z / se
    print(u2, long_run, one_sided_strict, z)
    return z_a, z_t


def p_tests(z: NDArray, lag: int, trend: str) -> Tuple[float, float]:
    x, y = z[:, 1:], z[:, 0]
    nobs = x.shape[0]
    x = add_trend(x, trend=trend)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    u = y - x @ beta
    z_lead = z[1:]
    z_lag = add_trend(z[:-1], trend=trend)
    phi = np.linalg.lstsq(z_lag, z_lead, rcond=None)[0]
    xi = z_lead - z_lag @ phi

    omega = xi.T @ xi / nobs
    for i in range(1, lag + 1):
        w = 1 - i / (lag + 1)
        gamma = xi[i:].T @ xi[:-i] / nobs
        omega += w * (gamma + gamma.T)
    omega21 = omega[0, 1:]
    omega22 = omega[1:, 1:]
    omega112 = omega[0, 0] - np.squeeze(omega21.T @ np.linalg.inv(omega22) @ omega21)
    denom = u.T @ u / nobs
    p_u = nobs * omega112 / denom

    tr = add_trend(nobs=z.shape[0], trend=trend)
    if tr.shape[1]:
        z = z - tr @ np.linalg.lstsq(tr, z, rcond=None)[0]
    else:
        z = z - z[0]
    m_zz = z.T @ z / nobs
    p_z = nobs * (omega @ np.linalg.inv(m_zz)).trace()
    return p_u, p_z


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("test_type", ["Za", "Zt", "Pu", "Pz"])
@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "quadratic-spectral"])
@pytest.mark.parametrize("bandwidth", [None, 10])
@pytest.mark.parametrize("force_int", [True, False])
def test_smoke(trivariate_data, trend, test_type, kernel, bandwidth, force_int):
    y, x = trivariate_data
    res = phillips_ouliaris(
        y,
        x,
        trend=trend,
        test_type=test_type,
        kernel=kernel,
        bandwidth=bandwidth,
        force_int=force_int,
    )
    assert isinstance(res.stat, float)


def test_errors(trivariate_data):
    y, x = trivariate_data
    with pytest.raises(ValueError, match="kernel is not a known estimator."):
        phillips_ouliaris(y, x, kernel="fancy-kernel")
    with pytest.raises(ValueError, match="Unknown test_type: z-alpha."):
        phillips_ouliaris(y, x, test_type="z-alpha")


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("bandwidth", [0, 1, 10])
def test_z_test_direct(trivariate_data, trend, bandwidth):
    y, x = trivariate_data
    y = np.asarray(y)
    x = np.asarray(x)
    z = np.column_stack([y, x])
    ref_z_a, ref_z_t = z_tests(z, bandwidth, trend)
    z_a = phillips_ouliaris(y, x, trend=trend, bandwidth=bandwidth, test_type="Za")
    z_t = phillips_ouliaris(y, x, trend=trend, bandwidth=bandwidth, test_type="Zt")
    assert_allclose(z_a.stat, ref_z_a)
    assert_allclose(z_t.stat, ref_z_t)
    assert isinstance(z_a.summary(), Summary)
    assert isinstance(z_t.summary(), Summary)


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("bandwidth", [0, 1, 10])
def test_p_test_direct(trivariate_data, trend, bandwidth):
    y, x = trivariate_data
    y = np.asarray(y)
    x = np.asarray(x)
    z = np.column_stack([y, x])
    ref_p_u, ref_p_z = p_tests(z, bandwidth, trend)
    p_u = phillips_ouliaris(y, x, trend=trend, bandwidth=bandwidth, test_type="Pu")
    p_z = phillips_ouliaris(y, x, trend=trend, bandwidth=bandwidth, test_type="Pz")
    assert_allclose(p_u.stat, ref_p_u)
    assert_allclose(p_z.stat, ref_p_z)
    assert isinstance(p_u.summary(), Summary)
    assert isinstance(p_z.summary(), Summary)


def test_cv_exceptions():
    with pytest.raises(ValueError, match="test_type must be one of"):
        phillips_ouliaris_cv("unknown", "c", 2, 500)
    with pytest.raises(ValueError, match="trend must by one of:"):
        phillips_ouliaris_cv("Pu", "unknown", 2, 500)
    with pytest.raises(ValueError, match="The number of stochastic trends must"):
        phillips_ouliaris_cv("Pu", "ct", 25, 500)
    with pytest.warns(CriticalValueWarning):
        phillips_ouliaris_cv("Pu", "ct", 2, 10)


def test_pval_exceptions():
    with pytest.raises(ValueError, match="test_type must be one of"):
        phillips_ouliaris_pval(3.0, "unknown", "c", 2)
    with pytest.raises(ValueError, match="trend must by one of:"):
        phillips_ouliaris_pval(3.0, "Pu", "unknown", 2)
    with pytest.raises(ValueError, match="The number of stochastic trends must"):
        phillips_ouliaris_pval(3.0, "Pu", "ct", 25)


def test_pval_extremes():
    assert phillips_ouliaris_pval(3.0, "Zt", "n", 2) == 1.0
    assert phillips_ouliaris_pval(-3000.0, "Zt", "n", 2) == 0.0
    # Above and below tau-star
    above = phillips_ouliaris_pval(1.0, "Zt", "n", 2)
    below = phillips_ouliaris_pval(0.0, "Zt", "n", 2)
    assert above > below


def test_auto_bandwidth(trivariate_data):
    y, x = trivariate_data
    res = phillips_ouliaris(y, x)
    assert isinstance(res.summary(), Summary)
    assert int(res.bandwidth) != res.bandwidth
    res = phillips_ouliaris(y, x, force_int=True)
    assert int(res.bandwidth) == res.bandwidth


REFERENCE = (
    (
        "n",
        [
            [-4.017712, 0.0062, -31.26709, 0.0079],
            [-4.207778, 0.0033, -33.83294, 0.0046],
            [-4.218211, 0.0031, -33.87041, 0.0045],
        ],
    ),
    (
        "c",
        [
            [-29.56978, 0.0000, -1210.576, 0.0001],
            [-32.09233, 0.0000, -976.0578, 0.0001],
            [-31.93710, 0.0000, -986.2292, 0.0001],
        ],
    ),
    (
        "ct",
        [
            [-27.55381, 0.0001, -1179.019, 0.0001],
            [-32.22650, 0.0001, -967.9824, 0.0001],
            [-31.75132, 0.0001, -999.3838, 0.0001],
        ],
    ),
    (
        "ctt",
        [
            [-27.61660, 0.0000, -1180.209, 0.0001],
            [-32.50041, 0.0000, -950.8720, 0.0001],
            [-31.94475, 0.0000, -983.4933, 0.0001],
        ],
    ),
)


@pytest.mark.parametrize("result", REFERENCE)
def test_against_ref(trivariate_data, result):
    trend = result[0]
    ref = result[1]
    for i in range(3):
        ref_row = ref[i]
        x_idx = list(np.arange(3))
        x_idx.pop(i)
        y_idx = [i]
        if isinstance(trivariate_data[0], pd.DataFrame):
            z = pd.concat(trivariate_data, axis=1)
            x = z.iloc[:, x_idx]
            y = z.iloc[:, y_idx]
        else:
            z = np.column_stack(trivariate_data)
            x = z[:, x_idx]
            y = z[:, y_idx]
        zt = phillips_ouliaris(y, x, trend=trend, test_type="Zt", bandwidth=9)
        za = phillips_ouliaris(y, x, trend=trend, test_type="Za", bandwidth=9)
        scale = y.shape[0] / (y.shape[0] - 1)
        assert_allclose(zt.stat, np.sqrt(scale) * ref_row[0], rtol=1e-4)
        assert_allclose(zt.pvalue, ref_row[1], atol=1e-3)
        assert_allclose(za.stat, scale * ref_row[2], rtol=1e-4)
        assert_allclose(za.pvalue, ref_row[3], atol=1e-3)
