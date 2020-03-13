from typing import Tuple

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.typing import ArrayLike2D, NDArray
from arch.unitroot.cointegration import DynamicOLS


@pytest.fixture(scope="module", params=[True, False])
def trivariate_data(request) -> Tuple[ArrayLike2D, ArrayLike2D]:

    rs = np.random.RandomState([922019, 12882912, 192010, 10189, 109981])
    nobs = 1000
    burn = 100
    e = rs.standard_normal((nobs + burn, 3))
    y = e.copy()
    for i in range(1, 3):
        roots = np.ones(3)
        roots[1:] = rs.random_sample(2)
        ar = -np.poly(roots)[1:]
        lags = np.arange(1, 4)
        for j in range(3, nobs + burn):
            y[j, i] = y[j - lags, i] @ ar + e[j, i]
    y[:, 0] = 10 + 0.75 * y[:, 1] + 0.25 * y[:, 2] + e[:, 0]
    y = y[burn:]
    theta = np.pi * (2 * rs.random_sample(3) - 1)
    rot = np.eye(3)
    idx = 0
    for i in range(3):
        for j in range(i + 1, 3):
            th = theta[idx]
            c = np.cos(th)
            s = np.sin(th)
            r = np.eye(3)
            r[j, j] = r[i, i] = c
            r[i, j] = -s
            r[j, i] = s
            rot = rot @ r
            idx += 1
    y = y @ rot
    if request.param:
        idx = pd.date_range("1-1-2000", periods=nobs, freq="M")
        cols = [f"y{i}" for i in range(1, 4)]
        data = pd.DataFrame(y, columns=cols, index=idx)
        # TODO: Remove
        data.to_csv("trivariate-data.csv")
        return data.iloc[:, :1], data.iloc[:, 1:]

    return y[:, :1], y[:, 1:]


@pytest.fixture(scope="module", params=[True, False])
def data(request) -> Tuple[NDArray, NDArray]:
    g = np.random.RandomState([12839028, 3092183, 902813])
    e = g.standard_normal((2000, 2))
    phi = g.random_sample((3, 2, 2))
    phi[:, 0, 0] *= 0.8 / phi[:, 0, 0].sum()
    phi[:, 1, 1] *= 0.8 / phi[:, 1, 1].sum()
    phi[:, 0, 1] *= 0.2 / phi[:, 0, 1].sum()
    phi[:, 1, 0] *= 0.2 / phi[:, 1, 0].sum()
    print(phi)
    print(np.linalg.eigvals(phi.sum(0) - np.eye(2)))
    y = e.copy()
    for i in range(3, y.shape[0]):
        y[i] = e[i]
        for j in range(3):
            y[i] += (phi[j] @ y[i - j - 1].T).T
    y = y[-1000:]
    if request.param:
        df = pd.DataFrame(y, columns=["y", "x"])
        return df.iloc[:, :1], df.iloc[:, 1:]
    return y[:, :1], y[:, 1:]


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("lags", [None, 2])
@pytest.mark.parametrize("leads", [None, 3])
@pytest.mark.parametrize("common", [True, False])
@pytest.mark.parametrize("max_lag", [None, 7])
@pytest.mark.parametrize("method", ["aic", "bic", "hqic"])
def test_smoke(data, trend, lags, leads, common, max_lag, method):
    y, x = data
    if common:
        leads = lags
    mod = DynamicOLS(y, x, trend, lags, leads, common, max_lag, max_lag, method)
    mod.fit()


@pytest.mark.parametrize("cov_type", ["unadjusted", "robust"])
@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "quadratic-spectral"])
@pytest.mark.parametrize("bandwidth", [None, 0, 5])
@pytest.mark.parametrize("force_int", [True, False])
@pytest.mark.parametrize("df_adjust", [True, False])
def test_smoke_fit(data, cov_type, kernel, bandwidth, force_int, df_adjust):
    y, x = data
    mod = DynamicOLS(y, x, "ct", 3, 5, False)
    res = mod.fit(cov_type, kernel, bandwidth, force_int, df_adjust)
    assert isinstance(res.leads, int)
    assert isinstance(res.lags, int)
    assert isinstance(res.bandwidth, (int, float))
    assert isinstance(res.params, pd.Series)
    assert isinstance(res.cov_type, str)
    assert isinstance(res.resid, pd.Series)
    assert isinstance(res.cov, pd.DataFrame)
    assert isinstance(res.kernel, str)

    print(res.summary())
    print(res.summary(True))


def test_mismatch_lead_lag(data):
    y, x = data
    with pytest.raises(ValueError, match="common is specified but leads"):
        DynamicOLS(y, x, "c", 4, 5, True)
    with pytest.raises(ValueError, match="common is specified but max_lead"):
        DynamicOLS(y, x, max_lag=6, max_lead=7, common=True)


def test_invalid_input(data):
    y, x = data
    with pytest.raises(ValueError, match="method must be one of"):
        DynamicOLS(y, x, method="unknown")
    with pytest.raises(ValueError, match="trend must of be one of"):
        DynamicOLS(y, x, trend="cttt")


def test_invalid_fit_options(data):
    y, x = data
    with pytest.raises(ValueError, match="kernel is not a "):
        DynamicOLS(y, x).fit(kernel="unknown")
    with pytest.raises(ValueError, match="Unknown cov_type"):
        DynamicOLS(y, x).fit(cov_type="unknown")


def test_basic(trivariate_data):
    # Tested against Eviews. Note: bandwidth is 1 less than Eviews bandwidth (2)
    y, x = trivariate_data
    res = DynamicOLS(y, x, leads=1, lags=1).fit(bandwidth=1, df_adjust=True)
    assert_allclose(res.params, [91.65392, -11.65535, 2.301629], rtol=1e-4)
    assert_allclose(res.std_errors, [1.341955, 0.102846, 0.017243], rtol=1e-4)
    assert_allclose(res.tvalues, [68.29882, -113.3279, 133.4834], rtol=1e-4)
    assert_allclose(res.pvalues, [0.0000, 0.0000, 0.0000], atol=1e-5)
    assert_allclose(res.long_run_variance, 39.35663, rtol=1e-4)
    assert_allclose(np.sqrt(res.residual_variance), 4.759419, rtol=1e-4)
    assert_allclose(res.rsquared, 0.998438, atol=1e-5)
    assert_allclose(res.rsquared_adj, 0.998425, atol=1e-5)


LEADS_LAGS = (
    ([0, 0], ((85.28087, 1.712124), (-11.16047, 0.131134), (2.218772, 0.021983),)),
    ([0, 1], ((89.31582, 1.545362), (-11.47526, 0.118405), (2.271718, 0.019851),)),
    ([1, 0], ((89.43907, 1.474557), (-11.48150, 0.112980), (2.272349, 0.018941),)),
    ([7, 3], ((96.03174, 0.893012), (-11.98434, 0.068532), (2.356085, 0.011497),)),
)


@pytest.mark.parametrize("config", LEADS_LAGS)
def test_direct_eviews(trivariate_data, config):
    # Tested against Eviews. Note: bandwidth is 1 less than Eviews bandwidth (2)
    y, x = trivariate_data
    leads, lags = config[0]
    expected = np.array(config[1])
    params = expected[:, 0]
    se = expected[:, 1]
    res = DynamicOLS(y, x, leads=leads, lags=lags).fit(bandwidth=1, df_adjust=True)
    assert res.leads == leads
    assert res.lags == lags
    assert_allclose(res.params, params, rtol=1e-4)
    assert_allclose(res.std_errors, se, rtol=1e-4)


AUTO = ([10, 4], ((96.33288, 0.870645), (-12.00320, 0.066865), (2.359063, 0.011222),))


@pytest.mark.parametrize("config", [AUTO])
def test_auto_eviews(trivariate_data, config):
    y, x = trivariate_data
    leads, lags = config[0]
    expected = np.array(config[1])
    params = expected[:, 0]
    se = expected[:, 1]
    res = DynamicOLS(y, x).fit(bandwidth=1, df_adjust=True)
    assert res.leads == leads
    assert res.lags == lags
    assert_allclose(res.params, params, rtol=1e-4)
    assert_allclose(res.std_errors, se, rtol=1e-4)


TRENDS = (
    (
        "ct",
        (11, 4),
        (
            (97.79006, 0.970597),
            (0.004077, 0.001256),
            (-12.25629, 0.102684),
            (2.398460, 0.016533),
        ),
        2.553272040644449,
        0.999443,
    ),
    (
        "ctt",
        (10, 4),
        (
            (95.88975, 0.959303),
            (0.014283, 0.001882),
            (-1.32e-05, 1.83e-06),
            (-12.18877, 0.098193),
            (2.389796, 0.015789),
        ),
        3.927197694670395,
        0.999492,
    ),
)


@pytest.mark.parametrize("config", TRENDS)
def test_auto_trends_eviews(trivariate_data, config):
    y, x = trivariate_data
    trend = config[0]
    leads, lags = config[1]
    expected = np.array(config[2])
    params = expected[:, 0]
    se = expected[:, 1]
    final_resid = config[3]
    r2 = config[4]
    res = DynamicOLS(y, x, trend=trend).fit(bandwidth=1, df_adjust=True)
    assert res.leads == leads
    assert res.lags == lags
    # Trends not checked since trends intrepreted differently
    assert_allclose(res.params.iloc[-2:], params[-2:], rtol=1e-4)
    assert_allclose(res.std_errors[-2:], se[-2:], rtol=1e-4)
    # Check resid to verify equivalent
    assert_allclose(res.resid.iloc[-1], final_resid, rtol=1e-4)
    assert_allclose(res.rsquared, r2, rtol=1e-5)


KERNELS = (("parzen", 2.821334, 42.75669), ("quadratic-spectral", 2.821334, 56.04873))


@pytest.mark.parametrize("config", KERNELS)
def test_kernels_eviews(trivariate_data, config):
    y, x = trivariate_data
    kernel = config[0]
    ser = float(config[1])
    lrvar = config[2]
    bw = 9 if kernel == "parzen" else 10
    res = DynamicOLS(y, x).fit(kernel=kernel, bandwidth=bw, df_adjust=True)
    assert_allclose(res.residual_variance, ser ** 2.0, rtol=1e-5)
    assert_allclose(res.long_run_variance, lrvar, rtol=1e-5)


HAC = (((96.33288, 1.331138), (-12.00320, 0.097762), (2.359063, 0.016323),),)


@pytest.mark.parametrize("config", HAC)
def test_hac_eviews(trivariate_data, config):
    y, x = trivariate_data
    res = DynamicOLS(y, x).fit(bandwidth=9, df_adjust=True, cov_type="robust")
    expected = np.array(config)
    se = expected[:, 1]
    assert_allclose(res.std_errors, se, rtol=1e-4)
