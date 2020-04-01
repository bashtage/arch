from typing import Optional, Tuple

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.covariance.kernel import CovarianceEstimate
from arch.covariance.var import PreWhitenedRecolored
from arch.typing import NDArray

KERNELS = [
    "Bartlett",
    "Parzen",
    "ParzenCauchy",
    "ParzenGeometric",
    "ParzenRiesz",
    "TukeyHamming",
    "TukeyHanning",
    "TukeyParzen",
    "QuadraticSpectral",
    "Andrews",
    "Gallant",
    "NeweyWest",
]


@pytest.fixture(params=KERNELS)
def kernel(request):
    return request.param


def direct_var(
    x, const: bool, full_order: int, diag_order: int, max_order: Optional[int] = None
) -> Tuple[NDArray, NDArray]:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    c = int(const)
    nobs, nvar = x.shape
    order = max_order if max_order is not None else max(full_order, diag_order)
    rhs = np.empty((nobs - order, c + nvar * order))
    lhs = np.empty((nobs - order, nvar))
    offset = 0
    if const:
        rhs[:, 0] = 1
        offset = 1
    from statsmodels.tsa.tsatools import lagmat

    for i in range(nvar):
        idx = offset + i + nvar * np.arange(order)
        rhs[:, idx], lhs[:, i : i + 1] = lagmat(
            x[:, i : i + 1], order, "both", "sep", False
        )
    idx = []
    if const:
        idx += [0]
    idx += (c + np.arange(full_order * nvar)).tolist()
    idx += [-9999] * (diag_order - full_order)
    locs = np.array(idx, dtype=np.int)
    diag_start = int(const) + full_order * nvar
    params = np.zeros((nvar, rhs.shape[1]))
    resids = np.empty_like(lhs)
    for i in range(nvar):
        if diag_order > full_order:
            locs[diag_start:] = c + i + nvar * np.arange(full_order, diag_order)
        _rhs = rhs[:, locs]
        if _rhs.shape[1] > 0:
            p = np.linalg.lstsq(_rhs, lhs[:, i : i + 1], rcond=None)[0]
            params[i : i + 1, locs] = p.T
            resids[:, i : i + 1] = lhs[:, i : i + 1] - _rhs @ p
        else:
            # Branch is a workaround of NumPy 1.15
            # TODO: Remove after NumPy 1.15 dropped
            resids[:, i : i + 1] = lhs[:, i : i + 1]
    return params, resids


def direct_ic(
    x,
    ic: str,
    const: bool,
    full_order: int,
    diag_order: int,
    max_order: Optional[int] = None,
) -> float:
    _, resids = direct_var(x, const, full_order, diag_order, max_order)
    nobs, nvar = resids.shape
    sigma = resids.T @ resids / nobs
    ndiag = max(0, diag_order - full_order)
    nparams = (int(const) + full_order * nvar + ndiag) * nvar
    if ic == "aic":
        penalty = 2
    elif ic == "hqc":
        penalty = 2 * np.log(np.log(nobs))
    else:  # bic
        penalty = np.log(nobs)
    _, ld = np.linalg.slogdet(sigma)
    return ld + penalty * nparams / nobs


@pytest.mark.parametrize("const", [True, False])
@pytest.mark.parametrize("full_order", [1, 3])
@pytest.mark.parametrize("diag_order", [3, 5])
@pytest.mark.parametrize("max_order", [None, 10])
@pytest.mark.parametrize("ic", ["aic", "bic", "hqc"])
def test_direct_var(covariance_data, const, full_order, diag_order, max_order, ic):
    direct_ic(covariance_data, ic, const, full_order, diag_order, max_order)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("diagonal", [True, False])
@pytest.mark.parametrize("method", ["aic", "bic", "hqc"])
def test_ic(covariance_data, center, diagonal, method):
    pwrc = PreWhitenedRecolored(
        covariance_data, center=center, diagonal=diagonal, method=method, bandwidth=0.0,
    )
    cov = pwrc.cov
    expected_type = (
        np.ndarray if isinstance(covariance_data, np.ndarray) else pd.DataFrame
    )
    assert isinstance(cov.short_run, expected_type)
    expected_max_lag = int(covariance_data.shape[0] ** (1 / 3))
    assert pwrc._max_lag == expected_max_lag
    expected_ics = {}
    for full_order in range(expected_max_lag + 1):
        diag_limit = expected_max_lag + 1 if diagonal else full_order + 1
        if covariance_data.ndim == 1 or covariance_data.shape[1] == 1:
            diag_limit = full_order + 1
        for diag_order in range(full_order, diag_limit):
            key = (full_order, diag_order)
            expected_ics[key] = direct_ic(
                covariance_data,
                method,
                center,
                full_order,
                diag_order,
                max_order=expected_max_lag,
            )
    assert tuple(sorted(pwrc._ics.keys())) == tuple(sorted(expected_ics.keys()))
    for key in expected_ics:
        assert_allclose(pwrc._ics[key], expected_ics[key])
    expected_order = pd.Series(expected_ics).idxmin()
    assert pwrc._order == expected_order


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("diagonal", [True, False])
@pytest.mark.parametrize("method", ["aic", "bic", "hqc"])
@pytest.mark.parametrize("lags", [0, 1, 3])
def test_short_long_run(covariance_data, center, diagonal, method, lags):
    pwrc = PreWhitenedRecolored(
        covariance_data,
        center=center,
        diagonal=diagonal,
        method=method,
        lags=lags,
        bandwidth=0.0,
    )
    cov = pwrc.cov
    full_order, diag_order = pwrc._order
    params, resids = direct_var(covariance_data, center, full_order, diag_order)
    nobs, nvar = resids.shape
    expected_short_run = resids.T @ resids / nobs
    assert_allclose(cov.short_run, expected_short_run)
    d = np.eye(nvar)
    c = int(center)
    for i in range(max(full_order, diag_order)):
        d -= params[:, c + i * nvar : c + (i + 1) * nvar]
    d_inv = np.linalg.inv(d)
    scale = nobs / (nobs - nvar * (pwrc._order != (0, 0)))
    expected_long_run = scale * d_inv @ expected_short_run @ d_inv.T
    assert_allclose(cov.long_run, expected_long_run)


@pytest.mark.parametrize("force_int", [True, False])
def test_pwrc_attributes(covariance_data, force_int):
    pwrc = PreWhitenedRecolored(covariance_data, force_int=force_int)
    assert isinstance(pwrc.bandwidth_scale, float)
    assert isinstance(pwrc.kernel_const, float)
    assert isinstance(pwrc.rate, float)
    assert isinstance(pwrc._weights(), np.ndarray)
    assert pwrc.force_int == force_int
    expected_type = (
        np.ndarray if isinstance(covariance_data, np.ndarray) else pd.DataFrame
    )
    assert isinstance(pwrc.cov.short_run, expected_type)
    assert isinstance(pwrc.cov.long_run, expected_type)
    assert isinstance(pwrc.cov.one_sided, expected_type)
    assert isinstance(pwrc.cov.one_sided_strict, expected_type)


@pytest.mark.parametrize("sample_autocov", [True, False])
def test_data(covariance_data, sample_autocov, kernel):
    pwrc = PreWhitenedRecolored(
        covariance_data, sample_autocov=sample_autocov, kernel=kernel, bandwidth=0.0
    )
    assert isinstance(pwrc.cov, CovarianceEstimate)


def test_pwrc_errors():
    x = np.random.standard_normal((500, 2))
    with pytest.raises(ValueError, match="lags must be a"):
        PreWhitenedRecolored(x, lags=-1)
    with pytest.raises(ValueError, match="lags must be a"):
        PreWhitenedRecolored(x, lags=np.array([2]))
    with pytest.raises(ValueError, match="lags must be a"):
        PreWhitenedRecolored(x, lags=3.5)


def test_pwrc_warnings():
    x = np.random.standard_normal((9, 5))
    with pytest.warns(RuntimeWarning, match="The maximum number of lags is 0"):
        assert isinstance(PreWhitenedRecolored(x).cov, CovarianceEstimate)


def test_unknown_kernel(covariance_data):
    with pytest.raises(ValueError, match=""):
        PreWhitenedRecolored(covariance_data, kernel="unknown")
