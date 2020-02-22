from itertools import product
from typing import Optional, Tuple

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.covariance.var import PreWhitenRecoloredCovariance
from arch.typing import NDArray

DATA_PARAMS = list(product([1, 3], [True, False], [0]))  # , 1, 3]))
DATA_IDS = [f"dim: {d}, pandas: {p}, order: {o}" for d, p, o in DATA_PARAMS]


@pytest.fixture(scope="module", params=DATA_PARAMS, ids=DATA_IDS)
def data(request):
    dim, pandas, order = request.param
    rs = np.random.RandomState([839084, 3823810, 982103, 829108])
    burn = 100
    shape = (burn + 500,)
    if dim > 1:
        shape += (3,)
    rvs = rs.standard_normal(shape)
    phi = np.zeros((order, dim, dim))
    if order > 0:
        phi[0] = np.eye(dim) * 0.4 + 0.1
        for i in range(1, order):
            phi[i] = 0.3 / (i + 1) * np.eye(dim)
        for i in range(order, burn + 500):
            for j in range(order):
                if dim == 1:
                    rvs[i] += np.squeeze(phi[j] * rvs[i - j - 1])
                else:
                    rvs[i] += phi[j] @ rvs[i - j - 1]
    if order > 1:
        p = np.eye(dim * order, dim * order, -dim)
        for j in range(order):
            p[:dim, j * dim : (j + 1) * dim] = phi[j]
        v, _ = np.linalg.eig(p)
        assert np.max(np.abs(v)) < 1
    rvs = rvs[burn:]
    if pandas and dim == 1:
        return pd.Series(rvs, name="x")
    elif pandas:
        return pd.DataFrame(rvs, columns=[f"x{i}" for i in range(dim)])
    return rvs


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
        p = np.linalg.lstsq(_rhs, lhs[:, i : i + 1], rcond=None)[0]
        params[i : i + 1, locs] = p.T
        resids[:, i : i + 1] = lhs[:, i : i + 1] - _rhs @ p
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
def test_direct_var(data, const, full_order, diag_order, max_order, ic):
    direct_ic(data, ic, const, full_order, diag_order, max_order)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("diagonal", [True, False])
@pytest.mark.parametrize("method", ["aic", "bic", "hqc"])
def test_ic(data, center, diagonal, method):
    pwrc = PreWhitenRecoloredCovariance(
        data, center=center, diagonal=diagonal, method=method
    )
    cov = pwrc.cov
    assert isinstance(cov.short_run, np.ndarray)
    expected_max_lag = int(data.shape[0] ** (1 / 3))
    assert pwrc._max_lag == expected_max_lag
    expected_ics = {}
    for full_order in range(expected_max_lag + 1):
        diag_limit = expected_max_lag + 1 if diagonal else full_order + 1
        for diag_order in range(full_order, diag_limit):
            key = (full_order, diag_order)
            expected_ics[key] = direct_ic(
                data, method, center, full_order, diag_order, max_order=expected_max_lag
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
def test_short_long_run(data, center, diagonal, method, lags):
    pwrc = PreWhitenRecoloredCovariance(
        data, center=center, diagonal=diagonal, method=method, lags=lags
    )
    cov = pwrc.cov
    full_order, diag_order = pwrc._order
    params, resids = direct_var(data, center, full_order, diag_order)
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


@pytest.mark.parametrize("sample_autocov", [True, False])
def test_data(data, sample_autocov):
    pwrc = PreWhitenRecoloredCovariance(data, sample_autocov=sample_autocov)
    pwrc.cov


def test_pwrc_errors():
    x = np.random.standard_normal((500, 2))
    with pytest.raises(ValueError, match="lags must be a"):
        PreWhitenRecoloredCovariance(x, lags=-1)
    with pytest.raises(ValueError, match="lags must be a"):
        PreWhitenRecoloredCovariance(x, lags=np.array([2]))
    with pytest.raises(ValueError, match="lags must be a"):
        PreWhitenRecoloredCovariance(x, lags=3.5)


def test_pwrc_warnings():
    x = np.random.standard_normal((9, 5))
    with pytest.warns(RuntimeWarning, match="The maximum number of lags is 0"):
        PreWhitenRecoloredCovariance(x).cov
