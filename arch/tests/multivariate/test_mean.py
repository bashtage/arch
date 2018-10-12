from itertools import product

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
import statsmodels.api as sm
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from arch.multivariate import VARX, MultivariateNormal, ConstantCovariance
from arch.tests.multivariate.utility import generate_data

NVAR = [1, 2, 10]
NOBS = [100, 500]
NEXOG = [0, 1, 3]
COMMON = [True, False]
LAGS = [1, [1], 3, [1, 3], None]
CONSTANT = [True, False]
PANDAS = [True, False]
params = list(product(NVAR, NOBS, NEXOG, COMMON, LAGS, CONSTANT, PANDAS))
ID = """nvar={0}, nobs={1}, nexog={2}, common={3}, lags={4}, const={5}, pandas={6}"""
ids = [ID.format(*map(str, p)) for p in params]

CONSTANT_COVARIANCE = ConstantCovariance()
MVN = MultivariateNormal()


@pytest.fixture(scope='module', params=params, ids=ids)
def var_data(request):
    nvar, nobs, nexog, common, lags, constant, pandas = request.param
    return generate_data(nvar, nobs, nexog, common, lags, constant, pandas)


def none_shape(a):
    if a is None:
        return 0
    return a.shape[1]


def empty_replace(a, nobs):
    if a is None:
        return np.empty((nobs, 0))
    return a


def construct_rhs_direct(y, x, lags, constant):
    y_orig = y
    y = np.asarray(y)
    nobs, nvar = y.shape
    lags = [] if lags is None else lags
    lags = list(np.arange(1, lags + 1)) if np.isscalar(lags) else lags
    nreg = nvar * (constant + nvar * len(lags))
    nx = 0
    common = False
    if isinstance(x, (np.ndarray, pd.DataFrame)) or x is None:
        nx = nvar * none_shape(x)
        if nx == 0:
            x = np.empty((nobs, 0))
        x = [x] * nvar
        common = True
    elif isinstance(x, list):
        nx = sum(map(lambda a: none_shape(a), x))
        x = [empty_replace(_x, nobs) for _x in x]
    else:  # dict
        nx = sum([none_shape(x[key]) for key in x])
        x = [empty_replace(x[key], nobs) for key in y_orig]
    nreg += nx
    if common:
        nreg //= nvar
    max_lag = max(lags) if lags else 0
    lag_mats = [sm.tsa.tsatools.lagmat(y[:, i], max_lag) for i in range(nvar)]
    all_rhs = []
    for i in range(nvar):
        rhs = [np.ones((nobs, int(constant)))]
        for j in range(len(lags)):
            lag = lags[j]
            for k in range(nvar):
                rhs.append(lag_mats[k][:, lag - 1:lag])
        rhs.append(x[i])
        all_rhs.append(np.hstack(rhs))
    rhs = all_rhs
    if common:
        rhs = rhs[0]

    return nreg, common, rhs


def fit_single(y, x, common, lags, idx):
    y = np.asarray(y)
    if not common:
        x = x[idx]
    if lags:
        max_lag = lags if np.isscalar(lags) else max(lags)
        y = y[max_lag:]
        x = x[max_lag:]
    if x.shape[1] > 0:
        return sm.OLS(y[:, idx:idx + 1], x).fit()
    return


def test_regressors(var_data):
    mod = VARX(var_data.y, var_data.x, var_data.lags, var_data.constant,
               volatility=CONSTANT_COVARIANCE,
               distribution=MVN)
    nreg, common, rhs = construct_rhs_direct(var_data.y, var_data.x, var_data.lags,
                                             var_data.constant)
    if isinstance(mod._rhs, (np.ndarray)):
        assert mod._rhs.shape[1] == nreg
    else:
        assert sum(map(lambda a: a.shape[1], mod._rhs)) == nreg
    assert mod._common_regressor is common
    if common:
        assert_array_almost_equal(rhs, mod._rhs)
    else:
        for i in range(var_data.y.shape[1]):
            assert_array_almost_equal(rhs[i], mod._rhs[i])


def test_coefficients_direct(var_data):
    mod = VARX(var_data.y, var_data.x, var_data.lags, var_data.constant,
               volatility=CONSTANT_COVARIANCE,
               distribution=MVN)
    res = mod.fit(cov_type='mle')
    var_params = res.params
    nreg, common, rhs = construct_rhs_direct(var_data.y, var_data.x, var_data.lags,
                                             var_data.constant)
    nvar = var_data.y.shape[1]
    offset = 0

    for i in range(nvar):
        ols_res = fit_single(var_data.y, rhs, common, var_data.lags, i)
        if ols_res is not None:
            ols_params = ols_res.params
            m = ols_params.shape[0]
            assert_array_almost_equal(ols_params, var_params[offset:offset + m])
            offset += m


def test_r2(var_data):
    mod = VARX(var_data.y, var_data.x, var_data.lags, var_data.constant,
               volatility=CONSTANT_COVARIANCE,
               distribution=MVN)
    nreg, common, rhs = construct_rhs_direct(var_data.y, var_data.x, var_data.lags,
                                             var_data.constant)
    res = mod.fit(cov_type='mle')
    nvar = var_data.y.shape[1]
    for i in range(nvar):
        ols_res = fit_single(var_data.y, rhs, common, var_data.lags, i)
        if ols_res is not None:
            assert_almost_equal(res.rsquared[i], ols_res.rsquared)
            assert_almost_equal(res.rsquared_adj[i], ols_res.rsquared_adj)


def getnonnull(a, idx):
    try:
        return a.dropna().iloc[:, idx]
    except AttributeError:
        a = a[:, idx]
        return a[~np.isnan(a)]


def test_residuals(var_data):
    mod = VARX(var_data.y, var_data.x, var_data.lags, var_data.constant,
               volatility=CONSTANT_COVARIANCE,
               distribution=MVN)
    nreg, common, rhs = construct_rhs_direct(var_data.y, var_data.x, var_data.lags,
                                             var_data.constant)
    res = mod.fit(cov_type='mle')
    nvar = var_data.y.shape[1]
    for i in range(nvar):
        ols_res = fit_single(var_data.y, rhs, common, var_data.lags, i)
        if ols_res is not None:
            assert_almost_equal(getnonnull(res.resid, i), ols_res.resid)


def test_basic_attributes(var_data):
    mod = VARX(var_data.y, var_data.x, var_data.lags, var_data.constant,
               volatility=CONSTANT_COVARIANCE,
               distribution=MVN)
    nreg, common, rhs = construct_rhs_direct(var_data.y, var_data.x, var_data.lags,
                                             var_data.constant)

    nvar = var_data.y.shape[1]
    if var_data.lags is None:
        maxlag = 0
    else:
        if isinstance(var_data.lags, int):
            maxlag = var_data.lags
        else:
            maxlag = max(var_data.lags)
    nobs = var_data.y.shape[0] - maxlag
    num_params = nreg * max(1, common * nvar) + (nvar * (nvar + 1)) // 2

    res = mod.fit(cov_type='mle')
    llf = res.loglikelihood
    cov = res.conditional_covariance
    resid = np.asarray(res.resid)[maxlag:]
    spllf = stats.multivariate_normal(cov=cov[-1]).logpdf(resid).sum()
    assert_almost_equal(llf, spllf)

    assert res.nvar == nvar
    assert res.nobs == nobs
    assert res.fit_start == maxlag
    assert res.fit_stop == var_data.y.shape[0]
    assert res.convergence_flag == 0
    assert res.num_params == num_params

    assert_almost_equal(res.aic, -2 * spllf + 2 * num_params)
    assert_almost_equal(res.bic, -2 * spllf + np.log(nobs) * num_params)


def test_attributes_smoke(var_data):
    mod = VARX(var_data.y, var_data.x, var_data.lags, var_data.constant,
               volatility=CONSTANT_COVARIANCE,
               distribution=MVN)
    res = mod.fit(cov_type='mle')

    # Still smoke, need test
    res.pvalues
    res.std_err
    res.tvalues
    res.param_cov


def test_str(var_data):
    mod = VARX(var_data.y, var_data.x, var_data.lags, var_data.constant,
               volatility=CONSTANT_COVARIANCE,
               distribution=MVN)
    s = mod.__str__()
    lags = var_data.lags
    if lags in (None, 0):
        lag_str = 'none'
    else:
        if isinstance(lags, int):
            lags = list(range(1, lags + 1))
        lag_str = ', '.join(map(str, lags))
    assert 'lags: {0}'.format(lag_str) in s
    assert 'volatility: Constant Covariance' in s
    assert 'distribution: Multivariate Normal' in s
    assert 'constant: {0}'.format('yes' if var_data.constant else 'no') in s
    exog = var_data.x is not None
    if exog and isinstance(var_data.x, list):
        exog = any([_x is not None for _x in var_data.x])
    elif exog and isinstance(var_data.x, dict):
        exog = any([var_data.x[key] is not None for key in var_data.x])
    assert 'exogenous: {0}'.format('yes' if exog else 'no') in s
    if exog:
        common = isinstance(var_data, (np.ndarray, pd.DataFrame))
        assert 'exogenous structure: {0}'.format('common' if common else 'heterogeneous')
    rep = mod.__repr__()
    assert str(hex(id(mod))) in rep
