from itertools import product

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.covariance.kernel import (
    Andrews,
    Bartlett,
    CovarianceEstimate,
    CovarianceEstimator,
    Gallant,
    NeweyWest,
    Parzen,
    ParzenCauchy,
    ParzenGeometric,
    ParzenRiesz,
    QuadraticSpectral,
    TukeyHamming,
    TukeyHanning,
    TukeyParzen,
)
from arch.typing import ArrayLike

ESTIMATORS = [
    Andrews,
    Bartlett,
    Gallant,
    NeweyWest,
    Parzen,
    ParzenCauchy,
    ParzenGeometric,
    ParzenRiesz,
    QuadraticSpectral,
    TukeyHamming,
    TukeyHanning,
    TukeyParzen,
]

KERNEL_PARAMS = {
    "Andrews": (1.3221, 2, 2 / 25),
    "Bartlett": (1.1447, 1, 2 / 9),
    "Gallant": (2.6614, 2, 4 / 25),
    "NeweyWest": (1.1447, 1, 2 / 9),
    "Parzen": (2.6614, 2, 4 / 25),
    "ParzenCauchy": (1.0924, 2, 4 / 25),
    "ParzenGeometric": (1.0000, 1, 2 / 9),
    "ParzenRiesz": (1.1340, 2, 4 / 25),
    "QuadraticSpectral": (1.3221, 2, 2 / 25),
    "TukeyHamming": (1.6694, 2, 4 / 25),
    "TukeyHanning": (1.7462, 2, 4 / 25),
    "TukeyParzen": (1.8576, 2, 4 / 25),
}


DATA_PARAMS = list(product([1, 2], [True, False]))
DATA_IDS = [f"{ndim}-{pandas}" for ndim, pandas in DATA_PARAMS]


@pytest.fixture(scope="module", params=DATA_PARAMS, ids=DATA_IDS)
def data(request) -> ArrayLike:
    rs = np.random.RandomState([3894830, 432841, 323297, 8927821])
    ndim, use_pandas = request.param
    burn = 100
    size: tuple[int, ...] = (500 + burn,)
    if ndim == 2:
        size += (3,)
    e = rs.standard_normal(size)
    if ndim == 1:
        phi = rs.uniform(0, 0.9)
        for i in range(1, size[0]):
            e[i] += e[i - 1] * phi
    else:
        phi_arr = np.diag(rs.uniform(0, 0.9, size[1]))
        for i in range(1, size[0]):
            e[i] += e[i - 1] @ phi_arr
    e = e[burn:]
    if use_pandas:
        if ndim == 1:
            return pd.Series(e, name="x")
        else:
            return pd.DataFrame(e, columns=[f"x{i}" for i in range(e.shape[1])])
    return e


@pytest.fixture(params=ESTIMATORS)
def estimator(request) -> CovarianceEstimator:
    return request.param


def test_covariance_smoke(data: ArrayLike, estimator: type[CovarianceEstimator]):
    cov = estimator(data)
    est_cov = cov.cov
    ndim = data.shape[1] if data.ndim > 1 else 1
    assert isinstance(est_cov, CovarianceEstimate)
    assert est_cov.long_run.shape == (ndim, ndim)
    assert est_cov.short_run.shape == (ndim, ndim)
    assert est_cov.one_sided_strict.shape == (ndim, ndim)
    assert est_cov.one_sided.shape == (ndim, ndim)
    assert isinstance(str(cov), str)
    assert isinstance(repr(cov), str)


def test_covariance_errors(data: ArrayLike, estimator: type[CovarianceEstimator]):
    with pytest.raises(ValueError, match="Degrees of freedom is <= 0"):
        estimator(data, df_adjust=data.shape[0] + 1)
    with pytest.raises(ValueError, match="df_adjust must be a non-negative"):
        estimator(data, df_adjust=-2)
    with pytest.raises(ValueError, match="df_adjust must be a non-negative"):
        # Type ignored due to invalid type used in test
        estimator(data, df_adjust=np.ones(2))  # type: ignore
    with pytest.raises(ValueError, match="bandwidth must be"):
        estimator(data, bandwidth=-3)
    with pytest.raises(ValueError, match="weights must be"):
        estimator(data, weights=np.ones(7))


def test_bartlett_auto(data: ArrayLike):
    nw = Bartlett(data, force_int=True)
    if data.ndim == 1:
        expected_bw = 11
    else:
        expected_bw = 12
    assert int(nw.bandwidth) == expected_bw
    expected = 1.0 - np.arange(nw.bandwidth + 1) / (nw.bandwidth + 1)
    assert_allclose(nw.kernel_weights, expected)
    resid = data - np.asarray(data.mean(axis=0))
    resid = np.asarray(resid)
    nobs = resid.shape[0]
    expected_cov = resid.T @ resid / nobs
    assert_allclose(expected_cov, np.squeeze(nw.cov.short_run))
    expected_oss = np.zeros_like(expected_cov)
    for i in range(1, int(nw.bandwidth) + 1):
        gamma = (resid[i:].T @ resid[:-i]) / nobs
        expected_oss += expected[i] * gamma
        expected_cov += expected[i] * (gamma + gamma.T)
    assert_allclose(expected_oss, np.squeeze(nw.cov.one_sided_strict))
    assert_allclose(expected_cov, np.squeeze(nw.cov.long_run))
    ce = CovarianceEstimate(
        short_run=np.asarray(nw.cov.short_run),
        one_sided_strict=np.asarray(nw.cov.one_sided_strict),
        long_run=np.asarray(nw.cov.long_run),
        one_sided=np.asarray(nw.cov.one_sided),
    )
    assert_allclose(ce.short_run, nw.cov.short_run)
    assert_allclose(ce.one_sided_strict, nw.cov.one_sided_strict)
    assert_allclose(ce.long_run, nw.cov.long_run)
    assert_allclose(ce.one_sided, nw.cov.one_sided)


def test_parzen_auto(data: ArrayLike):
    pz = Parzen(data, force_int=True)

    if data.ndim == 1:
        # This test is noisy
        expected_bw: tuple[int, ...] = (18, 19)
        expected_weights = [
            1.0000e00,
            9.8575e-01,
            9.4600e-01,
            8.8525e-01,
            8.0800e-01,
            7.1875e-01,
            6.2200e-01,
            5.2225e-01,
            4.2400e-01,
            3.3175e-01,
            2.5000e-01,
            1.8225e-01,
            1.2800e-01,
            8.5750e-02,
            5.4000e-02,
            3.1250e-02,
            1.6000e-02,
            6.7500e-03,
            2.0000e-03,
            2.5000e-04,
        ]
    else:
        expected_bw = (17,)
        expected_weights = [
            1.00000000e00,
            9.82510288e-01,
            9.34156379e-01,
            8.61111111e-01,
            7.69547325e-01,
            6.65637860e-01,
            5.55555556e-01,
            4.45473251e-01,
            3.41563786e-01,
            2.50000000e-01,
            1.75582990e-01,
            1.17626886e-01,
            7.40740741e-02,
            4.28669410e-02,
            2.19478738e-02,
            9.25925926e-03,
            2.74348422e-03,
            3.42935528e-04,
        ]
    assert int(pz.bandwidth) in expected_bw
    if data.ndim > 1 or (pz.bandwidth + 1) == len(expected_weights):
        assert_allclose(pz.kernel_weights, np.array(expected_weights), rtol=1e-4)


def test_qs_auto(data: ArrayLike):
    qs = QuadraticSpectral(data, force_int=True)
    if data.ndim == 1:
        expected_bw = 8
        expected_weights = [
            1.0,
            0.97796879,
            0.91394558,
            0.81388998,
            0.68693073,
            0.54427698,
            0.3979104,
            0.25923469,
            0.13786058,
            0.04067932,
        ]
    else:
        expected_bw = 9
        expected_weights = [
            1.0,
            0.98256363,
            0.93155267,
            0.85073648,
            0.74599423,
            0.62475681,
            0.49531303,
            0.36605493,
            0.24474206,
            0.13786058,
        ]
    assert int(qs.bandwidth) == expected_bw
    assert_allclose(qs.kernel_weights[:10], np.array(expected_weights))


def test_force_int(data: ArrayLike, estimator: type[CovarianceEstimator]):
    bw = estimator(data, force_int=False).bandwidth
    bw_int = estimator(data, force_int=True).bandwidth
    assert bw_int >= bw
    assert bw_int == int(bw_int)


def test_first_weights(data: ArrayLike, estimator: type[CovarianceEstimator]):
    w = estimator(data).kernel_weights
    assert_allclose(w[0], 1.0)


def test_constants(data: ArrayLike, estimator: type[CovarianceEstimator]):
    cov_est = estimator(data)
    kc, bs, rate = KERNEL_PARAMS[estimator.__name__]
    assert_allclose(cov_est.kernel_const, kc)
    assert cov_est.bandwidth_scale == bs
    assert_allclose(cov_est.rate, rate)


def test_weight_len(data: ArrayLike, estimator: type[CovarianceEstimator]):
    cov_est = estimator(data, force_int=True)
    name = estimator.__name__
    is_qs = name in ("QuadraticSpectral", "Andrews")
    if is_qs:
        exp_len = data.shape[0]
    else:
        exp_len = int(cov_est.bandwidth) + 1
    assert cov_est.kernel_weights.shape[0] == exp_len


def test_kernel_weights(data: ArrayLike, estimator: type[CovarianceEstimator]):
    if data.ndim == 1:
        return
    weights = np.arange(1, data.shape[1] + 1)
    weights = weights / weights.mean()
    wcov = estimator(data, weights=weights, force_int=False)
    cov = estimator(data, force_int=False)
    assert wcov.bandwidth != cov.bandwidth


def test_center(data: ArrayLike, estimator: type[CovarianceEstimator]):
    centered_cov = estimator(data, center=False, force_int=False)
    cov = estimator(data, force_int=False)
    assert centered_cov.bandwidth != cov.bandwidth
