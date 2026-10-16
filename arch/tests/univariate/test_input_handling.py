import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
from scipy.special import gammaln

from arch.univariate import (
    GARCH,
    HARCH,
    ConstantMean,
    MIDASHyperbolic,
    Normal,
    StudentsT,
    ZeroMean,
    arch_model,
)


def midas_weights(theta, m):
    # phi_i(theta) proportional to Gamma(i + theta) / (Gamma(i + 1) Gamma(theta))
    j = np.arange(1.0, m + 1)
    w = np.exp(gammaln(theta + j) - gammaln(j + 1) - gammaln(theta))
    return w / w.sum()


def midas_reference(params, errors, nobs, burn, m, asym):
    """Direct implementation of the MIDAS Hyperbolic recursion"""
    if asym:
        omega, alpha, gamma, theta = params
    else:
        omega, alpha, theta = params
        gamma = 0.0
    weights = midas_weights(theta, m)
    # The pre-sample values are the unconditional variance
    initial = omega / (1 - alpha - gamma / 2)
    total = nobs + burn
    sigma2 = np.empty(total)
    data = np.empty(total)
    sigma2[:m] = initial
    data[:m] = np.sqrt(initial)
    for t in range(m, total):
        sigma2[t] = omega
        for i in range(m):
            lagged = data[t - 1 - i]
            if t - 1 - i < m:
                coef = alpha + gamma / 2
            else:
                coef = alpha + gamma * (lagged < 0)
            sigma2[t] += coef * weights[i] * lagged**2
        data[t] = errors[t] * np.sqrt(sigma2[t])
    return data[burn:], sigma2[burn:]


@pytest.mark.parametrize("burn", [0, 5, 21, 22, 30])
@pytest.mark.parametrize("asym", [False, True], ids=["symmetric", "asymmetric"])
def test_midas_simulate_short_burn(burn, asym):
    m, nobs = 22, 40
    params = np.array([0.1, 0.3, 0.4, 0.8] if asym else [0.1, 0.5, 0.8])
    errors = np.random.default_rng(3).standard_normal(nobs + 500)

    def rng(size):
        return errors[:size]

    process = MIDASHyperbolic(m, asym=asym)
    data, sigma2 = process.simulate(params, nobs, rng, burn=burn)
    # The first m observations are always discarded
    expected_data, expected_sigma2 = midas_reference(
        params, errors, nobs, max(burn, m), m, asym
    )
    assert data.shape == (nobs,)
    assert sigma2.shape == (nobs,)
    assert_allclose(data, expected_data, rtol=1e-10)
    assert_allclose(sigma2, expected_sigma2, rtol=1e-10)


def test_midas_simulate_burn_zero_mean_model():
    mod = ConstantMean(volatility=MIDASHyperbolic(22), distribution=Normal(seed=None))
    sim = mod.simulate([0.0, 0.1, 0.5, 0.8], 50, burn=0)
    assert sim.shape[0] == 50
    assert np.all(np.isfinite(sim["data"]))


def test_midas_str():
    assert str(MIDASHyperbolic(22)) == "MIDAS Hyperbolic(lags: 22, asym: False)"
    assert str(MIDASHyperbolic(5, asym=True)) == "MIDAS Hyperbolic(lags: 5, asym: True)"


@pytest.mark.parametrize(
    "lags",
    [
        np.array([1, 5, 22]),
        np.array([1, 5, 22], dtype=np.int32),
        pd.Series([1, 5, 22]),
        (1, 5, 22),
        [1, 5, 22],
    ],
    ids=["int64-array", "int32-array", "series", "tuple", "list"],
)
def test_harch_lags_formats(lags):
    harch = HARCH(lags)
    assert_array_equal(harch.lags, [1, 5, 22])
    assert harch.num_params == 4
    assert str(harch) == "HARCH(lags: 1, 5, 22)"


@pytest.mark.parametrize(
    "lags", [3, np.int64(3), np.array(3)], ids=["int", "numpy-int", "0d-array"]
)
def test_harch_scalar_lags(lags):
    assert_array_equal(HARCH(lags).lags, [1, 2, 3])


def test_harch_array_lags_arch_model():
    y = np.random.default_rng(5).standard_normal(500)
    res_array = arch_model(y, vol="harch", p=np.array([1, 5])).fit(disp="off")
    res_list = arch_model(y, vol="harch", p=[1, 5]).fit(disp="off")
    assert_allclose(res_array.params, res_list.params)
    assert_allclose(res_array.loglikelihood, res_list.loglikelihood)


@pytest.mark.parametrize(
    "lags",
    [0, -1, [0, 1], [1, -5], [], np.array([0, 1]), np.ones((2, 2), dtype=int)],
    ids=[
        "zero",
        "negative",
        "zero-in-list",
        "negative-in-list",
        "empty",
        "array",
        "2d",
    ],
)
def test_harch_invalid_lags(lags):
    with pytest.raises(ValueError, match="lags must be a positive integer"):
        HARCH(lags)


@pytest.mark.parametrize("mean", [ConstantMean, ZeroMean])
@pytest.mark.parametrize("dist", [Normal, StudentsT])
def test_simulate_params_length(mean, dist):
    # [mu] + [omega, alpha, beta] + [nu]
    nmean = 1 if mean is ConstantMean else 0
    valid = [0.5] * nmean + [0.05, 0.1, 0.8] + ([8.0] if dist is StudentsT else [])
    mod = mean(volatility=GARCH(), distribution=dist(seed=np.random.RandomState(0)))
    sim = mod.simulate(valid, 25, burn=10)
    assert sim.shape == (25, 3)
    if nmean:
        assert_allclose(sim["data"] - 0.5, sim["errors"])
    else:
        assert_allclose(sim["data"], sim["errors"])
    for extra in (1, 2):
        with pytest.raises(ValueError, match=f"Expected {len(valid)}, got "):
            mod.simulate([*valid, *([0.5] * extra)], 25)
    for missing in (1, 2):
        with pytest.raises(ValueError, match=f"Expected {len(valid)}, got "):
            mod.simulate(valid[:-missing], 25)
    with pytest.raises(ValueError, match="Expected"):
        mod.simulate([], 25)


@pytest.mark.parametrize("mean", [ConstantMean, ZeroMean])
def test_simulate_params_shape(mean):
    mod = mean(volatility=GARCH())
    n = 4 if mean is ConstantMean else 3
    params = np.array([0.5, 0.05, 0.1, 0.8])[-n:]
    # a column or row vector is fine
    assert mod.simulate(params[:, None], 10, burn=5).shape[0] == 10
    assert mod.simulate(params[None, :], 10, burn=5).shape[0] == 10
    with pytest.raises(ValueError, match=r"1D|squeezable"):
        mod.simulate(np.tile(params, (2, 2)), 10)
