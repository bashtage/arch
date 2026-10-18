"""
Tests that the simulation of HARCH and MIDAS volatility processes, which is
vectorized over the lags, matches the original loop over lags
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.univariate import HARCH, MIDASHyperbolic


def harch_simulate_reference(parameters, lags, errors, nobs, burn):
    """Loop over the lags with scalar arithmetic"""
    lags = np.asarray(lags)
    initial_value = parameters[0] / (1.0 - np.sum(parameters[1:]))
    sigma2 = np.empty(nobs + burn)
    data = np.empty(nobs + burn)
    max_lag = int(np.max(lags))
    sigma2[:max_lag] = initial_value
    data[:max_lag] = np.sqrt(initial_value)
    for t in range(max_lag, nobs + burn):
        sigma2[t] = parameters[0]
        for i in range(lags.shape[0]):
            param = parameters[1 + i] / lags[i]
            for j in range(lags[i]):
                sigma2[t] += param * data[t - 1 - j] ** 2.0
        data[t] = errors[t] * np.sqrt(sigma2[t])
    return data[burn:], sigma2[burn:]


def midas_simulate_reference(vol, parameters, errors, nobs, burn, asym):
    """Loop over the lags with scalar arithmetic"""
    if asym:
        omega, alpha, gamma = parameters[:3]
    else:
        omega, alpha = parameters[:2]
        gamma = 0
    weights = vol._weights(parameters)
    aw = weights * alpha
    gw = weights * gamma
    m = weights.shape[0]
    burn = max(burn, m)
    initial_value = parameters[0] / (1.0 - alpha - 0.5 * gamma)
    sigma2 = np.empty(nobs + burn)
    data = np.empty(nobs + burn)
    sigma2[:m] = initial_value
    data[:m] = np.sqrt(initial_value)
    for t in range(m, nobs + burn):
        sigma2[t] = omega
        for i in range(m):
            if t - 1 - i < m:
                coef = aw[i] + 0.5 * gw[i]
            else:
                coef = aw[i] + gw[i] * (data[t - 1 - i] < 0)
            sigma2[t] += coef * data[t - 1 - i] ** 2.0
        data[t] = errors[t] * np.sqrt(sigma2[t])
    return data[burn:], sigma2[burn:]


@pytest.mark.parametrize("lags", [[1], [1, 5], [1, 5, 22], [3, 7], [2, 3, 4]], ids=str)
@pytest.mark.parametrize(("nobs", "burn"), [(1, 0), (10, 5), (60, 30), (7, 200)])
def test_harch_simulate(lags, nobs, burn):
    vol = HARCH(lags)
    rng = np.random.default_rng(len(lags) * 1000 + nobs)
    parameters = np.concatenate([[0.2], rng.uniform(0.02, 0.8 / len(lags), len(lags))])
    errors = rng.standard_normal(nobs + burn)
    data, sigma2 = vol.simulate(parameters, nobs, lambda size: errors, burn=burn)
    ref_data, ref_sigma2 = harch_simulate_reference(
        parameters, lags, errors, nobs, burn
    )
    assert_allclose(data, ref_data, rtol=1e-12)
    assert_allclose(sigma2, ref_sigma2, rtol=1e-12)


@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("m", [2, 5, 22])
@pytest.mark.parametrize(("nobs", "burn"), [(1, 0), (10, 5), (60, 30), (100, 250)])
def test_midas_simulate(asym, m, nobs, burn):
    vol = MIDASHyperbolic(m, asym)
    rng = np.random.default_rng(m * 1000 + nobs)
    if asym:
        parameters = np.array([0.1, 0.6, 0.3, 1.2])
    else:
        parameters = np.array([0.1, 0.7, 1.2])
    total = nobs + max(burn, m)
    errors = rng.standard_normal(total)
    data, sigma2 = vol.simulate(parameters, nobs, lambda size: errors, burn=burn)
    ref_data, ref_sigma2 = midas_simulate_reference(
        vol, parameters, errors, nobs, burn, asym
    )
    assert data.shape == (nobs,)
    assert_allclose(data, ref_data, rtol=1e-12)
    assert_allclose(sigma2, ref_sigma2, rtol=1e-12)
