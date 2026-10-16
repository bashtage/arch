import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import gammaln

from arch.univariate import ConstantMean, MIDASHyperbolic, Normal


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
