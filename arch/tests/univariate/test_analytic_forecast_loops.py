"""
Tests that the multi-step analytic forecasts of GARCH models, which are computed
for all forecast origins at once, match the loop over origins
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.univariate import ARCH, GARCH


def analytic_forecast_reference(vol, parameters, resids, backcast, start, horizon, w):
    """Loop over the origins with scalar arithmetic"""
    var_bounds = vol.variance_bounds(resids)
    sigma2, forecasts = vol._one_step_forecast(
        parameters, resids, backcast, var_bounds, horizon, start
    )
    if horizon == 1:
        # One-step forecasts come from the recursion, which does not use w
        return forecasts
    t = resids.shape[0]
    p, o, q = vol.p, vol.o, vol.q
    omega = parameters[0]
    alpha = parameters[1 : p + 1]
    gamma = parameters[p + 1 : p + o + 1]
    beta = parameters[p + o + 1 :]
    m = np.max([p, o, q])
    _resids = np.zeros(m + horizon)
    _asym_resids = np.zeros(m + horizon)
    _sigma2 = np.zeros(m + horizon)
    for i in range(start, t):
        if i - m + 1 >= 0:
            _resids[:m] = resids[i - m + 1 : i + 1]
            _asym_resids[:m] = _resids[:m] * (_resids[:m] < 0)
            _sigma2[:m] = sigma2[i - m + 1 : i + 1]
        else:
            _resids[: m - i - 1] = np.sqrt(backcast)
            _resids[m - i - 1 : m] = resids[0 : i + 1]
            _asym_resids = _resids * (_resids < 0)
            _asym_resids[: m - i - 1] = np.sqrt(w * backcast)
            _sigma2[:m] = backcast
            _sigma2[m - i - 1 : m] = sigma2[0 : i + 1]
        for h in range(horizon):
            loc = i - start
            forecasts[loc, h] = omega
            start_loc = h + m - 1
            for j in range(p):
                forecasts[loc, h] += alpha[j] * _resids[start_loc - j] ** 2
            for j in range(o):
                forecasts[loc, h] += gamma[j] * _asym_resids[start_loc - j] ** 2
            for j in range(q):
                forecasts[loc, h] += beta[j] * _sigma2[start_loc - j]
            _resids[h + m] = np.sqrt(forecasts[loc, h])
            _asym_resids[h + m] = np.sqrt(w * forecasts[loc, h])
            _sigma2[h + m] = forecasts[loc, h]
    return forecasts


ORDERS = {
    "1,0,1": (1, 0, 1),
    "1,1,1": (1, 1, 1),
    "2,0,2": (2, 0, 2),
    "1,2,1": (1, 2, 1),
    "3,1,2": (3, 1, 2),
    "0,1,1": (0, 1, 1),
    "1,0,0": (1, 0, 0),
    "1,1,0": (1, 1, 0),
}


@pytest.mark.parametrize("order", ORDERS.values(), ids=list(ORDERS))
@pytest.mark.parametrize("start", [0, 1, 2, 3, 4, 10, 37])
@pytest.mark.parametrize("horizon", [1, 2, 5])
@pytest.mark.parametrize("asym_weight", [0.5, 0.3], ids=lambda v: f"w{v}")
def test_garch_analytic_forecast(order, start, horizon, asym_weight):
    p, o, q = order
    rng = np.random.default_rng(p * 100 + o * 10 + q + start)
    resids = rng.standard_normal(40) * 1.5
    vol = GARCH(p, o, q)
    parameters = np.concatenate(
        [[0.2], np.full(p, 0.08), np.full(o, 0.06), np.full(q, 0.5 / max(q, 1))]
    )
    backcast = 1.3
    var_bounds = vol.variance_bounds(resids)
    forecast = vol._analytic_forecast(
        parameters,
        resids,
        backcast,
        var_bounds,
        start,
        horizon,
        asym_weight=asym_weight,
    )
    expected = analytic_forecast_reference(
        vol, parameters, resids, backcast, start, horizon, asym_weight
    )
    assert forecast.forecasts.shape == (40 - start, horizon)
    assert_allclose(forecast.forecasts, expected, rtol=1e-12)


@pytest.mark.parametrize("p", [1, 3], ids=["p1", "p3"])
def test_arch_analytic_forecast(p):
    resids = np.random.default_rng(4).standard_normal(30)
    vol = ARCH(p)
    parameters = np.concatenate([[0.3], np.full(p, 0.1)])
    backcast = 1.1
    var_bounds = vol.variance_bounds(resids)
    forecast = vol._analytic_forecast(parameters, resids, backcast, var_bounds, 0, 4)
    expected = analytic_forecast_reference(vol, parameters, resids, backcast, 0, 4, 0.5)
    assert_allclose(forecast.forecasts, expected, rtol=1e-12)
