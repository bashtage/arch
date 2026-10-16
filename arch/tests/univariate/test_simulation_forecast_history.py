"""
Tests that the simulation forecasts of FIGARCH and FIAPARCH, which compute the
contribution of the observed history once for all of the paths, match a
computation that repeats the history in every path
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.univariate import FIAPARCH, FIGARCH
import arch.univariate.recursions as rec


def figarch_reference(
    vol, parameters, resids, backcast, var_bounds, start, h, sims, rng
):
    """Every path holds the history and uses one dot product with all of the lags"""
    _, forecasts = vol._one_step_forecast(
        parameters, resids, backcast, var_bounds, h, start
    )
    t = resids.shape[0]
    paths = np.empty((t - start, sims, h))
    shocks = np.empty((t - start, sims, h))
    power, truncation, p, q = vol.power, vol.truncation, vol.p, vol.q
    lam_rev = rec.figarch_weights(parameters[1:], p, q, truncation)[::-1]
    beta = parameters[-1] if q else 0.0
    omega_tilde = parameters[0] / (1 - beta)
    fpath = np.empty((sims, truncation + h))
    fresids = np.absolute(resids) ** power
    for i in range(start, t):
        std_shocks = rng((sims, h))
        available = i + 1 - max(0, i - truncation + 1)
        fpath[:, truncation - available : truncation] = fresids[
            max(0, i + 1 - truncation) : i + 1
        ]
        if available < truncation:
            fpath[:, : (truncation - available)] = backcast
        for k in range(h):
            temp = omega_tilde + fpath[:, k : truncation + k].dot(lam_rev)
            sigma2_k = temp ** (2.0 / power)
            shocks[i - start, :, k] = std_shocks[:, k] * np.sqrt(sigma2_k)
            paths[i - start, :, k] = sigma2_k
            forecasts[i - start, k] = sigma2_k.mean()
            fpath[:, truncation + k] = np.absolute(shocks[i - start, :, k]) ** power
    return forecasts, paths, shocks


def fiaparch_reference(
    vol, parameters, resids, backcast, var_bounds, start, h, sims, rng
):
    """Every path holds the history and uses one dot product with all of the lags"""
    _, forecasts = vol._one_step_forecast(
        parameters, resids, backcast, var_bounds, h, start
    )
    t = resids.shape[0]
    paths = np.empty((t - start, sims, h))
    shocks = np.empty((t - start, sims, h))
    p, o, q, truncation = vol.p, vol.o, vol.q, vol.truncation
    gamma = float(parameters[1 + p + 1 + q]) if o else 0.0
    delta = float(parameters[-1]) if vol._est_delta else vol.delta
    lam_rev = rec.figarch_weights(parameters[: 1 + p + 1 + q][1:], p, q, truncation)[
        ::-1
    ]
    beta = float(parameters[1 + p + q]) if q else 0.0
    omega_tilde = float(parameters[0]) / (1 - beta)
    backcast_delta = backcast ** (delta / 2.0)
    fshocks_hist = np.power(
        np.absolute(resids) - gamma * resids, np.full(resids.shape[0], delta)
    )
    fpath = np.empty((sims, truncation + h))
    for i in range(start, t):
        std_shocks = rng((sims, h))
        available = i + 1 - max(0, i - truncation + 1)
        fpath[:, truncation - available : truncation] = fshocks_hist[
            max(0, i + 1 - truncation) : i + 1
        ]
        if available < truncation:
            fpath[:, : (truncation - available)] = backcast_delta
        for k in range(h):
            temp = omega_tilde + fpath[:, k : truncation + k].dot(lam_rev)
            sigma2_k = temp ** (2.0 / delta)
            shocks[i - start, :, k] = std_shocks[:, k] * np.sqrt(sigma2_k)
            paths[i - start, :, k] = sigma2_k
            forecasts[i - start, k] = sigma2_k.mean()
            new_shocks = np.absolute(shocks[i - start, :, k])
            fpath[:, truncation + k] = (
                new_shocks - gamma * shocks[i - start, :, k]
            ) ** delta
    return forecasts, paths, shocks


def compare(vol, reference, parameters, nobs, start, horizon, seed):
    resids = np.random.default_rng(seed).standard_normal(nobs) * 1.2
    var_bounds = vol.variance_bounds(resids)
    backcast = vol.backcast(resids)
    sims = 6
    ref = reference(
        vol,
        parameters,
        resids,
        backcast,
        var_bounds,
        start,
        horizon,
        sims,
        np.random.RandomState(7).standard_normal,
    )
    result = vol._simulation_forecast(
        parameters,
        resids,
        backcast,
        var_bounds,
        start,
        horizon,
        sims,
        np.random.RandomState(7).standard_normal,
    )
    assert_allclose(result.forecasts, ref[0], rtol=1e-11)
    assert_allclose(result.forecast_paths, ref[1], rtol=1e-11)
    assert_allclose(result.shocks, ref[2], rtol=1e-11)


# horizon larger than, equal to and smaller than the truncation
@pytest.mark.parametrize("truncation", [1, 2, 3, 10, 40], ids=lambda v: f"T{v}")
@pytest.mark.parametrize("horizon", [1, 2, 3, 7], ids=lambda v: f"h{v}")
@pytest.mark.parametrize("start", [0, 1, 12, 44], ids=lambda v: f"start{v}")
@pytest.mark.parametrize("power", [2.0, 1.0, 1.5], ids=lambda v: f"power{v}")
def test_figarch_simulation_forecast(truncation, horizon, start, power):
    vol = FIGARCH(1, 1, power=power, truncation=truncation)
    parameters = np.array([0.2, 0.1, 0.3, 0.25])
    compare(vol, figarch_reference, parameters, 45, start, horizon, truncation + start)


@pytest.mark.parametrize("pq", [(0, 1), (1, 0), (0, 0)], ids=str)
@pytest.mark.parametrize("truncation", [2, 10], ids=lambda v: f"T{v}")
def test_figarch_simulation_forecast_orders(pq, truncation):
    p, q = pq
    vol = FIGARCH(p, q, truncation=truncation)
    parameters = np.array([0.2] + [0.1] * p + [0.3] + [0.2] * q)
    compare(vol, figarch_reference, parameters, 30, 10, 4, 3)


@pytest.mark.parametrize("truncation", [1, 2, 3, 10, 40], ids=lambda v: f"T{v}")
@pytest.mark.parametrize("horizon", [1, 2, 3, 7], ids=lambda v: f"h{v}")
@pytest.mark.parametrize("start", [0, 1, 12, 44], ids=lambda v: f"start{v}")
@pytest.mark.parametrize("delta", [None, 1.5, 2.0, 0.5], ids=lambda v: f"delta{v}")
def test_fiaparch_simulation_forecast(truncation, horizon, start, delta):
    vol = FIAPARCH(1, 1, 1, delta=delta, truncation=truncation)
    parameters = np.array([0.2, 0.1, 0.3, 0.25, 0.3] + ([1.4] if delta is None else []))
    compare(vol, fiaparch_reference, parameters, 45, start, horizon, truncation + start)


@pytest.mark.parametrize("pqo", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 0)], ids=str)
def test_fiaparch_simulation_forecast_orders(pqo):
    p, q, o = pqo
    vol = FIAPARCH(p, o, q, delta=1.5, truncation=8)
    parameters = np.array([0.2] + [0.1] * p + [0.3] + [0.2] * q + [0.3] * o)
    compare(vol, fiaparch_reference, parameters, 30, 10, 4, 5)
