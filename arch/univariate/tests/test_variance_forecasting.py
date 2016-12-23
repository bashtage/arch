from unittest import TestCase

import numpy as np
import pytest
try:
    from arch.univariate.recursions import garch_recursion
except ImportError:
    from arch.univariate.recursions_python import garch_recursion
from numpy.testing import assert_allclose

from arch.univariate.distribution import Normal, StudentsT
from arch.univariate.volatility import GARCH, ConstantVariance, HARCH, EWMAVariance, \
    RiskMetrics2006, BootstrapRng, EGARCH


def _compare_truncated_forecasts(full, trunc, start):

    assert np.all(np.isnan(trunc.forecasts[:start]))
    assert_allclose(trunc.forecasts[start:], full.forecasts[start:])

    if full.forecast_paths is None:
        assert trunc.forecast_paths is None
        assert trunc.shocks is None
        return


class PreservedState(object):
    """
    Context manager that will save NumPy's random generator's state when entering and restore
    the original state when exiting.
    """
    _state = None

    def __enter__(self):
        self._state = np.random.get_state()

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self._state)


preserved_state = PreservedState


def _simple_direct_gjrgarch_forecaster(resids, params, p, o, q, backcast, var_bounds, horizon):
    """
    Simple GARCH forecasting for use when testing model-based forecasts

    Parameters
    ----------
    resids : arrah
    params : arrah
    p : int
    o : int
    q : int
    backcast : float
    var_bounds : array
    horizon : int

    Returns
    -------
    forecasts : array

    """
    m = max([p, o, q])
    t = resids.shape[0]
    _resids = resids
    _sigma2 = np.empty(t)
    garch_recursion(params, resids ** 2.0, np.sign(resids), _sigma2,
                    p, o, q, t, backcast, var_bounds)
    resids = np.empty((t, m + horizon))
    asymresids = resids.copy()
    sigma2 = np.empty((t, m + horizon))

    resids[:, :m] = np.sqrt(backcast)
    asymresids[:, :m] = np.sqrt(0.5 * backcast)
    sigma2[:, :m] = backcast

    for i in range(m):
        resids[m - 1 - i:, i] = _resids[:(t - (m - 1) + i)]
        asymresids[m - 1 - i:, i] = _resids[:(t - (m - 1) + i)] * (_resids[:(t - (m - 1) + i)] < 0)
        sigma2[m - 1 - i:, i] = _sigma2[:(t - (m - 1) + i)]

    for i in range(m, m + horizon):
        sigma2[:, i] = params[0]
        ind = 1
        for j in range(p):
            sigma2[:, i] += params[ind] * resids[:, i - j - 1] ** 2.0
            ind += 1

        for j in range(o):
            sigma2[:, i] += params[ind] * asymresids[:, i - j - 1] ** 2.0
            ind += 1

        for j in range(q):
            sigma2[:, i] += params[ind] * sigma2[:, i - j - 1]
            ind += 1
        resids[:, i] = np.sqrt(sigma2[:, i])
        asymresids[:, i] = np.sqrt(0.5 * sigma2[:, i])

    return sigma2[:, m:]


class TestVarianceForecasts(TestCase):
    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        cls.t = 1000
        cls.resid = np.random.randn(cls.t) * np.sqrt(10)

    def test_constant_variance_forecast(self):
        vol = ConstantVariance()
        params = np.array([10.0])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds)
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert forecast.forecasts[-1] == params[0]
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=5)
        assert forecast.forecasts.shape == (1000, 5)
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert np.all(forecast.forecasts[-1] == params[0])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, start=100)
        assert forecast.forecasts.shape == (1000, 1)
        assert np.all(np.isnan(forecast.forecasts[:100]))
        assert np.all(forecast.forecasts[100:] == params[0])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, start=100, horizon=3)
        assert forecast.forecasts.shape == (1000, 3)
        assert np.all(np.isnan(forecast.forecasts[:100]))
        assert np.all(forecast.forecasts[100:] == params[0])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with pytest.raises(ValueError):
            vol.forecast(params, self.resid, backcast, var_bounds, horizon=5,
                         method='bootstrap', start=0, simulations=2000)

    def test_arch_1_forecast(self):
        t = self.t
        vol = GARCH(p=1, o=0, q=0)
        params = np.array([10.0, 0.4])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds)
        expected = np.empty((t, 1))
        expected.fill(np.nan)
        expected[-1] = params[0] + params[1] * self.resid[-1] ** 2
        assert_allclose(forecast.forecasts, expected)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=5)
        assert forecast.forecasts.shape == (1000, 5)
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert forecast.forecast_paths is None
        assert forecast.shocks is None
        expected = np.zeros(5)
        expected[0] = params[0] + params[1] * self.resid[-1] ** 2
        for i in range(1, 5):
            expected[i] = params[0] + params[1] * expected[i - 1]
        assert_allclose(forecast.forecasts[-1], expected)

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, start=0)
        assert forecast.forecasts.shape == (1000, 1)
        assert np.all(np.isfinite(forecast.forecasts))
        assert forecast.forecast_paths is None
        assert forecast.shocks is None
        expected = params[0] + params[1] * self.resid ** 2
        expected.shape = (1000, 1)
        assert_allclose(forecast.forecasts, expected)

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, start=0, horizon=3)
        assert forecast.forecasts.shape == (1000, 3)
        assert np.all(np.isfinite(forecast.forecasts))
        assert forecast.forecast_paths is None
        assert forecast.shocks is None
        expected = np.zeros((1000, 3))
        expected[:, 0] = params[0] + params[1] * self.resid ** 2
        for i in range(1, 3):
            expected[:, i] = params[0] + params[1] * expected[:, i - 1]
        assert_allclose(forecast.forecasts, expected)

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, start=100, horizon=3)
        expected[:100] = np.nan
        assert_allclose(forecast.forecasts, expected)

    def test_arch_1_forecast_simulation(self):
        dist = Normal()
        rng = dist.simulate([])
        vol = GARCH(p=1, o=0, q=0)
        params = np.array([10.0, 0.4])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=1, method='simulation', rng=rng)
        assert forecast.forecasts.shape == (1000, 1)
        assert forecast.forecast_paths.shape == (1000, 1000, 1)
        assert forecast.shocks.shape == (1000, 1000, 1)
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert np.all(np.isnan(forecast.forecast_paths[:-1]))
        expected = (params[0] + params[1] * self.resid[-1] ** 2.0)
        assert_allclose(forecast.forecasts[-1], expected)
        assert_allclose(forecast.forecast_paths[-1], expected * np.ones((1000, 1)))
        assert_allclose(forecast.shocks[-1], np.sqrt(expected) * rng((1000, 1)))

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=5, method='simulation', rng=rng)

        paths = np.zeros((1000, 5))
        paths[:, 0] = expected
        std_shocks = rng((1000, 5))
        shocks = np.zeros((1000, 5))
        shocks[:, 0] = np.sqrt(paths[:, 0]) * std_shocks[:, 0]
        for i in range(1, 5):
            paths[:, i] = params[0] + params[1] * shocks[:, i - 1] ** 2
            shocks[:, i] = np.sqrt(paths[:, i]) * std_shocks[:, i]

        assert forecast.forecasts.shape == (1000, 5)
        assert forecast.forecast_paths.shape == (1000, 1000, 5)
        assert forecast.shocks.shape == (1000, 1000, 5)
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert np.all(np.isnan(forecast.forecast_paths[:-1]))
        assert np.all(np.isnan(forecast.shocks[:-1]))
        assert_allclose(forecast.forecasts[-1], paths.mean(0))
        assert_allclose(forecast.forecast_paths[-1], paths)
        assert_allclose(forecast.shocks[-1], shocks)

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=5, start=100, method='simulation',
                                    simulations=2000, rng=rng)

        paths = np.zeros((1000, 2000, 5))
        paths.fill(np.nan)
        shocks = np.zeros((1000, 2000, 5))
        shocks.fill(np.nan)
        for i in range(100, 1000):
            std_shocks = rng((2000, 5))
            paths[i, :, 0] = params[0] + params[1] * self.resid[i] ** 2.0
            shocks[i, :, 0] = np.sqrt(paths[i, :, 0]) * std_shocks[:, 0]
            for j in range(1, 5):
                paths[i, :, j] = params[0] + params[1] * shocks[i, :, j - 1] ** 2
                shocks[i, :, j] = np.sqrt(paths[i, :, j]) * std_shocks[:, j]

        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.shocks, shocks)

    def test_arch_1_forecast_bootstrap(self):
        vol = GARCH(p=1, o=0, q=0)
        params = np.array([10.0, 0.4])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast,
                                    var_bounds, horizon=5, method='bootstrap')
        sigma2 = np.zeros(1000)
        sigma2[0] = params[0] + params[1] * backcast
        for i in range(1, 1000):
            sigma2[i] = params[0] + params[1] * self.resid[i - 1] ** 2.0
        std_resids = self.resid / np.sqrt(sigma2)
        locs = np.floor(1000 * np.random.random_sample((1000, 5))).astype(np.int64)
        std_shocks = std_resids[locs]

        paths = np.zeros((1000, 5))
        paths[:, 0] = params[0] + params[1] * self.resid[-1] ** 2.0
        shocks = np.zeros((1000, 5))
        shocks[:, 0] = np.sqrt(paths[:, 0]) * std_shocks[:, 0]
        for i in range(1, 5):
            paths[:, i] = params[0] + params[1] * shocks[:, i - 1] ** 2
            shocks[:, i] = np.sqrt(paths[:, i]) * std_shocks[:, i]

        assert forecast.forecasts.shape == (1000, 5)
        assert forecast.forecast_paths.shape == (1000, 1000, 5)
        assert forecast.shocks.shape == (1000, 1000, 5)
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert np.all(np.isnan(forecast.forecast_paths[:-1]))
        assert np.all(np.isnan(forecast.shocks[:-1]))
        assert_allclose(forecast.forecasts[-1], paths.mean(0))
        assert_allclose(forecast.forecast_paths[-1], paths)
        assert_allclose(forecast.shocks[-1], shocks)

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=5, method='bootstrap', start=333,
                                    simulations=2000)

        paths = np.zeros((1000, 2000, 5))
        paths.fill(np.nan)
        shocks = np.zeros((1000, 2000, 5))
        shocks.fill(np.nan)
        for i in range(333, 1000):
            locs = np.random.random_sample((2000, 5))
            int_locs = np.floor((i+1) * locs).astype(np.int64)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = params[0] + params[1] * self.resid[i] ** 2.0
            shocks[i, :, 0] = np.sqrt(paths[i, :, 0]) * std_shocks[:, 0]
            for j in range(1, 5):
                paths[i, :, j] = params[0] + params[1] * shocks[i, :, j - 1] ** 2
                shocks[i, :, j] = np.sqrt(paths[i, :, j]) * std_shocks[:, j]

        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.shocks, shocks)

    def test_arch_2_forecast(self):
        vol = GARCH(p=2, o=0, q=0)
        params = np.array([10.0, 0.4, 0.3])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=10,
                                start=0)
        expected = _simple_direct_gjrgarch_forecaster(self.resid, params, 2, 0, 0,
                                                      backcast, var_bounds, 10)
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_garch_11_forecast(self):
        vol = GARCH(p=1, o=0, q=1)
        params = np.array([10.0, 0.1, 0.85])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=10,
                                start=0)
        expected = _simple_direct_gjrgarch_forecaster(self.resid, params, 1, 0, 1,
                                                      backcast, var_bounds, 10)
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_gjrgarch_111_forecast(self):
        vol = GARCH(p=1, o=1, q=1)
        params = np.array([10.0, 0.05, 0.1, 0.85])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=10,
                                start=0)
        expected = _simple_direct_gjrgarch_forecaster(self.resid, params, 1, 1, 1,
                                                      backcast, var_bounds, 10)
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_garch_21_forecast(self):
        vol = GARCH(p=2, o=0, q=1)
        params = np.array([10.0, 0.05, 0.1, 0.85])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=10,
                                start=0)
        expected = _simple_direct_gjrgarch_forecaster(self.resid, params, 2, 0, 1,
                                                      backcast, var_bounds, 10)
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_garch_12_forecast(self):
        vol = GARCH(p=1, o=0, q=2)
        params = np.array([10.0, 0.1, 0.55, 0.3])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=10,
                                start=0)
        expected = _simple_direct_gjrgarch_forecaster(self.resid, params, 1, 0, 2,
                                                      backcast, var_bounds, 10)
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_garch_22_forecast(self):
        vol = GARCH(p=2, o=0, q=2)
        params = np.array([10.0, 0.1, 0.05, 0.4, 0.3])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=10,
                                start=0)
        expected = _simple_direct_gjrgarch_forecaster(self.resid, params, 2, 0, 2,
                                                      backcast, var_bounds, 10)
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_harch_forecast(self):
        vol = HARCH(lags=[1, 5, 22])
        params = np.array([10.0, 0.4, 0.3, 0.2])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=10,
                                start=0)
        trans_params = np.zeros(23)
        trans_params[0] = params[0]
        trans_params[1:2] += params[1]
        trans_params[1:6] += params[2] / 5
        trans_params[1:23] += params[3] / 22
        expected = _simple_direct_gjrgarch_forecaster(self.resid, trans_params, 22, 0, 0,
                                                      backcast, var_bounds, 10)
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_tarch_111_forecast(self):
        t = self.t
        vol = GARCH(p=1, o=1, q=1, power=1.0)
        params = np.array([3.0, 0.1, 0.1, 0.80])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid, power=1.0)

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=1, start=0)
        sigma = np.zeros((t + 1, 1))
        sigma[0] = params[0] + (params[1] + 0.5 * params[2] + params[3]) * backcast

        resid = self.resid
        abs_resid = np.abs(resid)
        sresid = resid < 0
        for i in range(1, t + 1):
            sigma[i] = params[0]
            sigma[i] += params[1] * abs_resid[i - 1]
            sigma[i] += params[2] * abs_resid[i - 1] * sresid[i - 1]
            sigma[i] += params[3] * sigma[i - 1]

        expected = sigma[1:] ** 2.0
        assert_allclose(forecast.forecasts, expected)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with pytest.raises(ValueError):
            vol.forecast(params, self.resid, backcast, var_bounds, horizon=10, start=0)

    def test_tarch_111_forecast_simulation(self):
        t = self.t
        vol = GARCH(p=1, o=1, q=1, power=1.0)
        dist = StudentsT()
        rng = dist.simulate([8.0])
        params = np.array([3.0, 0.1, 0.1, 0.80])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids, power=1.0)
        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=10, start=0, rng=rng, method='simulation')
        one_step = vol.forecast(params, self.resid, backcast, var_bounds, horizon=1,
                                start=0)

        paths = np.zeros((1000, 10))
        shocks = np.zeros((1000, 10))
        for j in range(t):
            std_shocks = rng((1000, 10))
            paths[:, 0] = one_step.forecasts[j, 0]
            shocks[:, 0] = std_shocks[:, 0] * np.sqrt(paths[:, 0])
            sqrt_path = np.sqrt(paths[:, 0])
            for i in range(1, 10):
                abs_err = np.abs(shocks[:, i - 1])
                err_neg = shocks[:, i - 1] < 0
                sqrt_path = params[0] + \
                    params[1] * abs_err + \
                    params[2] * abs_err * err_neg + \
                    params[3] * sqrt_path
                paths[:, i] = sqrt_path ** 2
                shocks[:, i] = sqrt_path * std_shocks[:, i]
            assert_allclose(paths.mean(0), forecast.forecasts[j])
            assert_allclose(paths, forecast.forecast_paths[j])
            assert_allclose(shocks, forecast.shocks[j])

    def test_tarch_111_forecast_bootstrap(self):
        t = self.t
        vol = GARCH(p=1, o=1, q=1, power=1.0)
        dist = StudentsT()
        rng = dist.simulate([8.0])
        params = np.array([3.0, 0.1, 0.1, 0.80])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids, power=1.0)
        sigma2 = np.zeros(t)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=10, start=100, rng=rng, method='bootstrap')

        std_resids = resids / np.sqrt(sigma2)

        one_step = vol.forecast(params, self.resid, backcast, var_bounds, horizon=1,
                                start=100)
        paths = np.zeros((1000, 10))
        shocks = np.zeros((1000, 10))
        for j in range(100, t):
            locs = np.random.random_sample((1000, 10))
            int_locs = np.floor(locs * (j + 1)).astype(np.int)
            std_shocks = std_resids[int_locs]

            paths[:, 0] = one_step.forecasts[j, 0]
            shocks[:, 0] = std_shocks[:, 0] * np.sqrt(paths[:, 0])
            sqrt_path = np.sqrt(paths[:, 0])
            for i in range(1, 10):
                abs_err = np.abs(shocks[:, i - 1])
                err_neg = shocks[:, i - 1] < 0
                sqrt_path = params[0] + \
                    params[1] * abs_err + \
                    params[2] * abs_err * err_neg + \
                    params[3] * sqrt_path
                paths[:, i] = sqrt_path ** 2
                shocks[:, i] = sqrt_path * std_shocks[:, i]
            assert_allclose(paths.mean(0), forecast.forecasts[j])
            assert_allclose(paths, forecast.forecast_paths[j])
            assert_allclose(shocks, forecast.shocks[j])

    def test_harch_forecast_simulation(self):
        t = self.t
        vol = HARCH(lags=[1, 5, 22])
        dist = Normal()
        rng = dist.simulate([])
        params = np.array([3.0, 0.4, 0.3, 0.2])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        with preserved_state():
            forecast = vol.forecast(params, resids, backcast, var_bounds,
                                    horizon=10, start=0, rng=rng, method='simulation')

        resids2 = np.zeros((t, 22 + 10))
        resids2.fill(backcast)
        for i in range(22):
            resids2[21 - i:, i] = resids[:(t - 21 + i)] ** 2.0
        paths = np.zeros((t, 1000, 10))
        shocks = np.zeros((t, 1000, 10))
        const = 3.0
        arch = np.zeros(22)
        arch[0] = 0.4
        arch[:5] += 0.3 / 5
        arch[:22] += 0.2 / 22
        for i in range(t):
            std_shocks = rng((1000, 10))
            temp = np.empty((1000, 32))
            temp[:, :22] = resids2[i:i + 1, :22]
            for j in range(10):
                paths[i, :, j] = const + temp[:, j:22 + j].dot(arch[::-1])
                shocks[i, :, j] = std_shocks[:, j] * np.sqrt(paths[i, :, j])
                temp[:, 22 + j] = shocks[i, :, j] ** 2.0

        forecasts = paths.mean(1)

        assert_allclose(forecast.forecasts, forecasts)
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

    def test_harch_forecast_bootstrap(self):
        t = self.t
        vol = HARCH(lags=[1, 5, 22])
        dist = Normal()
        rng = dist.simulate([])
        params = np.array([3.0, 0.4, 0.3, 0.2])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        sigma2 = np.empty_like(resids)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=10, start=100, rng=rng, method='bootstrap')

        std_resid = resids / np.sqrt(sigma2)
        resids2 = np.zeros((t, 22 + 10))
        resids2.fill(backcast)
        for i in range(22):
            resids2[21 - i:, i] = resids[:(t - 21 + i)] ** 2.0
        paths = np.zeros((t, 1000, 10))
        paths.fill(np.nan)
        shocks = np.zeros((t, 1000, 10))
        shocks.fill(np.nan)
        const = 3.0
        arch = np.zeros(22)
        arch[0] = 0.4
        arch[:5] += 0.3 / 5
        arch[:22] += 0.2 / 22
        for i in range(100, t):
            locs = np.random.random_sample((1000, 10))
            locs *= (i+1)
            int_locs = np.floor(locs).astype(np.int64)
            std_shocks = std_resid[int_locs]
            temp = np.empty((1000, 32))
            temp[:, :22] = resids2[i:i + 1, :22]
            for j in range(10):
                paths[i, :, j] = const + temp[:, j:22 + j].dot(arch[::-1])
                shocks[i, :, j] = std_shocks[:, j] * np.sqrt(paths[i, :, j])
                temp[:, 22 + j] = shocks[i, :, j] ** 2.0

        forecasts = paths.mean(1)

        assert_allclose(forecast.forecasts, forecasts)
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

    def test_egarch_111_forecast(self):
        t = self.t
        dist = Normal()
        rng = dist.simulate([])
        vol = EGARCH(p=1, o=1, q=1)
        params = np.array([0.0, 0.1, 0.1, 0.95])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)

        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1, start=0)

        _resids = np.array(resids.tolist() + [0])
        _sigma2 = np.empty_like(_resids)
        _var_bounds = np.array(var_bounds.tolist() + [[0, np.inf]])
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        sigma2 = _sigma2[:-1]
        one_step = _sigma2[1:][:, None]

        assert_allclose(forecast.forecasts, one_step)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        expected = forecast.forecasts.copy()
        expected[:-1] = np.nan
        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1)
        assert_allclose(forecast.forecasts, expected)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with preserved_state():
            forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=5, start=0,
                                    method='simulation', rng=rng)
        paths = np.empty((t, 1000, 5))
        shocks = np.empty((t, 1000, 5))
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(t):
            std_shocks = rng((1000, 5))
            paths[i, :, 0] = np.log(one_step[i])
            for j in range(1, 5):
                paths[i, :, j] = params[0] + \
                    params[1] * (np.abs(std_shocks[:, j - 1]) - sqrt2pi) + \
                    params[2] * std_shocks[:, j - 1] + \
                    params[3] * paths[i, :, j - 1]
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

        with preserved_state():
            forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=5,
                                    start=100, method='bootstrap')

        std_resids = resids / np.sqrt(sigma2)
        paths = np.empty((t, 1000, 5))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 5))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(100, t):
            locs = np.random.random_sample((1000, 5))
            int_locs = np.floor((i + 1) * locs).astype(np.int64)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = np.log(one_step[i])
            for j in range(1, 5):
                paths[i, :, j] = params[0] + \
                    params[1] * (np.abs(std_shocks[:, j - 1]) - sqrt2pi) + \
                    params[2] * std_shocks[:, j - 1] + \
                    params[3] * paths[i, :, j - 1]
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

    def test_egarch_101_forecast(self):
        t = self.t
        dist = Normal()
        rng = dist.simulate([])
        vol = EGARCH(p=1, o=0, q=1)
        params = np.array([0.0, 0.1, 0.95])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1, start=0)

        _resids = np.array(resids.tolist() + [0])
        _sigma2 = np.empty_like(_resids)
        _var_bounds = np.array(var_bounds.tolist() + [[0, np.inf]])
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        sigma2 = _sigma2[:-1]
        one_step = _sigma2[1:][:, None]

        assert_allclose(forecast.forecasts, one_step)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        expected = forecast.forecasts.copy()
        expected[:-1] = np.nan
        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1)
        assert_allclose(forecast.forecasts, expected)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with preserved_state():
            forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=5, start=0,
                                    method='simulation', rng=rng)
        paths = np.empty((t, 1000, 5))
        shocks = np.empty((t, 1000, 5))
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(t):
            std_shocks = rng((1000, 5))
            paths[i, :, 0] = np.log(one_step[i])
            for j in range(1, 5):
                paths[i, :, j] = params[0] + params[1] * (np.abs(std_shocks[:, j - 1]) - sqrt2pi) \
                                 + params[2] * paths[i, :, j - 1]
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

        with preserved_state():
            forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=5,
                                    start=100, method='bootstrap')

        std_resids = resids / np.sqrt(sigma2)
        paths = np.empty((t, 1000, 5))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 5))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(100, t):
            locs = np.random.random_sample((1000, 5))
            int_locs = np.floor((i + 1) * locs).astype(np.int64)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = np.log(one_step[i])
            for j in range(1, 5):
                paths[i, :, j] = params[0] + params[1] * (np.abs(std_shocks[:, j - 1]) - sqrt2pi) \
                                 + params[2] * paths[i, :, j - 1]
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

    def test_egarch_211_forecast(self):
        t = self.t
        dist = Normal()
        rng = dist.simulate([])
        vol = EGARCH(p=2, o=1, q=1)
        params = np.array([0.0, 0.15, 0.05, 0.1, 0.95])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)

        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1, start=0)

        _resids = np.array(resids.tolist() + [0])
        _sigma2 = np.empty_like(_resids)
        _var_bounds = np.array(var_bounds.tolist() + [[0, np.inf]])
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        one_step = _sigma2[1:][:, None]

        assert_allclose(forecast.forecasts, one_step)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        expected = forecast.forecasts.copy()
        expected[:-1] = np.nan
        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1)
        assert_allclose(forecast.forecasts, expected)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with preserved_state():
            forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=7, start=433,
                                    method='simulation', rng=rng)

        sigma2 = np.empty_like(self.resid)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        std_resids = resids / np.sqrt(sigma2)
        paths = np.empty((t, 1000, 7))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 7))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(433, t):
            std_shocks = rng((1000, 7))
            paths[i, :, 0] = np.log(one_step[i])

            for j in range(1, 7):
                lag_1_sym = np.abs(std_shocks[:, j - 1]) - sqrt2pi
                if j > 1:
                    lag_2_sym = np.abs(std_shocks[:, j - 2]) - sqrt2pi
                else:
                    lag_2_sym = np.abs(std_resids[i]) - sqrt2pi

                paths[i, :, j] = params[0] + \
                    params[1] * lag_1_sym + \
                    params[2] * lag_2_sym + \
                    params[3] * std_shocks[:, j - 1] + \
                    params[4] * paths[i, :, j - 1]
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

        with preserved_state():
            forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=3, start=731,
                                    method='bootstrap', simulations=1111)
        paths = np.empty((t, 1111, 3))
        paths.fill(np.nan)
        shocks = np.empty((t, 1111, 3))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(731, t):
            locs = np.random.random_sample((1111, 3))
            int_locs = np.floor((i+1) * locs).astype(np.int64)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = np.log(one_step[i])

            for j in range(1, 3):
                lag_1_sym = np.abs(std_shocks[:, j - 1]) - sqrt2pi
                if j > 1:
                    lag_2_sym = np.abs(std_shocks[:, j - 2]) - sqrt2pi
                else:
                    lag_2_sym = np.abs(std_resids[i]) - sqrt2pi

                paths[i, :, j] = params[0] + \
                    params[1] * lag_1_sym + \
                    params[2] * lag_2_sym + \
                    params[3] * std_shocks[:, j - 1] + \
                    params[4] * paths[i, :, j - 1]
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

    def test_egarch_212_forecast_smoke(self):
        t = self.t
        dist = Normal()
        rng = dist.simulate([])
        vol = EGARCH(p=2, o=1, q=2)
        params = np.array([0.0, 0.15, 0.05, 0.1, 0.55, 0.4])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)

        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1, start=0)

        _resids = np.array(resids.tolist() + [0])
        _sigma2 = np.empty_like(_resids)
        _var_bounds = np.array(var_bounds.tolist() + [[0, np.inf]])
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        one_step = _sigma2[1:][:, None]

        assert_allclose(forecast.forecasts, one_step)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        expected = forecast.forecasts.copy()
        expected[:-1] = np.nan
        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1)
        assert_allclose(forecast.forecasts, expected)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with preserved_state():
            forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=7, start=433,
                                    method='simulation', rng=rng)

        sigma2 = np.empty_like(self.resid)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        std_resids = resids / np.sqrt(sigma2)
        paths = np.empty((t, 1000, 7))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 7))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(433, t):
            std_shocks = rng((1000, 7))
            paths[i, :, 0] = np.log(one_step[i])

            for j in range(1, 7):
                lag_1_sym = np.abs(std_shocks[:, j - 1]) - sqrt2pi
                lag_1_path = paths[i, :, j - 1]
                if j > 1:
                    lag_2_sym = np.abs(std_shocks[:, j - 2]) - sqrt2pi
                    lag_2_path = paths[i, :, j - 2]
                else:
                    lag_2_sym = np.abs(std_resids[i]) - sqrt2pi
                    lag_2_path = np.log(sigma2[i])

                paths[i, :, j] = params[0] + \
                    params[1] * lag_1_sym + \
                    params[2] * lag_2_sym + \
                    params[3] * std_shocks[:, j - 1] + \
                    params[4] * lag_1_path + \
                    params[5] * lag_2_path
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

        with preserved_state():
            forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=3, start=731,
                                    method='bootstrap', simulations=1111)
        paths = np.empty((t, 1111, 3))
        paths.fill(np.nan)
        shocks = np.empty((t, 1111, 3))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(731, t):
            locs = np.random.random_sample((1111, 3))
            int_locs = np.floor((i+1) * locs).astype(np.int64)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = np.log(one_step[i])

            for j in range(1, 3):
                lag_1_sym = np.abs(std_shocks[:, j - 1]) - sqrt2pi
                lag_1_path = paths[i, :, j - 1]
                if j > 1:
                    lag_2_sym = np.abs(std_shocks[:, j - 2]) - sqrt2pi
                    lag_2_path = paths[i, :, j - 2]
                else:
                    lag_2_sym = np.abs(std_resids[i]) - sqrt2pi
                    lag_2_path = np.log(sigma2[i])

                paths[i, :, j] = params[0] + \
                    params[1] * lag_1_sym + \
                    params[2] * lag_2_sym + \
                    params[3] * std_shocks[:, j - 1] + \
                    params[4] * lag_1_path + \
                    params[5] * lag_2_path
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

    def test_constant_variance_simulation(self):
        t = self.t
        dist = Normal()
        rng = dist.simulate([])
        vol = ConstantVariance()
        params = np.array([10.0])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=1, method='simulation', rng=rng)
        assert forecast.forecasts.shape == (1000, 1)
        assert forecast.forecast_paths.shape == (1000, 1000, 1)
        assert forecast.shocks.shape == (1000, 1000, 1)
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert forecast.forecasts[-1] == params[0]
        assert np.all(np.isnan(forecast.forecast_paths[:-1]))
        assert np.all(forecast.forecast_paths[-1] == params[0])
        assert np.all(np.isnan(forecast.shocks[:-1]))
        assert_allclose(forecast.shocks[-1], np.sqrt(params[0]) * rng((1000, 1)))

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=5, method='simulation', rng=rng)
        assert forecast.forecasts.shape == (1000, 5)
        assert forecast.forecast_paths.shape == (1000, 1000, 5)
        assert forecast.shocks.shape == (1000, 1000, 5)
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert np.all(forecast.forecasts[-1] == params[0])
        assert np.all(np.isnan(forecast.forecast_paths[:-1]))
        assert np.all(forecast.forecast_paths[-1] == params[0])
        assert_allclose(forecast.shocks[-1], np.sqrt(params[0]) * rng((1000, 5)))

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=5, start=100, method='simulation',
                                    simulations=2000, rng=rng)
        assert forecast.forecasts.shape == (1000, 5)
        assert forecast.forecast_paths.shape == (1000, 2000, 5)
        assert forecast.shocks.shape == (1000, 2000, 5)
        assert np.all(np.isnan(forecast.forecasts[:100]))
        assert np.all(forecast.forecasts[100:] == params[0])
        assert np.all(np.isnan(forecast.forecast_paths[:100]))
        assert np.all(forecast.forecast_paths[100:] == params[0])
        expected = np.sqrt(params[0]) * rng((t - 100, 2000, 5))
        expected = np.concatenate((np.empty((100, 2000, 5)), expected))
        expected[:100] = np.nan
        assert_allclose(forecast.shocks, expected)

    def test_constant_variance_bootstrap(self):
        t = self.t
        vol = ConstantVariance()
        params = np.array([10.0])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=5, method='bootstrap')
        assert forecast.forecasts.shape == (1000, 5)
        assert forecast.forecast_paths.shape == (1000, 1000, 5)
        assert forecast.shocks.shape == (1000, 1000, 5)
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert np.all(forecast.forecasts[-1] == params[0])
        assert np.all(np.isnan(forecast.forecast_paths[:-1]))
        assert np.all(forecast.forecast_paths[-1] == params[0])
        index = np.floor(np.random.random_sample((1000, 5)) * t)
        index = index.astype(np.int64)
        assert_allclose(forecast.shocks[-1], self.resid[index])

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=5, method='bootstrap', start=100,
                                    simulations=2000)
        assert forecast.forecasts.shape == (1000, 5)
        assert forecast.forecast_paths.shape == (1000, 2000, 5)
        assert forecast.shocks.shape == (1000, 2000, 5)
        assert np.all(np.isnan(forecast.forecasts[:100]))
        assert np.all(forecast.forecasts[100:] == params[0])
        assert np.all(np.isnan(forecast.forecast_paths[:100]))
        assert np.all(forecast.forecast_paths[100:] == params[0])
        expected = np.empty((1000, 2000, 5))
        expected.fill(np.nan)
        for i in range(100, 1000):
            index = np.random.random_sample((2000, 5))
            int_index = np.floor((i + 1) * index).astype(np.int64)
            expected[i] = self.resid[int_index]
        assert_allclose(forecast.shocks, expected)

    def test_garch11_simulation(self):
        t = self.t
        vol = GARCH(p=1, o=0, q=1)
        dist = Normal()
        rng = dist.simulate([])
        params = np.array([10.0, 0.1, 0.85])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=10, method='simulation', rng=rng)
        assert forecast.forecasts.shape == (t, 10)
        assert forecast.forecast_paths.shape == (t, 1000, 10)
        assert forecast.shocks.shape == (t, 1000, 10)
        assert np.all(np.isnan(forecast.forecast_paths[:-1]))
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert np.all(np.isnan(forecast.shocks[:-1]))
        std_shocks = rng((1000, 10))
        one_step = vol.forecast(params, self.resid, backcast, var_bounds,
                                horizon=1, start=0)
        paths = np.zeros((1000, 10))
        shocks = np.zeros((1000, 10))
        paths[:, 0] = one_step.forecasts[-1]
        shocks[:, 0] = np.sqrt(paths[:, 0]) * std_shocks[:, 0]
        for i in range(1, 10):
            paths[:, i] = params[0]
            paths[:, i] += params[1] * shocks[:, i - 1] ** 2.0
            paths[:, i] += params[2] * paths[:, i - 1]
            shocks[:, i] = np.sqrt(paths[:, i]) * std_shocks[:, i]
        forecasts = paths.mean(0)
        assert_allclose(forecast.forecasts[-1], forecasts)
        assert_allclose(forecast.forecast_paths[-1], paths)
        assert_allclose(forecast.shocks[-1], shocks)

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=10, method='simulation',
                                    simulations=2000, rng=rng)
        assert np.all(np.isnan(forecast.forecast_paths[:-1]))
        assert np.all(np.isnan(forecast.forecasts[:-1]))
        assert np.all(np.isnan(forecast.shocks[:-1]))
        assert forecast.forecasts.shape == (t, 10)
        assert forecast.forecast_paths.shape == (t, 2000, 10)
        assert forecast.shocks.shape == (t, 2000, 10)
        std_shocks = rng((2000, 10))
        paths = np.zeros((2000, 10))
        shocks = np.zeros((2000, 10))
        paths[:, 0] = one_step.forecasts[-1]
        shocks[:, 0] = np.sqrt(paths[:, 0]) * std_shocks[:, 0]
        for i in range(1, 10):
            paths[:, i] = params[0]
            paths[:, i] += params[1] * shocks[:, i - 1] ** 2.0
            paths[:, i] += params[2] * paths[:, i - 1]
            shocks[:, i] = np.sqrt(paths[:, i]) * std_shocks[:, i]
        forecasts = paths.mean(0)
        assert_allclose(forecast.forecasts[-1], forecasts)
        assert_allclose(forecast.forecast_paths[-1], paths)
        assert_allclose(forecast.shocks[-1], shocks)

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=3, method='simulation', start=0,
                                    rng=rng)
        paths = np.zeros((1000, 3))
        shocks = np.zeros((1000, 3))
        for j in range(t):
            std_shocks = rng((1000, 3))
            paths[:, 0] = one_step.forecasts[j]
            shocks[:, 0] = np.sqrt(paths[:, 0]) * std_shocks[:, 0]
            for i in range(1, 3):
                paths[:, i] = params[0]
                paths[:, i] += params[1] * shocks[:, i - 1] ** 2.0
                paths[:, i] += params[2] * paths[:, i - 1]
                shocks[:, i] = np.sqrt(paths[:, i]) * std_shocks[:, i]
            forecasts = paths.mean(0)
            assert_allclose(forecast.forecasts[j], forecasts)
            assert_allclose(forecast.forecast_paths[j], paths)
            assert_allclose(forecast.shocks[j], shocks)

    def test_garch11_bootstrap(self):
        t = self.t
        vol = GARCH(p=1, o=0, q=1)
        params = np.array([10.0, 0.1, 0.85])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=10, method='bootstrap', start=100)

        paths = np.empty((t, 1000, 10))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 10))
        shocks.fill(np.nan)

        _sigma2 = np.empty(t + 1)
        _resids = np.concatenate((self.resid, [0]))
        _var_bounds = np.concatenate((var_bounds, [[0, np.inf]]))
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        std_resid = self.resid / np.sqrt(_sigma2[:-1])
        one_step = _sigma2[1:]
        omega, alpha, beta = params
        for i in range(100, t):
            locs = np.floor((i + 1) * np.random.random_sample((1000, 10))).astype(np.int64)
            std_shocks = std_resid[locs]
            paths[i, :, 0] = one_step[i]
            shocks[i, :, 0] = np.sqrt(paths[i, :, 0]) * std_shocks[:, 0]
            for j in range(1, 10):
                paths[i, :, j] = omega + \
                    alpha * shocks[i, :, j - 1] ** 2.0 + \
                    beta * paths[i, :, j - 1]
                shocks[i, :, j] = np.sqrt(paths[i, :, j]) * std_shocks[:, j]

        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(paths, forecast.forecast_paths)
        assert_allclose(shocks, forecast.shocks)

        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=10,
                                    method='bootstrap', simulations=2000, start=100)

        assert forecast.forecasts.shape == (t, 10)
        assert forecast.forecast_paths.shape == (t, 2000, 10)
        assert forecast.shocks.shape == (t, 2000, 10)

        assert np.all(np.isnan(forecast.forecasts[:100]))
        assert np.all(np.isnan(forecast.forecast_paths[:100]))
        assert np.all(np.isnan(forecast.shocks[:100]))

    def test_gjrgarch222_simulation(self):
        t = self.t
        vol = GARCH(p=2, o=2, q=2)
        params = np.array([10.0, 0.05, 0.03, 0.1, 0.05, 0.3, 0.2])
        dist = Normal()
        rng = dist.simulate([])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        one_step = vol.forecast(params, self.resid, backcast, var_bounds,
                                horizon=1, start=0)
        with preserved_state():
            forecast = vol.forecast(params, self.resid, backcast, var_bounds,
                                    horizon=3, method='simulation', start=0,
                                    rng=rng, simulations=100)
        paths = np.zeros((100, 3))
        shocks = np.zeros((100, 3))

        resids = self.resid
        sigma2_0 = params[0] + (np.sum(params[1:]) - 0.5 * np.sum(params[3:5])) * backcast
        _sigma2 = np.concatenate([[backcast], [sigma2_0], one_step.forecasts[:-1, 0]])
        _resids = np.concatenate([[np.sqrt(backcast)], resids])
        _asymresids = np.concatenate(([[np.sqrt(0.5 * backcast)], resids * (resids < 0)]))

        for t in range(resids.shape[0]):
            std_shocks = rng((100, 3))

            j = 0
            tau = t + 1 + j + 1
            r1, r2 = _resids[tau - 1], _resids[tau - 2]
            a1, a2 = _asymresids[tau - 1], _asymresids[tau - 2]
            s21, s22 = _sigma2[tau - 1], _sigma2[tau - 2]

            fcast = params[0] + params[1] * r1 ** 2 + params[2] * r2 ** 2 + \
                params[3] * a1 ** 2 + params[4] * a2 ** 2 + \
                params[5] * s21 + params[6] * s22
            paths[:, j] = fcast
            shocks[:, j] = std_shocks[:, j] * np.sqrt(fcast)

            j = 1
            r1, r2 = shocks[:, 0], _resids[tau - 1]
            a1, a2 = shocks[:, 0] * (shocks[:, 0] < 0), _asymresids[tau - 1]
            s21, s22 = paths[:, 0], _sigma2[tau - 1]
            fcast = params[0] + params[1] * r1 ** 2 + params[2] * r2 ** 2 + \
                params[3] * a1 ** 2 + params[4] * a2 ** 2 + \
                params[5] * s21 + params[6] * s22
            paths[:, j] = fcast
            shocks[:, j] = std_shocks[:, j] * np.sqrt(fcast)

            j = 2
            r1, r2 = shocks[:, 1], shocks[:, 0]
            a1, a2 = shocks[:, 1] * (shocks[:, 1] < 0), shocks[:, 0] * (shocks[:, 0] < 0)
            s21, s22 = paths[:, 1], paths[:, 0]
            fcast = params[0] + params[1] * r1 ** 2 + params[2] * r2 ** 2 + \
                params[3] * a1 ** 2 + params[4] * a2 ** 2 + \
                params[5] * s21 + params[6] * s22
            paths[:, j] = fcast
            shocks[:, j] = std_shocks[:, j] * np.sqrt(fcast)

            forecasts = paths.mean(0)
            assert_allclose(forecast.forecasts[t], forecasts)
            assert_allclose(forecast.forecast_paths[t], paths)
            assert_allclose(forecast.shocks[t], shocks)

    def test_ewma_forecast(self):
        t = self.t
        vol = EWMAVariance()
        params = np.array([])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)

        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=10, start=0)

        expected = np.zeros((t+1))
        expected[0] = backcast
        lam = 0.94
        for i in range(1, t+1):
            expected[i] = lam * expected[i-1] + (1-lam) * resids[i-1] ** 2
        for i in range(10):
            assert_allclose(forecast.forecasts[:, i], expected[1:])

        alt_forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=10, start=500)
        _compare_truncated_forecasts(forecast, alt_forecast, 500)

    def test_ewma_simulation(self):
        t = self.t
        vol = EWMAVariance()
        params = np.array([])
        resids = self.resid
        dist = Normal()
        rng = dist.simulate([])
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        with preserved_state():
            forecasts = vol.forecast(params, resids, backcast, var_bounds, horizon=10,
                                     start=0, method='simulation', rng=rng)

        one_step = np.empty(t + 1)
        one_step[0] = backcast
        for i in range(1, t + 1):
            one_step[i] = vol.lam * one_step[i - 1] + (1 - vol.lam) * resids[i - 1] ** 2
        one_step = one_step[1:]
        paths = np.empty((t, 1000, 10))
        shocks = np.empty((t, 1000, 10))
        for i in range(t):
            std_shocks = rng((1000, 10))
            paths[i, :, 0] = one_step[i]
            shocks[i, :, 0] = np.sqrt(paths[i, :, 0]) * std_shocks[:, 0]
            for j in range(1, 10):
                paths[i, :, j] = vol.lam * paths[i, :, j - 1] + \
                    (1 - vol.lam) * shocks[i, :, j - 1] ** 2
                shocks[i, :, j] = np.sqrt(paths[i, :, j]) * std_shocks[:, j]

        assert_allclose(forecasts.forecasts, paths.mean(1))
        assert_allclose(forecasts.forecast_paths, paths)
        assert_allclose(forecasts.shocks, shocks)

        forecasts = vol.forecast(params, resids, backcast, var_bounds, horizon=10,
                                 start=252, method='simulation', rng=rng)

        assert np.all(np.isnan(forecasts.forecasts[:252]))
        assert np.all(np.isnan(forecasts.forecast_paths[:252]))
        assert np.all(np.isnan(forecasts.shocks[:252]))

        assert np.all(np.isfinite(forecasts.forecasts[252:]))
        assert np.all(np.isfinite(forecasts.forecast_paths[252:]))
        assert np.all(np.isfinite(forecasts.shocks[252:]))

    def test_ewma_bootstrap(self):
        t = self.t
        vol = EWMAVariance()
        params = np.array([])
        resids = self.resid
        dist = Normal()
        rng = dist.simulate([])
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        with preserved_state():
            forecasts = vol.forecast(params, resids, backcast, var_bounds, horizon=10,
                                     start=131, method='bootstrap', rng=rng)

        sigma2 = np.empty_like(self.resid)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        std_resids = resids / np.sqrt(sigma2)

        one_step = np.empty(t + 1)
        one_step[0] = backcast
        for i in range(1, t + 1):
            one_step[i] = vol.lam * one_step[i - 1] + (1 - vol.lam) * resids[i - 1] ** 2
        one_step = one_step[1:]
        paths = np.empty((t, 1000, 10))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 10))
        shocks.fill(np.nan)
        for i in range(131, t):
            locs = np.random.random_sample((1000, 10))
            int_locs = np.floor(locs * (i + 1)).astype(np.int)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = one_step[i]
            shocks[i, :, 0] = np.sqrt(paths[i, :, 0]) * std_shocks[:, 0]
            for j in range(1, 10):
                paths[i, :, j] = vol.lam * paths[i, :, j - 1] + \
                                 (1 - vol.lam) * shocks[i, :, j - 1] ** 2
                shocks[i, :, j] = np.sqrt(paths[i, :, j]) * std_shocks[:, j]

        assert_allclose(forecasts.forecasts, paths.mean(1))
        assert_allclose(forecasts.forecast_paths, paths)
        assert_allclose(forecasts.shocks, shocks)

        forecasts = vol.forecast(params, resids, backcast, var_bounds, horizon=10,
                                 start=252, method='simulation', rng=rng)

        assert np.all(np.isnan(forecasts.forecasts[:252]))
        assert np.all(np.isnan(forecasts.forecast_paths[:252]))
        assert np.all(np.isnan(forecasts.shocks[:252]))

        assert np.all(np.isfinite(forecasts.forecasts[252:]))
        assert np.all(np.isfinite(forecasts.forecast_paths[252:]))
        assert np.all(np.isfinite(forecasts.shocks[252:]))

    def test_rm2006_forecast(self):
        vol = RiskMetrics2006()
        params = np.array([])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        forecasts = vol.forecast(params, resids, backcast, var_bounds, horizon=10, start=0)
        _resids = np.array(resids.tolist() + [0])
        _sigma2 = np.empty_like(_resids)
        _var_bounds = np.array(var_bounds.tolist() + [[0, np.inf]])
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        assert forecasts.forecast_paths is None
        assert forecasts.shocks is None
        one_step = _sigma2[1:]
        for i in range(10):
            assert_allclose(forecasts.forecasts[:, i], one_step, rtol=1e-4)

        alt_forecasts = vol.forecast(params, resids, backcast, var_bounds, horizon=10, start=500)
        _compare_truncated_forecasts(forecasts, alt_forecasts, 500)

    def test_rm2006_simulation_smoke(self):
        dist = Normal()
        rng = dist.simulate([])
        vol = RiskMetrics2006()
        params = np.array([])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        with preserved_state():
            vol.forecast(params, resids, backcast, var_bounds, horizon=10, start=0,
                         method='simulation', rng=rng)

    def test_rm2006_bootstrap_smoke(self):
        vol = RiskMetrics2006()
        params = np.array([])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        with preserved_state():
            vol.forecast(params, resids, backcast, var_bounds, horizon=10, start=100,
                         method='bootstrap')


class TestBootstrapRng(TestCase):
    def test_bs_rng(self):
        y = np.random.rand(1000)
        bs_rng = BootstrapRng(y, 100)
        rng = bs_rng.rng()
        size = (1231, 13)
        with preserved_state():
            output = {i: rng(size) for i in range(100, 1000)}

        expected = {}
        for i in range(100, 1000):
            locs = np.random.random_sample(size)
            expected[i] = y[np.floor(locs * (i + 1)).astype(np.int64)]

        for i in range(100, 1000):
            assert_allclose(expected[i], output[i])

    def test_bs_rng_errors(self):
        y = np.random.rand(1000)
        bs_rng = BootstrapRng(y, 100)
        rng = bs_rng.rng()
        with pytest.raises(IndexError):
            for i in range(100, 1001):
                rng(1)

        y = np.random.rand(1000)
        with pytest.raises(ValueError):
            BootstrapRng(y, 0)
