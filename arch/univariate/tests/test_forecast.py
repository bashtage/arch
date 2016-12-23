import datetime as dt
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.util.testing import assert_frame_equal

from ..mean import _ar_to_impulse, _ar_forecast
from ...univariate import arch_model
from .test_variance_forecasting import preserved_state


class TestForecasting(TestCase):
    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        am = arch_model(None, mean='Constant', vol='Constant')
        data = am.simulate(np.array([0.0, 1.0]), 1000)
        data.index = pd.date_range('2000-01-01', periods=data.index.shape[0])
        cls.zero_mean = data.data

        am = arch_model(None, mean='AR', vol='Constant', lags=[1])
        data = am.simulate(np.array([1.0, 0.9, 0.1]), 1000)
        data.index = pd.date_range('2000-01-01', periods=data.index.shape[0])
        cls.ar1 = data.data

        am = arch_model(None, mean='AR', vol='Constant', lags=[1, 2])
        data = am.simulate(np.array([1.0, 1.9, -0.95, 0.1]), 1000)
        data.index = pd.date_range('2000-01-01', periods=data.index.shape[0])
        cls.ar2 = data.data

        am = arch_model(None, mean='HAR', vol='Constant', lags=[1, 5, 22])
        data = am.simulate(np.array([1.0, 0.4, 0.3, 0.2, 0.1]), 1000)
        data.index = pd.date_range('2000-01-01', periods=data.index.shape[0])
        cls.har3 = data.data

        am = arch_model(None, mean='AR', vol='GARCH', lags=[1, 2], p=1, q=1)
        data = am.simulate(np.array([1.0, 1.9, -0.95, 0.1, 0.1, 0.88]), 1000)
        data.index = pd.date_range('2000-01-01', periods=data.index.shape[0])
        cls.ar2_garch = data.data

    def test_ar_forecasting(self):
        params = np.array([0.9])
        forecasts = _ar_forecast(self.zero_mean, 5, 0, 0.0, params)
        expected = np.zeros((1000, 5))
        expected[:, 0] = 0.9 * self.zero_mean.values
        for i in range(1, 5):
            expected[:, i] = 0.9 * expected[:, i - 1]
        assert_allclose(forecasts, expected)

        params = np.array([0.5, -0.3, 0.2])
        forecasts = _ar_forecast(self.zero_mean, 5, 2, 0.0, params)
        expected = np.zeros((998, 8))
        expected[:, 0] = self.zero_mean.iloc[0:-2]
        expected[:, 1] = self.zero_mean.iloc[1:-1]
        expected[:, 2] = self.zero_mean.iloc[2:]
        for i in range(3, 8):
            expected[:, i] = 0.5 * expected[:, i - 1] - \
                             0.3 * expected[:, i - 2] + \
                             0.2 * expected[:, i - 3]
        fill = np.empty((2, 5))
        fill.fill(np.nan)
        expected = np.concatenate((fill, expected[:, 3:]))
        assert_allclose(forecasts, expected)

    def test_ar_to_impulse(self):
        arp = np.array([0.9])
        impulses = _ar_to_impulse(20, arp)
        expected = 0.9 ** np.arange(20)
        assert_allclose(impulses, expected)

        arp = np.array([0.5, 0.3])
        impulses = _ar_to_impulse(20, arp)
        comp = np.array([arp, [1, 0]])
        a = comp.copy()
        expected = np.ones(20)
        for i in range(1, 20):
            expected[i] = a[0, 0]
            a = a.dot(comp)
        assert_allclose(impulses, expected)

        arp = np.array([1.5, 0.0, -0.7])
        impulses = _ar_to_impulse(20, arp)
        comp = np.array([arp, [1, 0, 0], [0, 1, 0]])
        a = comp.copy()
        expected = np.ones(20)
        for i in range(1, 20):
            expected[i] = a[0, 0]
            a = a.dot(comp)
        assert_allclose(impulses, expected)

    def test_zero_mean_forecast(self):
        am = arch_model(self.zero_mean, mean='Zero', vol='Constant')
        res = am.fit()
        fcast = res.forecast(res.params, horizon=3)
        alt_fcast = res.forecast(horizon=3)
        assert_frame_equal(fcast.mean, alt_fcast.mean)
        assert_frame_equal(fcast.variance, alt_fcast.variance)
        assert_frame_equal(fcast.residual_variance,
                           alt_fcast.residual_variance)

        assert np.all(np.isnan(fcast.mean[:-1]))
        assert np.all(np.isnan(fcast.variance[:-1]))
        assert np.all(np.isnan(fcast.residual_variance[:-1]))

        params = np.asarray(res.params)
        assert np.all(0.0 == fcast.mean.iloc[-1])
        assert_allclose(fcast.variance.iloc[-1],
                        np.ones(3) * params[0])
        assert_allclose(fcast.residual_variance.iloc[-1],
                        np.ones(3) * params[0])

        res = am.fit(last_obs=500)
        params = np.asarray(res.params)
        fcast = res.forecast(horizon=3)
        assert np.all(np.isnan(fcast.mean[:499]))
        assert np.all(np.isnan(fcast.variance[:499]))
        assert np.all(np.isnan(fcast.residual_variance[:499]))

        assert np.all(0.0 == fcast.mean[499:])
        assert_allclose(fcast.variance.iloc[499:],
                        np.ones((501, 3)) * params[0])
        assert_allclose(fcast.residual_variance.iloc[499:],
                        np.ones((501, 3)) * params[0])

    def test_frame_labels(self):
        am = arch_model(self.zero_mean, mean='Zero', vol='Constant')
        res = am.fit()
        fcast = res.forecast(horizon=12)
        assert fcast.mean.shape[1] == 12
        assert fcast.variance.shape[1] == 12
        assert fcast.residual_variance.shape[1] == 12
        for i in range(1, 13):
            if i < 10:
                col = 'h.0' + str(i)
            else:
                col = 'h.' + str(i)

            assert col in fcast.mean.columns
            assert col in fcast.variance.columns
            assert col in fcast.residual_variance.columns

    def test_ar1_forecast(self):
        am = arch_model(self.ar1, mean='AR', vol='Constant', lags=[1])
        res = am.fit()

        fcast = res.forecast(horizon=5, start=0)
        params = np.asarray(res.params)
        direct = self.ar1.values

        for i in range(5):
            direct = params[0] + params[1] * direct
            assert_allclose(direct, fcast.mean.iloc[:, i])
            scale = np.sum((params[1] ** np.arange(i + 1)) ** 2.0)
            var = fcast.variance.iloc[1:, i]
            assert_allclose(var, scale * params[2] * np.ones_like(var))

        assert np.all(fcast.residual_variance[1:] == params[2])

        fcast = res.forecast(horizon=5)
        params = np.asarray(res.params)
        assert np.all(np.isnan(fcast.mean[:-1]))
        assert np.all(np.isnan(fcast.variance[:-1]))
        assert np.all(np.isnan(fcast.residual_variance[:-1]))
        assert np.all(fcast.residual_variance.iloc[-1] == params[-1])
        means = np.zeros(5)
        means[0] = params[0] + params[1] * self.ar1.iloc[-1]
        for i in range(1, 5):
            means[i] = params[0] + params[1] * means[i - 1]
        assert_allclose(means, fcast.mean.iloc[-1].values)

    def test_constant_mean_forecast(self):
        am = arch_model(self.zero_mean, mean='Constant', vol='Constant')
        res = am.fit()
        fcast = res.forecast(horizon=5)

        assert np.all(np.isnan(fcast.mean[:-1]))
        assert np.all(np.isnan(fcast.variance[:-1]))
        assert np.all(np.isnan(fcast.residual_variance[:-1]))
        params = np.asarray(res.params)
        assert_allclose(params[0] * np.ones(5), fcast.mean.iloc[-1])
        assert_allclose(params[1] * np.ones(5), fcast.variance.iloc[-1])
        assert_allclose(params[1] * np.ones(5), fcast.residual_variance.iloc[-1])

        assert fcast.mean.shape == (self.zero_mean.shape[0], 5)
        assert fcast.variance.shape == (self.zero_mean.shape[0], 5)
        assert fcast.residual_variance.shape == (self.zero_mean.shape[0], 5)

    def test_ar2_forecast(self):
        am = arch_model(self.ar2, mean='AR', vol='Constant', lags=[1, 2])
        res = am.fit()

        fcast = res.forecast(horizon=5)
        params = np.asarray(res.params)
        expected = np.zeros(7)
        expected[:2] = self.ar2.iloc[-2:]
        for i in range(2, 7):
            expected[i] = params[0] + \
                          params[1] * expected[i - 1] + \
                          params[2] * expected[i - 2]

        expected = expected[2:]
        assert np.all(np.isnan(fcast.mean.iloc[:-1]))
        assert_allclose(fcast.mean.iloc[-1], expected)

        expected = np.zeros(5)
        comp = np.array([res.params.iloc[1:3], [1, 0]])
        a = np.eye(2)
        for i in range(5):
            expected[i] = a[0, 0]
            a = a.dot(comp)
        expected = res.params.iloc[-1] * np.cumsum(expected ** 2)
        assert_allclose(fcast.variance.iloc[-1], expected)

        expected = np.empty((1000, 5))
        expected[:2] = np.nan
        expected[2:] = res.params.iloc[-1]

        fcast = res.forecast(horizon=5, start=1)
        expected = np.zeros((999, 7))
        expected[:, 0] = self.ar2.iloc[0:-1]
        expected[:, 1] = self.ar2.iloc[1:]
        for i in range(2, 7):
            expected[:, i] = params[0] + \
                             params[1] * expected[:, i - 1] + \
                             params[2] * expected[:, i - 2]
        fill = np.empty((1, 5))
        fill.fill(np.nan)
        expected = np.concatenate((fill, expected[:, 2:]))
        assert_allclose(fcast.mean.values, expected)

        expected = np.empty((1000, 5))
        expected[:2] = np.nan
        expected[2:] = res.params.iloc[-1]
        assert_allclose(fcast.residual_variance.values, expected)

        with pytest.raises(ValueError):
            fcast = res.forecast(horizon=5, start=0)

    def test_har_forecast(self):
        am = arch_model(self.har3, mean='HAR', vol='Constant', lags=[1, 5, 22])
        res = am.fit()
        fcast_1 = res.forecast(horizon=1)
        fcast_5 = res.forecast(horizon=5)
        assert_allclose(fcast_1.mean, fcast_5.mean.iloc[:, :1])

        with pytest.raises(ValueError):
            res.forecast(horizon=1, start=0)
        with pytest.raises(ValueError):
            res.forecast(horizon=1, start=20)

        fcast_66 = res.forecast(horizon=66, start=21)
        expected = np.empty((1000, 66 + 22))
        expected.fill(np.nan)
        for i in range(22):
            if i < 21:
                expected[21:, i] = self.har3.iloc[i:(-21 + i)]
            else:
                expected[21:, i] = self.har3.iloc[i:]
        params = np.asarray(res.params)
        const = params[0]
        arp = np.zeros(22)
        arp[0] = params[1]
        arp[:5] += params[2] / 5
        arp[:22] += params[3] / 22
        arp_rev = arp[::-1]
        for i in range(22, 88):
            expected[:, i] = const + expected[:, i - 22:i].dot(arp_rev)
        expected = expected[:, 22:]
        assert_allclose(fcast_66.mean, expected)

        expected[:22] = np.nan
        expected[22:] = res.params.iloc[-1]
        assert_allclose(fcast_66.residual_variance, expected)

        impulse = _ar_to_impulse(66, arp)
        expected = expected * np.cumsum(impulse ** 2)
        assert_allclose(fcast_66.variance, expected)

    def test_forecast_start_alternatives(self):
        am = arch_model(self.har3, mean='HAR', vol='Constant', lags=[1, 5, 22])
        res = am.fit()
        date = self.har3.index[21]
        fcast_1 = res.forecast(start=21)
        fcast_2 = res.forecast(start=date)
        for field in ('mean', 'variance', 'residual_variance'):
            assert_frame_equal(getattr(fcast_1, field),
                               getattr(fcast_2, field))
        pydt = dt.datetime(date.year, date.month, date.day)
        fcast_2 = res.forecast(start=pydt)
        for field in ('mean', 'variance', 'residual_variance'):
            assert_frame_equal(getattr(fcast_1, field),
                               getattr(fcast_2, field))

        strdt = pydt.strftime('%Y-%m-%d')
        fcast_2 = res.forecast(start=strdt)
        for field in ('mean', 'variance', 'residual_variance'):
            assert_frame_equal(getattr(fcast_1, field),
                               getattr(fcast_2, field))

        npydt = np.datetime64(pydt).astype('M8[ns]')
        fcast_2 = res.forecast(start=npydt)
        for field in ('mean', 'variance', 'residual_variance'):
            assert_frame_equal(getattr(fcast_1, field),
                               getattr(fcast_2, field))

        with pytest.raises(ValueError):
            date = self.har3.index[20]
            res.forecast(start=date)

        with pytest.raises(ValueError):
            date = self.har3.index[0]
            res.forecast(start=date)

        fcast_0 = res.forecast()
        fcast_1 = res.forecast(start=999)
        fcast_2 = res.forecast(start=self.har3.index[999])
        for field in ('mean', 'variance', 'residual_variance'):
            assert_frame_equal(getattr(fcast_0, field),
                               getattr(fcast_1, field))

            assert_frame_equal(getattr(fcast_0, field),
                               getattr(fcast_2, field))

    def test_fit_options(self):
        am = arch_model(self.zero_mean, mean='Constant', vol='Constant')
        res = am.fit(first_obs=100)
        res.forecast()
        res = am.fit(last_obs=900)
        res.forecast()
        res = am.fit(first_obs=100, last_obs=900)
        res.forecast()
        res.forecast(start=100)
        res.forecast(start=200)
        am = arch_model(self.zero_mean, mean='Constant', vol='Constant',
                        hold_back=20)
        res = am.fit(first_obs=100)
        res.forecast()

    def test_ar1_forecast_simulation(self):
        am = arch_model(self.ar1, mean='AR', vol='GARCH', lags=[1])
        res = am.fit(disp='off')

        with preserved_state():
            forecast = res.forecast(horizon=5, start=0, method='simulation')

        y = np.asarray(self.ar1)
        index = self.ar1.index
        t = y.shape[0]
        params = np.array(res.params)
        resids = np.asarray(y[1:] - params[0] - params[1] * y[:-1])
        vol = am.volatility
        params = np.array(res.params)
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        rng = am.distribution.simulate([])
        vfcast = vol.forecast(params[2:], resids, backcast, var_bounds, start=0,
                              method='simulation', rng=rng, horizon=5)
        const, ar = params[0], params[1]
        means = np.zeros((t, 5))
        means[:, 0] = const + ar * y
        for i in range(1, 5):
            means[:, i] = const + ar * means[:, i-1]
        means = pd.DataFrame(means, index=index,
                             columns=['h.{0}'.format(j) for j in range(1, 6)])
        assert_frame_equal(means, forecast.mean)
        var = np.concatenate([[[np.nan]*5], vfcast.forecasts])
        rv = pd.DataFrame(var, index=index,
                          columns=['h.{0}'.format(j) for j in range(1, 6)])
        assert_frame_equal(rv, forecast.residual_variance)

        lrv = rv.copy()
        for i in range(5):
            weights = (ar ** np.arange(i+1)) ** 2
            weights = weights[:, None]
            lrv.iloc[:, i:i+1] = rv.values[:, :i+1].dot(weights[::-1])
        assert_frame_equal(lrv, forecast.variance)

    def test_ar2_garch11(self):
        pass

    def test_first_obs(self):
        y = self.ar2_garch
        mod = arch_model(y)
        res = mod.fit(disp='off', first_obs=y.index[100])
        mod = arch_model(y[100:])
        res2 = mod.fit(disp='off')
        assert_allclose(res.params, res2.params)
        mod = arch_model(y)
        res3 = mod.fit(disp='off', first_obs=100)
        assert res.fit_start == 100
        assert_allclose(res.params, res3.params)

        forecast = res.forecast(horizon=3)
        assert np.all(np.isnan(forecast.mean.iloc[:-1]))
        assert np.all(np.isfinite(forecast.mean.iloc[-1]))
        assert np.all(np.isnan(forecast.variance.iloc[:-1]))
        assert np.all(np.isfinite(forecast.variance.iloc[-1]))

        forecast = res.forecast(horizon=3, start=y.index[100])
        assert np.all(np.isnan(forecast.mean.iloc[:100]))
        assert np.all(np.isfinite(forecast.mean.iloc[100:]))
        assert np.all(np.isnan(forecast.variance.iloc[:100]))
        assert np.all(np.isfinite(forecast.variance.iloc[100:]))

        forecast = res.forecast(horizon=3, start=100)
        assert np.all(np.isnan(forecast.mean.iloc[:100]))
        assert np.all(np.isfinite(forecast.mean.iloc[100:]))
        assert np.all(np.isnan(forecast.variance.iloc[:100]))
        assert np.all(np.isfinite(forecast.variance.iloc[100:]))

        with pytest.raises(ValueError):
            res.forecast(horizon=3, start=y.index[98])

        res = mod.fit(disp='off')
        forecast = res.forecast(horizon=3)
        assert np.all(np.isnan(forecast.mean.iloc[:-1]))
        assert np.all(np.isfinite(forecast.mean.iloc[-1]))
        assert np.all(np.isnan(forecast.variance.iloc[:-1]))
        assert np.all(np.isfinite(forecast.variance.iloc[-1]))

        forecast = res.forecast(horizon=3, start=y.index[100])
        assert np.all(np.isnan(forecast.mean.iloc[:100]))
        assert np.all(np.isfinite(forecast.mean.iloc[100:]))
        assert np.all(np.isnan(forecast.variance.iloc[:100]))
        assert np.all(np.isfinite(forecast.variance.iloc[100:]))
        forecast = res.forecast(horizon=3, start=0)
        assert np.all(np.isfinite(forecast.mean))
        assert np.all(np.isfinite(forecast.variance))

        mod = arch_model(y, mean='AR', lags=[1, 2])
        res = mod.fit(disp='off')
        with pytest.raises(ValueError):
            res.forecast(horizon=3, start=0)

        forecast = res.forecast(horizon=3, start=1)
        assert np.all(np.isnan(forecast.mean.iloc[0]))
        assert np.all(np.isfinite(forecast.mean.iloc[1:]))
        assert np.all(np.isnan(forecast.variance.iloc[0]))
        assert np.all(np.isfinite(forecast.variance.iloc[2:]))

    def test_last_obs(self):
        y = self.ar2_garch
        mod = arch_model(y)
        res = mod.fit(disp='off', last_obs=y.index[900])
        res_2 = mod.fit(disp='off', last_obs=900)
        assert_allclose(res.params, res_2.params)
        mod = arch_model(y[:900])
        res_3 = mod.fit(disp='off')
        assert_allclose(res.params, res_3.params)

    def test_first_last_obs(self):
        y = self.ar2_garch
        mod = arch_model(y)
        res = mod.fit(disp='off', first_obs=y.index[100], last_obs=y.index[900])
        res_2 = mod.fit(disp='off', first_obs=100, last_obs=900)
        assert_allclose(res.params, res_2.params)
        mod = arch_model(y.iloc[100:900])
        res_3 = mod.fit(disp='off')
        assert_allclose(res.params, res_3.params)

        mod = arch_model(y)
        res_4 = mod.fit(disp='off', first_obs=100, last_obs=y.index[900])
        assert_allclose(res.params, res_4.params)

    def test_holdback_first_obs(self):
        y = self.ar2_garch
        mod = arch_model(y, hold_back=20)
        res_holdback = mod.fit(disp='off')
        mod = arch_model(y)
        res_first_obs = mod.fit(disp='off', first_obs=20)
        assert_allclose(res_holdback.params, res_first_obs.params)

        with pytest.raises(ValueError):
            res_holdback.forecast(start=18)

    def test_holdback_lastobs(self):
        y = self.ar2_garch
        mod = arch_model(y, hold_back=20)
        res_holdback_last_obs = mod.fit(disp='off', last_obs=800)
        mod = arch_model(y)
        res_first_obs_last_obs = mod.fit(disp='off', first_obs=20, last_obs=800)
        assert_allclose(res_holdback_last_obs.params, res_first_obs_last_obs.params)
        mod = arch_model(y[20:800])
        res_direct = mod.fit(disp='off')
        assert_allclose(res_direct.params, res_first_obs_last_obs.params)

        with pytest.raises(ValueError):
            res_holdback_last_obs.forecast(start=18)

    def test_holdback_ar(self):
        y = self.ar2_garch
        mod = arch_model(y, mean='AR', lags=1, hold_back=1)
        res_holdback = mod.fit()
        mod = arch_model(y, mean='AR', lags=1)
        res = mod.fit()
        assert_allclose(res_holdback.params, res.params)
