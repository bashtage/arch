from __future__ import absolute_import, division
from arch.compat.python import range, iteritems

import warnings
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
import pytest
from numpy.random import randn
from numpy.testing import assert_almost_equal, assert_equal

try:
    import arch.univariate.recursions as rec
except ImportError:
    import arch.univariate.recursions_python as rec  # noqa
from arch.univariate.base import ARCHModelResult, ARCHModelForecast, \
    _align_forecast
from arch.univariate.mean import HARX, ConstantMean, ARX, ZeroMean, LS, \
    arch_model
from arch.univariate.volatility import ConstantVariance, GARCH, HARCH, ARCH, \
    RiskMetrics2006, EWMAVariance, EGARCH, FixedVariance
from arch.univariate.distribution import Normal, StudentsT
try:
    import matplotlib.pyplot  # noqa
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DISPLAY = 'off'


class TestMeanModel(TestCase):
    @classmethod
    def setup_class(cls):
        np.random.seed(1234)
        cls.T = 1000
        cls.resids = randn(cls.T)
        np.random.seed(1234)
        zm = ZeroMean()
        zm.volatility = GARCH()
        sim_data = zm.simulate(np.array([0.1, 0.1, 0.8]), 1000)
        date_index = pd.date_range('2000-12-31', periods=1000, freq='W')
        cls.y = sim_data.data.values
        cls.y_df = pd.DataFrame(cls.y[:, None],
                                columns=['LongVariableName'],
                                index=date_index)

        cls.y_series = pd.Series(cls.y,
                                 name='VeryVeryLongLongVariableName',
                                 index=date_index)
        x = cls.resids + randn(cls.T)
        cls.x = x[:, None]
        cls.x_df = pd.DataFrame(cls.x, columns=['LongExogenousName'])
        cls.resid_var = np.var(cls.resids)
        cls.sigma2 = np.zeros_like(cls.resids)
        cls.backcast = 1.0

    def test_constant_mean(self):
        cm = ConstantMean(self.y)
        parameters = np.array([5.0, 1.0])
        cm.simulate(parameters, self.T)
        assert_equal(cm.num_params, 1)
        bounds = cm.bounds()
        assert_equal(bounds, [(-np.inf, np.inf)])
        assert_equal(cm.constant, True)
        a, b = cm.constraints()
        assert_equal(a, np.empty((0, 1)))
        assert_equal(b, np.empty((0,)))
        assert isinstance(cm.volatility, ConstantVariance)
        assert isinstance(cm.distribution, Normal)
        assert_equal(cm.lags, None)
        res = cm.fit(disp='off')
        expected = np.array([self.y.mean(), self.y.var()])
        assert_almost_equal(res.params, expected)

        forecasts = res.forecast(horizon=20, start=20)
        direct = pd.DataFrame(index=np.arange(self.y.shape[0]),
                              columns=['h.{0:>02d}'.format(i + 1) for i in
                                       range(20)],
                              dtype=np.float64)
        direct.iloc[20:, :] = res.params.iloc[0]
        # TODO
        # assert_frame_equal(direct, forecasts)
        assert isinstance(forecasts, ARCHModelForecast)

    def test_zero_mean(self):
        zm = ZeroMean(self.y)
        parameters = np.array([1.0])
        data = zm.simulate(parameters, self.T)
        assert_equal(data.shape, (self.T, 3))
        assert_equal(data['data'].shape[0], self.T)
        assert_equal(zm.num_params, 0)
        bounds = zm.bounds()
        assert_equal(bounds, [])
        assert_equal(zm.constant, False)
        a, b = zm.constraints()
        assert_equal(a, np.empty((0, 0)))
        assert_equal(b, np.empty((0,)))
        assert isinstance(zm.volatility, ConstantVariance)
        assert isinstance(zm.distribution, Normal)
        assert_equal(zm.lags, None)
        res = zm.fit(disp='off')
        assert_almost_equal(res.params, np.array([np.mean(self.y ** 2)]))

        forecasts = res.forecast(horizon=99)
        direct = pd.DataFrame(index=np.arange(self.y.shape[0]),
                              columns=['h.{0:>02d}'.format(i + 1) for i in
                                       range(99)],
                              dtype=np.float64)
        direct.iloc[:, :] = 0.0
        assert isinstance(forecasts, ARCHModelForecast)
        # TODO
        # assert_frame_equal(direct, forecasts)
        garch = GARCH()
        zm.volatility = garch
        zm.fit(update_freq=0, disp=DISPLAY)

    def test_harx(self):
        harx = HARX(self.y, self.x, lags=[1, 5, 22])
        params = np.array([1.0, 0.4, 0.3, 0.2, 1.0, 1.0])
        data = harx.simulate(params, self.T, x=randn(self.T + 500, 1))
        iv = randn(22, 1)
        data = harx.simulate(params, self.T, x=randn(self.T + 500, 1),
                             initial_value=iv)
        assert_equal(data.shape, (self.T, 3))
        cols = ['data', 'volatility', 'errors']
        for c in cols:
            assert c in data

        bounds = harx.bounds()
        for b in bounds:
            assert_equal(b[0], -np.inf)
            assert_equal(b[1], np.inf)
        assert_equal(len(bounds), 5)

        assert_equal(harx.num_params, 1 + 3 + self.x.shape[1])
        assert_equal(harx.constant, True)
        a, b = harx.constraints()
        assert_equal(a, np.empty((0, 5)))
        assert_equal(b, np.empty(0))
        res = harx.fit(disp=DISPLAY)
        with pytest.raises(RuntimeError):
            res.forecast(horizon=10)
        with pytest.raises(ValueError):
            res.forecast(params=np.array([1.0, 1.0]))
        nobs = self.T - 22
        rhs = np.ones((nobs, 5))
        y = self.y
        lhs = y[22:]
        for i in range(self.T - 22):
            rhs[i, 1] = y[i + 21]
            rhs[i, 2] = np.mean(y[i + 17:i + 22])
            rhs[i, 3] = np.mean(y[i:i + 22])
        rhs[:, 4] = self.x[22:, 0]
        params = np.linalg.pinv(rhs).dot(lhs)
        assert_almost_equal(params, res.params[:-1])

        assert_equal(harx.hold_back, None)
        assert_equal(harx.lags, [1, 5, 22])
        assert_equal(harx.name, 'HAR-X')
        assert_equal(harx.use_rotated, False)
        harx
        harx._repr_html_()
        res = harx.fit(cov_type='mle', disp=DISPLAY)
        res

    def test_har(self):
        har = HARX(self.y, lags=[1, 5, 22])
        params = np.array([1.0, 0.4, 0.3, 0.2, 1.0])
        data = har.simulate(params, self.T)
        assert_equal(data.shape, (self.T, 3))
        cols = ['data', 'volatility', 'errors']
        for c in cols:
            assert c in data

        bounds = har.bounds()
        for b in bounds:
            assert_equal(b[0], -np.inf)
            assert_equal(b[1], np.inf)
        assert_equal(len(bounds), 4)

        assert_equal(har.num_params, 4)
        assert_equal(har.constant, True)
        a, b = har.constraints()
        assert_equal(a, np.empty((0, 4)))
        assert_equal(b, np.empty(0))
        res = har.fit(disp=DISPLAY)
        nobs = self.T - 22
        rhs = np.ones((nobs, 4))
        y = self.y
        lhs = y[22:]
        for i in range(self.T - 22):
            rhs[i, 1] = y[i + 21]
            rhs[i, 2] = np.mean(y[i + 17:i + 22])
            rhs[i, 3] = np.mean(y[i:i + 22])
        params = np.linalg.pinv(rhs).dot(lhs)
        assert_almost_equal(params, res.params[:-1])

        with pytest.raises(ValueError):
            res.forecast(horizon=6, start=0)
        forecasts = res.forecast(horizon=6)
        t = self.y.shape[0]
        direct = pd.DataFrame(index=np.arange(t),
                              columns=['h.' + str(i + 1) for i in range(6)],
                              dtype=np.float64)

        params = np.asarray(res.params)
        fcast = np.zeros(t + 6)
        for i in range(21, t):
            fcast[:i + 1] = self.y[:i + 1]
            fcast[i + 1:] = 0.0
            for h in range(6):
                fcast[i + h + 1] = params[0]
                fcast[i + h + 1] += params[1] * fcast[i + h:i + h + 1]
                fcast[i + h + 1] += params[2] * fcast[
                                                i + h - 4:i + h + 1].mean()
                fcast[i + h + 1] += params[3] * fcast[
                                                i + h - 21:i + h + 1].mean()
            direct.iloc[i, :] = fcast[i + 1:i + 7]
        assert isinstance(forecasts, ARCHModelForecast)
        # TODO
        # assert_frame_equal(direct, forecasts)
        forecasts = res.forecast(res.params, horizon=6)
        assert isinstance(forecasts, ARCHModelForecast)
        # TODO
        # assert_frame_equal(direct, forecasts)

        assert_equal(har.hold_back, None)
        assert_equal(har.lags, [1, 5, 22])
        assert_equal(har.name, 'HAR')
        assert_equal(har.use_rotated, False)

        har = HARX(self.y_series, lags=[1, 5, 22])
        res = har.fit(disp=DISPLAY)
        direct = pd.DataFrame(index=self.y_series.index,
                              columns=['h.' + str(i + 1) for i in range(6)],
                              dtype=np.float64)
        forecasts = res.forecast(horizon=6)
        params = np.asarray(res.params)
        fcast = np.zeros(t + 6)
        for i in range(21, t):
            fcast[:i + 1] = self.y[:i + 1]
            fcast[i + 1:] = 0.0
            for h in range(6):
                fcast[i + h + 1] = params[0]
                fcast[i + h + 1] += params[1] * fcast[i + h:i + h + 1]
                fcast[i + h + 1] += params[2] * fcast[
                                                i + h - 4:i + h + 1].mean()
                fcast[i + h + 1] += params[3] * fcast[
                                                i + h - 21:i + h + 1].mean()
            direct.iloc[i, :] = fcast[i + 1:i + 7]
        assert isinstance(forecasts, ARCHModelForecast)
        # TODO
        # assert_frame_equal(direct, forecasts)

    def test_arx(self):
        arx = ARX(self.y, self.x, lags=3, hold_back=10, constant=False)
        params = np.array([0.4, 0.3, 0.2, 1.0, 1.0])
        data = arx.simulate(params, self.T, x=randn(self.T + 500, 1))
        assert isinstance(data, pd.DataFrame)
        bounds = arx.bounds()
        for b in bounds:
            assert_equal(b[0], -np.inf)
            assert_equal(b[1], np.inf)
        assert_equal(len(bounds), 4)

        assert_equal(arx.num_params, 4)
        assert not arx.constant
        a, b = arx.constraints()
        assert_equal(a, np.empty((0, 4)))
        assert_equal(b, np.empty(0))
        res = arx.fit(last_obs=900, disp=DISPLAY)
        assert res.fit_stop == 900

        nobs = 900 - 10
        rhs = np.zeros((nobs, 4))
        y = self.y
        lhs = y[10:900]
        for i in range(10, 900):
            rhs[i - 10, 0] = y[i - 1]
            rhs[i - 10, 1] = y[i - 2]
            rhs[i - 10, 2] = y[i - 3]
        rhs[:, 3] = self.x[10:900, 0]
        params = np.linalg.pinv(rhs).dot(lhs)
        assert_almost_equal(params, res.params[:-1])
        with pytest.raises(RuntimeError):
            res.forecast()
        assert_equal(arx.hold_back, 10)
        assert_equal(arx.lags, np.array([[0, 1, 2], [1, 2, 3]]))
        assert_equal(arx.name, 'AR-X')
        assert_equal(arx.use_rotated, False)
        arx
        arx._repr_html_()

    def test_ar(self):
        ar = ARX(self.y, lags=3)
        params = np.array([1.0, 0.4, 0.3, 0.2, 1.0])
        data = ar.simulate(params, self.T)
        assert len(data) == self.T
        assert_equal(self.y, ar.y)

        bounds = ar.bounds()
        for b in bounds:
            assert_equal(b[0], -np.inf)
            assert_equal(b[1], np.inf)
        assert_equal(len(bounds), 4)

        assert_equal(ar.num_params, 4)
        assert ar.constant
        a, b = ar.constraints()
        assert_equal(a, np.empty((0, 4)))
        assert_equal(b, np.empty(0))
        res = ar.fit(disp=DISPLAY)

        nobs = 1000 - 3
        rhs = np.ones((nobs, 4))
        y = self.y
        lhs = y[3:1000]
        for i in range(3, 1000):
            rhs[i - 3, 1] = y[i - 1]
            rhs[i - 3, 2] = y[i - 2]
            rhs[i - 3, 3] = y[i - 3]
        params = np.linalg.pinv(rhs).dot(lhs)
        assert_almost_equal(params, res.params[:-1])

        forecasts = res.forecast(horizon=5)
        direct = pd.DataFrame(index=np.arange(y.shape[0]),
                              columns=['h.' + str(i + 1) for i in range(5)],
                              dtype=np.float64)
        params = res.params.iloc[:-1]
        for i in range(2, y.shape[0]):
            fcast = np.zeros(y.shape[0] + 5)
            fcast[:y.shape[0]] = y.copy()
            for h in range(1, 6):
                reg = np.array([1.0, fcast[i + h - 1],
                                fcast[i + h - 2], fcast[i + h - 3]])
                fcast[i + h] = reg.dot(params)
            direct.iloc[i, :] = fcast[i + 1:i + 6]
        assert isinstance(forecasts, ARCHModelForecast)
        # TODO
        # assert_frame_equal(direct, forecasts)

        assert_equal(ar.hold_back, None)
        assert_equal(ar.lags, np.array([[0, 1, 2], [1, 2, 3]]))
        assert_equal(ar.name, 'AR')
        assert_equal(ar.use_rotated, False)
        ar.__repr__()
        ar._repr_html_()

        ar = ARX(self.y_df, lags=5)
        ar.__repr__()
        ar = ARX(self.y_series, lags=5)
        ar.__repr__()
        res = ar.fit(disp=DISPLAY)
        assert isinstance(res.resid, pd.Series)
        assert isinstance(res.conditional_volatility, pd.Series)
        # Smoke tests
        summ = ar.fit(disp=DISPLAY).summary()
        assert 'Constant Variance' in str(summ)
        ar = ARX(self.y, lags=1, volatility=GARCH(), distribution=StudentsT())
        res = ar.fit(disp=DISPLAY, update_freq=5, cov_type='mle')
        res.param_cov

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not installed')
    def test_ar_plot(self):
        ar = ARX(self.y, lags=1, volatility=GARCH(), distribution=StudentsT())
        res = ar.fit(disp=DISPLAY, update_freq=5, cov_type='mle')
        res.plot()
        res.plot(annualize='D')
        res.plot(annualize='W')
        res.plot(annualize='M')
        with pytest.raises(ValueError):
            res.plot(annualize='unknown')

        res.plot(scale=360)
        res.hedgehog_plot(start=500)
        res.hedgehog_plot(start=500, type='mean')
        res.hedgehog_plot(type='volatility')
        res.hedgehog_plot(start=500, method='simulation', simulations=100)

    def test_arch_arx(self):
        np.random.seed(12345)
        x = np.random.randn(500, 3)
        y = x.sum(1) + 3 * np.random.randn(500)

        am = ARX(y=y, x=x)
        am.fit(disp=DISPLAY).summary()
        am.volatility = ARCH(p=2)
        results = am.fit(update_freq=0, disp='off')
        assert isinstance(results.pvalues, pd.Series), True
        assert_equal(list(results.pvalues.index),
                     ['Const', 'x0', 'x1', 'x2',
                      'omega', 'alpha[1]', 'alpha[2]'])

        am = ARX(y=y, lags=2, x=x)
        am.fit(disp=DISPLAY).summary()
        am.volatility = ARCH(p=2)
        results = am.fit(update_freq=0, disp='off')
        assert isinstance(results.pvalues, pd.Series)
        assert_equal(list(results.pvalues.index),
                     ['Const', 'y[1]', 'y[2]', 'x0', 'x1', 'x2',
                      'omega', 'alpha[1]', 'alpha[2]'])

        x = pd.DataFrame(x, columns=['x0', 'x1', 'x2'])
        y = pd.Series(y, name='y')
        am = ARX(y=y, x=x)
        am.fit(disp=DISPLAY).summary()
        am.volatility = ARCH(p=2)
        results = am.fit(update_freq=0, disp='off')
        assert isinstance(results.pvalues, pd.Series)
        assert_equal(list(results.pvalues.index),
                     ['Const', 'x0', 'x1', 'x2',
                      'omega', 'alpha[1]', 'alpha[2]'])

    def test_arch_model(self):
        am = arch_model(self.y)
        assert isinstance(am, ConstantMean)
        assert isinstance(am.volatility, GARCH)
        assert isinstance(am.distribution, Normal)

        am = arch_model(self.y, mean='harx', lags=[1, 5, 22])
        assert isinstance(am, HARX)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, mean='har', lags=[1, 5, 22])
        assert isinstance(am, HARX)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, self.x, mean='ls')
        assert isinstance(am, LS)
        assert isinstance(am.volatility, GARCH)
        am.__repr__()

        am = arch_model(self.y, mean='arx', lags=[1, 5, 22])
        assert isinstance(am, ARX)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, mean='ar', lags=[1, 5, 22])
        assert isinstance(am, ARX)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, mean='ar', lags=None)
        assert isinstance(am, ARX)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, mean='zero')
        assert isinstance(am, ZeroMean)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, vol='Harch')
        assert isinstance(am, ConstantMean)
        assert isinstance(am.volatility, HARCH)

        am = arch_model(self.y, vol='Constant')
        assert isinstance(am, ConstantMean)
        assert isinstance(am.volatility, ConstantVariance)

        am = arch_model(self.y, vol='arch')
        assert isinstance(am.volatility, ARCH)

        am = arch_model(self.y, vol='egarch')
        assert isinstance(am.volatility, EGARCH)

        with pytest.raises(ValueError):
            arch_model(self.y, mean='unknown')
        with pytest.raises(ValueError):
            arch_model(self.y, vol='unknown')
        with pytest.raises(ValueError):
            arch_model(self.y, dist='unknown')

        am.fit(disp=DISPLAY)

    def test_pandas(self):
        am = arch_model(self.y_df, self.x_df, mean='ls')
        assert isinstance(am, LS)

    def test_summary(self):
        am = arch_model(self.y, mean='ar', lags=[1, 3, 5])
        res = am.fit(update_freq=0, disp=DISPLAY)
        res.summary()

        am = arch_model(self.y, mean='ar', lags=[1, 3, 5], dist='studentst')
        res = am.fit(update_freq=0, disp=DISPLAY)
        res.summary()

    def test_errors(self):
        with pytest.raises(ValueError):
            ARX(self.y, lags=np.array([[1, 2], [3, 4]]))
        x = randn(self.y.shape[0] + 1, 1)
        with pytest.raises(ValueError):
            ARX(self.y, x=x)
        with pytest.raises(ValueError):
            HARX(self.y, lags=np.eye(3))
        with pytest.raises(ValueError):
            ARX(self.y, lags=-1)
        with pytest.raises(ValueError):
            ARX(self.y, x=randn(1, 1), lags=-1)

        ar = ARX(self.y, lags=1)
        with self.assertRaises(ValueError):
            d = Normal()
            ar.volatility = d

        with self.assertRaises(ValueError):
            v = GARCH()
            ar.distribution = v
        x = randn(1000, 1)
        with pytest.raises(ValueError):
            ar.simulate(np.ones(5), 100, x=x)
        with pytest.raises(ValueError):
            ar.simulate(np.ones(5), 100)
        with pytest.raises(ValueError):
            ar.simulate(np.ones(3), 100, initial_value=randn(10))

        with self.assertRaises(ValueError):
            ar.volatility = ConstantVariance()
            ar.fit(cov_type='unknown')

    def test_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            ARX(self.y, lags=[1, 2, 3, 12], hold_back=5)
            assert_equal(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            HARX(self.y, lags=[[1, 1, 1], [2, 5, 22]], use_rotated=True)
            assert_equal(len(w), 1)

    def test_har_lag_specifications(self):
        """ Test equivalence of alternative lag specifications"""
        har = HARX(self.y, lags=[1, 2, 3])
        har_r = HARX(self.y, lags=[1, 2, 3], use_rotated=True)
        har_r_v2 = HARX(self.y, lags=3, use_rotated=True)
        ar = ARX(self.y, lags=[1, 2, 3])
        ar_v2 = ARX(self.y, lags=3)

        res_har = har.fit(disp=DISPLAY)
        res_har_r = har_r.fit(disp=DISPLAY)
        res_har_r_v2 = har_r_v2.fit(disp=DISPLAY)
        res_ar = ar.fit(disp=DISPLAY)
        res_ar_v2 = ar_v2.fit(disp=DISPLAY)
        assert_almost_equal(res_har.rsquared, res_har_r.rsquared)
        assert_almost_equal(res_har_r_v2.rsquared, res_har_r.rsquared)
        assert_almost_equal(np.asarray(res_ar.params),
                            np.asarray(res_ar_v2.params))
        assert_almost_equal(np.asarray(res_ar.params),
                            np.asarray(res_har_r_v2.params))
        assert_almost_equal(np.asarray(res_ar.param_cov),
                            np.asarray(res_har_r_v2.param_cov))
        assert_almost_equal(res_ar.conditional_volatility,
                            res_har_r_v2.conditional_volatility)
        assert_almost_equal(res_ar.resid, res_har_r_v2.resid)

    def test_starting_values(self):
        am = arch_model(self.y, mean='ar', lags=[1, 3, 5])
        res = am.fit(cov_type='mle', update_freq=0, disp=DISPLAY)
        res2 = am.fit(starting_values=res.params, update_freq=0, disp=DISPLAY)
        assert isinstance(res, ARCHModelResult)
        assert isinstance(res2, ARCHModelResult)
        assert len(res.params) == 7
        assert len(res2.params) == 7

        am = arch_model(self.y, mean='zero')
        sv = np.array([1.0, 0.3, 0.8])
        with warnings.catch_warnings(record=True) as w:
            am.fit(starting_values=sv, update_freq=0, disp=DISPLAY)
            assert_equal(len(w), 1)

    def test_no_param_volatility(self):
        cm = ConstantMean(self.y)
        cm.volatility = EWMAVariance()
        cm.fit(update_freq=0, disp=DISPLAY)
        cm.volatility = RiskMetrics2006()
        cm.fit(update_freq=0, disp=DISPLAY)

        ar = ARX(self.y, lags=5)
        ar.volatility = EWMAVariance()
        ar.fit(update_freq=0, disp=DISPLAY)
        ar.volatility = RiskMetrics2006()
        ar.fit(update_freq=0, disp=DISPLAY)
        assert 'tau0' in str(ar.volatility)
        assert 'tau1' in str(ar.volatility)
        assert 'kmax' in str(ar.volatility)

    def test_egarch(self):
        cm = ConstantMean(self.y)
        cm.volatility = EGARCH()
        cm.fit(update_freq=0, disp=DISPLAY)
        cm.distribution = StudentsT()
        cm.fit(update_freq=0, disp=DISPLAY)

    def test_multiple_lags(self):
        """Smoke test to ensure models estimate with multiple lags"""
        vp = {'garch': GARCH,
              'egarch': EGARCH,
              'harch': HARCH,
              'arch': ARCH}
        cm = ConstantMean(self.y)
        for name, process in iteritems(vp):
            cm.volatility = process()
            cm.fit(update_freq=0, disp='off')
            for p in [1, 2, 3]:
                for o in [1, 2, 3]:
                    for q in [1, 2, 3]:
                        if name in ('arch',):
                            cm.volatility = process(p=p + o + q)
                            cm.fit(update_freq=0, disp='off')
                        elif name in ('harch',):
                            cm.volatility = process(lags=[p, p + o, p + o + q])
                            cm.fit(update_freq=0, disp='off')
                        else:
                            cm.volatility = process(p=p, o=o, q=q)
                            cm.fit(update_freq=0, disp='off')

    def test_first_last_obs(self):
        ar = ARX(self.y, lags=5, hold_back=100)
        res = ar.fit(update_freq=0, disp=DISPLAY)
        resids = res.resid
        resid_copy = resids.copy()
        resid_copy[:100] = np.nan
        assert_equal(resids, resid_copy)

        ar.volatility = GARCH()
        res = ar.fit(update_freq=0, disp=DISPLAY)
        resids = res.resid
        resid_copy = resids.copy()
        resid_copy[:100] = np.nan
        assert_equal(resids, resid_copy)

        ar = ARX(self.y, lags=5)
        ar.volatility = GARCH()
        res = ar.fit(update_freq=0, last_obs=500, disp=DISPLAY)
        resids = res.resid
        resid_copy = resids.copy()
        resid_copy[500:] = np.nan
        assert_equal(resids, resid_copy)

        ar = ARX(self.y, lags=5, hold_back=100)
        ar.volatility = GARCH()
        res = ar.fit(update_freq=0, last_obs=500, disp=DISPLAY)
        resids = res.resid
        resid_copy = resids.copy()
        resid_copy[:100] = np.nan
        resid_copy[500:] = np.nan
        assert_equal(resids, resid_copy)

        vol = res.conditional_volatility
        vol_copy = vol.copy()
        vol_copy[:100] = np.nan
        vol_copy[500:] = np.nan
        assert_equal(vol, vol_copy)
        assert_equal(self.y.shape[0], vol.shape[0])

        ar = ARX(self.y, lags=5)
        ar.volatility = GARCH()
        res = ar.fit(update_freq=0, last_obs=500, disp=DISPLAY)
        resids = res.resid
        resid_copy = resids.copy()
        resid_copy[:5] = np.nan
        resid_copy[500:] = np.nan
        assert_equal(resids, resid_copy)

    def test_date_first_last_obs(self):
        y = self.y_series

        cm = ConstantMean(y)
        res = cm.fit(last_obs=y.index[900], disp=DISPLAY)

        cm = ConstantMean(y)
        res2 = cm.fit(last_obs=900, disp=DISPLAY)

        assert_equal(res.resid.values, res2.resid.values)

    def test_align(self):
        dates = pd.date_range('2000-01-01', '2010-01-01', freq='M')
        columns = ['h.' + '{0:>02}'.format(str(h + 1)) for h in range(10)]
        forecasts = pd.DataFrame(np.random.randn(120, 10),
                                 index=dates,
                                 columns=columns)

        aligned = _align_forecast(forecasts.copy(), align='origin')
        assert_frame_equal(aligned, forecasts)

        aligned = _align_forecast(forecasts.copy(), align='target')
        direct = forecasts.copy()
        for i in range(10):
            direct.iloc[(i + 1):, i] = direct.iloc[:(120 - i - 1), i].values
            direct.iloc[:(i + 1), i] = np.nan
        assert_frame_equal(aligned, direct)

        with pytest.raises(ValueError):
            _align_forecast(forecasts, align='unknown')

    def test_fixed_user_parameters(self):
        am = arch_model(self.y_series)
        res = am.fit(disp=DISPLAY)
        fixed_res = am.fix(res.params)
        assert_series_equal(res.conditional_volatility,
                            fixed_res.conditional_volatility)
        assert_series_equal(res.params, fixed_res.params)
        assert_equal(res.aic, fixed_res.aic)
        assert_equal(res.bic, fixed_res.bic)
        assert_equal(res.loglikelihood, fixed_res.loglikelihood)
        assert_equal(res.num_params, fixed_res.num_params)
        # Smoke for summary
        fixed_res.summary()

    def test_fixed_user_parameters_new_model(self):
        am = arch_model(self.y_series)
        res = am.fit(disp=DISPLAY)
        new_am = arch_model(self.y_series)
        fixed_res = new_am .fix(res.params)
        assert_series_equal(res.conditional_volatility,
                            fixed_res.conditional_volatility)
        assert_series_equal(res.params, fixed_res.params)
        assert_equal(res.aic, fixed_res.aic)
        assert_equal(res.bic, fixed_res.bic)
        assert_equal(res.loglikelihood, fixed_res.loglikelihood)
        assert_equal(res.num_params, fixed_res.num_params)

        # Test first and last dates
        am = arch_model(self.y_series)
        res = am.fit(disp=DISPLAY, first_obs=100, last_obs=900)
        new_am = arch_model(self.y_series)
        fixed_res = new_am .fix(res.params, first_obs=100, last_obs=900)
        assert_series_equal(res.params, fixed_res.params)
        assert_equal(res.aic, fixed_res.aic)
        assert_equal(res.bic, fixed_res.bic)
        assert_equal(res.loglikelihood, fixed_res.loglikelihood)
        assert_equal(res.num_params, fixed_res.num_params)

    def test_output_options(self):
        import sys
        from arch.compat.python import StringIO
        am = arch_model(self.y_series)
        orig_stdout = sys.stdout
        try:
            sio = StringIO()
            sys.stdout = sio
            am.fit(disp='final')
            sio.seek(0)
            print('SIO!')
            print(sio.read())
        finally:
            sys.stdout = orig_stdout

        try:
            sio = StringIO()
            sys.stdout = sio
            am.fit(disp='off')
            sio.seek(0)
            output = sio.read()
            assert len(output) == 0
        finally:
            sys.stdout = orig_stdout

    def test_convergence_warning(self):
        y = np.array([0.83277114, 0.45194014, -0.33475561, -0.49463896,
                      0.54715787, 1.11895382, 1.31280266, 0.81464021,
                      0.8532107, 1.0967188, 0.9346354, 0.92289249, 1.01339085,
                      1.071065, 1.42413486, 1.15392453, 1.10929691, 0.96162061,
                      0.96489515, 0.93250153, 1.34509807, 1.80951607,
                      1.66313783, 1.38610821, 1.26381761])
        am = arch_model(y, mean='ARX', lags=10, p=5, q=0)
        with warnings.catch_warnings(record=True) as w:
            am.fit(disp=DISPLAY)
            assert_equal(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            am.fit(show_warning=False, disp=DISPLAY)
            assert_equal(len(w), 0)

        with warnings.catch_warnings(record=True) as w:
            am.fit(show_warning=True, disp=DISPLAY)
            assert_equal(len(w), 1)

    def test_first_after_last(self):
        am = arch_model(self.y_series)
        with pytest.raises(ValueError):
            am.fit(disp='off', first_obs=500, last_obs=480)

        with pytest.raises(ValueError):
            am.fit(disp='off',
                   first_obs=self.y_series.index[500],
                   last_obs=self.y_series.index[480])

    def test_sample_adjustment(self):
        am = arch_model(self.y_series, vol='Constant')
        res = am.fit(disp='off')

        res_adj = am.fit(disp='off',
                         first_obs=0,
                         last_obs=self.y_series.shape[0] + 1)
        assert_equal(res.resid.values, res_adj.resid.values)
        assert_equal(res.params.values, res_adj.params.values)

        res = am.fit(disp='off', first_obs=100)
        assert res.fit_start == 100
        res_adj = am.fit(disp='off', first_obs=self.y_series.index[100])
        assert_equal(res.params.values, res_adj.params.values)
        assert_equal(res.resid.values, res_adj.resid.values)

        res = am.fit(disp='off', last_obs=900)
        res2 = am.fit(disp='off', last_obs=self.y_series.index[900])
        assert_equal(res.params.values, res2.params.values)
        assert_equal(res.resid.values, res2.resid.values)

        res = am.fit(disp='off', first_obs=100, last_obs=900)
        res2 = am.fit(disp='off',
                      first_obs=self.y_series.index[100],
                      last_obs=self.y_series.index[900])
        assert_equal(res.params.values, res2.params.values)
        assert_equal(res.resid.values, res2.resid.values)

    def test_model_obs_equivalence(self):
        """Tests models that should use the same observation"""
        am = arch_model(self.y_series.iloc[100:900])
        res = am.fit(disp='off')
        am = arch_model(self.y_series)
        res2 = am.fit(disp='off', first_obs=100, last_obs=900)
        index = self.y_series.index
        res3 = am.fit(disp='off', first_obs=index[100], last_obs=index[900])
        assert_equal(res.params.values, res2.params.values)
        assert_equal(res2.params.values, res3.params.values)

        am = arch_model(self.y_series, hold_back=100)
        res4 = am.fit(disp='off', last_obs=900)
        assert_equal(res.params.values, res4.params.values)

    def test_model_obs_equivalence_ar(self):
        """Tests models that should use the same observation"""
        am = arch_model(self.y_series.iloc[100:900], mean='AR', lags=[1, 2, 4])
        res = am.fit(disp='off')
        am = arch_model(self.y_series, mean='AR', lags=[1, 2, 4])
        res2 = am.fit(disp='off', first_obs=100, last_obs=900)
        index = self.y_series.index
        res3 = am.fit(disp='off', first_obs=index[100], last_obs=index[900])
        assert_almost_equal(res.params.values, res2.params.values)
        assert_almost_equal(res2.params.values, res3.params.values)

        am = arch_model(self.y_series, mean='AR', lags=[1, 2, 4],
                        hold_back=100)
        res4 = am.fit(disp='off', first_obs=4, last_obs=900)
        assert_almost_equal(res.params.values, res4.params.values, decimal=4)
        assert am.hold_back == 100

    def test_constant_mean_fixed_variance(self):
        variance = 2 + np.random.standard_normal(self.y.shape[0]) ** 2.0
        mod = ConstantMean(self.y_series, volatility=FixedVariance(variance))
        res = mod.fit()
        print(res.summary())
        assert len(res.params) == 2
        assert 'scale' in res.params.index

        mod = ARX(self.y_series, lags=[1, 2, 3], volatility=FixedVariance(variance))
        res = mod.fit()
        assert len(res.params) == 5
        assert 'scale' in res.params.index

        mod = ARX(self.y_series, lags=[1, 2, 3],
                  volatility=FixedVariance(variance, unit_scale=True))
        res = mod.fit()
        assert len(res.params) == 4
        assert 'scale' not in res.params.index
