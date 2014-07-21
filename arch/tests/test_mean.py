import unittest
import warnings

import numpy as np
from numpy.random import randn
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
from nose.tools import assert_true
import pandas as pd


try:
    import arch.recursions as rec
except ImportError:
    import arch.recursions_python as rec
from arch.mean import HARX, ConstantMean, ARX, ZeroMean, arch_model, LS
from arch.volatility import ConstantVariance, GARCH, HARCH, ARCH, \
    RiskMetrics2006, EWMAVariance, EGARCH
from arch.distribution import Normal, StudentsT
from arch.compat.python import range
from pandas.compat import iteritems


class TestMeanModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1234)
        cls.T = 1000
        cls.resids = randn(cls.T)
        np.random.seed(1234)
        zm = ZeroMean()
        zm.volatility = GARCH()
        sim_data = zm.simulate(np.array([0.1, 0.1, 0.8]), 1000)
        cls.y = sim_data.data.values
        cls.y_df = pd.DataFrame(cls.y[:, None], columns=['LongVariableName'])
        cls.y_series = pd.Series(cls.y, name='VeryVeryLongLongVariableName')
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
        assert_true(isinstance(cm.volatility, ConstantVariance))
        assert_true(isinstance(cm.distribution, Normal))
        assert_equal(cm.first_obs, 0)
        assert_equal(cm.last_obs, 1000)
        assert_equal(cm.lags, None)
        res = cm.fit()
        assert_almost_equal(res.params, np.array([self.y.mean(), self.y.var()]))

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
        assert_true(isinstance(zm.volatility, ConstantVariance))
        assert_true(isinstance(zm.distribution, Normal))
        assert_equal(zm.first_obs, 0)
        assert_equal(zm.last_obs, 1000)
        assert_equal(zm.lags, None)
        res = zm.fit()
        assert_almost_equal(res.params, np.array([np.mean(self.y ** 2)]))
        garch = GARCH()
        zm.volatility = garch
        zm.fit(iter=5)

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
            assert_true(c in data)

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
        res = harx.fit()
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

        assert_equal(harx.first_obs, 22)
        assert_equal(harx.last_obs, 1000)
        assert_equal(harx.hold_back, None)
        assert_equal(harx.lags, [1, 5, 22])
        assert_equal(harx.nobs, self.T - 22)
        assert_equal(harx.name, 'HAR-X')
        assert_equal(harx.use_rotated, False)
        harx
        harx._repr_html_()
        res = harx.fit(cov_type='mle')
        res

    def test_har(self):
        har = HARX(self.y, lags=[1, 5, 22])
        params = np.array([1.0, 0.4, 0.3, 0.2, 1.0])
        data = har.simulate(params, self.T)
        assert_equal(data.shape, (self.T, 3))
        cols = ['data', 'volatility', 'errors']
        for c in cols:
            assert_true(c in data)

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
        res = har.fit()
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

        assert_equal(har.first_obs, 22)
        assert_equal(har.last_obs, 1000)
        assert_equal(har.hold_back, None)
        assert_equal(har.lags, [1, 5, 22])
        assert_equal(har.nobs, self.T - 22)
        assert_equal(har.name, 'HAR')
        assert_equal(har.use_rotated, False)

    def test_arx(self):
        arx = ARX(self.y, self.x, lags=3, hold_back=10, last_obs=900,
                  constant=False)
        params = np.array([0.4, 0.3, 0.2, 1.0, 1.0])
        data = arx.simulate(params, self.T, x=randn(self.T + 500, 1))
        bounds = arx.bounds()
        for b in bounds:
            assert_equal(b[0], -np.inf)
            assert_equal(b[1], np.inf)
        assert_equal(len(bounds), 4)

        assert_equal(arx.num_params, 4)
        assert_true(not arx.constant)
        a, b = arx.constraints()
        assert_equal(a, np.empty((0, 4)))
        assert_equal(b, np.empty(0))
        res = arx.fit()

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

        assert_equal(arx.first_obs, 10)
        assert_equal(arx.last_obs, 900)
        assert_equal(arx.hold_back, 10)
        assert_equal(arx.lags, np.array([[0, 1, 2], [1, 2, 3]]))
        assert_equal(arx.nobs, 890)
        assert_equal(arx.name, 'AR-X')
        assert_equal(arx.use_rotated, False)
        arx
        arx._repr_html_()


    def test_ar(self):
        ar = ARX(self.y, lags=3)
        params = np.array([1.0, 0.4, 0.3, 0.2, 1.0])
        data = ar.simulate(params, self.T)

        bounds = ar.bounds()
        for b in bounds:
            assert_equal(b[0], -np.inf)
            assert_equal(b[1], np.inf)
        assert_equal(len(bounds), 4)

        assert_equal(ar.num_params, 4)
        assert_true(ar.constant)
        a, b = ar.constraints()
        assert_equal(a, np.empty((0, 4)))
        assert_equal(b, np.empty(0))
        res = ar.fit()

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

        assert_equal(ar.first_obs, 3)
        assert_equal(ar.last_obs, 1000)
        assert_equal(ar.hold_back, None)
        assert_equal(ar.lags, np.array([[0, 1, 2], [1, 2, 3]]))
        assert_equal(ar.nobs, 997)
        assert_equal(ar.name, 'AR')
        assert_equal(ar.use_rotated, False)
        ar.__repr__()
        ar._repr_html_()

        ar = ARX(self.y_df, lags=5)
        ar.__repr__()
        ar = ARX(self.y_series, lags=5)
        ar.__repr__()
        summ = ar.fit().summary()
        ar = ARX(self.y, lags=1, volatility=GARCH(), distribution=StudentsT())
        res = ar.fit(iter=5)
        res.plot()


    def test_arch_model(self):
        am = arch_model(self.y)
        assert_true(isinstance(am, ConstantMean))
        assert_true(isinstance(am.volatility, GARCH))
        assert_true(isinstance(am.distribution, Normal))

        am = arch_model(self.y, mean='harx', lags=[1, 5, 22])
        assert_true(isinstance(am, HARX))
        assert_true(isinstance(am.volatility, GARCH))

        am = arch_model(self.y, mean='har', lags=[1, 5, 22])
        assert_true(isinstance(am, HARX))
        assert_true(isinstance(am.volatility, GARCH))

        am = arch_model(self.y, self.x, mean='ls')
        assert_true(isinstance(am, LS))
        assert_true(isinstance(am.volatility, GARCH))
        am.__repr__()

        am = arch_model(self.y, mean='arx', lags=[1, 5, 22])
        assert_true(isinstance(am, ARX))
        assert_true(isinstance(am.volatility, GARCH))

        am = arch_model(self.y, mean='ar', lags=[1, 5, 22])
        assert_true(isinstance(am, ARX))
        assert_true(isinstance(am.volatility, GARCH))

        am = arch_model(self.y, mean='ar', lags=None)
        assert_true(isinstance(am, ARX))
        assert_true(isinstance(am.volatility, GARCH))

        am = arch_model(self.y, mean='zero')
        assert_true(isinstance(am, ZeroMean))
        assert_true(isinstance(am.volatility, GARCH))

        am = arch_model(self.y, vol='Harch')
        assert_true(isinstance(am, ConstantMean))
        assert_true(isinstance(am.volatility, HARCH))

        am = arch_model(self.y, vol='Constant')
        assert_true(isinstance(am, ConstantMean))
        assert_true(isinstance(am.volatility, ConstantVariance))

        am = arch_model(self.y, vol='arch')
        assert_true(isinstance(am.volatility, ARCH))

        assert_raises(ValueError, arch_model, self.y, mean='unknown')
        assert_raises(ValueError, arch_model, self.y, vol='unknown')
        assert_raises(ValueError, arch_model, self.y, dist='unknown')

        am.fit()

    def test_pandas(self):
        am = arch_model(self.y_df, self.x_df, mean='ls')
        assert_true(isinstance(am, LS))

    def test_summary(self):
        am = arch_model(self.y, mean='ar', lags=[1, 3, 5])
        res = am.fit()
        res.summary()

        am = arch_model(self.y, mean='ar', lags=[1, 3, 5], dist='studentst')
        res = am.fit()
        res.summary()

    def test_errors(self):
        assert_raises(ValueError, ARX, self.y, lags=np.array([[1, 2], [3, 4]]))
        x = randn(self.y.shape[0] + 1, 1)
        assert_raises(ValueError, ARX, self.y, x=x)
        assert_raises(ValueError, HARX, self.y, lags=np.eye(3))
        assert_raises(ValueError, ARX, self.y, lags=-1)
        assert_raises(ValueError, ARX, self.y, x=randn(1, 1), lags=-1)

        ar = ARX(self.y, lags=1)
        with self.assertRaises(ValueError):
            d = Normal()
            ar.volatility = d

        with self.assertRaises(ValueError):
            v = GARCH()
            ar.distribution = v
        x = randn(1000, 1)
        assert_raises(ValueError, ar.simulate, np.ones(5), 100, x=x)
        assert_raises(ValueError, ar.simulate, np.ones(5), 100)
        assert_raises(ValueError, ar.simulate, np.ones(3), 100,
                      initial_value=randn(10))

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

        har = HARX()
        with warnings.catch_warnings(record=True) as w:
            har.fit()
        assert_equal(len(w), 1)

    def test_har_lag_specifications(self):
        """ Test equivalence of alternative lag specifications"""
        har = HARX(self.y, lags=[1, 2, 3])
        har_r = HARX(self.y, lags=[1, 2, 3], use_rotated=True)
        har_r_v2 = HARX(self.y, lags=3, use_rotated=True)
        ar = ARX(self.y, lags=[1, 2, 3])
        ar_v2 = ARX(self.y, lags=3)

        res_har = har.fit()
        res_har_r = har_r.fit()
        res_har_r_v2 = har_r_v2.fit()
        res_ar = ar.fit()
        res_ar_v2 = ar_v2.fit()
        assert_almost_equal(res_har.rsquared, res_har_r.rsquared)
        assert_almost_equal(res_har_r_v2.rsquared, res_har_r.rsquared)
        assert_almost_equal(res_ar.params, res_ar_v2.params)
        assert_almost_equal(res_ar.params, res_har_r_v2.params)
        assert_almost_equal(res_ar.param_cov, res_har_r_v2.param_cov)
        assert_almost_equal(res_ar.conditional_volatility,
                            res_har_r_v2.conditional_volatility)
        assert_almost_equal(res_ar.resid, res_har_r_v2.resid)


    def test_starting_values(self):
        am = arch_model(self.y, mean='ar', lags=[1, 3, 5])
        res = am.fit(cov_type='mle')
        res2 = am.fit(starting_values=res.params, iter=0)

        am = arch_model(self.y, mean='zero')
        sv = np.array([1.0, 0.3, 0.8])
        with warnings.catch_warnings(record=True) as w:
            am.fit(starting_values=sv)
        assert_equal(len(w), 1)

    def test_no_param_volatility(self):
        cm = ConstantMean(self.y)
        cm.volatility = EWMAVariance()
        cm.fit()
        cm.volatility = RiskMetrics2006()
        cm.fit()

        ar = ARX(self.y, lags=5)
        ar.volatility = EWMAVariance()
        ar.fit()
        ar.volatility = RiskMetrics2006()
        ar.fit()

    def test_egarch(self):
        cm = ConstantMean(self.y)
        cm.volatility = EGARCH()
        cm.fit(iter=5)
        cm.distribution = StudentsT()
        cm.fit(iter=5)

    def test_multiple_lags(self):
        """Smoke test to ensure models estimate with multiple lags"""
        vp = {'garch': GARCH,
              'egarch': EGARCH,
              'harch': HARCH,
              'arch': ARCH}
        cm = ConstantMean(self.y)
        for name, process in iteritems(vp):
            cm.volatility = process()
            cm.fit(iter=0, disp='off')
            for p in [1, 2, 3]:
                for o in [1, 2, 3]:
                    for q in [1, 2, 3]:
                        print(name + ':' + str(p) + ',' + str(o) + ',' + str(q))
                        if name in ('arch',):
                            cm.volatility = process(p=p + o + q)
                            cm.fit(iter=0, disp='off')
                        elif name in ('harch',):
                            cm.volatility = process(lags=[p, p + o, p + o + q])
                            cm.fit(iter=0, disp='off')
                        else:
                            cm.volatility = process(p=p, o=o, q=q)
                            cm.fit(iter=0, disp='off')

