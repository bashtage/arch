import unittest

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_almost_equal
import arch.univariate.recursions as rec

import arch.univariate.recursions_python as recpy
from arch.compat.python import range


class TestRecursions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T = 1000
        cls.resids = np.random.randn(cls.T)
        cls.sigma2 = np.zeros_like(cls.resids)
        var = cls.resids.var()
        var_bounds = np.array([var / 1000000.0, var * 1000000.0])
        cls.var_bounds = np.ones((cls.T, 2)) * var_bounds
        cls.backcast = 1.0

    def test_garch(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3, .2])
        fresids = resids ** 2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(parameters, fresids, sresids, sigma2,
                              1, 1, 1, T, backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1,
                            1, T, backcast, self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)

    def test_harch(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3, .2])
        lags = np.array([1, 5, 22], dtype=np.int32)
        recpy.harch_recursion(parameters, resids, sigma2, lags, T, backcast,
                              self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.harch_recursion(parameters, resids, sigma2, lags, T, backcast,
                            self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)

    def test_arch(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3, .2])
        p = 3

        recpy.arch_recursion(parameters, resids, sigma2, p, T,
                             backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.arch_recursion(parameters, resids, sigma2, p, T, backcast,
                           self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_power_1(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3, .2])
        lags = np.array([1, 5, 22])
        fresids = np.abs(resids) ** 1.0
        sresids = np.sign(resids)

        recpy.garch_recursion(parameters, fresids, sresids, sigma2,
                              1, 1, 1, T, backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1,
                            1, T, backcast, self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_direct(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3, .2])
        fresids = np.abs(resids) ** 2.0
        sresids = np.sign(resids)

        for t in range(T):
            if t == 0:
                sigma2[t] = parameters.dot(
                    np.array([1.0, backcast, 0.5 * backcast, backcast]))
            else:
                vars = np.array([1.0,
                                 resids[t - 1] ** 2.0,
                                 resids[t - 1] ** 2.0 * (resids[t - 1] < 0),
                                 sigma2[t - 1]])
                sigma2[t] = parameters.dot(vars)

        sigma2_python = sigma2.copy()
        rec.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1,
                            1, T, backcast, self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_no_q(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3])
        fresids = resids ** 2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(parameters, fresids, sresids, sigma2,
                              1, 1, 0, T, backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1,
                            0, T, backcast, self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_no_p(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3])
        fresids = resids ** 2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(parameters, fresids, sresids, sigma2,
                              0, 1, 1, T, backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.garch_recursion(parameters, fresids, sresids, sigma2, 0, 1,
                            1, T, backcast, self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_no_o(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3, .2])
        fresids = resids ** 2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(parameters, fresids, sresids, sigma2,
                              1, 0, 1, T, backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.garch_recursion(parameters, fresids, sresids, sigma2, 1, 0,
                            1, T, backcast, self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_arch(self):
        backcast = self.backcast
        T, resids, sigma2 = self.T, self.resids, self.sigma2

        parameters = np.array([.1, .4, .3, .2])
        fresids = resids ** 2.0
        sresids = np.sign(resids)

        rec.garch_recursion(parameters, fresids, sresids, sigma2,
                            3, 0, 0, T, backcast, self.var_bounds)
        sigma2_garch = sigma2.copy()
        rec.arch_recursion(parameters, resids, sigma2, 3, T, backcast,
                           self.var_bounds)

        assert_almost_equal(sigma2_garch, sigma2)

    def test_bounds(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([1e100, .4, .3, .2])
        lags = np.array([1, 5, 22], dtype=np.int32)
        recpy.harch_recursion(parameters, resids, sigma2, lags, T, backcast,
                              self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.harch_recursion(parameters, resids, sigma2, lags, T, backcast,
                            self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)
        assert_true((sigma2 >= self.var_bounds[:, 1]).all())

        parameters = np.array([-1e100, .4, .3, .2])
        recpy.harch_recursion(parameters, resids, sigma2, lags, T, backcast,
                              self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.harch_recursion(parameters, resids, sigma2, lags, T, backcast,
                            self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)
        assert_almost_equal(sigma2, self.var_bounds[:, 0])

        parameters = np.array([1e100, .4, .3, .2])
        fresids = resids ** 2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(parameters, fresids, sresids, sigma2,
                              1, 1, 1, T, backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1,
                            1, T, backcast, self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)
        assert_true((sigma2 >= self.var_bounds[:, 1]).all())

        parameters = np.array([-1e100, .4, .3, .2])
        recpy.garch_recursion(parameters, fresids, sresids, sigma2,
                              1, 1, 1, T, backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1,
                            1, T, backcast, self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)
        assert_almost_equal(sigma2, self.var_bounds[:, 0])

        parameters = np.array([1e100, .4, .3, .2])
        recpy.arch_recursion(parameters, resids, sigma2, 3, T, backcast,
                             self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.arch_recursion(parameters, resids, sigma2, 3, T, backcast,
                           self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)
        assert_true((sigma2 >= self.var_bounds[:, 1]).all())

        parameters = np.array([-1e100, .4, .3, .2])
        recpy.arch_recursion(parameters, resids, sigma2, 3, T, backcast,
                             self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.arch_recursion(parameters, resids, sigma2, 3, T, backcast,
                           self.var_bounds)
        assert_almost_equal(sigma2_python, sigma2)
        assert_almost_equal(sigma2, self.var_bounds[:, 0])

    def test_egarch(self):
        nobs = self.T
        parameters = np.array([0.0, 0.1, -0.1, 0.95])
        resids, sigma2 = self.resids, self.sigma2
        p = o = q = 1
        backcast = 0.0
        var_bounds = self.var_bounds
        lnsigma2 = np.empty_like(sigma2)
        std_resids = np.empty_like(sigma2)
        abs_std_resids = np.empty_like(sigma2)
        rec.egarch_recursion(parameters, resids, sigma2, p, o, q, nobs,
                             backcast,
                             var_bounds, lnsigma2, std_resids, abs_std_resids)
        sigma2_python = sigma2.copy()
        recpy.egarch_recursion(parameters, resids, sigma2_python, p, o, q, nobs,
                               backcast, var_bounds, lnsigma2, std_resids,
                               abs_std_resids)
        assert_almost_equal(sigma2_python, sigma2)

        norm_const = np.sqrt(2 / np.pi)
        for t in range(nobs):
            lnsigma2[t] = parameters[0]
            if t == 0:
                lnsigma2[t] += parameters[3] * backcast
            else:
                stdresid = resids[t - 1] / np.sqrt(sigma2[t - 1])
                lnsigma2[t] += parameters[1] * (np.abs(stdresid) - norm_const)
                lnsigma2[t] += parameters[2] * stdresid
                lnsigma2[t] += parameters[3] * lnsigma2[t - 1]
            sigma2[t] = np.exp(lnsigma2[t])
        assert_almost_equal(sigma2_python, sigma2)



