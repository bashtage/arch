from unittest import TestCase
from arch.compat.python import range

import timeit

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import arch.univariate.recursions_python as recpy
try:
    import arch.univariate.recursions as rec_cython
    missing_extension = False
except ImportError:
    missing_extension = True

if missing_extension:
    rec = recpy
else:
    rec = rec_cython

try:
    import numba  # noqa
    missing_numba = False
except ImportError:
    missing_numba = True


class Timer(object):
    def __init__(self, first, first_name, second, second_name, model_name,
                 setup, repeat=5, number=10):
        self.first_code = first
        self.second_code = second
        self.setup = setup
        self.first_name = first_name
        self.second_name = second_name
        self.model_name = model_name
        self.repeat = repeat
        self.number = number
        self._run = False
        self.times = []
        self._codes = [first, second]
        self.ratio = np.inf

    def display(self):
        if not self._run:
            self.time()
        self.ratio = self.times[0] / self.times[1]

        print(self.model_name + ' timing')
        print(self.first_name + ': ' + str(self.times[0]) + 's')
        print(self.second_name + ': ' + str(self.times[1]) + 's')
        print(self.first_name + '/' + self.second_name + ' Ratio: ' +
              str(self.ratio) + 's')
        print('\n')

    def time(self):
        self.times = []
        for code in self._codes:
            timer = timeit.Timer(code, setup=self.setup)
            self.times.append(min(timer.repeat(self.repeat, self.number)))
        return None


class TestRecursions(TestCase):
    @classmethod
    def setup_class(cls):
        cls.T = 1000
        cls.resids = np.random.randn(cls.T)
        cls.sigma2 = np.zeros_like(cls.resids)
        var = cls.resids.var()
        var_bounds = np.array([var / 1000000.0, var * 1000000.0])
        cls.var_bounds = np.ones((cls.T, 2)) * var_bounds
        cls.backcast = 1.0
        cls.timer_setup = """
import numpy as np
import arch.univariate.recursions as rec
import arch.univariate.recursions_python as recpy
from arch.compat.python import range

T = 10000
resids = np.random.randn(T)
sigma2 = np.zeros_like(resids)
var = resids.var()
backcast = 1.0
var_bounds = np.array([var / 1000000.0, var * 1000000.0])
var_bounds = np.ones((T, 2)) * var_bounds
        """

    def test_garch(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3, .2])
        fresids = resids ** 2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(parameters, fresids, sresids, sigma2,
                              1, 1, 1, T, backcast, self.var_bounds)
        sigma2_numba = sigma2.copy()
        recpy.garch_recursion_python(parameters, fresids, sresids, sigma2, 1,
                                     1, 1, T, backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        rec.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1,
                            1, T, backcast, self.var_bounds)
        assert_almost_equal(sigma2_numba, sigma2)
        assert_almost_equal(sigma2_python, sigma2)

    def test_harch(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3, .2])
        lags = np.array([1, 5, 22], dtype=np.int32)
        recpy.harch_recursion_python(parameters, resids, sigma2, lags, T,
                                     backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        recpy.harch_recursion(parameters, resids, sigma2, lags, T, backcast,
                              self.var_bounds)
        sigma2_numba = sigma2.copy()
        rec.harch_recursion(parameters, resids, sigma2, lags, T, backcast,
                            self.var_bounds)
        assert_almost_equal(sigma2_numba, sigma2)
        assert_almost_equal(sigma2_python, sigma2)

    def test_arch(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3, .2])
        p = 3

        recpy.arch_recursion_python(parameters, resids, sigma2, p, T,
                                    backcast, self.var_bounds)
        sigma2_python = sigma2.copy()
        recpy.arch_recursion(parameters, resids, sigma2, p, T,
                             backcast, self.var_bounds)
        sigma2_numba = sigma2.copy()
        rec.arch_recursion(parameters, resids, sigma2, p, T, backcast,
                           self.var_bounds)
        assert_almost_equal(sigma2_numba, sigma2)
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_power_1(self):
        T, resids, = self.T, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([.1, .4, .3, .2])
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
        assert (sigma2 >= self.var_bounds[:, 1]).all()

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
        assert (sigma2 >= self.var_bounds[:, 1]).all()

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
        assert (sigma2 >= self.var_bounds[:, 1]).all()

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
        recpy.egarch_recursion(parameters, resids, sigma2, p, o, q, nobs,
                               backcast, var_bounds, lnsigma2, std_resids,
                               abs_std_resids)
        sigma2_numba = sigma2.copy()
        recpy.egarch_recursion_python(parameters, resids, sigma2, p, o, q,
                                      nobs, backcast, var_bounds, lnsigma2,
                                      std_resids, abs_std_resids)
        sigma2_python = sigma2.copy()
        rec.egarch_recursion(parameters, resids, sigma2, p, o, q, nobs,
                             backcast, var_bounds, lnsigma2, std_resids,
                             abs_std_resids)
        assert_almost_equal(sigma2_numba, sigma2)
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

    @pytest.mark.skipif(missing_numba or missing_extension, reason='numba not installed')
    def test_garch_performance(self):
        garch_setup = """
parameters = np.array([.1, .4, .3, .2])
fresids = resids ** 2.0
sresids = np.sign(resids)
        """

        garch_first = """
recpy.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1, 1, T,
backcast, var_bounds)
        """
        garch_second = """
rec.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1, 1, T, backcast,
var_bounds)
        """
        timer = Timer(garch_first, 'Numba', garch_second, 'Cython', 'GARCH',
                      self.timer_setup + garch_setup)
        timer.display()
        assert timer.ratio < 10.0

    @pytest.mark.skipif(missing_numba or missing_extension, reason='numba not installed')
    def test_harch_performance(self):
        harch_setup = """
parameters = np.array([.1, .4, .3, .2])
lags = np.array([1, 5, 22], dtype=np.int32)
        """

        harch_first = """
recpy.harch_recursion(parameters, resids, sigma2, lags, T, backcast,
var_bounds)
        """

        harch_second = """
rec.harch_recursion(parameters, resids, sigma2, lags, T, backcast, var_bounds)
        """

        timer = Timer(harch_first, 'Numba', harch_second, 'Cython', 'HARCH',
                      self.timer_setup + harch_setup)
        timer.display()
        assert timer.ratio < 10.0

    @pytest.mark.skipif(missing_numba or missing_extension, reason='numba not installed')
    def test_egarch_performance(self):
        egarch_setup = """
nobs = T
parameters = np.array([0.0, 0.1, -0.1, 0.95])
p = o = q = 1
backcast = 0.0
lnsigma2 = np.empty_like(sigma2)
std_resids = np.empty_like(sigma2)
abs_std_resids = np.empty_like(sigma2)
        """

        egarch_first = """
rec.egarch_recursion(parameters, resids, sigma2, p, o, q, nobs, backcast,
var_bounds, lnsigma2, std_resids, abs_std_resids)
"""

        egarch_second = """
recpy.egarch_recursion(parameters, resids, sigma2, p, o, q, nobs, backcast,
var_bounds, lnsigma2, std_resids, abs_std_resids)
"""
        timer = Timer(egarch_first, 'Numba', egarch_second, 'Cython', 'EGARCH',
                      self.timer_setup + egarch_setup)
        timer.display()
