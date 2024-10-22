import os
import pickle
import timeit
import types

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy.special import gamma

import arch.univariate.recursions_python as recpy
from arch.univariate.volatility import RiskMetrics2006

CYTHON_COVERAGE = os.environ.get("ARCH_CYTHON_COVERAGE", "0") in ("true", "1", "True")
DISABLE_NUMBA = os.environ.get("ARCH_DISABLE_NUMBA", False) in ("1", "true", "True")

try:
    import arch.univariate.recursions as rec_cython

    MISSING_EXTENSION = False
except ImportError:
    MISSING_EXTENSION = True

if MISSING_EXTENSION:
    rec: types.ModuleType = recpy
else:
    rec = rec_cython

try:
    import numba  # noqa

    MISSING_NUMBA = False or DISABLE_NUMBA
except ImportError:
    MISSING_NUMBA = True

pytestmark = pytest.mark.filterwarnings("ignore::arch.compat.numba.PerformanceWarning")


class Timer:
    def __init__(
        self,
        first,
        first_name,
        second,
        second_name,
        model_name,
        setup,
        repeat=5,
        number=10,
    ) -> None:
        self.first_code = first
        self.second_code = second
        self.setup = setup
        self.first_name = first_name
        self.second_name = second_name
        self.model_name = model_name
        self.repeat = repeat
        self.number = number
        self._run = False
        self.times: list[float] = []
        self._codes = [first, second]
        self.ratio = np.inf

    def display(self):
        if not self._run:
            self.time()
        self.ratio = self.times[0] / self.times[1]
        title = self.model_name + " timing"
        print("\n" + title)
        print("-" * len(title))
        print(self.first_name + ": " + f"{1000 * self.times[0]:0.3f} ms")
        print(self.second_name + ": " + f"{1000 * self.times[1]:0.3f} ms")
        if self.ratio < 1:
            print(f"{self.first_name} is {100 * (1 / self.ratio - 1):0.1f}% faster")
        else:
            print(f"{self.second_name} is {100 * (self.ratio - 1):0.1f}% faster")
        print(self.first_name + "/" + self.second_name + f" Ratio: {self.ratio:0.3f}\n")

    def time(self):
        self.times = []
        for code in self._codes:
            timer = timeit.Timer(code, setup=self.setup)
            self.times.append(min(timer.repeat(self.repeat, self.number)))


class TestRecursions:
    @classmethod
    def setup_class(cls):
        cls.nobs = 1000
        cls.rng = RandomState(12345)
        cls.resids = cls.rng.standard_normal(cls.nobs)
        cls.sigma2 = np.zeros_like(cls.resids)
        var = cls.resids.var()
        var_bounds = np.array([var / 1000000.0, var * 1000000.0])
        cls.var_bounds = np.ones((cls.nobs, 2)) * var_bounds
        cls.backcast = 1.0
        cls.timer_setup = """
import numpy as np
import arch.univariate.recursions as rec
import arch.univariate.recursions_python as recpy

nobs = 10000
resids = np.random.standard_normal(nobs)
sigma2 = np.zeros_like(resids)
var = resids.var()
backcast = 1.0
var_bounds = np.array([var / 1000000.0, var * 1000000.0])
var_bounds = np.ones((nobs, 2)) * var_bounds
"""

    def test_garch(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.4, 0.3, 0.2])
        fresids = resids**2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        sigma2_numba = sigma2.copy()
        recpy.garch_recursion_python(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        sigma2_python = sigma2.copy()
        rec.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert_almost_equal(sigma2_numba, sigma2)
        assert_almost_equal(sigma2_python, sigma2)

        parameters = np.array([0.1, -0.4, 0.3, 0.2])
        recpy.garch_recursion_python(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([0.1, 0.4, 3, 2])
        recpy.garch_recursion_python(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([0.1, 0.4, 0.3, 0.2])
        mod_fresids = fresids.copy()
        mod_fresids[:1] = np.inf
        recpy.garch_recursion_python(
            parameters,
            mod_fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])
        rec.garch_recursion(
            parameters,
            mod_fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

    def test_garch_update(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.4, 0.3, 0.2])
        fresids = resids**2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        sigma2_ref = sigma2.copy()
        sigma2[:] = np.nan
        for t in range(nobs):
            sigma2[t] = recpy.garch_core_python(
                t, parameters, resids, sigma2, backcast, self.var_bounds, 1, 1, 1, 2.0
            )
        assert_allclose(sigma2, sigma2_ref)
        sigma2[:] = np.nan
        for t in range(nobs):
            sigma2[t] = recpy.garch_core(
                t, parameters, resids, sigma2, backcast, self.var_bounds, 1, 1, 1, 2.0
            )
        assert_allclose(sigma2, sigma2_ref)

        for t in range(nobs):
            sigma2[t] = rec.garch_core(
                t, parameters, resids, sigma2, backcast, self.var_bounds, 1, 1, 1, 2.0
            )
        assert_allclose(sigma2, sigma2_ref)

        sigma2[:] = np.nan
        gu = recpy.GARCHUpdater(1, 1, 1, 2.0)
        gu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            gu.update(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                gu = pickle.loads(pickle.dumps(gu))
        gu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_ref)

        sigma2[:] = np.nan
        gu = rec.GARCHUpdater(1, 1, 1, 2.0)
        gu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            gu._update_tester(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                gu = pickle.loads(pickle.dumps(gu))
        gu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_ref)

    def test_harch(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.4, 0.3, 0.2])
        lags = np.array([1, 5, 22], dtype=np.int32)
        recpy.harch_recursion_python(
            parameters, resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        sigma2_python = sigma2.copy()
        recpy.harch_recursion(
            parameters, resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        sigma2_numba = sigma2.copy()
        rec.harch_recursion(
            parameters, resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        assert_almost_equal(sigma2_numba, sigma2)
        assert_almost_equal(sigma2_python, sigma2)

        parameters = np.array([-0.1, -0.4, 0.3, 0.2])
        recpy.harch_recursion_python(
            parameters, resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([0.1, 4e8, 3, 2])
        recpy.harch_recursion_python(
            parameters, resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([0.1, 4e8, 3, 2])
        mod_resids = resids.copy()
        mod_resids[:10] = np.inf
        recpy.harch_recursion_python(
            parameters, mod_resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])
        rec.harch_recursion(
            parameters, mod_resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

    def test_harch_update(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.4, 0.3, 0.2])
        lags = np.array([1, 5, 22], dtype=np.int32)
        rec.harch_recursion(
            parameters, resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        sigma2_direct = sigma2.copy()
        sigma2[:] = np.nan
        for t in range(nobs):
            recpy.harch_core_python(
                t, parameters, resids, sigma2, lags, backcast, self.var_bounds
            )
        assert_allclose(sigma2, sigma2_direct)

        for t in range(nobs):
            recpy.harch_core(
                t, parameters, resids, sigma2, lags, backcast, self.var_bounds
            )
        assert_allclose(sigma2, sigma2_direct)

        for t in range(nobs):
            rec.harch_core(
                t, parameters, resids, sigma2, lags, backcast, self.var_bounds
            )
        assert_allclose(sigma2, sigma2_direct)

        sigma2[:] = np.nan
        hu = recpy.HARCHUpdater(lags)
        hu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            hu.update(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                hu = pickle.loads(pickle.dumps(hu))
        hu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_direct)

        sigma2[:] = np.nan
        hu = rec.HARCHUpdater(lags)
        hu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            hu._update_tester(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                hu = pickle.loads(pickle.dumps(hu))
        hu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_direct)

    def test_arch(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.4, 0.3, 0.2])
        p = 3

        recpy.arch_recursion_python(
            parameters, resids, sigma2, p, nobs, backcast, self.var_bounds
        )
        sigma2_python = sigma2.copy()
        recpy.arch_recursion(
            parameters, resids, sigma2, p, nobs, backcast, self.var_bounds
        )
        sigma2_numba = sigma2.copy()
        rec.arch_recursion(
            parameters, resids, sigma2, p, nobs, backcast, self.var_bounds
        )
        assert_almost_equal(sigma2_numba, sigma2)
        assert_almost_equal(sigma2_python, sigma2)

        parameters = np.array([-0.1, -0.4, 0.3, 0.2])
        recpy.arch_recursion_python(
            parameters, resids, sigma2, p, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([0.1, 4e8, 3, 2])
        recpy.arch_recursion_python(
            parameters, resids, sigma2, p, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        mod_resids = resids.copy()
        mod_resids[:10] = np.inf
        recpy.arch_recursion_python(
            parameters, mod_resids, sigma2, p, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])
        rec.arch_recursion(
            parameters, mod_resids, sigma2, p, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

    def test_garch_power_1(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.4, 0.3, 0.2])
        fresids = np.abs(resids) ** 1.0
        sresids = np.sign(resids)

        recpy.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        sigma2_python = sigma2.copy()
        rec.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_direct(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.4, 0.3, 0.2])
        fresids = np.abs(resids) ** 2.0
        sresids = np.sign(resids)

        for t in range(nobs):
            if t == 0:
                sigma2[t] = parameters.dot(
                    np.array([1.0, backcast, 0.5 * backcast, backcast])
                )
            else:
                var = np.array(
                    [
                        1.0,
                        resids[t - 1] ** 2.0,
                        resids[t - 1] ** 2.0 * (resids[t - 1] < 0),
                        sigma2[t - 1],
                    ]
                )
                sigma2[t] = parameters.dot(var)

        sigma2_python = sigma2.copy()
        rec.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_no_q(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.4, 0.3])
        fresids = resids**2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            0,
            nobs,
            backcast,
            self.var_bounds,
        )
        sigma2_python = sigma2.copy()
        rec.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            0,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_no_p(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.4, 0.3])
        fresids = resids**2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            0,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        sigma2_python = sigma2.copy()
        rec.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            0,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_no_o(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.4, 0.3, 0.2])
        fresids = resids**2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            0,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        sigma2_python = sigma2.copy()
        rec.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            0,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert_almost_equal(sigma2_python, sigma2)

    def test_garch_arch(self):
        backcast = self.backcast
        nobs, resids, sigma2 = self.nobs, self.resids, self.sigma2

        parameters = np.array([0.1, 0.4, 0.3, 0.2])
        fresids = resids**2.0
        sresids = np.sign(resids)

        rec.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            3,
            0,
            0,
            nobs,
            backcast,
            self.var_bounds,
        )
        sigma2_garch = sigma2.copy()
        rec.arch_recursion(
            parameters, resids, sigma2, 3, nobs, backcast, self.var_bounds
        )

        assert_almost_equal(sigma2_garch, sigma2)

    def test_bounds(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([1e100, 0.4, 0.3, 0.2])
        lags = np.array([1, 5, 22], dtype=np.int32)
        recpy.harch_recursion(
            parameters, resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        sigma2_python = sigma2.copy()
        rec.harch_recursion(
            parameters, resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        assert_almost_equal(sigma2_python, sigma2)
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([-1e100, 0.4, 0.3, 0.2])
        recpy.harch_recursion(
            parameters, resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        sigma2_python = sigma2.copy()
        rec.harch_recursion(
            parameters, resids, sigma2, lags, nobs, backcast, self.var_bounds
        )
        assert_almost_equal(sigma2_python, sigma2)
        assert_almost_equal(sigma2, self.var_bounds[:, 0])

        parameters = np.array([1e100, 0.4, 0.3, 0.2])
        fresids = resids**2.0
        sresids = np.sign(resids)

        recpy.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        sigma2_python = sigma2.copy()
        rec.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert_almost_equal(sigma2_python, sigma2)
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([-1e100, 0.4, 0.3, 0.2])
        recpy.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        sigma2_python = sigma2.copy()
        rec.garch_recursion(
            parameters,
            fresids,
            sresids,
            sigma2,
            1,
            1,
            1,
            nobs,
            backcast,
            self.var_bounds,
        )
        assert_almost_equal(sigma2_python, sigma2)
        assert_almost_equal(sigma2, self.var_bounds[:, 0])

        parameters = np.array([1e100, 0.4, 0.3, 0.2])
        recpy.arch_recursion(
            parameters, resids, sigma2, 3, nobs, backcast, self.var_bounds
        )
        sigma2_python = sigma2.copy()
        rec.arch_recursion(
            parameters, resids, sigma2, 3, nobs, backcast, self.var_bounds
        )
        assert_almost_equal(sigma2_python, sigma2)
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([-1e100, 0.4, 0.3, 0.2])
        recpy.arch_recursion(
            parameters, resids, sigma2, 3, nobs, backcast, self.var_bounds
        )
        sigma2_python = sigma2.copy()
        rec.arch_recursion(
            parameters, resids, sigma2, 3, nobs, backcast, self.var_bounds
        )
        assert_almost_equal(sigma2_python, sigma2)
        assert_almost_equal(sigma2, self.var_bounds[:, 0])

    def test_egarch(self):
        nobs = self.nobs
        parameters = np.array([0.0, 0.1, -0.1, 0.95])
        resids, sigma2 = self.resids, self.sigma2
        p = o = q = 1
        backcast = 0.0
        var_bounds = self.var_bounds
        lnsigma2 = np.empty_like(sigma2)
        std_resids = np.empty_like(sigma2)
        abs_std_resids = np.empty_like(sigma2)
        recpy.egarch_recursion(
            parameters,
            resids,
            sigma2,
            p,
            o,
            q,
            nobs,
            backcast,
            var_bounds,
            lnsigma2,
            std_resids,
            abs_std_resids,
        )
        sigma2_numba = sigma2.copy()
        recpy.egarch_recursion_python(
            parameters,
            resids,
            sigma2,
            p,
            o,
            q,
            nobs,
            backcast,
            var_bounds,
            lnsigma2,
            std_resids,
            abs_std_resids,
        )
        sigma2_python = sigma2.copy()
        rec.egarch_recursion(
            parameters,
            resids,
            sigma2,
            p,
            o,
            q,
            nobs,
            backcast,
            var_bounds,
            lnsigma2,
            std_resids,
            abs_std_resids,
        )
        assert_almost_equal(sigma2_numba, sigma2)
        assert_almost_equal(sigma2_python, sigma2)

        sigma2[:] = np.nan
        eu = recpy.EGARCHUpdater(p, o, q)
        eu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            eu._update_tester(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                eu = pickle.loads(pickle.dumps(eu))
        eu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_python)

        sigma2[:] = np.nan
        eu = rec.EGARCHUpdater(p, o, q)
        eu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            eu._update_tester(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                eu = pickle.loads(pickle.dumps(eu))
        eu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_python)

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

        parameters = np.array([-100.0, 0.1, -0.1, 0.95])
        recpy.egarch_recursion_python(
            parameters,
            resids,
            sigma2,
            p,
            o,
            q,
            nobs,
            backcast,
            var_bounds,
            lnsigma2,
            std_resids,
            abs_std_resids,
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([0.0, 0.1, -0.1, 9.5])
        recpy.egarch_recursion_python(
            parameters,
            resids,
            sigma2,
            p,
            o,
            q,
            nobs,
            backcast,
            var_bounds,
            lnsigma2,
            std_resids,
            abs_std_resids,
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([0.0, 0.1, -0.1, 0.95])
        mod_resids = resids.copy()
        mod_resids[:1] = np.inf
        recpy.egarch_recursion_python(
            parameters,
            resids,
            sigma2,
            p,
            o,
            q,
            nobs,
            backcast,
            var_bounds,
            lnsigma2,
            std_resids,
            abs_std_resids,
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([0.0, 31, -0.1, 0.95])
        mod_resids = resids.copy()
        mod_resids[100] = np.finfo(float).max
        recpy.egarch_recursion_python(
            parameters,
            mod_resids,
            sigma2,
            p,
            o,
            q,
            nobs,
            backcast,
            var_bounds,
            lnsigma2,
            std_resids,
            abs_std_resids,
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        rec.egarch_recursion(
            parameters,
            mod_resids,
            sigma2,
            p,
            o,
            q,
            nobs,
            backcast,
            var_bounds,
            lnsigma2,
            std_resids,
            abs_std_resids,
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

    def test_midas_hyperbolic(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.8, 0])
        j = np.arange(1, 22 + 1)
        weights = gamma(j + 0.6) / (gamma(j + 1) * gamma(0.6))
        weights = weights / weights.sum()
        recpy.midas_recursion(
            parameters, weights, resids, sigma2, nobs, backcast, self.var_bounds
        )
        sigma2_numba = sigma2.copy()
        recpy.midas_recursion_python(
            parameters, weights, resids, sigma2, nobs, backcast, self.var_bounds
        )
        sigma2_python = sigma2.copy()
        rec.midas_recursion(
            parameters, weights, resids, sigma2, nobs, backcast, self.var_bounds
        )
        assert_almost_equal(sigma2_numba, sigma2)
        assert_almost_equal(sigma2_python, sigma2)

        mod_resids = resids.copy()
        mod_resids[:10] = np.inf
        recpy.midas_recursion_python(
            parameters, weights, mod_resids, sigma2, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([0.1, 10e10, 0])
        j = np.arange(1, 22 + 1)
        weights = gamma(j + 0.6) / (gamma(j + 1) * gamma(0.6))
        weights = weights / weights.sum()
        recpy.midas_recursion_python(
            parameters, weights, resids, sigma2, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])
        rec.midas_recursion(
            parameters, weights, resids, sigma2, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

        parameters = np.array([0.1, -0.4, 0])
        recpy.midas_recursion_python(
            parameters, weights, resids, sigma2, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])
        rec.midas_recursion(
            parameters, weights, resids, sigma2, nobs, backcast, self.var_bounds
        )
        assert np.all(sigma2 >= self.var_bounds[:, 0])
        assert np.all(sigma2 <= 1.1 * self.var_bounds[:, 1])

    def test_midas_hyperbolic_update(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast

        parameters = np.array([0.1, 0.6, 0.2])
        j = np.arange(1, 22 + 1)
        weights = gamma(j + 0.6) / (gamma(j + 1) * gamma(0.6))
        weights = weights / weights.sum()
        recpy.midas_recursion(
            parameters, weights, resids, sigma2, nobs, backcast, self.var_bounds
        )
        sigma2_ref = sigma2.copy()

        sigma2[:] = np.nan
        parameters = np.array([0.1, 0.6, 0.2, 0.6])
        mu = recpy.MIDASUpdater(22, True)
        mu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            mu._update_tester(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                mu = pickle.loads(pickle.dumps(mu))
        mu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_ref)

        sigma2[:] = np.nan
        mu = rec.MIDASUpdater(22, True)
        mu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            mu._update_tester(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                mu = pickle.loads(pickle.dumps(mu))
        mu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_ref)

    def test_figarch_recursion(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast
        parameters = np.array([1.0, 0.2, 0.4, 0.3])
        fresids = resids**2
        p = q = 1
        trunc_lag = 1000
        rec.figarch_recursion(
            parameters,
            fresids,
            sigma2,
            p,
            q,
            nobs,
            trunc_lag,
            backcast,
            self.var_bounds,
        )
        lam = rec.figarch_weights(parameters[1:], p, q, trunc_lag=trunc_lag)
        lam_rev = lam[::-1]
        omega_tilde = parameters[0] / (1 - parameters[-1])
        sigma2_direct = np.empty_like(sigma2)
        for t in range(nobs):
            backcasts = trunc_lag - t
            sigma2_direct[t] = omega_tilde
            if backcasts:
                sigma2_direct[t] += backcast * lam_rev[:backcasts].sum()
            if t:
                sigma2_direct[t] += np.sum(lam_rev[-t:] * fresids[max(0, t - 1000) : t])
        assert_almost_equal(sigma2_direct, sigma2)

        recpy.figarch_recursion(
            parameters,
            fresids,
            sigma2,
            p,
            q,
            nobs,
            trunc_lag,
            backcast,
            self.var_bounds,
        )
        sigma2_numba = sigma2.copy()
        recpy.figarch_recursion_python(
            parameters,
            fresids,
            sigma2,
            p,
            q,
            nobs,
            trunc_lag,
            backcast,
            self.var_bounds,
        )
        sigma2_python = sigma2.copy()
        rec.figarch_recursion(
            parameters,
            fresids,
            sigma2,
            p,
            q,
            nobs,
            trunc_lag,
            backcast,
            self.var_bounds,
        )
        assert_almost_equal(sigma2_numba, sigma2)
        assert_almost_equal(sigma2_python, sigma2)

    def test_figarch_weights(self):
        parameters = np.array([1.0, 0.4])
        lam = rec.figarch_weights(parameters[1:], 0, 0, trunc_lag=1000)
        lam_direct = np.empty_like(lam)
        lam_direct[0] = parameters[-1]
        for i in range(1, 1000):
            lam_direct[i] = (i - parameters[-1]) / (i + 1) * lam_direct[i - 1]
        assert_almost_equal(lam, lam_direct)

    @pytest.mark.skipif(
        MISSING_NUMBA or MISSING_EXTENSION, reason="numba not installed"
    )
    def test_garch_performance(self):
        garch_setup = """
parameters = np.array([.1, .4, .3, .2])
fresids = resids ** 2.0
sresids = np.sign(resids)
        """

        garch_first = """
recpy.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1, 1, nobs,
backcast, var_bounds)
        """
        garch_second = """
rec.garch_recursion(parameters, fresids, sresids, sigma2, 1, 1, 1, nobs, backcast,
var_bounds)
        """
        timer = Timer(
            garch_first,
            "Numba",
            garch_second,
            "Cython",
            "GARCH",
            self.timer_setup + garch_setup,
        )
        timer.display()
        assert timer.ratio < 10.0
        if not (MISSING_NUMBA or CYTHON_COVERAGE):
            assert 0.1 < timer.ratio

    @pytest.mark.skipif(
        MISSING_NUMBA or MISSING_EXTENSION, reason="numba not installed"
    )
    def test_harch_performance(self):
        harch_setup = """
parameters = np.array([.1, .4, .3, .2])
lags = np.array([1, 5, 22], dtype=np.int32)
        """

        harch_first = """
recpy.harch_recursion(parameters, resids, sigma2, lags, nobs, backcast,
var_bounds)
        """

        harch_second = """
rec.harch_recursion(parameters, resids, sigma2, lags, nobs, backcast, var_bounds)
        """

        timer = Timer(
            harch_first,
            "Numba",
            harch_second,
            "Cython",
            "HARCH",
            self.timer_setup + harch_setup,
        )
        timer.display()
        assert timer.ratio < 10.0
        if not (MISSING_NUMBA or CYTHON_COVERAGE):
            assert 0.1 < timer.ratio

    @pytest.mark.skipif(
        MISSING_NUMBA or MISSING_EXTENSION, reason="numba not installed"
    )
    def test_egarch_performance(self):
        egarch_setup = """
parameters = np.array([0.0, 0.1, -0.1, 0.95])
p = o = q = 1
backcast = 0.0
lnsigma2 = np.empty_like(sigma2)
std_resids = np.empty_like(sigma2)
abs_std_resids = np.empty_like(sigma2)
        """

        egarch_first = """
recpy.egarch_recursion(parameters, resids, sigma2, p, o, q, nobs, backcast,
var_bounds, lnsigma2, std_resids, abs_std_resids)
"""

        egarch_second = """
rec.egarch_recursion(parameters, resids, sigma2, p, o, q, nobs, backcast,
var_bounds, lnsigma2, std_resids, abs_std_resids)
"""
        timer = Timer(
            egarch_first,
            "Numba",
            egarch_second,
            "Cython",
            "EGARCH",
            self.timer_setup + egarch_setup,
        )
        timer.display()
        assert timer.ratio < 10.0
        if not (MISSING_NUMBA or CYTHON_COVERAGE):
            assert 0.1 < timer.ratio

    @pytest.mark.skipif(
        MISSING_NUMBA or MISSING_EXTENSION, reason="numba not installed"
    )
    def test_midas_performance(self):
        midas_setup = """
from scipy.special import gamma
parameters = np.array([.1, 0.8, 0])
j = np.arange(1,22+1)
weights = gamma(j+0.6) / (gamma(j+1) * gamma(0.6))
weights = weights / weights.sum()
"""

        midas_first = """
recpy.midas_recursion(parameters, weights, resids, sigma2, nobs, backcast, var_bounds)
                """
        midas_second = """
rec.midas_recursion(parameters, weights, resids, sigma2, nobs, backcast, var_bounds)
"""
        timer = Timer(
            midas_first,
            "Numba",
            midas_second,
            "Cython",
            "MIDAS",
            self.timer_setup + midas_setup,
        )
        timer.display()
        assert timer.ratio < 10.0
        if not (MISSING_NUMBA or CYTHON_COVERAGE):
            assert 0.1 < timer.ratio

    @pytest.mark.skipif(
        MISSING_NUMBA or MISSING_EXTENSION, reason="numba not installed"
    )
    def test_figarch_performance(self):
        midas_setup = """
p = q = 1
trunc_lag = 1000
parameters = np.array([1.0, 0.2, 0.2, 0.04])
fresids = resids ** 2.0
"""

        midas_first = """
recpy.figarch_recursion(parameters, fresids, sigma2, p, q,
                        nobs, trunc_lag, backcast, var_bounds)
"""
        midas_second = """
rec.figarch_recursion(parameters, fresids, sigma2, p, q,
                      nobs, trunc_lag, backcast, var_bounds)
"""
        timer = Timer(
            midas_first,
            "Numba",
            midas_second,
            "Cython",
            "FIGARCH",
            self.timer_setup + midas_setup,
        )
        timer.display()
        assert timer.ratio < 10.0
        if not (MISSING_NUMBA or CYTHON_COVERAGE):
            assert 0.1 < timer.ratio

    def test_garch_aparch_equiv(self):
        parameters = np.array([0.1, 0.1, 0.8])
        fresids = self.resids**2
        sresids = np.sign(self.resids)
        sigma2 = np.empty(1000)
        p = q = 1
        o = 0
        recpy.garch_recursion_python(
            parameters,
            fresids,
            sresids,
            sigma2,
            p,
            o,
            q,
            self.nobs,
            self.backcast,
            self.var_bounds,
        )
        sigma2_garch = sigma2.copy()

        parameters = np.array([0.1, 0.1, 0.8, 2])
        sigma2[:] = np.nan
        sigma2_delta = np.empty_like(sigma2)
        recpy.aparch_recursion_python(
            parameters,
            self.resids,
            np.abs(self.resids),
            sigma2,
            sigma2_delta,
            p,
            o,
            q,
            self.nobs,
            self.backcast,
            self.var_bounds,
        )
        assert_allclose(sigma2_garch, sigma2, atol=1e-6)

        sigma2[:] = np.nan
        recpy.aparch_recursion(
            parameters,
            self.resids,
            np.abs(self.resids),
            sigma2,
            sigma2_delta,
            p,
            o,
            q,
            self.nobs,
            self.backcast,
            self.var_bounds,
        )
        assert_allclose(sigma2_garch, sigma2, atol=1e-6)

        sigma2[:] = np.nan
        rec.aparch_recursion(
            parameters,
            self.resids,
            np.abs(self.resids),
            sigma2,
            sigma2_delta,
            p,
            o,
            q,
            self.nobs,
            self.backcast,
            self.var_bounds,
        )
        assert_allclose(sigma2_garch, sigma2, atol=1e-6)

    def test_asym_aparch_smoke(self):
        sigma2 = np.empty(1000)
        p = o = q = 1
        parameters = np.array([0.1, 0.1, 0.1, 0.8, 1.3])
        sigma2[:] = np.nan
        sigma2_delta = np.empty_like(sigma2)
        recpy.aparch_recursion_python(
            parameters,
            self.resids,
            np.abs(self.resids),
            sigma2,
            sigma2_delta,
            p,
            o,
            q,
            self.nobs,
            self.backcast,
            self.var_bounds,
        )
        assert np.all(np.isfinite(sigma2))
        sigma2_py = sigma2.copy()
        sigma2[:] = np.nan
        recpy.aparch_recursion(
            parameters,
            self.resids,
            np.abs(self.resids),
            sigma2,
            sigma2_delta,
            p,
            o,
            q,
            self.nobs,
            self.backcast,
            self.var_bounds,
        )
        assert np.all(np.isfinite(sigma2))
        assert_allclose(sigma2_py, sigma2)

        sigma2[:] = np.nan
        rec.aparch_recursion(
            parameters,
            self.resids,
            np.abs(self.resids),
            sigma2,
            sigma2_delta,
            p,
            o,
            q,
            self.nobs,
            self.backcast,
            self.var_bounds,
        )
        assert np.all(np.isfinite(sigma2))
        assert_allclose(sigma2_py, sigma2)

    @pytest.mark.parametrize("lam", [None, 0.94])
    def test_ewma_update(self, lam):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast
        if lam is None:
            parameters = np.array([0.9])
        else:
            parameters = np.empty(0)
        sigma2[:] = np.nan
        eu = recpy.EWMAUpdater(lam)
        eu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            eu.update(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                eu = pickle.loads(pickle.dumps(eu))
        eu.initialize_update(parameters, backcast, nobs)
        sigma2_ref = sigma2.copy()

        sigma2[:] = np.nan
        eu = rec.EWMAUpdater(lam)
        eu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            eu._update_tester(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                eu = pickle.loads(pickle.dumps(eu))
        eu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_ref)

        sigma2[:] = np.nan
        gu = rec.GARCHUpdater(1, 0, 1, 2.0)
        _lam = 0.9 if lam is None else lam
        parameters = np.array([0, 1 - _lam, _lam])
        gu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            gu._update_tester(t, parameters, resids, sigma2, self.var_bounds)
        gu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_ref)

    def test_figarch_update(self):
        nobs, resids = self.nobs, self.resids
        sigma2, backcast = self.sigma2, self.backcast
        parameters = np.array([1.0, 0.2, 0.4, 0.3])
        fresids = resids**2
        p = q = 1
        trunc_lag = 1000
        rec.figarch_recursion(
            parameters,
            fresids,
            sigma2,
            p,
            q,
            nobs,
            trunc_lag,
            backcast,
            self.var_bounds,
        )
        sigma2_ref = sigma2.copy()

        fu = recpy.FIGARCHUpdater(p, q, 2.0, trunc_lag)
        fu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            fu._update_tester(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                fu = pickle.loads(pickle.dumps(fu))
        fu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_ref)

        fu = rec.FIGARCHUpdater(p, q, 2.0, trunc_lag)
        fu.initialize_update(parameters, backcast, nobs)
        for t in range(nobs):
            fu._update_tester(t, parameters, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                fu = pickle.loads(pickle.dumps(fu))
        fu.initialize_update(parameters, backcast, nobs)
        assert_allclose(sigma2, sigma2_ref)

    def test_rm_2006(self):
        rm2006 = RiskMetrics2006()
        params = np.empty(0)
        resids = self.resids
        sigma2 = self.sigma2
        nobs = self.nobs
        backcast = rm2006.backcast(self.resids)
        rm2006.compute_variance(params, resids, sigma2, backcast, self.var_bounds)

        sigma2_ref = sigma2.copy()
        sigma2[:] = np.nan
        cw = rm2006._ewma_combination_weights()
        sp = rm2006._ewma_smoothing_parameters()
        ru = recpy.RiskMetrics2006Updater(rm2006.kmax, cw, sp)
        ru.initialize_update(None, backcast, None)
        for t in range(nobs):
            ru._update_tester(t, params, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                ru = pickle.loads(pickle.dumps(ru))
        ru.initialize_update(None, backcast, None)
        assert_allclose(sigma2, sigma2_ref)
        sigma2_py = sigma2.copy()

        sigma2[:] = np.nan
        ru = rec.RiskMetrics2006Updater(rm2006.kmax, cw, sp)
        ru.initialize_update(None, backcast, None)
        for t in range(nobs):
            ru._update_tester(t, params, resids, sigma2, self.var_bounds)
            if t == nobs // 2:
                ru = pickle.loads(pickle.dumps(ru))
        ru.initialize_update(None, backcast, None)
        ru.initialize_update(None, 3.0, None)
        assert_allclose(sigma2, sigma2_py)


def test_bounds_check():
    var_bounds = np.array([0.1, 10])
    assert_almost_equal(recpy.bounds_check_python(-1.0, var_bounds), 0.1)
    assert_almost_equal(
        recpy.bounds_check_python(20.0, var_bounds), 10 + np.log(20.0 / 10.0)
    )
    assert_almost_equal(recpy.bounds_check_python(np.inf, var_bounds), 1010.0)
