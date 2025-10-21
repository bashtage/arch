from arch.compat.statsmodels import dataset_loader

from numpy import diff, log
from numpy.random import RandomState
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import macrodata

from arch.utility.cov import cov_nw


class TestVarNW:
    @classmethod
    def setup_class(cls):
        cls.rng = RandomState(12345)
        cls.cpi = log(dataset_loader(macrodata)["cpi"])
        cls.inflation = diff(cls.cpi)

    def test_cov_nw(self):
        y = self.inflation
        simple_cov = cov_nw(y, lags=0)
        e = y - y.mean()
        assert_almost_equal(e.dot(e) / e.shape[0], simple_cov)

    def test_cov_nw_ddof(self):
        y = self.inflation
        simple_cov = cov_nw(y, lags=0, ddof=1)
        e = y - y.mean()
        n = e.shape[0]
        assert_almost_equal(e.dot(e) / (n - 1), simple_cov)

    def test_cov_nw_no_demean(self):
        y = self.inflation
        simple_cov = cov_nw(y, lags=0, demean=False)
        assert_almost_equal(y.dot(y) / y.shape[0], simple_cov)

    def test_cov_nw_2d(self):
        y = self.rng.randn(100, 2)
        simple_cov = cov_nw(y, lags=0)
        e = y - y.mean(0)
        assert_almost_equal(e.T.dot(e) / e.shape[0], simple_cov)

    def test_cov_nw_2d_2lags(self):
        y = self.rng.randn(100, 2)
        e = y - y.mean(0)
        gamma_0 = e.T.dot(e)
        gamma_1 = e[1:].T.dot(e[:-1])
        gamma_2 = e[2:].T.dot(e[:-2])
        w1, w2 = 1.0 - (1.0 / 3.0), 1.0 - (2.0 / 3.0)
        expected = (
            gamma_0 + w1 * (gamma_1 + gamma_1.T) + w2 * (gamma_2 + gamma_2.T)
        ) / 100.0
        assert_almost_equal(cov_nw(y, lags=2), expected)

    def test_cov_nw_axis(self):
        y = self.rng.randn(100, 2)
        e = y - y.mean(0)
        gamma_0 = e.T.dot(e)
        gamma_1 = e[1:].T.dot(e[:-1])
        gamma_2 = e[2:].T.dot(e[:-2])
        w1, w2 = 1.0 - (1.0 / 3.0), 1.0 - (2.0 / 3.0)
        expected = (
            gamma_0 + w1 * (gamma_1 + gamma_1.T) + w2 * (gamma_2 + gamma_2.T)
        ) / 100.0
        assert_almost_equal(cov_nw(y.T, lags=2, axis=1), expected)

    def test_errors(self):
        y = self.rng.randn(100, 2)
        with pytest.raises(ValueError, match=r"lags must be weakly smaller than"):
            cov_nw(y, 200)
        with pytest.raises(ValueError, match=r"axis must be less than the"):
            cov_nw(y, axis=3)
        with pytest.raises(ValueError, match=r"ddof must be strictly smaller"):
            cov_nw(y, ddof=200)
