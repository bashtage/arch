import unittest

import scipy.stats as stats
from scipy.special import gammaln
from numpy.testing import assert_almost_equal, assert_equal, \
    assert_array_equal, assert_raises
import numpy as np

from arch.univariate.distribution import Normal, StudentsT


class TestDistributions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T = 1000
        cls.resids = np.random.randn(cls.T)
        cls.sigma2 = 1 + np.random.random(cls.resids.shape)

    def test_normal(self):
        dist = Normal()
        ll1 = dist.loglikelihoood([], self.resids, self.sigma2)
        scipy_dist = stats.norm
        ll2 = scipy_dist.logpdf(self.resids, scale=np.sqrt(self.sigma2)).sum()
        assert_almost_equal(ll1, ll2)

        assert_equal(dist.num_params, 0)

        bounds = dist.bounds(self.resids)
        assert_equal(len(bounds), 0)

        a, b = dist.constraints()
        assert_equal(len(a), 0)

        assert_array_equal(dist.starting_values(self.resids), np.empty((0,)))

    def test_studentst(self):
        dist = StudentsT()
        v = 4.0
        ll1 = dist.loglikelihoood(np.array([v]), self.resids, self.sigma2)
        # Direct calculation of PDF, then log
        constant = np.exp(gammaln(0.5 * (v + 1)) - gammaln(0.5 * v))
        pdf = constant / np.sqrt(np.pi * (v - 2) * self.sigma2)
        pdf *= (1 + self.resids ** 2.0 / (self.sigma2 * (v - 2))) ** (
            -(v + 1) / 2)
        ll2 = np.log(pdf).sum()
        assert_almost_equal(ll1, ll2)

        assert_equal(dist.num_params, 1)

        bounds = dist.bounds(self.resids)
        assert_equal(len(bounds), 1)

        A, b = dist.constraints()
        assert_equal(A.shape, (2, 1))

        k = stats.kurtosis(self.resids, fisher=False)
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)
        assert_array_equal(dist.starting_values(self.resids), np.array([sv]))

        assert_raises(ValueError, dist.simulate, np.array([1.5]))

