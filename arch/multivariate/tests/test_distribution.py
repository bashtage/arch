from unittest import TestCase

import numpy as np
from arch.multivariate.distribution import Normal
from arch.univariate import distribution as uni_distribution
from numpy.testing import assert_equal, assert_almost_equal


class TestNormal(TestCase):
    def test_simulation(self):
        n = Normal(5)
        sim = n.simulate([])
        data = sim(100)
        assert_equal(data.shape, (100, 5))

    def test_simulation_random_state(self):
        rs = np.random.RandomState()
        state = rs.get_state()
        n = Normal(5)
        sim = n.simulate([], random_state=rs)
        data = sim(100)
        assert_equal(data.shape, (100, 5))
        rs.set_state(state)
        rvs = rs.standard_normal(5 * 100)
        assert_equal(np.reshape(rvs, (100, 5)), data)

    def test_name(self):
        n = Normal(11)
        assert_equal(n.name[:2], '11')
        assert_equal(n.__str__(), '11-dimensional Multivariate Normal distribution')

    def test_names(self):
        n = Normal(6)
        assert_equal(n.parameter_names(), [])

    def test_starting_values(self):
        n = Normal(7)
        assert_equal(n.starting_values(np.random.standard_normal((1000, 7))),
                     np.empty(0))

    def test_loglikelihood(self):
        np.random.seed(12345)
        t, k = 1000, 7
        data = np.random.standard_normal((t, k))
        n = Normal(k)
        sigma2 = np.tile(np.eye(k), (t, 1, 1))
        llf = n.loglikelihoood([], data, sigma2)
        llfs = n.loglikelihoood([], data, sigma2, True)
        assert_almost_equal(llf, llfs.sum(), 4)

        uni_normal = uni_distribution.Normal()
        uni_llf = 0.0
        uni_llfs = np.zeros(t)
        for i in range(7):
            uni_llf += uni_normal.loglikelihoood([], data[:, i], np.ones(t))
            uni_llfs += uni_normal.loglikelihoood([], data[:, i], np.ones(t), True)
        assert_almost_equal(uni_llf, llf)
        assert_almost_equal(uni_llfs, llfs)

    def test_constraints_and_bounds(self):
        n = Normal(13)
        r = np.random.standard_normal((100, 13))
        assert_equal(n.constraints(), (np.empty(0), np.empty(0)))
        assert_equal(n.bounds(r), tuple([]))
