import numpy as np
from numpy.random import RandomState, default_rng
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
import pytest
from scipy import stats
from scipy.special import gamma, gammaln

from arch.univariate.distribution import (
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
)


@pytest.fixture(params=[GeneralizedError, SkewStudent, StudentsT])
def distribution(request):
    return request.param


@pytest.fixture(params=[None, 12345, RandomState, default_rng])
def seed(request):
    if isinstance(request.param, int) or request.param is None:
        return request.param
    else:
        return request.param(12345)


class TestDistributions:
    @classmethod
    def setup_class(cls):
        cls.rng = RandomState(12345)
        cls.T = 1000
        cls.resids = cls.rng.standard_normal(cls.T)
        cls.sigma2 = 1 + cls.rng.random_sample(cls.resids.shape)

    def test_normal(self, seed):
        dist = Normal(seed=seed)
        ll1 = dist.loglikelihood([], self.resids, self.sigma2)
        scipy_dist = stats.norm
        ll2 = scipy_dist.logpdf(self.resids, scale=np.sqrt(self.sigma2)).sum()
        assert_almost_equal(ll1, ll2)

        assert_equal(dist.num_params, 0)

        bounds = dist.bounds(self.resids)
        assert_equal(len(bounds), 0)

        a, _ = dist.constraints()
        assert_equal(len(a), 0)

        assert_array_equal(dist.starting_values(self.resids), np.empty((0,)))

    def test_studentst(self, seed):
        dist = StudentsT(seed=seed)
        v = 4.0
        ll1 = dist.loglikelihood(np.array([v]), self.resids, self.sigma2)
        # Direct calculation of PDF, then log
        constant = np.exp(gammaln(0.5 * (v + 1)) - gammaln(0.5 * v))
        pdf = constant / np.sqrt(np.pi * (v - 2) * self.sigma2)
        pdf *= (1 + self.resids**2.0 / (self.sigma2 * (v - 2))) ** (-(v + 1) / 2)
        ll2 = np.log(pdf).sum()
        assert_almost_equal(ll1, ll2)

        assert_equal(dist.num_params, 1)

        bounds = dist.bounds(self.resids)
        assert_equal(len(bounds), 1)

        a, _ = dist.constraints()
        assert_equal(a.shape, (2, 1))

        k = stats.kurtosis(self.resids, fisher=False)
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)
        assert_array_equal(dist.starting_values(self.resids), np.array([sv]))

        with pytest.raises(ValueError, match=r"The shape parameter must be larger than 2"):
            dist.simulate(np.array([1.5]))
        sim = dist.simulate(8.0)
        assert isinstance(sim(100), np.ndarray)

    def test_skewstudent(self, seed):
        dist = SkewStudent(seed=seed)
        assert dist.parameter_names() == ["eta", "lambda"]
        eta, lam = 4.0, 0.5
        ll1 = dist.loglikelihood(np.array([eta, lam]), self.resids, self.sigma2)
        # Direct calculation of PDF, then log
        const_c = gamma((eta + 1) / 2) / ((np.pi * (eta - 2)) ** 0.5 * gamma(eta / 2))
        const_a = 4 * lam * const_c * (eta - 2) / (eta - 1)
        const_b = (1 + 3 * lam**2 - const_a**2) ** 0.5

        resids = self.resids / self.sigma2**0.5
        power = -(eta + 1) / 2
        pdf = (
            const_b
            * const_c
            / self.sigma2**0.5
            * (
                1
                + 1
                / (eta - 2)
                * (
                    (const_b * resids + const_a)
                    / (1 + np.sign(resids + const_a / const_b) * lam)
                )
                ** 2
            )
            ** power
        )

        ll2 = np.log(pdf).sum()
        assert_almost_equal(ll1, ll2)

        assert_equal(dist.num_params, 2)

        bounds = dist.bounds(self.resids)
        assert_equal(len(bounds), 2)

        a, _ = dist.constraints()
        assert_equal(a.shape, (4, 2))

        k = stats.kurtosis(self.resids, fisher=False)
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)
        assert_array_equal(dist.starting_values(self.resids), np.array([sv, 0.0]))

        with pytest.raises(ValueError, match=r"The shape parameter must be larger"):
            dist.simulate(np.array([1.5, 0.0]))
        with pytest.raises(ValueError, match=r"The skew parameter must be smaller than 1"):
            dist.simulate(np.array([4.0, 1.5]))
        with pytest.raises(ValueError, match=r"The skew parameter must be smaller than 1"):
            dist.simulate(np.array([4.0, -1.5]))
        with pytest.raises(ValueError, match=r"The shape parameter must be larger than 2"):
            dist.simulate(np.array([1.5, 1.5]))

        sim = dist.simulate([8.0, -0.2])
        assert isinstance(sim(100), np.ndarray)

    def test_ged(self, seed):
        dist = GeneralizedError(seed=seed)
        nu = 1.7
        ll1 = dist.loglikelihood(np.array([nu]), self.resids, self.sigma2)

        sigma = np.sqrt(self.sigma2)
        x = self.resids

        c = (2 ** (-2 / nu) * gamma(1 / nu) / gamma(3 / nu)) ** 0.5
        pdf = nu / (c * gamma(1 / nu) * 2 ** (1 + 1 / nu) * sigma)
        pdf *= np.exp(-(1 / 2) * np.abs(x / (c * sigma)) ** nu)
        ll2 = np.log(pdf).sum()
        assert_almost_equal(ll1, ll2)
        lls1 = dist.loglikelihood(
            np.array([nu]), self.resids, self.sigma2, individual=True
        )
        assert_almost_equal(lls1, np.log(pdf))

        assert_equal(dist.num_params, 1)

        bounds = dist.bounds(self.resids)
        assert_equal(len(bounds), 1)

        a, _ = dist.constraints()
        assert_equal(a.shape, (2, 1))

        assert_array_equal(dist.starting_values(self.resids), np.array([1.5]))

        with pytest.raises(ValueError, match=r"The shape parameter must be larger than 1"):
            dist.simulate(np.array([0.9]))
        simulator = dist.simulate(1.5)
        rvs = simulator(1000)
        assert rvs.shape[0] == 1000
        assert str(hex(id(dist))) in dist.__repr__()
        assert dist.parameter_names() == ["nu"]


def test_bad_input():
    with pytest.raises(TypeError, match=r"Normal.__init__\(\) got an unexpected"):
        Normal(random_state="random_state")


DISTRIBUTIONS = [
    (Normal, ()),
    (StudentsT, (8.0,)),
    (StudentsT, (3.0,)),
    (GeneralizedError, (1.5,)),
    (GeneralizedError, (2.1,)),
    (SkewStudent, (8.0, -0.5)),
]


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_roundtrip_cdf_ppf(distribution):
    pits = np.arange(1, 100.0) / 100.0
    dist, param = distribution
    dist = dist()
    x = dist.ppf(pits, param)
    p = dist.cdf(x, param)
    assert_almost_equal(pits, p)

    pits = 0.3
    x = dist.ppf(pits, param)
    p = dist.cdf(x, param)
    assert_almost_equal(pits, p)


def test_invalid_params():
    pits = np.arange(1, 100.0) / 100.0
    dist = Normal()
    with pytest.raises(ValueError, match=r"parameters must have 0 elements"):
        dist.ppf(pits, [1.0])
    dist = StudentsT()
    with pytest.raises(ValueError, match=r"S does not satisfy the bounds requirement"):
        dist.ppf(pits, [1.0])


def test_no_parameters_error(distribution):
    dist = distribution()
    pits = np.arange(1, 100.0) / 100.0
    with pytest.raises(ValueError, match=r"parameters must have"):
        dist.ppf(pits, None)
    with pytest.raises(ValueError, match=r"parameters must have"):
        dist.cdf(pits, None)


def test_random_state_seed_transition():
    with pytest.raises(TypeError, match=r"seed must by a NumPy Generator or RandomState"):
        Normal(seed="1234")
