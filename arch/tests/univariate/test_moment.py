from numpy import exp, inf, log, nan, ones_like, pi
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.integrate import quad
from scipy.special import gammaln

from arch.univariate.distribution import (
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
)

DISTRIBUTIONS = [
    (SkewStudent(), [6, -0.1]),
    (SkewStudent(), [6, -0.5]),
    (SkewStudent(), [6, 0.1]),
    (SkewStudent(), [6, 0.5]),
    (GeneralizedError(), [1.5]),
    (GeneralizedError(), [2.1]),
    (StudentsT(), [6]),
    (StudentsT(), [7]),
    (Normal(), None),
]


@pytest.mark.parametrize("dist, params", DISTRIBUTIONS)
def test_moment(dist, params):
    """
    Ensures that Distribtion.moment and .partial_moment agree
    with numeric integrals for order n=0,...,5 and z=+/-1,...,+/-5

    Parameters
    ----------
    dist : distribution.Distribution
        The distribution whose moments are being checked
    params : List
        List of parameters
    """

    assert_equal(dist.moment(-1, params), nan)
    assert_equal(dist.partial_moment(-1, 0.0, params), nan)

    # verify moments that exist
    def f(x, n):
        sigma2 = ones_like(x)
        return (x**n) * exp(dist.loglikelihood(params, x, sigma2, True))

    for n in range(6):  # moments 0-5
        # complete moments
        m_quad = quad(f, -inf, inf, args=(n,))[0]
        m_method = dist.moment(n, params)
        assert_almost_equal(m_quad, m_method)

        # partial moments at z=+/-1,...,+/-5
        # SkewT integral is broken up for numerical stability
        for z in range(-5, 5):  # partial moments at +-1,...,+-5
            if isinstance(dist, SkewStudent):
                eta, lam = params
                c = gammaln((eta + 1) / 2) - gammaln(eta / 2) - log(pi * (eta - 2)) / 2
                a = 4 * lam * exp(c) * (eta - 2) / (eta - 1)
                b = (1 + 3 * lam**2 - a**2) ** 0.5
                loc = -a / b
                if z < loc:
                    m_quad = quad(f, -inf, z, args=(n,))[0]
                else:
                    m_quad = (
                        quad(f, -inf, loc - 1e-9, args=(n,))[0]
                        + quad(f, loc + 1e-9, z, args=(n,))[0]
                    )

            else:
                m_quad = quad(f, -inf, z, args=(n,))[0]

            m_method = dist.partial_moment(n, z, params)
            assert_almost_equal(m_quad, m_method)
