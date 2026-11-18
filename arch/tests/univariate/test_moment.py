from numpy import errstate, exp, inf, log, nan, ones_like, pi
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
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


@pytest.mark.parametrize(("dist", "params"), DISTRIBUTIONS)
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


@pytest.mark.parametrize("nu", [342.0, 400.0, 500.0])
def test_students_t_odd_partial_moment_large_nu(nu):
    dist = StudentsT()

    def f(x, n):
        return (x**n) * exp(dist.loglikelihood([nu], x, ones_like(x), True))

    for n in (1, 3):
        for z in (-2.0, 0.0, 1.5):
            m_quad = quad(f, -inf, z, args=(n,))[0]
            assert_almost_equal(dist.partial_moment(n, z, [nu]), m_quad)


@pytest.mark.parametrize("eta", [5.0, 8.0])
def test_skew_student_partial_moment_lambda_one(eta):
    # GH 1024: lambda = 1 leaves the left half with no mass and a zero lscale, so
    # partial_moment returned nan for every z at or above the support endpoint.
    dist = SkewStudent()
    lam = 1.0
    loc = dist.ppf(0.0, [eta, lam])
    assert_almost_equal(dist.partial_moment(0, loc, [eta, lam]), 0.0)

    # The partial moments join continuously to the values just below the
    # degenerate skew, and the upper limit is the complete moment.
    for n in range(3):
        for z in (0.0, 2.0, 5.0):
            m_method = dist.partial_moment(n, z, [eta, lam])
            m_near = dist.partial_moment(n, z, [eta, 0.999999])
            assert_almost_equal(m_method, m_near, decimal=6)
        assert_almost_equal(
            dist.partial_moment(n, inf, [eta, lam]), dist.moment(n, [eta, lam])
        )


@pytest.mark.parametrize(("dist", "params"), DISTRIBUTIONS)
def test_partial_moment_infinite_bound(dist, params):
    # The lower partial moment to -inf is 0 and to inf is the moment. Normal and
    # StudentsT gave nan, and StudentsT also gave nan for n=1 with large nu
    for n in range(5):
        assert_almost_equal(dist.partial_moment(n, -inf, params), 0.0)
        assert_almost_equal(dist.partial_moment(n, inf, params), dist.moment(n, params))


@pytest.mark.parametrize("nu", [5.0, 400.0])
def test_students_t_partial_moment_infinite_bound(nu):
    dist = StudentsT()
    for n in range(5):
        assert dist.partial_moment(n, -inf, [nu]) == 0.0
        assert_almost_equal(dist.partial_moment(n, inf, [nu]), dist.moment(n, [nu]))


@pytest.mark.parametrize("eta", [3.0, 8.0, 30.0])
@pytest.mark.parametrize(("order", "expected"), [(0, 1.0), (1, 0.0), (2, 1.0)])
@pytest.mark.parametrize("bound", [2.0, 5.0, inf])
def test_skew_student_partial_moment_above_left_half_support(
    eta, order, expected, bound
):
    # At lambda=-1 the standardized left half-t has a finite upper endpoint.
    # Above it, the first three partial moments equal mass, mean and variance.
    with errstate(divide="raise", invalid="raise"):
        actual = SkewStudent().partial_moment(order, bound, [eta, -1.0])
    assert_allclose(actual, expected, atol=1e-12)
