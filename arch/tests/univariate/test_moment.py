from arch.univariate.distribution import SkewStudent, Normal, StudentsT, \
    GeneralizedError
from numpy import power, inf
from scipy.integrate import quad

DISTRIBUTIONS = [
    (SkewStudent(), [6, -0.1]),
    (SkewStudent(), [5, -0.5]),
    (SkewStudent(), [6, 0.1]),
    (SkewStudent(), [5, 0.5]),
    (GeneralizedError(), [1.5]),
    (GeneralizedError(), [2.1]),
    (StudentsT(), [5]),
    (StudentsT(), [6]),
    (Normal(), None)]


def test_moments():
    ''' Ensure that moment formula agrees with:
        integral [0, inf]  h * x^(h-1) * (sf(x) + (-1^h)*cdf(x)) dx
        to around 3 decimal points
    '''
    for distribution in DISTRIBUTIONS:
        s, params = distribution

        max_h = 4
        epsabs = 1e-4
        epsrel = 1e-4
        verbose = True

        def f(x, h):
            # factor of 1e-10 is hackey way to get stable quaderature for higher moments
            return h * (1e-10)*power(x, h-1) * (1 - s.cdf(x, params) + power(-1,h)*s.cdf(-x, params))

        for h in range(1, max_h+1):
            integral = quad(f, 0, inf, limit=10000, epsabs=1e-6, epsrel=1e-6, args=h)

            m_hat, error = integral
            m_hat *= 1e10
            m_act = s.moment(h, params)

            abs_err = abs(m_hat-m_act)
            rel_err = abs(m_hat-m_act)/abs(m_act+1e-15)
            all_good = True
            if abs_err<epsabs and (rel_err < epsrel or m_act < 1e-15) and verbose:
                print('{:} Moment {:d} PASSED, m_act={:.4f}, m_hat={:.4f}'.format(s.name, h, m_act, m_hat))
            else:
                if verbose:
                    print('{:} Moment {:d} FAILED, m_act={:.4f}, m_hat={:.4f}'.format(s.name, h, m_act, m_hat))
                all_good = False

        assert all_good == True, '{:}, {:}, {:}'.format(s, params, all_good)


if __name__ == '__main__':
    test_moments()
