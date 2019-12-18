import scipy.stats as stats
from arch.univariate.distribution import (SkewStudent, Normal, StudentsT,
                                          GeneralizedError)
from itertools import product
from numpy import arange, exp, inf, log, pi, sqrt
from scipy.integrate import quad
from scipy.special import gammaln

DISTRIBUTIONS = [
    (SkewStudent(), [9, -0.1]),
    (SkewStudent(), [9, -0.5]),
    (SkewStudent(), [9, 0.1]),
    (SkewStudent(), [9, 0.5]),
    (GeneralizedError(), [1.5]),
    (GeneralizedError(), [2.1]),
    (StudentsT(), [9]),
    (StudentsT(), [8]),
    (Normal(), None)]


class MomentTests(object):
    """
    Wrapper around moment and partial moment tests. All it
    does is ensure the results agree with numeric integrals.
    For moments 1-5 and partial moments 1-5 evaluated at
    z= +-1,...,+-5
    """

    @staticmethod
    def test_moments(s, params):
        eps_abs = 1e-4
        eps_rel = 1e-4

        def f(x, h):
            return (x**h) * exp(s.loglikelihood(params, x, 1, True))

        all_passed = True
        msg = []
        msg_template = '{:},{:} Moment h={:d} {:}, m_formula={:.4f},' + \
                       ' m_integral={:.4f}'
        for h in range(1, 6):
            m_integral = quad(f, -inf, inf, args=h)[0]
            m_formula = s.moment(h, params)

            abs_err = abs(m_integral - m_formula)
            rel_err = abs_err / abs(m_formula + 1e-15)

            if h == 1:
                abs_err = abs(m_formula)
            elif h == 2:
                abs_err = abs(1. - m_formula)
                rel_err = abs_err

            if abs_err < eps_abs and (rel_err < eps_rel or
                                      abs(m_formula) < 1e-8):
                cycle_status = 'PASSED'
            else:
                cycle_status = 'FAILED'
                all_passed = False

            msg.append(msg_template.format(s.name, params, h, cycle_status,
                                           round(m_formula, 4),
                                           round(m_integral, 4)))

        return '\n'.join(msg), all_passed

    @staticmethod
    def test_partial_moments(s, params):
        """
        Ensure that partial moment formula agrees with numeric integral
        for h=1,2,3,4,5,  and z=+-1,+-2,...,+-10
        """
        eps_abs = 1e-4
        eps_rel = 1e-4

        def f(x, h):
            return (x**h) * exp(s.loglikelihood(params, x, 1, True))

        all_passed = True
        msg = []
        msg_template = '{:},{:} Partial Moment {:d} (-inf,{:}) {:},' + \
                       ' m_formula={:.4f}, m_integral={:.4f}'
        for h, z in product(range(1, 6), arange(-5, 6)):

            if isinstance(s, SkewStudent):
                eta, lam = params
                c = (gammaln((eta + 1) / 2) - gammaln(eta / 2) -
                     log(pi * (eta - 2)) / 2)
                a = 4 * lam * exp(c) * (eta - 2) / (eta - 1)
                b = (1 + 3 * lam ** 2 - a ** 2) ** .5
                loc = -a/b
                if z < loc:
                    m_integral = quad(f, -inf, z, args=h)[0]
                else:
                    m_integral = quad(f, -inf, loc-1e-6, args=h)[0] + \
                        quad(f, loc+1e-6, z, args=h)[0]

            elif isinstance(s, Normal):
                m_integral = stats.norm.expect(lambda x: x**h, lb=-inf, ub=z)

            elif isinstance(s, StudentsT):
                scale = sqrt((params[0]-2)/params[0])
                m_integral = stats.t.expect(lambda x: x**h, args=(params[0],),
                                            lb=-inf, ub=z, scale=scale)

            else:  # Generalized Normal
                if z <= 0:
                    m_integral = quad(f, -inf, z, args=h)[0]

                else:
                    m_integral = quad(f, -inf, 0, args=h)[0] + \
                        quad(f, 0, z, args=h)[0]

            m_formula = s.partial_moment(h, z, params)

            abs_err = abs(m_formula - m_integral)
            rel_err = abs_err / (m_formula + 1e-15)

            if abs_err < eps_abs and (rel_err < eps_rel or
                                      abs(m_formula) < 1e-8):
                cycle_status = 'PASSED'
            else:
                cycle_status = 'FAILED'
                all_passed = False

            msg.append(msg_template.format(s.name, params, h, z, cycle_status,
                                           round(m_formula, 4),
                                           round(m_integral, 4)))

        return '\n'.join(msg), all_passed


def test_moments(verbose=True):
    moment_failures = []
    ptl_moment_failures = []
    all_good = True
    for d in DISTRIBUTIONS:
        mom_msg, mom_good = MomentTests.test_moments(*d)
        pmom_msg, pmom_good = MomentTests.test_partial_moments(*d)

        if not mom_good:
            moment_failures.append('{:},{:} Moment FAILED'.format(
                d[0].name, d[1]))
            all_good = False

        if not pmom_good:
            ptl_moment_failures.append('{:},{:} Partial Moment FAILED'.format(
                d[0].name, d[1]))
            all_good = False

        if verbose:
            print(mom_msg)
            print(pmom_msg)

    if all_good:
        print(60*'-')
        print('All Moment Checks PASSED')
        print('All Partial Moment Checks PASSED')

    else:
        print(60*'')
        print('\n'.join(moment_failures))
        print('\n'.join(ptl_moment_failures))

    assert all_good


if __name__ == '__main__':

    test_moments()
