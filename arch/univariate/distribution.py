# -*- coding: utf-8 -*-
"""
Distributions to use in ARCH models.  All distributions must inherit from
:class:`Distribution` and provide the same methods with the same inputs.
"""
from __future__ import division, absolute_import
from ..compat.python import add_metaclass

from numpy.random import standard_normal, standard_t
from numpy import (empty, array, sqrt, log, exp, sign, pi, sum, asarray,
                   ones_like)
from scipy.special import gammaln
import scipy.stats as stats

from ..utility.array import DocStringInheritor


__all__ = ['Distribution', 'Normal', 'StudentsT', 'SkewStudent']


@add_metaclass(DocStringInheritor)
class Distribution(object):
    """
    Template for subclassing only
    """

    def __init__(self, name):
        self.name = name
        self.num_params = 0
        self._parameters = None

    def _simulator(self, size):
        """
        Simulate i.i.d. draws from the distribution

        Parameters
        ----------
        size : int or tuple
            Shape of the draws from the distribution

        Returns
        -------
        rvs : array
            Simulated pseudo random variables

        Notes
        -----
        Must call `simulate` before using `_simulator`
        """
        raise NotImplementedError(
            'Subclasses must implement')  # pragma: no cover

    def simulate(self, parameters):
        """
        Simulates i.i.d. draws from the distribution

        Parameters
        ----------
        parameters : array
            Distribution parameters

        Returns
        -------
        simulator : callable
            Callable that take a single output size argument and returns i.i.d.
            draws from the distribution
        """
        raise NotImplementedError(
            'Subclasses must implement')  # pragma: no cover

    def constraints(self):
        """
        Returns
        -------
        A : array
            Constraint loadings
        b : array
            Constraint values

        Notes
        -----
        Parameters satisfy the constraints A.dot(parameters)-b >= 0
        """
        raise NotImplementedError(
            'Subclasses must implement')  # pragma: no cover

    def bounds(self, resids):
        """
        Parameters
        ----------
        resids : array
             Residuals to use when computing the bounds

        Returns
        -------
        bounds : list
            List containing a single tuple with (lower, upper) bounds
        """
        raise NotImplementedError(
            'Subclasses must implement')  # pragma: no cover

    def loglikelihoood(self, parameters, resids, sigma2, individual=False):
        """
        Parameters
        ----------
        parameters : array
            Distribution shape parameters
        resids : array
            nobs array of model residuals
        sigma2 : array
            nobs array of conditional variances
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Notes
        -----
        Returns the loglikelihood where resids are the "data",
        and parameters and sigma2 are inputs.
        """
        raise NotImplementedError(
            'Subclasses must implement')  # pragma: no cover

    def starting_values(self, std_resid):
        """
        Parameters
        ----------
        std_resid : array
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : array
            The estimated shape parameters for the distribution

        Notes
        -----
        Size of sv depends on the distribution
        """
        raise NotImplementedError(
            'Subclasses must implement')  # pragma: no cover

    def parameter_names(self):
        """
        Names of distribution shepe parameters

        Returns
        -------
        names : list (str)
            Parameter names
        """
        raise NotImplementedError(
            'Subclasses must implement')  # pragma: no cover

    def __str__(self):
        return self._description()

    def __repr__(self):
        return self.__str__() + ', id: ' + hex(id(self)) + ''

    def _description(self):
        return self.name + ' distribution'


class Normal(Distribution):
    """
    Standard normal distribution for use with ARCH models
    """

    def __init__(self):
        super(Normal, self).__init__('Normal')
        self.name = 'Normal'

    def constraints(self):
        return empty(0), empty(0)

    def bounds(self, resids):
        return tuple([])

    def loglikelihoood(self, parameters, resids, sigma2, individual=False):
        r"""Computes the log-likelihood of assuming residuals are normally
        distributed, conditional on the variance

        Parameters
        ----------
        parameters : empty array
            The normal likelihood has no shape parameters
        resids  : array
            The residuals to use in the log-likelihood calculation
        sigma2 : array
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln f\left(x\right)=-\frac{1}{2}\left(\ln2\pi+\ln\sigma^{2}
            +\frac{x^{2}}{\sigma^{2}}\right)

        """
        lls = -0.5 * (log(2 * pi) + log(sigma2) + resids ** 2.0 / sigma2)
        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid):
        return empty(0)

    def _simulator(self, size):
        return standard_normal(size)

    def simulate(self, parameters):
        return self._simulator

    def parameter_names(self):
        return []


class StudentsT(Distribution):
    """
    Standardized Student's nobs distribution for use with ARCH models
    """

    def __init__(self):
        super(StudentsT, self).__init__('Standardized Student\'s t')
        self.num_params = 1

    def constraints(self):
        return array([[1], [-1]]), array([2.05, -500.0])

    def bounds(self, resids):
        return [(2.05, 500.0)]

    def loglikelihoood(self, parameters, resids, sigma2, individual=False):
        r"""Computes the log-likelihood of assuming residuals are have a
        standardized (to have unit variance) Student's t distribution,
        conditional on the variance.

        Parameters
        ----------
        parameters : array
            Shape parameter of the t distribution
        resids  : array
            The residuals to use in the log-likelihood calculation
        sigma2 : array
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln\Gamma\left(\frac{\nu+1}{2}\right)
            -\ln\Gamma\left(\frac{\nu}{2}\right)
            -\frac{1}{2}\ln(\pi\left(\nu-2\right)\sigma^{2})
            -\frac{\nu+1}{2}\ln(1+x^{2}/(\sigma^{2}(\nu-2)))

        where :math:`\Gamma` is the gamma function.
        """
        nu = parameters[0]
        lls = gammaln((nu + 1) / 2) - gammaln(nu / 2) - log(pi * (nu - 2)) / 2
        lls -= 0.5 * (log(sigma2))
        lls -= ((nu + 1) / 2) * \
               (log(1 + (resids ** 2.0) / (sigma2 * (nu - 2))))

        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid):
        """
        Parameters
        ----------
        std_resid : array
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : array
            Array containing starting valuer for shape parameter

        Notes
        -----
        Uses relationship between kurtosis and degree of freedom parameter to
        produce a moment-based estimator for the starting values.
        """
        k = stats.kurtosis(std_resid, fisher=False)
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)
        return array([sv])

    def _simulator(self, size):
        parameters = self._parameters
        std_dev = sqrt(parameters[0] / (parameters[0] - 2))
        return standard_t(self._parameters[0], size=size) / std_dev

    def simulate(self, parameters):
        parameters = asarray(parameters)[None]
        if parameters[0] <= 2.0:
            raise ValueError('The shape parameter must be larger than 2')
        self._parameters = parameters
        return self._simulator

    def parameter_names(self):
        return ['nu']


class SkewStudent(Distribution):
    r"""
    Standardized Skewed Student's [1]_ distribution for use with ARCH models

    Notes
    -----
    The Standardized Skewed Student's distribution takes two parameters,
    :math:`\eta` and :math:`\lambda`. :math:`\eta` controls the tail shape
    and is similar to the shape parameter in a Standardized Student's t.
    :math:`\lambda` controls the skewness. When :math:`\lambda=0` the
    distribution is identical to a standardized Student's t.

    References
    ----------

    .. [1] Hansen, B. E. (1994). Autoregressive conditional density estimation.
        *International Economic Review*, 35(3), 705–730.
        <http://www.ssc.wisc.edu/~bhansen/papers/ier_94.pdf>

    """

    def __init__(self):
        super(SkewStudent, self).__init__('Standardized Skew Student\'s t')
        self.num_params = 2

    def constraints(self):
        return array([[1, 0], [-1, 0], [0, 1], [0, -1]]), \
               array([2.05, -300.0, -1, -1])

    def bounds(self, resids):
        return [(2.05, 300.0), (-1, 1)]

    def loglikelihoood(self, parameters, resids, sigma2, individual=False):
        r"""Computes the log-likelihood of assuming residuals are have a
        standardized (to have unit variance) Skew Student's t distribution,
        conditional on the variance.

        Parameters
        ----------
        parameters : array
            Shape parameter of the skew-t distribution
        resids  : array
            The residuals to use in the log-likelihood calculation
        sigma2 : array
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln\left[\frac{bc}{\sigma}\left(1+\frac{1}{\eta-2}
                \left(\frac{a+bx/\sigma}
                {1+sgn(x/\sigma+a/b)\lambda}\right)^{2}\right)
                ^{-\left(\eta+1\right)/2}\right],

        where :math:`2<\eta<\infty`, and :math:`-1<\lambda<1`.
        The constants :math:`a`, :math:`b`, and :math:`c` are given by

        .. math::

            a=4\lambda c\frac{\eta-2}{\eta-1},
                \quad b^{2}=1+3\lambda^{2}-a^{2},
                \quad c=\frac{\Gamma\left(\frac{\eta+1}{2}\right)}
                {\sqrt{\pi\left(\eta-2\right)}
                \Gamma\left(\frac{\eta}{2}\right)},

        and :math:`\Gamma` is the gamma function.
        """
        eta, lam = parameters

        const_c = self.__const_c(parameters)
        const_a = self.__const_a(parameters)
        const_b = self.__const_b(parameters)

        resids = resids / sigma2 ** .5
        lls = log(const_b) + const_c - log(sigma2) / 2
        llf_resid = ((const_b * resids + const_a) /
                     (1 + sign(resids + const_a / const_b) * lam)) ** 2
        lls -= (eta + 1) / 2 * log(1 + llf_resid / (eta - 2))

        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid):
        """
        Parameters
        ----------
        std_resid : array
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : array
            Array containing starting valuer for shape parameter

        Notes
        -----
        Uses relationship between kurtosis and degree of freedom parameter to
        produce a moment-based estimator for the starting values.
        """
        k = stats.kurtosis(std_resid, fisher=False)
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)
        return array([sv, 0.])

    def _simulator(self, size):
        # No need to normalize since it is already done in parameterization
        return self.ppf(stats.uniform.rvs(size=size), self._parameters[0])

    def simulate(self, parameters):
        parameters = asarray(parameters)[None]
        if parameters[0, 0] <= 2.0:
            raise ValueError('The shape parameter must be larger than 2')
        if abs(parameters[0, 1]) > 1.0:
            raise ValueError('The skew parameter must be ' +
                             'smaller than 1 in absolute value')
        self._parameters = parameters
        return self._simulator

    def parameter_names(self):
        return ['nu', 'lambda']

    def __const_a(self, parameters):
        """Compute a constant.

        Parameters
        ----------
        parameters : array
            Shape parameters of the skew-t distribution

        Returns
        -------
        a: float
            Constant used in the distribution

        """
        eta, lam = parameters
        c = self.__const_c(parameters)
        return 4 * lam * exp(c) * (eta-2) / (eta - 1)

    def __const_b(self, parameters):
        """Compute b constant.

        Parameters
        ----------
        parameters : array
            Shape parameters of the skew-t distribution

        Returns
        -------
        b: float
            Constant used in the distribution
        """
        eta, lam = parameters
        a = self.__const_a(parameters)
        return (1 + 3*lam**2 - a**2)**.5

    def __const_c(self, parameters):
        """Compute c constant.

        Parameters
        ----------
        parameters : array
            Shape parameters of the skew-t distribution

        Returns
        -------
        c : float
            Log of the constant used in loglikelihood
        """
        eta, lam = parameters
#        return gamma((eta+1)/2) / ((pi*(eta-2))**.5 * gamma(eta/2))
        return gammaln((eta+1)/2) - gammaln(eta/2) - log(pi*(eta-2))/2

    def ppf(self, arg, parameters):
        """Inverse cumulative density function (ICDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at. Must belong to (0, 1)
        parameters : array
            Shape parameters of the skew-t distribution

        Returns
        -------
        array
            ICDF values. Same shape as the input.

        """
        eta, lam = parameters

        a = self.__const_a(parameters)
        b = self.__const_b(parameters)

        cond = arg < (1 - lam) / 2

        icdf1 = stats.t.ppf(arg[cond] / (1 - lam), eta)
        icdf2 = stats.t.ppf(.5 + (arg[~cond] - (1 - lam) / 2) / (1 + lam), eta)
        icdf = -999.99 * ones_like(arg)
        icdf[cond] = icdf1
        icdf[~cond] = icdf2
        icdf = (icdf *
                (1 + sign(arg - (1 - lam) / 2) * lam) * (1 - 2 / eta) ** .5 -
                a)
        icdf = icdf / b

        return icdf
