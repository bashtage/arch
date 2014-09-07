"""
Distributions to use in ARCH models.  All distributions must inherit from
:class:`Distribution` and provide the same methods with the same inputs.
"""
from __future__ import division, absolute_import

from numpy.random import standard_normal, standard_t
from numpy import empty, array, sqrt, log, pi, sum, asarray
from scipy.special import gammaln
import scipy.stats as stats

from ..compat.python import add_metaclass
from ..utils import DocStringInheritor


__all__ = ['Distribution', 'Normal', 'StudentsT']


@add_metaclass(DocStringInheritor)
class Distribution(object):
    """
    Template for subclassing only
    """

    def __init__(self, name):
        self.name = name
        self.num_params = 0
        self._parameters = None

    def _simulator(self, nobs):
        """
        Simulate i.i.d. draws from the distribution

        Parameters
        ----------
        nobs: int
            Number of draws from the distribution

        Returns
        -------
        rvs : 1-d array
            Simulated pseudo random normal variables

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
        parameters : 1-d array
            Distribution parameters

        Returns
        -------
        simulator : callable
            Callable that take a single, integer argument and returns i.i.d.
            draws from the distribution
        """
        raise NotImplementedError(
            'Subclasses must implement')  # pragma: no cover

    def constraints(self):
        """
        Returns
        -------
        A : 2-d array
            Constraint loadings
        b : 1-d array
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
        resids : 1-d array
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
        parameters : 1-d array
            Distribution parameters
        resids : 1-d array
            nobs array of model residuals
        sigma2 : 1-d array
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
        Compute starting values from standardized residuals
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
        """Computes the log-likelihood of assuming residuals are normally
        distribution.rst, conditional on the variance

        Parameters
        ----------
        parameters : empty array
            The normal likelihood has no shape parameters
        resids  : 1-d array, float
            The residuals to use in the log-likelihood calculation
        sigma2 : 1-d array, float
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float64
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \\ln f\\left(x\\right)=-\\frac{1}{2}\\ln2\\pi\\sigma^{2}
            -\\frac{x^{2}}{2\\sigma^{2}}

        """
        lls = -0.5 * (log(2 * pi) + log(sigma2) + resids ** 2.0 / sigma2)
        if individual:
            return lls
        else:
            return sum(lls)


    def starting_values(self, std_resid):
        """
        Parameters
        ----------
        std_resid : 1-d array
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : empty array
            THe normal distribution has no shape parameters

        """
        return empty(0)

    def _simulator(self, nobs):
        return standard_normal(nobs)

    def simulate(self, parameters):
        return self._simulator

    def parameter_names(self):
        return []


class StudentsT(Distribution):
    """
    Standardized Student's nobs distribution for use with ARCH models
    """

    def __init__(self):
        super(StudentsT, self).__init__('Normal')
        self.num_params = 1
        self.name = 'Standardized Student\'s t'

    def constraints(self):
        return array([[1], [-1]]), array([2.05, -500.0])

    def bounds(self, resids):
        return [(2.05, 500.0)]

    def loglikelihoood(self, parameters, resids, sigma2, individual=False):
        """Computes the log-likelihood of assuming residuals are have a
        standardized (to have unit variance) Student's t distribution,
        conditional on the variance.

        Parameters
        ----------
        parameters : 1-d array
            Shape parameter of the t distribution
        resids  : 1-d array, float
            The residuals to use in the log-likelihood calculation
        sigma2 : 1-d array, float
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float64
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \\ln\\Gamma\\left(\\frac{\\nu+1}{2}\\right)
            -\\ln\\Gamma\\left(\\frac{\\nu}{2}\\right)
            -\\frac{1}{2}\\ln(\\pi\\left(\\nu-2\\right)\\sigma^{2})
            -\\frac{\\nu+1}{2}\\ln(1+x^{2}/(\\sigma^{2}(\\nu-2)))

        where :math:`\Gamma` is the gamma function.
        """
        nu = parameters[0]
        lls = gammaln(0.5 * (nu + 1)) - gammaln(nu / 2) - 1 / 2 * log(
            pi * (nu - 2))
        lls -= 0.5 * (log(sigma2))
        lls -= ((nu + 1) / 2) * (log(1 + (resids ** 2.0) / (sigma2 * (nu - 2))))

        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid):
        """
        Parameters
        ----------
        std_resid : 1-d array
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : 1-d array
            Array containing starting valuer for shape parameter

        Notes
        -----
        Uses relationship between kurtosis and degree of freedom parameter to
        produce a moment-based estimator for the starting values.
        """
        k = stats.kurtosis(std_resid, fisher=False)
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)
        return array([sv])

    def _simulator(self, nobs):
        parameters = self._parameters
        std_dev = sqrt(parameters[0] / (parameters[0] - 2))
        return standard_t(self._parameters[0], nobs) / std_dev

    def simulate(self, parameters):
        parameters = asarray(parameters)[None]
        if parameters[0] <= 2.0:
            raise ValueError('The shape parameter must be larger than 2')
        self._parameters = parameters
        return self._simulator

    def parameter_names(self):
        return ['nu']


