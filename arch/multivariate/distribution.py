# -*- coding: utf-8 -*-
"""
Distributions to use in ARCH models.  All distributions must inherit from
:class:`Distribution` and provide the same methods with the same inputs.
"""
from __future__ import absolute_import, division
from abc import abstractmethod

from numpy import empty, log, pi, r_
from numpy.linalg import slogdet, inv, multi_dot
from numpy.random import RandomState

from arch.compat.python import add_metaclass
from arch.utility.array import AbstractDocStringInheritor

__all__ = ['MultivariateDistribution', 'MultivariateNormal']


@add_metaclass(AbstractDocStringInheritor)
class MultivariateDistribution(object):
    """
    Template for subclassing only
    """

    def __init__(self, name, random_state=None):
        self.name = name
        self.num_params = 0
        self._parameters = None
        self._random_state = random_state
        self._nvar = None
        if random_state is None:
            self._random_state = RandomState()
        if not isinstance(self._random_state, RandomState):
            raise TypeError('random_state must by a NumPy RandomState instance')

    @property
    def nvar(self):
        """Set of get the number of variables in the distribution"""
        return self._nvar

    @nvar.setter
    def nvar(self, value):
        self._nvar = int(value)

    @property
    def random_state(self):
        """The NumPy RandomState attached to the distribution"""
        return self._random_state

    @abstractmethod
    def _simulator(self, size):
        """
        Simulate i.i.d. draws from the distribution

        Parameters
        ----------
        size : int or tuple
            Shape of the draws from the distribution

        Returns
        -------
        rvs : ndarray
            Simulated pseudo random variables

        Notes
        -----
        Must call `simulate` before using `_simulator`
        """
        pass

    @abstractmethod
    def simulate(self, parameters):
        """
        Simulates i.i.d. draws from the distribution

        Parameters
        ----------
        parameters : ndarray
            Distribution parameters

        Returns
        -------
        simulator : callable
            Callable that take a single output size argument and returns i.i.d.
            draws from the distribution. The size of the simulated values is
            (size, nvar).
        """
        pass

    def constraints(self):
        """
        Returns
        -------
        A : ndarray
            Constraint loadings
        b : ndarray
            Constraint values

        Notes
        -----
        Parameters satisfy the constraints A.dot(parameters)-b >= 0
        """
        pass

    def bounds(self, resids):
        """
        Parameters
        ----------
        resids : ndarray
             Residuals to use when computing the bounds

        Returns
        -------
        bounds : list
            List containing a single tuple with (lower, upper) bounds
        """
        pass

    @abstractmethod
    def loglikelihood(self, parameters, resids, sigma, individual=False):
        """
        Parameters
        ----------
        parameters : ndarray
            Distribution shape parameters
        resids : ndarray
            nobs array of model residuals
        sigma : ndarray
            nobs array of conditional covariances
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Notes
        -----
        Returns the loglikelihood where resids are the "data",
        and parameters and sigma are inputs.
        """
        pass

    @abstractmethod
    def starting_values(self, std_resid):
        """
        Parameters
        ----------
        std_resid : ndarray
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : ndarray
            The estimated shape parameters for the distribution

        Notes
        -----
        Size of sv depends on the distribution
        """
        pass

    @abstractmethod
    def parameter_names(self):
        """
        Names of distribution shape parameters

        Returns
        -------
        names : list (str)
            Parameter names
        """
        pass

    def __str__(self):
        return self._description()

    def __repr__(self):
        return self.__str__() + ', id: ' + hex(id(self)) + ''

    def _description(self):
        return self.name + ' distribution'


class MultivariateNormal(MultivariateDistribution):
    """
    Standard normal distribution for use with ARCH models
    """
    def __init__(self, random_state=None):
        super(MultivariateNormal, self).__init__('Multivariate Normal', random_state=random_state)

    def constraints(self):
        return empty(0), empty(0)

    def bounds(self, resids):
        return tuple([])

    def loglikelihood(self, parameters, resids, sigma, individual=False):
        r"""Computes the log-likelihood of assuming residuals are normally
        distributed, conditional on the covariance

        Parameters
        ----------
        parameters : ndarray
            The normal likelihood has no shape parameters. Empty since the
            standard normal has no shape parameters.
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma : ndarray
            Conditional covariances of resids
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

            \ln f_{X}\left(x;\Sigma\right) =
            -\frac{k}{2}\left(\ln2\pi+\ln|\Sigma|+x^{\prime}\Sigma^{-1}x\right)

        where k is the dimension of x.

        """
        _, logdet = slogdet(sigma)
        sigma_inv = inv(sigma)
        nobs, nvar = resids.shape[:2]
        lls = empty(nobs)
        for i in range(len(sigma)):
            r = resids[i:(i+1), :]
            lls[i] = -0.5 * multi_dot((r, sigma_inv[i], r.T))
        lls += -0.5 * nvar * log(2 * pi) - 0.5 * logdet
        if individual:
            return lls
        else:
            return lls.sum()

    def starting_values(self, std_resid):
        return empty(0)

    def _simulator(self, size):
        return self._random_state.standard_normal(r_[size, self._nvar])

    def simulate(self, parameters):
        if self._nvar is None:
            raise ValueError('nvar is currently None.  This value must be set before calling '
                             'simulate.')
        return self._simulator

    def parameter_names(self):
        return []
