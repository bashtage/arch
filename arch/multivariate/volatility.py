from __future__ import absolute_import, division

from abc import abstractmethod

import numpy as np
from arch.compat.python import add_metaclass
from arch.multivariate.distribution import MultivariateNormal
from arch.multivariate.utility import inv_vech, vech, symmetric_matrix_root
from arch.utility.array import AbstractDocStringInheritor
from numpy import (sum)


@add_metaclass(AbstractDocStringInheritor)
class MultivariateVolatilityProcess(object):
    """
    Abstract base class for ARCH models.  Allows the conditional mean model to be specified
    separately from the conditional variance, even though parameters are estimated jointly.
    """

    def __init__(self):
        self.name = ''
        self.closed_form = False
        self._mvnormal = MultivariateNormal()
        self._min_bootstrap_obs = 100
        self._start = 0
        self._stop = -1
        self._nvar = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__() + ', id: ' + hex(id(self))

    @property
    def nvar(self):
        """Set of get the number of variables in the distribution"""
        return self._nvar

    @nvar.setter
    def nvar(self, value):
        self._nvar = int(value)

    def _requires_nvar(self):
        if self.nvar is None:
            raise TypeError('nvar must be set before calling this function.')

    @property
    @abstractmethod
    def num_params(self):
        """Number of parameters in process"""
        pass

    @property
    def start(self):
        """Index to use to start variance subarray selection"""
        return self._start

    @start.setter
    def start(self, value):
        self._start = value

    @property
    def stop(self):
        """Index to use to stop variance subarray selection"""
        return self._stop

    @stop.setter
    def stop(self, value):
        self._stop = value

    @abstractmethod
    def transform_parameters(self, *args):
        """
        Transform parameters from natural representation to a vector suitable
        for estimation.
        """
        pass

    @abstractmethod
    def starting_values(self, resids):
        """
        Returns starting values for the ARCH model

        Parameters
        ----------
        resids : ndarray
            Array of (approximate) residuals to use when computing starting
            values

        Returns
        -------
        sv : ndarray
            Array of starting values
        """
        pass

    def backcast(self, resids):
        """
        Construct values for backcasting to start the recursion

        Parameters
        ----------
        resids : ndarray
            Vector of (approximate) residuals

        Returns
        -------
        backcast : ndarray
            Value to use in backcasting in the volatility recursion
        """
        nobs, nvar = resids.shape
        tau = min(75, nobs)
        w = (0.94 ** np.arange(tau))
        w = w / sum(w)
        bc = np.zeros((nvar, nvar))
        for i in range(tau):
            bc += w[i] * resids[i:i + 1].T.dot(resids[i:i + 1])
        return bc

    @abstractmethod
    def bounds(self, resids):
        """
        Returns bounds for parameters

        Parameters
        ----------
        resids : ndarray
            Vector of (approximate) residuals

        Returns
        -------
        bounds : list[tuple[float, float]]
            List of tuples containing bounds for each parameter of the
            form (low, up)
        """
        pass

    @abstractmethod
    def compute_covariance(self, parameters, resids, sigma, backcast):
        """
        Compute the covariance for the ARCH model

        Parameters
        ----------
        parameters : ndarray
            Model parameters
        resids : ndarray
            Vector of mean zero residuals
        sigma : ndarray
            Array with size (nobs, nvar, nvar) to store the conditional variance
        backcast : {float, ndarray}
            Value to use when initializing ARCH recursion. Can be an ndarray
            when the model contains multiple components.
        """
        pass

    @abstractmethod
    def constraints(self):
        """
        Construct parameter constraints arrays for parameter estimation

        Returns
        -------
        A : ndarray
            Parameters loadings in constraint. Shape is number of constraints
            by number of parameters
        b : ndarray
            Constraint values, one for each constraint

        Notes
        -----
        Values returned are used in constructing linear inequality
        constraints of the form A.dot(parameters) - b >= 0
        """
        pass

    @abstractmethod
    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        """
        Simulate data from the model

        Parameters
        ----------
        parameters : {ndarray, Series}
            Parameters required to simulate the volatility model
        nobs : int
            Number of data points to simulate
        rng : callable
            Callable function that takes a single integer input and returns
            a vector of random numbers
        burn : int, optional
            Number of additional observations to generate when initializing
            the simulation
        initial_value : {float, ndarray}, optional
            Scalar or array of initial values to use when initializing the
            simulation

        Returns
        -------
        simulations : CovarianceSimulation
            Object with two attributes:

            * `resids`: The array of simulated residuals (nobs, nvar)
            * `covariance`: Array of simulated conditional covariances
              (nobs, nvar, nvar)
        """
        pass

    def _gaussian_loglikelihood(self, parameters, resids, backcast,
                                var_bounds):
        """
        Private implementation of a Gaussian log-likelihood for use in constructing starting
        values or other quantities that do not depend on the distribution used by the model.
        """
        nobs, nvar = resids.shape
        sigma = np.empty_like((nobs, nvar, nvar))
        self.compute_covariance(parameters, resids, sigma, backcast, var_bounds)
        return self._mvnormal.loglikelihood([], resids, sigma)

    @abstractmethod
    def parameter_names(self):
        """
        Names of model parameters

        Returns
        -------
         names : list (str)
            Variables names
        """
        pass


class ConstantCovariance(MultivariateVolatilityProcess):
    r"""
    Constant covariance process

    Notes
    -----
    Model has the same covariance in all periods
    """

    def __init__(self):
        super(ConstantCovariance, self).__init__()
        self.name = 'Constant Covariance'
        self.closed_form = True

    @property
    def num_params(self):
        self._requires_nvar()
        return (self.nvar * (self.nvar + 1)) // 2

    def compute_covariance(self, parameters, resids, sigma, backcast):
        parameters = inv_vech(parameters, symmetric=False)
        sigma[:] = parameters.dot(parameters.T)
        return sigma

    def transform_parameters(self, *args):
        """
        transform_parameters(cov)

        Parameters
        ----------
        cov : array-like
            Covariance matrix

        Returns
        -------
        tcov : ndarray
            vech of the Cholesky factor of the covariance
        """
        return vech(np.linalg.cholesky(args[0]))

    def starting_values(self, resids):
        cov = resids.T.dot(resids) / resids.shape[0]
        return vech(cov)

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        errors = rng(nobs + burn)  # errors are nobs + burn by nvar
        parameters = inv_vech(parameters, symmetric=False)
        sigma = parameters.dot(parameters.T)
        sigma_12 = symmetric_matrix_root(sigma)
        sigma = np.tile(sigma, (nobs, 1, 1))
        data = errors.dot(sigma_12)
        return CovarianceSimulation(data[burn:], sigma)

    def constraints(self):
        self._requires_nvar()
        return np.empty((0, self.nvar)), np.empty(0)

    def backcast(self, resids):
        return resids.T.dot(resids) / resids.shape[0]

    def bounds(self, resids):
        cov = resids.T.dot(resids) / resids.shape[0]
        lower = vech(np.linalg.cholesky(cov / 100000.0))
        upper = vech(np.linalg.cholesky(cov * 100000.0))
        return [(l, u) for l, u in zip(lower, upper)]

    def parameter_names(self):
        self._requires_nvar()
        return ['c[{0},{1}]'.format(i, j) for i in range(self.nvar) for j in range(i + 1)]


class EWMACovariance(MultivariateVolatilityProcess):
    def __init__(self, lam=0.94):
        super(EWMACovariance, self).__init__()
        self.lam = lam
        self._estimate_lam = lam is None

    @property
    def num_params(self):
        return 1 if self._estimate_lam else 0

    def starting_values(self, resids):
        if self._estimate_lam:
            return np.array([0.94])
        return np.empty(0)

    def bounds(self, resids):
        if self._estimate_lam:
            return [(0.0, 1.0)]
        return []

    def compute_covariance(self, parameters, resids, sigma, backcast):
        sigma[0] = backcast
        nobs = resids.shape[0]
        lam = parameters[0] if self._estimate_lam else self.lam
        for i in range(1, nobs):
            r = resids[(i - 1):i]
            sigma[i] = (1 - lam) * r.T.dot(r) + lam * sigma[i - 1]
        return sigma

    def constraints(self):
        if self._estimate_lam:
            a = np.ones((1, 1))
            b = np.zeros((1,))
            return a, b
        return np.empty((0, 0)), np.empty((0,))

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        """
        Simulate data from the model

        Parameters
        ----------
        parameters : {ndarray, Series}
            Parameters required to simulate the volatility model
        nobs : int
            Number of data points to simulate
        rng : callable
            Callable function that takes a single integer input and returns
            a vector of random numbers
        burn : int, optional
            Number of additional observations to generate when initializing
            the simulation
        initial_value : {float, ndarray}
            Array to use as the initial value (nvar, nvar).  This value is
            **required** for this process.

        Returns
        -------
        simulations : CovarianceSimulation
            Object with two attributes:

            * `resids`: The array of simulated residuals (nobs, nvar)
            * `covariance`: Array of simulated conditional covariances
              (nobs, nvar, nvar)

        """
        if initial_value is None:
            raise ValueError('Initial value is required when simulating an EWMA covariance.')
        lam = self.lam if not self._estimate_lam else np.asarray(parameters)[0]
        nvar = initial_value.shape[0]
        sigma = np.empty((nobs + burn, nvar, nvar))
        sigma[0] = initial_value
        std_resids = rng(nobs + burn)
        resids = np.empty_like(std_resids)
        resids[:1] = std_resids[:1].dot(symmetric_matrix_root(initial_value))
        for i in range(1, nobs + burn):
            r = resids[i - 1:i]
            sigma[i] = (1 - lam) * r.T.dot(r) + lam * sigma[i - 1]
            resids[i:i + 1] = std_resids[i:i + 1].dot(symmetric_matrix_root(sigma[i]))

        return CovarianceSimulation(resids[burn:], sigma[burn:])

    def parameter_names(self):
        if self._estimate_lam:
            return ['lambda']
        return []

    def transform_parameters(self, *args):
        """
        transform_parameters(lam)

        :param args:
        :return:
        """
        return args[0]


class CovarianceSimulation(object):
    __slots__ = ('resids', 'covariance')

    def __init__(self, resids, covariance):
        self.resids = resids
        self.covariance = covariance
