"""
Volatility processes for ARCH model estimation.  All volatility processes must
inherit from :class:`VolatilityProcess` and provide the same methods with the
same inputs.
"""
from __future__ import absolute_import, division

from arch.compat.python import add_metaclass, range

from abc import abstractmethod
import itertools
from warnings import warn

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln

from arch.univariate.distribution import Normal
from arch.utility.array import AbstractDocStringInheritor, ensure1d
from arch.utility.exceptions import InitialValueWarning, initial_value_warning

try:
    from arch.univariate.recursions import (garch_recursion, harch_recursion,
                                            egarch_recursion, midas_recursion,
                                            figarch_weights, figarch_recursion)
except ImportError:  # pragma: no cover
    from arch.univariate.recursions_python import (garch_recursion, harch_recursion,
                                                   egarch_recursion, midas_recursion,
                                                   figarch_recursion, figarch_weights)

__all__ = ['GARCH', 'ARCH', 'HARCH', 'ConstantVariance', 'EWMAVariance', 'RiskMetrics2006',
           'EGARCH', 'FixedVariance', 'BootstrapRng', 'MIDASHyperbolic', 'VolatilityProcess']


def _common_names(p, o, q):
    names = ['omega']
    names.extend(['alpha[' + str(i + 1) + ']' for i in range(p)])
    names.extend(['gamma[' + str(i + 1) + ']' for i in range(o)])
    names.extend(['beta[' + str(i + 1) + ']' for i in range(q)])
    return names


class BootstrapRng(object):
    """
    Simple fake RNG used to transform bootstrap-based forecasting into a standard
    simulation forecasting problem

    Parameters
    ----------
    std_resid : ndarray
        Array containing standardized residuals
    start : int
        Location of first forecast
    random_state : RandomState, optional
        NumPy RandomState instance
    """

    def __init__(self, std_resid, start, random_state=None):
        if start <= 0 or start > std_resid.shape[0]:
            raise ValueError('start must be > 0 and <= len(std_resid).')

        self.std_resid = std_resid
        self.start = start
        self._index = start
        if random_state is None:
            random_state = RandomState()
        if not isinstance(random_state, RandomState):
            raise TypeError('random_state must be a RandomState instance')
        self._random_state = random_state

    @property
    def random_state(self):
        return self._random_state

    def rng(self):
        def _rng(size):
            if self._index >= self.std_resid.shape[0]:
                raise IndexError('not enough data points.')
            index = self._random_state.random_sample(size)
            int_index = np.floor((self._index + 1) * index)
            int_index = int_index.astype(np.int64)
            self._index += 1
            return self.std_resid[int_index]

        return _rng


def ewma_recursion(lam, resids, sigma2, nobs, backcast):
    """
    Compute variance recursion for EWMA/RiskMetrics Variance

    Parameters
    ----------
    lam : float
        Smoothing parameter
    resids : ndarray
        Residuals to use in the recursion
    sigma2 : ndarray
        Conditional variances with same shape as resids
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    """

    # Throw away bounds
    var_bounds = np.ones((nobs, 2)) * np.array([-1.0, 1.7e308])

    garch_recursion(np.array([0.0, 1.0 - lam, lam]), resids ** 2.0, resids, sigma2, 1, 0, 1, nobs,
                    backcast, var_bounds)
    return sigma2


class VarianceForecast(object):
    _forecasts = None
    _forecast_paths = None

    def __init__(self, forecasts, forecast_paths=None, shocks=None):
        self._forecasts = forecasts
        self._forecast_paths = forecast_paths
        self._shocks = shocks

    @property
    def forecasts(self):
        return self._forecasts

    @property
    def forecast_paths(self):
        return self._forecast_paths

    @property
    def shocks(self):
        return self._shocks


@add_metaclass(AbstractDocStringInheritor)
class VolatilityProcess(object):
    """
    Abstract base class for ARCH models.  Allows the conditional mean model to be specified
    separately from the conditional variance, even though parameters are estimated jointly.
    """

    def __init__(self):
        self.num_params = 0
        self.name = ''
        self.closed_form = False
        self._normal = Normal()
        self._min_bootstrap_obs = 100
        self._start = 0
        self._stop = -1

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__() + ', id: ' + hex(id(self))

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
    def _check_forecasting_method(self, method, horizon):
        """
        Verify the requested forecasting method as valid for the specification

        Parameters
        ----------
        method : str
            Forecasting method
        horizon : int
            Forecast horizon

        Raises
        ------
        NotImplementedError
            * If method is not known or not supported
        """
        pass

    def _one_step_forecast(self, parameters, resids, backcast, var_bounds, horizon):
        """
        One-step ahead forecast

        Parameters
        ----------
        parameters : ndarray
            Parameters required to forecast the volatility model
        resids : ndarray
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : ndarray
            Array containing columns of lower and upper bounds
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].

        Returns
        -------
        sigma2 : ndarray
            t element array containing the one-step ahead forecasts
        forecasts : ndarray
            t by horizon array containing the one-step ahead forecasts in the first location
        """
        t = resids.shape[0]
        _resids = np.concatenate((resids, [0]))
        _var_bounds = np.concatenate((var_bounds, [[0, np.inf]]))
        sigma2 = np.zeros(t + 1)
        self.compute_variance(parameters, _resids, sigma2, backcast, _var_bounds)
        forecasts = np.zeros((t, horizon))
        forecasts[:, 0] = sigma2[1:]
        sigma2 = sigma2[:-1]

        return sigma2, forecasts

    @abstractmethod
    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        """
        Analytic multi-step volatility forecasts from the model

        Parameters
        ----------
        parameters : ndarray
            Parameters required to forecast the volatility model
        resids : ndarray
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : ndarray
            Array containing columns of lower and upper bounds
        start : int
            Index of the first observation to use as the starting point for
            the forecast.  Default is 0.
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].

        Returns
        -------
        forecasts : VarianceForecast
            Class containing the variance forecasts, and, if using simulation
            or bootstrap, the simulated paths.
        """
        pass

    @abstractmethod
    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        """
        Simulation-based volatility forecasts from the model

        Parameters
        ----------
        parameters : ndarray
            Parameters required to forecast the volatility model
        resids : ndarray
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : ndarray
            Array containing columns of lower and upper bounds
        start : int
            Index of the first observation to use as the starting point for
            the forecast.  Default is 0.
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.
        rng : callable
            Callable random number generator required if method is
            'simulation'. Must take a single shape input and return random
            samples numbers with that shape.

        Returns
        -------
        forecasts : VarianceForecast
            Class containing the variance forecasts, and, if using simulation
            or bootstrap, the simulated paths.
        """
        pass

    def _bootstrap_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                            simulations, random_state):
        """
        Simulation-based volatility forecasts using model residuals

        Parameters
        ----------
        parameters : ndarray
            Parameters required to forecast the volatility model
        resids : ndarray
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : ndarray
            Array containing columns of lower and upper bounds
        start : int
            Index of the first observation to use as the starting point for
            the forecast.  Default is 0.
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.
        random_state : {RandomState, None}
            NumPy RandomState instance to use in the BootstrapRng

        Returns
        -------
        forecasts : VarianceForecast
            Class containing the variance forecasts, and, if using simulation
            or bootstrap, the simulated paths.
        """
        sigma2 = np.empty_like(resids)
        self.compute_variance(parameters, resids, sigma2, backcast, var_bounds)
        std_resid = resids / np.sqrt(sigma2)
        if start < self._min_bootstrap_obs:
            raise ValueError('start must include more than {0} '
                             'observations'.format(self._min_bootstrap_obs))
        rng = BootstrapRng(std_resid, start, random_state=random_state).rng()
        return self._simulation_forecast(parameters, resids, backcast, var_bounds,
                                         start, horizon, simulations, rng)

    def variance_bounds(self, resids, power=2.0):
        """
        Parameters
        ----------
        resids : ndarray
            Approximate residuals to use to compute the lower and upper bounds
            on the conditional variance
        power : float, optional
            Power used in the model. 2.0, the default corresponds to standard
            ARCH models that evolve in squares.

        Returns
        -------
        var_bounds : ndarray
            Array containing columns of lower and upper bounds with the same
            number of elements as resids
        """
        nobs = resids.shape[0]

        tau = min(75, nobs)
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        var_bound = np.zeros(nobs)
        initial_value = w.dot(resids[:tau] ** 2.0)
        ewma_recursion(0.94, resids, var_bound, resids.shape[0], initial_value)

        var_bounds = np.vstack((var_bound / 1e6, var_bound * 1e6)).T
        var = resids.var()
        min_upper_bound = 1 + (resids ** 2.0).max()
        lower_bound, upper_bound = var / 1e8, 1e7 * (1 + (resids ** 2.0).max())
        var_bounds[var_bounds[:, 0] < lower_bound, 0] = lower_bound
        var_bounds[var_bounds[:, 1] < min_upper_bound, 1] = min_upper_bound
        var_bounds[var_bounds[:, 1] > upper_bound, 1] = upper_bound

        if power != 2.0:
            var_bounds **= (power / 2.0)

        return np.ascontiguousarray(var_bounds)

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
        backcast : float
            Value to use in backcasting in the volatility recursion
        """
        tau = min(75, resids.shape[0])
        w = (0.94 ** np.arange(tau))
        w = w / sum(w)

        return np.sum((resids[:tau] ** 2.0) * w)

    def backcast_transform(self, backcast):
        """
        Transformation to apply to user-provided backcast values

        Parameters
        ----------
        backcast : {float, ndarray}
            User-provided ``backcast`` that approximates sigma2[0].

        Returns
        -------
        backcast : {float, ndarray}
            Backcast transformed to the model-appropriate scale
        """
        if np.any(backcast < 0):
            raise ValueError('User backcast value must be strictly positive.')
        return backcast

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
        bounds : list[tuple[float,float]]
            List of bounds where each element is (lower, upper).
        """
        pass

    @abstractmethod
    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        """
        Compute the variance for the ARCH model

        Parameters
        ----------
        parameters : ndarray
            Model parameters
        resids : ndarray
            Vector of mean zero residuals
        sigma2 : ndarray
            Array with same size as resids to store the conditional variance
        backcast : {float, ndarray}
            Value to use when initializing ARCH recursion. Can be an ndarray
            when the model contains multiple components.
        var_bounds : ndarray
            Array containing columns of lower and upper bounds
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

    def forecast(self, parameters, resids, backcast, var_bounds, start=None, horizon=1,
                 method='analytic', simulations=1000, rng=None, random_state=None):
        """
        Forecast volatility from the model

        Parameters
        ----------
        parameters : {ndarray, Series}
            Parameters required to forecast the volatility model
        resids : ndarray
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : ndarray, 2-d
            Array containing columns of lower and upper bounds
        start : {None, int}
            Index of the first observation to use as the starting point for
            the forecast.  Default is len(resids).
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].
        method : {'analytic', 'simulation', 'bootstrap'}
            Method to use when producing the forecast. The default is analytic.
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.
        rng : callable
            Callable random number generator required if method is
            'simulation'. Must take a single shape input and return random
            samples numbers with that shape.
        random_state : RandomState, optional
            NumPy RandomState instance to use when method is 'bootstrap'

        Returns
        -------
        forecasts : VarianceForecast
            Class containing the variance forecasts, and, if using simulation
            or bootstrap, the simulated paths.

        Raises
        ------
        NotImplementedError
            * If method is not supported
        ValueError
            * If the method is not known

        Notes
        -----
        The analytic ``method`` is not supported for all models.  Attempting
        to use this method when not available will raise a ValueError.
        """
        parameters = np.asarray(parameters)
        method = method.lower()
        if method not in ('analytic', 'simulation', 'bootstrap'):
            raise ValueError('{0} is not a known forecasting method'.format(method))

        self._check_forecasting_method(method, horizon)

        start = len(resids) - 1 if start is None else start
        if method == 'analytic':
            return self._analytic_forecast(parameters, resids, backcast, var_bounds, start,
                                           horizon)
        elif method == 'simulation':
            return self._simulation_forecast(parameters, resids, backcast, var_bounds, start,
                                             horizon, simulations, rng)
        else:
            if start < 10 or (horizon / start) >= .2:
                raise ValueError('Bootstrap forecasting requires at least 10 initial '
                                 'observations, and the ratio of horizon-to-start < 20%.')

            return self._bootstrap_forecast(parameters, resids, backcast, var_bounds, start,
                                            horizon, simulations, random_state)

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
        resids : ndarray
            The simulated residuals
        variance : ndarray
            The simulated variance
        """
        pass

    def _gaussian_loglikelihood(self, parameters, resids, backcast,
                                var_bounds):
        """
        Private implementation of a Gaussian log-likelihood for use in constructing starting
        values or other quantities that do not depend on the distribution used by the model.
        """
        sigma2 = np.zeros_like(resids)
        self.compute_variance(parameters, resids, sigma2, backcast, var_bounds)
        return self._normal.loglikelihood([], resids, sigma2)

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


class ConstantVariance(VolatilityProcess):
    r"""
    Constant volatility process

    Notes
    -----
    Model has the same variance in all periods
    """

    def __init__(self):
        super(ConstantVariance, self).__init__()
        self.num_params = 1
        self.name = 'Constant Variance'
        self.closed_form = True

    def compute_variance(self, parameters, resids, sigma2, backcast, var_bounds):
        sigma2[:] = parameters[0]
        return sigma2

    def starting_values(self, resids):
        return np.array([resids.var()])

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        errors = rng(nobs + burn)
        sigma2 = np.ones(nobs + burn) * parameters[0]
        data = np.sqrt(sigma2) * errors
        return data[burn:], sigma2[burn:]

    def constraints(self):
        return np.ones((1, 1)), np.zeros(1)

    def backcast_transform(self, backcast):
        backcast = super(ConstantVariance, self).backcast_transform(backcast)
        return backcast

    def backcast(self, resids):
        return resids.var()

    def bounds(self, resids):
        v = resids.var()
        return [(v / 100000.0, 10.0 * (v + resids.mean() ** 2.0))]

    def parameter_names(self):
        return ['sigma2']

    def _check_forecasting_method(self, method, horizon):
        return

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        t = resids.shape[0]
        forecasts = np.full((t, horizon), np.nan)

        forecasts[start:, :] = parameters[0]
        forecast_paths = None
        return VarianceForecast(forecasts, forecast_paths)

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        t = resids.shape[0]
        forecasts = np.full((t, horizon), np.nan)
        forecast_paths = np.full((t, simulations, horizon), np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)

        for i in range(start, t):
            shocks[i, :, :] = np.sqrt(parameters[0]) * rng((simulations, horizon))

        forecasts[start:, :] = parameters[0]
        forecast_paths[start:, :, :] = parameters[0]

        return VarianceForecast(forecasts, forecast_paths, shocks)


class GARCH(VolatilityProcess):
    r"""
    GARCH and related model estimation

    The following models can be specified using GARCH:
        * ARCH(p)
        * GARCH(p,q)
        * GJR-GARCH(p,o,q)
        * AVARCH(p)
        * AVGARCH(p,q)
        * TARCH(p,o,q)
        * Models with arbitrary, pre-specified powers

    Parameters
    ----------
    p : int
        Order of the symmetric innovation
    o : int
        Order of the asymmetric innovation
    q : int
        Order of the lagged (transformed) conditional variance
    power : float, optional
        Power to use with the innovations, abs(e) ** power.  Default is 2.0, which produces ARCH
        and related models. Using 1.0 produces AVARCH and related models.  Other powers can be
        specified, although these should be strictly positive, and usually larger than 0.25.

    Attributes
    ----------
    num_params : int
        The number of parameters in the model

    Examples
    --------
    >>> from arch.univariate import GARCH

    Standard GARCH(1,1)

    >>> garch = GARCH(p=1, q=1)

    Asymmetric GJR-GARCH process

    >>> gjr = GARCH(p=1, o=1, q=1)

    Asymmetric TARCH process

    >>> tarch = GARCH(p=1, o=1, q=1, power=1.0)

    Notes
    -----
    In this class of processes, the variance dynamics are

    .. math::

        \sigma_{t}^{\lambda}=\omega
        + \sum_{i=1}^{p}\alpha_{i}\left|\epsilon_{t-i}\right|^{\lambda}
        +\sum_{j=1}^{o}\gamma_{j}\left|\epsilon_{t-j}\right|^{\lambda}
        I\left[\epsilon_{t-j}<0\right]+\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{\lambda}
    """

    def __init__(self, p=1, o=0, q=1, power=2.0):
        super(GARCH, self).__init__()
        self.p = int(p)
        self.o = int(o)
        self.q = int(q)
        self.power = power
        self.num_params = 1 + p + o + q
        if p < 0 or o < 0 or q < 0:
            raise ValueError('All lags lengths must be non-negative')
        if p == 0 and o == 0:
            raise ValueError('One of p or o must be strictly positive')
        if power <= 0.0:
            raise ValueError('power must be strictly positive, usually larger than 0.25')
        self.name = self._name()

    def __str__(self):
        descr = self.name

        if self.power != 1.0 and self.power != 2.0:
            descr = descr[:-1] + ', '
        else:
            descr += '('

        for k, v in (('p', self.p), ('o', self.o), ('q', self.q)):
            if v > 0:
                descr += k + ': ' + str(v) + ', '

        descr = descr[:-2] + ')'
        return descr

    def variance_bounds(self, resids, power=2.0):
        return super(GARCH, self).variance_bounds(resids, self.power)

    def _name(self):
        p, o, q, power = self.p, self.o, self.q, self.power  # noqa: F841
        if power == 2.0:
            if o == 0 and q == 0:
                return 'ARCH'
            elif o == 0:
                return 'GARCH'
            else:
                return 'GJR-GARCH'
        elif power == 1.0:
            if o == 0 and q == 0:
                return 'AVARCH'
            elif o == 0:
                return 'AVGARCH'
            else:
                return 'TARCH/ZARCH'
        else:
            if o == 0 and q == 0:
                return 'Power ARCH (power: {0:0.1f})'.format(self.power)
            elif o == 0:
                return 'Power GARCH (power: {0:0.1f})'.format(self.power)
            else:
                return 'Asym. Power GARCH (power: {0:0.1f})'.format(self.power)

    def bounds(self, resids):
        v = np.mean(abs(resids) ** self.power)

        bounds = [(0.0, 10.0 * v)]
        bounds.extend([(0.0, 1.0)] * self.p)
        for i in range(self.o):
            if i < self.p:
                bounds.append((-1.0, 2.0))
            else:
                bounds.append((0.0, 2.0))

        bounds.extend([(0.0, 1.0)] * self.q)

        return bounds

    def constraints(self):
        p, o, q = self.p, self.o, self.q
        k_arch = p + o + q
        # alpha[i] >0
        # alpha[i] + gamma[i] > 0 for i<=p, otherwise gamma[i]>0
        # beta[i] >0
        # sum(alpha) + 0.5 sum(gamma) + sum(beta) < 1
        a = np.zeros((k_arch + 2, k_arch + 1))
        for i in range(k_arch + 1):
            a[i, i] = 1.0
        for i in range(o):
            if i < p:
                a[i + p + 1, i + 1] = 1.0

        a[k_arch + 1, 1:] = -1.0
        a[k_arch + 1, p + 1:p + o + 1] = -0.5
        b = np.zeros(k_arch + 2)
        b[k_arch + 1] = -1.0
        return a, b

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        # fresids is abs(resids) ** power
        # sresids is I(resids<0)
        power = self.power
        fresids = np.abs(resids) ** power
        sresids = np.sign(resids)

        p, o, q = self.p, self.o, self.q
        nobs = resids.shape[0]

        garch_recursion(parameters, fresids, sresids, sigma2, p, o, q, nobs,
                        backcast, var_bounds)
        inv_power = 2.0 / power
        sigma2 **= inv_power

        return sigma2

    def backcast_transform(self, backcast):
        backcast = super(GARCH, self).backcast_transform(backcast)
        return np.sqrt(backcast) ** self.power

    def backcast(self, resids):
        power = self.power
        tau = min(75, resids.shape[0])
        w = (0.94 ** np.arange(tau))
        w = w / sum(w)
        backcast = np.sum((abs(resids[:tau]) ** power) * w)

        return backcast

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        p, o, q, power = self.p, self.o, self.q, self.power
        errors = rng(nobs + burn)

        if initial_value is None:
            scale = np.ones_like(parameters)
            scale[p + 1:p + o + 1] = 0.5

            persistence = np.sum(parameters[1:] * scale[1:])
            if (1.0 - persistence) > 0:
                initial_value = parameters[0] / (1.0 - persistence)
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)
        fsigma = np.zeros(nobs + burn)
        fdata = np.zeros(nobs + burn)

        max_lag = np.max([p, o, q])
        fsigma[:max_lag] = initial_value
        sigma2[:max_lag] = initial_value ** (2.0 / power)
        data[:max_lag] = np.sqrt(sigma2[:max_lag]) * errors[:max_lag]
        fdata[:max_lag] = abs(data[:max_lag]) ** power

        for t in range(max_lag, nobs + burn):
            loc = 0
            fsigma[t] = parameters[loc]
            loc += 1
            for j in range(p):
                fsigma[t] += parameters[loc] * fdata[t - 1 - j]
                loc += 1
            for j in range(o):
                fsigma[t] += parameters[loc] * \
                             fdata[t - 1 - j] * (data[t - 1 - j] < 0)
                loc += 1
            for j in range(q):
                fsigma[t] += parameters[loc] * fsigma[t - 1 - j]
                loc += 1

            sigma2[t] = fsigma[t] ** (2.0 / power)
            data[t] = errors[t] * np.sqrt(sigma2[t])
            fdata[t] = abs(data[t]) ** power

        return data[burn:], sigma2[burn:]

    def starting_values(self, resids):
        p, o, q = self.p, self.o, self.q
        power = self.power
        alphas = [.01, .05, .1, .2]
        gammas = alphas
        abg = [.5, .7, .9, .98]
        abgs = list(itertools.product(*[alphas, gammas, abg]))

        target = np.mean(abs(resids) ** power)
        scale = np.mean(resids ** 2) / (target ** (2.0 / power))
        target *= (scale ** (power / 2))

        svs = []
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = np.zeros(len(abgs))
        for i, values in enumerate(abgs):
            alpha, gamma, agb = values
            sv = (1.0 - agb) * target * np.ones(p + o + q + 1)
            if p > 0:
                sv[1:1 + p] = alpha / p
                agb -= alpha
            if o > 0:
                sv[1 + p:1 + p + o] = gamma / o
                agb -= gamma / 2.0
            if q > 0:
                sv[1 + p + o:1 + p + o + q] = agb / q
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)

        return svs[int(loc)]

    def parameter_names(self):
        return _common_names(self.p, self.o, self.q)

    def _check_forecasting_method(self, method, horizon):
        if horizon == 1:
            return

        if method == 'analytic' and self.power != 2.0:
            raise ValueError('Analytic forecasts not available for horizon > 1 when power != 2')
        return

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):

        sigma2, forecasts = self._one_step_forecast(parameters, resids, backcast,
                                                    var_bounds, horizon)
        if horizon == 1:
            forecasts[:start] = np.nan
            return VarianceForecast(forecasts)

        t = resids.shape[0]
        p, o, q = self.p, self.o, self.q
        omega = parameters[0]
        alpha = parameters[1:p + 1]
        gamma = parameters[p + 1: p + o + 1]
        beta = parameters[p + o + 1:]

        m = np.max([p, o, q])
        _resids = np.zeros(m + horizon)
        _asym_resids = np.zeros(m + horizon)
        _sigma2 = np.zeros(m + horizon)

        for i in range(start, t):
            if i - m + 1 >= 0:
                _resids[:m] = resids[i - m + 1:i + 1]
                _asym_resids[:m] = _resids[:m] * (_resids[:m] < 0)
                _sigma2[:m] = sigma2[i - m + 1:i + 1]
            else:  # Back-casting needed
                _resids[:m - i - 1] = np.sqrt(backcast)
                _resids[m - i - 1: m] = resids[0:i + 1]
                _asym_resids = _resids * (_resids < 0)
                _asym_resids[:m - i - 1] = np.sqrt(0.5 * backcast)
                _sigma2[:m] = backcast
                _sigma2[m - i - 1: m] = sigma2[0:i + 1]

            for h in range(0, horizon):
                forecasts[i, h] = omega
                start_loc = h + m - 1

                for j in range(p):
                    forecasts[i, h] += alpha[j] * _resids[start_loc - j] ** 2

                for j in range(o):
                    forecasts[i, h] += gamma[j] * _asym_resids[start_loc - j] ** 2

                for j in range(q):
                    forecasts[i, h] += beta[j] * _sigma2[start_loc - j]

                _resids[h + m] = np.sqrt(forecasts[i, h])
                _asym_resids[h + m] = np.sqrt(0.5 * forecasts[i, h])
                _sigma2[h + m] = forecasts[i, h]

        forecasts[:start] = np.nan
        return VarianceForecast(forecasts)

    def _simulate_paths(self, m, parameters, horizon, std_shocks,
                        scaled_forecast_paths, scaled_shock, asym_scaled_shock):

        power = self.power
        p, o, q = self.p, self.o, self.q
        omega = parameters[0]
        alpha = parameters[1:p + 1]
        gamma = parameters[p + 1: p + o + 1]
        beta = parameters[p + o + 1:]
        shock = np.full_like(scaled_forecast_paths, np.nan)

        for h in range(horizon):
            loc = h + m - 1

            scaled_forecast_paths[:, h + m] = omega
            for j in range(p):
                scaled_forecast_paths[:, h + m] += alpha[j] * scaled_shock[:, loc - j]

            for j in range(o):
                scaled_forecast_paths[:, h + m] += gamma[j] * asym_scaled_shock[:, loc - j]

            for j in range(q):
                scaled_forecast_paths[:, h + m] += beta[j] * scaled_forecast_paths[:, loc - j]

            shock[:, h + m] = std_shocks[:, h] * scaled_forecast_paths[:, h + m] ** (1.0 / power)
            lt_zero = shock[:, h + m] < 0
            scaled_shock[:, h + m] = np.abs(shock[:, h + m]) ** power
            asym_scaled_shock[:, h + m] = scaled_shock[:, h + m] * lt_zero

        forecast_paths = scaled_forecast_paths[:, m:] ** (2.0 / power)

        return np.mean(forecast_paths, 0), forecast_paths, shock[:, m:]

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):

        sigma2, forecasts = self._one_step_forecast(parameters, resids, backcast,
                                                    var_bounds, horizon)
        t = resids.shape[0]
        paths = np.full((t, simulations, horizon), np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)

        power = self.power
        m = np.max([self.p, self.o, self.q])
        scaled_forecast_paths = np.zeros((simulations, m + horizon))
        scaled_shock = np.zeros((simulations, m + horizon))
        asym_scaled_shock = np.zeros((simulations, m + horizon))

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            if i - m < 0:
                scaled_forecast_paths[:, :m] = backcast ** (power / 2.0)
                scaled_shock[:, :m] = backcast ** (power / 2.0)
                asym_scaled_shock[:, :m] = (0.5 * backcast) ** (power / 2.0)

                count = i + 1
                scaled_forecast_paths[:, m - count:m] = sigma2[:count] ** (power / 2.0)
                scaled_shock[:, m - count:m] = np.abs(resids[:count]) ** power
                asym = np.abs(resids[:count]) ** power * (resids[:count] < 0)
                asym_scaled_shock[:, m - count:m] = asym
            else:
                scaled_forecast_paths[:, :m] = sigma2[i - m + 1:i + 1] ** (power / 2.0)
                scaled_shock[:, :m] = np.abs(resids[i - m + 1:i + 1]) ** power
                asym_scaled_shock[:, :m] = scaled_shock[:, :m] * (resids[i - m + 1:i + 1] < 0)

            f, p, s = self._simulate_paths(m, parameters, horizon, std_shocks,
                                           scaled_forecast_paths, scaled_shock, asym_scaled_shock)
            forecasts[i, :], paths[i], shocks[i] = f, p, s

        forecasts[:start] = np.nan
        return VarianceForecast(forecasts, paths, shocks)


class HARCH(VolatilityProcess):
    r"""
    Heterogeneous ARCH process

    Parameters
    ----------
    lags : {list, array, int}
        List of lags to include in the model, or if scalar, includes all lags up the value

    Attributes
    ----------
    num_params : int
        The number of parameters in the model

    Examples
    --------
    >>> from arch.univariate import HARCH

    Lag-1 HARCH, which is identical to an ARCH(1)

    >>> harch = HARCH()

    More useful and realistic lag lengths

    >>> harch = HARCH(lags=[1, 5, 22])

    Notes
    -----
    In a Heterogeneous ARCH process, variance dynamics are

    .. math::

        \sigma_{t}^{2}=\omega + \sum_{i=1}^{m}\alpha_{l_{i}}
        \left(l_{i}^{-1}\sum_{j=1}^{l_{i}}\epsilon_{t-j}^{2}\right)

    In the common case where lags=[1,5,22], the model is

    .. math::

        \sigma_{t}^{2}=\omega+\alpha_{1}\epsilon_{t-1}^{2}
        +\alpha_{5} \left(\frac{1}{5}\sum_{j=1}^{5}\epsilon_{t-j}^{2}\right)
        +\alpha_{22} \left(\frac{1}{22}\sum_{j=1}^{22}\epsilon_{t-j}^{2}\right)

    A HARCH process is a special case of an ARCH process where parameters in the more general
    ARCH process have been restricted.
    """

    def __init__(self, lags=1):
        super(HARCH, self).__init__()
        if np.isscalar(lags):
            lags = np.arange(1, lags + 1)
        lags = ensure1d(lags, 'lags')
        self.lags = np.array(lags, dtype=np.int32)
        self._num_lags = lags.shape[0]
        self.num_params = self._num_lags + 1
        self.name = 'HARCH'

    def __str__(self):
        descr = self.name + '(lags: '
        descr += ', '.join([str(l) for l in self.lags])
        descr += ')'

        return descr

    def bounds(self, resids):
        lags = self.lags
        k_arch = lags.shape[0]

        bounds = [(0.0, 10 * np.mean(resids ** 2.0))]
        bounds.extend([(0.0, 1.0)] * k_arch)

        return bounds

    def constraints(self):
        k_arch = self._num_lags
        a = np.zeros((k_arch + 2, k_arch + 1))
        for i in range(k_arch + 1):
            a[i, i] = 1.0
        a[k_arch + 1, 1:] = -1.0
        b = np.zeros(k_arch + 2)
        b[k_arch + 1] = -1.0
        return a, b

    def compute_variance(self, parameters, resids,
                         sigma2, backcast, var_bounds):
        lags = self.lags
        nobs = resids.shape[0]

        harch_recursion(parameters, resids, sigma2, lags, nobs, backcast, var_bounds)
        return sigma2

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        lags = self.lags
        errors = rng(nobs + burn)

        if initial_value is None:
            if (1.0 - np.sum(parameters[1:])) > 0:
                initial_value = parameters[0] / (1.0 - np.sum(parameters[1:]))
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.empty(nobs + burn)
        data = np.empty(nobs + burn)
        max_lag = np.max(lags)
        sigma2[:max_lag] = initial_value
        data[:max_lag] = np.sqrt(initial_value)
        for t in range(max_lag, nobs + burn):
            sigma2[t] = parameters[0]
            for i in range(lags.shape[0]):
                param = parameters[1 + i] / lags[i]
                for j in range(lags[i]):
                    sigma2[t] += param * data[t - 1 - j] ** 2.0
            data[t] = errors[t] * np.sqrt(sigma2[t])

        return data[burn:], sigma2[burn:]

    def starting_values(self, resids):
        k_arch = self._num_lags

        alpha = 0.9
        sv = (1.0 - alpha) * resids.var() * np.ones((k_arch + 1))
        sv[1:] = alpha / k_arch

        return sv

    def parameter_names(self):
        names = ['omega']
        lags = self.lags
        names.extend(['alpha[' + str(lags[i]) + ']' for i in range(self._num_lags)])
        return names

    def _harch_to_arch(self, params):
        arch_params = np.zeros((1 + self.lags.max()))
        arch_params[0] = params[0]
        for param, lag in zip(params[1:], self.lags):
            arch_params[1:lag + 1] += param / lag

        return arch_params

    def _common_forecast_components(self, parameters, resids, backcast, horizon):
        arch_params = self._harch_to_arch(parameters)
        t = resids.shape[0]
        m = self.lags.max()
        resids2 = np.empty((t, m + horizon))
        resids2[:m, :m] = backcast
        sq_resids = resids ** 2.0
        for i in range(m):
            resids2[m - i - 1:, i] = sq_resids[:(t - (m - i - 1))]
        const = arch_params[0]
        arch = arch_params[1:]

        return const, arch, resids2

    def _check_forecasting_method(self, method, horizon):
        return

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        const, arch, resids2 = self._common_forecast_components(parameters, resids, backcast,
                                                                horizon)
        m = self.lags.max()
        resids2[:start] = np.nan
        arch_rev = arch[::-1]
        for i in range(horizon):
            resids2[:, m + i] = const + resids2[:, i:(m + i)].dot(arch_rev)

        return VarianceForecast(resids2[:, m:].copy())

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        const, arch, resids2 = self._common_forecast_components(parameters, resids, backcast,
                                                                horizon)
        t, m = resids.shape[0], self.lags.max()

        shocks = np.full((t, simulations, horizon), np.nan)
        paths = np.full((t, simulations, horizon), np.nan)

        temp_resids2 = np.empty((simulations, m + horizon))
        arch_rev = arch[::-1]
        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            temp_resids2[:, :] = resids2[i:(i + 1)]
            for j in range(horizon):
                paths[i, :, j] = const + temp_resids2[:, j:(m + j)].dot(arch_rev)
                shocks[i, :, j] = std_shocks[:, j] * np.sqrt(paths[i, :, j])
                temp_resids2[:, m + j] = shocks[i, :, j] ** 2.0

        return VarianceForecast(paths.mean(1), paths, shocks)


class MIDASHyperbolic(VolatilityProcess):
    r"""
    MIDAS Hyperbolic ARCH process

    Parameters
    ----------
    m : int
        Length of maximum lag to include in the model
    asym : bool
        Flag indicating whether to include an asymmetric term

    Attributes
    ----------
    num_params : int
        The number of parameters in the model

    Examples
    --------
    >>> from arch.univariate import MIDASHyperbolic

    22-lag MIDAS Hyperbolic process

    >>> harch = MIDASHyperbolic()

    Longer 66-period lag

    >>> harch = MIDASHyperbolic(m=66)

    Asymmetric MIDAS Hyperbolic process

    >>> harch = MIDASHyperbolic(asym=True)

    Notes
    -----
    In a MIDAS Hyperbolic process, the variance evolves according to

    .. math::

        \sigma_{t}^{2}=\omega+
        \sum_{i=1}^{m}\left(\alpha+\gamma I\left[\epsilon_{t-j}<0\right]\right)
        \phi_{i}(\theta)\epsilon_{t-i}^{2}

    where

    .. math::

        \phi_{i}(\theta) \propto \Gamma(i+\theta)/(\Gamma(i+1)\Gamma(\theta))

    where :math:`\Gamma` is the gamma function. :math:`\{\phi_i(\theta)\}` is
    normalized so that :math:`\sum \phi_i(\theta)=1`

    References
    ----------
    .. [*] Foroni, Claudia, and Massimiliano Marcellino. "A survey of
       Econometric Methods for Mixed-Frequency Data". Norges Bank. (2013).
    .. [*] Sheppard, Kevin. "Direct volatility modeling". Manuscript. (2018).
    """

    def __init__(self, m=22, asym=False):
        super(MIDASHyperbolic, self).__init__()
        self.m = m
        self._asym = bool(asym)
        self.num_params = 3 + self._asym
        self.name = 'MIDAS Hyperbolic'

    def __str__(self):
        descr = self.name
        descr += '(lags: {0}, asym: {1}'.format(self.m, self._asym)

        return descr

    def bounds(self, resids):
        bounds = [(0.0, 10 * np.mean(resids ** 2.0))]  # omega
        bounds.extend([(0.0, 1.0)])  # 0 <= alpha < 1
        if self._asym:
            bounds.extend([(-1.0, 2.0)])  # -1 <= gamma < 2
        bounds.extend([(0.0, 1.0)])  # theta

        return bounds

    def constraints(self):
        """
        Constraints

        Notes
        -----
        Parameters are (omega, alpha, gamma, theta)

        A.dot(parameters) - b >= 0

        1. omega >0
        2. alpha>0 or alpha + gamma > 0
        3. alpha<1 or alpha+0.5*gamma<1
        4. theta > 0
        5. theta < 1
        """
        symm = not self._asym
        k = 3 + self._asym
        a = np.zeros((5, k))
        b = np.zeros(5)
        # omega
        a[0, 0] = 1.0
        # alpha >0 or alpha+gamma>0
        # alpha<1 or alpha+0.5*gamma<1
        if symm:
            a[1, 1] = 1.0
            a[2, 1] = -1.0
        else:
            a[1, 1:3] = 1.0
            a[2, 1:3] = [-1, -0.5]
        b[2] = -1.0
        # theta
        a[3, k - 1] = 1.0
        a[4, k - 1] = -1.0
        b[4] = -1.0

        return a, b

    def compute_variance(self, parameters, resids,
                         sigma2, backcast, var_bounds):
        nobs = resids.shape[0]
        weights = self._weights(parameters)
        if not self._asym:
            params = np.zeros(3)
            params[:2] = parameters[:2]
        else:
            params = parameters[:3]

        midas_recursion(params, weights, resids, sigma2, nobs, backcast, var_bounds)
        return sigma2

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        if self._asym:
            omega, alpha, gamma = parameters[:3]
        else:
            omega, alpha = parameters[:2]
            gamma = 0
        weights = self._weights(parameters)
        aw = weights * alpha
        gw = weights * gamma

        errors = rng(nobs + burn)

        if initial_value is None:
            if (1.0 - alpha - 0.5 * gamma) > 0:
                initial_value = parameters[0] / (1.0 - alpha - 0.5 * gamma)
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        m = weights.shape[0]
        burn = max(burn, m)
        sigma2 = np.empty(nobs + burn)
        data = np.empty(nobs + burn)

        sigma2[:m] = initial_value
        data[:m] = np.sqrt(initial_value)
        for t in range(m, nobs + burn):
            sigma2[t] = omega
            for i in range(m):
                if t - 1 - i < m:
                    coef = aw[i] + 0.5 * gw[i]
                else:
                    coef = aw[i] + gw[i] * (data[t - 1 - i] < 0)
                sigma2[t] += coef * data[t - 1 - i] ** 2.0
            data[t] = errors[t] * np.sqrt(sigma2[t])

        return data[burn:], sigma2[burn:]

    def starting_values(self, resids):
        theta = [.1, .5, .8, .9]
        alpha = [0.8, 0.9, 0.95, 0.98]
        var = (resids ** 2).mean()
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = []
        svs = []
        for a, t in itertools.product(alpha, theta):
            gamma = [0.0]
            if self._asym:
                gamma.extend([0.5, 0.9])
            for g in gamma:
                total = a + g / 2
                o = (1 - min(total, 0.99)) * var
                if self._asym:
                    sv = np.array([o, a, g, t])
                else:
                    sv = np.array([o, a, t])

                svs.append(sv)

                llf = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
                llfs.append(llf)
        llfs = np.array(llfs)
        loc = np.argmax(llfs)

        return svs[int(loc)]

    def parameter_names(self):
        names = ['omega', 'alpha', 'theta']
        if self._asym:
            names.insert(2, 'gamma')

        return names

    def _weights(self, params):
        m = self.m
        # Prevent 0
        theta = max(params[-1], np.finfo(np.float64).eps)
        j = np.arange(1.0, m + 1)
        w = gammaln(theta + j) - gammaln(j + 1) - gammaln(theta)
        w = np.exp(w)
        return w / w.sum()

    def _common_forecast_components(self, parameters, resids, backcast, horizon):
        if self._asym:
            omega, alpha, gamma = parameters[:3]
        else:
            omega, alpha = parameters[:2]
            gamma = 0.0
        weights = self._weights(parameters)
        aw = weights * alpha
        gw = weights * gamma

        t = resids.shape[0]
        m = self.m
        resids2 = np.empty((t, m + horizon))
        resids2[:m, :m] = backcast
        indicator = np.empty((t, m + horizon))
        indicator[:m, :m] = 0.5
        sq_resids = resids ** 2.0
        for i in range(m):
            resids2[m - i - 1:, i] = sq_resids[:(t - (m - i - 1))]
            indicator[m - i - 1:, i] = resids[:(t - (m - i - 1))] < 0

        return omega, aw, gw, resids2, indicator

    def _check_forecasting_method(self, method, horizon):
        return

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        omega, aw, gw, resids2, indicator = self._common_forecast_components(parameters, resids,
                                                                             backcast, horizon)
        m = self.m
        resids2[:start] = np.nan
        aw_rev = aw[::-1]
        gw_rev = gw[::-1]

        for i in range(horizon):
            resids2[:, m + i] = omega + resids2[:, i:(m + i)].dot(aw_rev)
            if self._asym:
                resids2_ind = resids2[:, i:(m + i)] * indicator[:, i:(m + i)]
                resids2[:, m + i] += resids2_ind.dot(gw_rev)
                indicator[:, m + i] = 0.5

        return VarianceForecast(resids2[:, m:].copy())

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        omega, aw, gw, resids2, indicator = self._common_forecast_components(parameters, resids,
                                                                             backcast, horizon)
        t = resids.shape[0]
        m = self.m

        shocks = np.full((t, simulations, horizon), np.nan)
        paths = np.full((t, simulations, horizon), np.nan)

        temp_resids2 = np.empty((simulations, m + horizon))
        temp_indicator = np.empty((simulations, m + horizon))
        aw_rev = aw[::-1]
        gw_rev = gw[::-1]
        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            temp_resids2[:, :] = resids2[i:(i + 1)]
            temp_indicator[:, :] = indicator[i:(i + 1)]
            for j in range(horizon):
                paths[i, :, j] = omega + temp_resids2[:, j:(m + j)].dot(aw_rev)
                if self._asym:
                    temp_resids2_ind = temp_resids2[:, j:(m + j)] * temp_indicator[:, j:(m + j)]
                    paths[i, :, j] += temp_resids2_ind.dot(gw_rev)

                shocks[i, :, j] = std_shocks[:, j] * np.sqrt(paths[i, :, j])
                temp_resids2[:, m + j] = shocks[i, :, j] ** 2.0
                temp_indicator[:, m + j] = (shocks[i, :, j] < 0).astype(np.double)

        return VarianceForecast(paths.mean(1), paths, shocks)


class ARCH(GARCH):
    r"""
    ARCH process

    Parameters
    ----------
    p : int
        Order of the symmetric innovation

    Attributes
    ----------
    num_params : int
        The number of parameters in the model

    Examples
    --------
    ARCH(1) process

    >>> from arch.univariate import ARCH

    ARCH(5) process

    >>> arch = ARCH(p=5)

    Notes
    -----
    The variance dynamics of the model estimated

    .. math::

        \sigma_t^{2}=\omega+\sum_{i=1}^{p}\alpha_{i}\epsilon_{t-i}^{2}

    """

    def __init__(self, p=1):
        super(ARCH, self).__init__(p, 0, 0, 2.0)
        self.num_params = p + 1

    def starting_values(self, resids):
        p = self.p

        alphas = np.arange(.1, .95, .05)
        svs = []
        backcast = self.backcast(resids)
        llfs = alphas.copy()
        var_bounds = self.variance_bounds(resids)
        for i, alpha in enumerate(alphas):
            sv = (1.0 - alpha) * resids.var() * np.ones((p + 1))
            sv[1:] = alpha / p
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)
        return svs[int(loc)]


class EWMAVariance(VolatilityProcess):
    r"""
    Exponentially Weighted Moving-Average (RiskMetrics) Variance process

    Parameters
    ----------
    lam : {float, None}, optional
        Smoothing parameter. Default is 0.94. Set to None to estimate lam
        jointly with other model parameters

    Attributes
    ----------
    num_params : int
        The number of parameters in the model

    Examples
    --------
    Daily RiskMetrics EWMA process

    >>> from arch.univariate import EWMAVariance
    >>> rm = EWMAVariance(0.94)

    Notes
    -----
    The variance dynamics of the model

    .. math::

        \sigma_t^{2}=\lambda\sigma_{t-1}^2 + (1-\lambda)\epsilon^2_{t-1}

    When lam is provided, this model has no parameters since the smoothing
    parameter is treated as fixed. Sel lam to ``None`` to jointly estimate this
    parameter when fitting the model.
    """

    def __init__(self, lam=0.94):
        super(EWMAVariance, self).__init__()
        self.lam = lam
        self._estimate_lam = lam is None
        self.num_params = 1 if self._estimate_lam else 0
        if lam is not None and not 0.0 < lam < 1.0:
            raise ValueError('lam must be strictly between 0 and 1')
        self.name = 'EWMA/RiskMetrics'

    def __str__(self):
        if self._estimate_lam:
            descr = self.name + '(lam: Estimated)'
        else:
            descr = self.name + '(lam: ' + '{0:0.2f}'.format(self.lam) + ')'
        return descr

    def starting_values(self, resids):
        if self._estimate_lam:
            return np.array([0.94])
        return np.array([])

    def parameter_names(self):
        if self._estimate_lam:
            return ['lam']
        return []

    def bounds(self, resids):
        if self._estimate_lam:
            return [(0, 1)]
        return []

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        lam = parameters[0] if self._estimate_lam else self.lam
        return ewma_recursion(lam, resids, sigma2, resids.shape[0], backcast)

    def constraints(self):
        if self._estimate_lam:
            a = np.ones((1, 1))
            b = np.zeros((1,))
            return a, b
        return np.empty((0, 0)), np.empty((0,))

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        errors = rng(nobs + burn)

        if initial_value is None:
            initial_value = 1.0

        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)

        sigma2[0] = initial_value
        data[0] = np.sqrt(sigma2[0])
        if self._estimate_lam:
            lam = parameters[0]
        else:
            lam = self.lam
        one_m_lam = 1.0 - lam
        for t in range(1, nobs + burn):
            sigma2[t] = lam * sigma2[t - 1] + one_m_lam * data[t - 1] ** 2.0
            data[t] = np.sqrt(sigma2[t]) * errors[t]

        return data[burn:], sigma2[burn:]

    def _check_forecasting_method(self, method, horizon):
        return

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        t = resids.shape[0]
        _resids = np.empty(t + 1)
        _resids[:t] = resids
        sigma2 = np.empty(t + 1)
        self.compute_variance(parameters, _resids, sigma2, backcast, var_bounds)
        sigma2.shape = (t + 1, 1)
        forecasts = sigma2[1:]
        forecasts[:start] = np.nan
        forecasts = np.tile(forecasts, (1, horizon))
        return VarianceForecast(forecasts)

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        one_step = self._analytic_forecast(parameters, resids, backcast, var_bounds,
                                           start, 1)
        t = resids.shape[0]
        paths = np.full((t, simulations, horizon), np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)
        if self._estimate_lam:
            lam = parameters[0]
        else:
            lam = self.lam

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            paths[i, :, 0] = one_step.forecasts[i]
            shocks[i, :, 0] = np.sqrt(one_step.forecasts[i]) * std_shocks[:, 0]
            for h in range(1, horizon):
                paths[i, :, h] = (1 - lam) * shocks[i, :, h - 1] ** 2.0 + lam * paths[i, :, h - 1]
                shocks[i, :, h] = np.sqrt(paths[i, :, h]) * std_shocks[:, h]

        return VarianceForecast(paths.mean(1), paths, shocks)


class RiskMetrics2006(VolatilityProcess):
    """
    RiskMetrics 2006 Variance process

    Parameters
    ----------
    tau0 : int, optional
        Length of long cycle. Default is 1560.
    tau1 : int, optional
        Length of short cycle. Default is 4.
    kmax : int, optional
        Number of components. Default is 14.
    rho : float, optional
        Relative scale of adjacent cycles. Default is sqrt(2)

    Attributes
    ----------
    num_params : int
        The number of parameters in the model

    Examples
    --------
    Daily RiskMetrics 2006 process

    >>> from arch.univariate import RiskMetrics2006
    >>> rm = RiskMetrics2006()

    Notes
    -----
    The variance dynamics of the model are given as a weighted average of kmax EWMA variance
    processes where the smoothing parameters and weights are determined by tau0, tau1 and rho.

    This model has no parameters since the smoothing parameter is fixed.
    """

    def __init__(self, tau0=1560, tau1=4, kmax=14, rho=1.4142135623730951):
        super(RiskMetrics2006, self).__init__()
        self.tau0 = tau0
        self.tau1 = tau1
        self.kmax = kmax
        self.rho = rho
        self.num_params = 0

        if tau0 <= tau1 or tau1 <= 0:
            raise ValueError('tau0 must be greater than tau1 and tau1 > 0')
        if tau1 * rho ** (kmax - 1) > tau0:
            raise ValueError('tau1 * rho ** (kmax-1) smaller than tau0')
        if not kmax >= 1:
            raise ValueError('kmax must be a positive integer')
        if not rho > 1:
            raise ValueError('rho must be a positive number larger than 1')
        self.name = 'RiskMetrics2006'

    def __str__(self):
        descr = self.name
        descr += '(tau0: {0:d}, tau1: {1:d}, kmax: {2:d}, ' \
                 'rho: {3:0.3f}'.format(self.tau0, self.tau1,
                                        self.kmax, self.rho)
        descr += ')'
        return descr

    def _ewma_combination_weights(self):
        """
        Returns
        -------
        weights : ndarray
            Combination weights for EWMA components
        """
        tau0, tau1, kmax, rho = self.tau0, self.tau1, self.kmax, self.rho
        taus = tau1 * (rho ** np.arange(kmax))
        w = 1 - np.log(taus) / np.log(tau0)
        w = w / w.sum()

        return w

    def _ewma_smoothing_parameters(self):
        tau1, kmax, rho = self.tau1, self.kmax, self.rho
        taus = tau1 * (rho ** np.arange(kmax))
        mus = np.exp(-1.0 / taus)
        return mus

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
            Backcast values for each EWMA component
        """

        nobs = resids.shape[0]
        mus = self._ewma_smoothing_parameters()

        resids2 = resids ** 2.0
        backcast = np.zeros(mus.shape[0])
        for k in range(int(self.kmax)):
            mu = mus[k]
            end_point = int(max(min(np.floor(np.log(.01) / np.log(mu)), nobs), k))
            weights = mu ** np.arange(end_point)
            weights = weights / weights.sum()
            backcast[k] = weights.dot(resids2[:end_point])

        return backcast

    def backcast_transform(self, backcast):
        backcast = super(RiskMetrics2006, self).backcast_transform(backcast)
        mus = self._ewma_smoothing_parameters()
        backcast = np.asarray(backcast)
        if backcast.ndim == 0:
            backcast = backcast * np.ones(mus.shape[0])
        if backcast.shape[0] != mus.shape[0] and backcast.ndim != 0:
            raise ValueError('User backcast mut be either a scalar or an vector containing the '
                             'number of\ncomponent EWMAs in the model.')

        return backcast

    def starting_values(self, resids):
        return np.empty((0,))

    def parameter_names(self):
        return []

    def variance_bounds(self, resids, power=2.0):
        return np.ones((resids.shape[0], 1)) * np.array([-1.0, np.finfo(np.float64).max])

    def bounds(self, resids):
        return []

    def constraints(self):
        return np.empty((0, 0)), np.empty((0,))

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        nobs = resids.shape[0]
        kmax = self.kmax
        w = self._ewma_combination_weights()
        mus = self._ewma_smoothing_parameters()

        sigma2_temp = np.zeros_like(sigma2)
        for k in range(kmax):
            mu = mus[k]
            ewma_recursion(mu, resids, sigma2_temp, nobs, backcast[k])
            if k == 0:
                sigma2[:] = w[k] * sigma2_temp
            else:
                sigma2 += w[k] * sigma2_temp

        return sigma2

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        errors = rng(nobs + burn)

        kmax = self.kmax
        w = self._ewma_combination_weights()
        mus = self._ewma_smoothing_parameters()

        if initial_value is None:
            initial_value = 1.0
        sigma2s = np.zeros((nobs + burn, kmax))
        sigma2s[0, :] = initial_value
        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)
        data[0] = np.sqrt(initial_value)
        sigma2[0] = w.dot(sigma2s[0])
        for t in range(1, nobs + burn):
            sigma2s[t] = mus * sigma2s[t - 1] + (1 - mus) * data[t - 1] ** 2.0
            sigma2[t] = w.dot(sigma2s[t])
            data[t] = np.sqrt(sigma2[t]) * errors[t]

        return data[burn:], sigma2[burn:]

    def _check_forecasting_method(self, method, horizon):
        return

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        backcast = np.asarray(backcast)
        t = resids.shape[0]
        _resids = np.empty(t + 1)
        _resids[:t] = resids
        sigma2 = np.empty(t + 1)
        self.compute_variance(parameters, _resids, sigma2, backcast, var_bounds)
        sigma2.shape = (t + 1, 1)
        forecasts = sigma2[1:]
        forecasts[:start] = np.nan
        forecasts = np.tile(forecasts, (1, horizon))
        return VarianceForecast(forecasts)

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        kmax = self.kmax
        w = self._ewma_combination_weights()
        mus = self._ewma_smoothing_parameters()
        backcast = np.asarray(backcast)

        t = resids.shape[0]
        paths = np.full((t, simulations, horizon), np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)

        temp_paths = np.empty((kmax, simulations, horizon))
        # We use the transpose here to get C-contiguous arrays
        component_one_step = np.empty((kmax, t + 1))
        _resids = np.empty((t + 1))
        _resids[:-1] = resids
        for k in range(kmax):
            mu = mus[k]
            ewma_recursion(mu, _resids, component_one_step[k, :], t + 1, backcast[k])
        # Transpose to be (t+1, kmax)
        component_one_step = component_one_step.T

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            for k in range(kmax):
                temp_paths[k, :, 0] = component_one_step[i, k]
            paths[i, :, 0] = w.dot(temp_paths[:, :, 0])
            shocks[i, :, 0] = std_shocks[:, 0] * np.sqrt(paths[i, :, 0])
            for j in range(1, horizon):
                for k in range(kmax):
                    mu = mus[k]
                    temp_paths[k, :, j] = (mu * temp_paths[k, :, j - 1] +
                                           (1 - mu) * shocks[i, :, j - 1] ** 2.0)
                paths[i, :, j] = w.dot(temp_paths[:, :, j])
                shocks[i, :, j] = std_shocks[:, j] * np.sqrt(paths[i, :, j])

        return VarianceForecast(paths.mean(1), paths, shocks)


class EGARCH(VolatilityProcess):
    r"""
    EGARCH model estimation

    Parameters
    ----------
    p : int
        Order of the symmetric innovation
    o : int
        Order of the asymmetric innovation
    q : int
        Order of the lagged (transformed) conditional variance

    Attributes
    ----------
    num_params : int
        The number of parameters in the model

    Examples
    --------
    >>> from arch.univariate import EGARCH

    Symmetric EGARCH(1,1)

    >>> egarch = EGARCH(p=1, q=1)

    Standard EGARCH process

    >>> egarch = EGARCH(p=1, o=1, q=1)

    Exponential ARCH process

    >>> earch = EGARCH(p=5)

    Notes
    -----
    In this class of processes, the variance dynamics are

    .. math::

        \ln\sigma_{t}^{2}=\omega
        +\sum_{i=1}^{p}\alpha_{i}
        \left(\left|e_{t-i}\right|-\sqrt{2/\pi}\right)
        +\sum_{j=1}^{o}\gamma_{j} e_{t-j}
        +\sum_{k=1}^{q}\beta_{k}\ln\sigma_{t-k}^{2}

    where :math:`e_{t}=\epsilon_{t}/\sigma_{t}`.
    """

    def __init__(self, p=1, o=0, q=1):
        super(EGARCH, self).__init__()
        self.p = int(p)
        self.o = int(o)
        self.q = int(q)
        self.num_params = 1 + p + o + q
        if p < 0 or o < 0 or q < 0:
            raise ValueError('All lags lengths must be non-negative')
        if p == 0 and o == 0:
            raise ValueError('One of p or o must be strictly positive')
        self.name = 'EGARCH' if q > 0 else 'EARCH'
        self._arrays = None  # Helpers for fitting variance

    def __str__(self):
        descr = self.name + '('
        for k, v in (('p', self.p), ('o', self.o), ('q', self.q)):
            if v > 0:
                descr += k + ': ' + str(v) + ', '
        descr = descr[:-2] + ')'
        return descr

    def variance_bounds(self, resids, power=2.0):
        return super(EGARCH, self).variance_bounds(resids, 2.0)

    def bounds(self, resids):
        v = np.mean(resids ** 2.0)
        log_const = np.log(10000.0)
        lnv = np.log(v)
        bounds = [(lnv - log_const, lnv + log_const)]
        bounds.extend([(-np.inf, np.inf)] * (self.p + self.o))
        bounds.extend([(0.0, float(self.q))] * self.q)

        return bounds

    def constraints(self):
        p, o, q = self.p, self.o, self.q
        k_arch = p + o + q
        a = np.zeros((1, k_arch + 1))
        a[0, p + o + 1:] = -1.0
        b = np.zeros((1,))
        b[0] = -1.0
        return a, b

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        p, o, q = self.p, self.o, self.q
        nobs = resids.shape[0]
        if (self._arrays is not None) and (self._arrays[0].shape[0] == nobs):
            lnsigma2, std_resids, abs_std_resids = self._arrays
        else:
            lnsigma2 = np.empty(nobs)
            abs_std_resids = np.empty(nobs)
            std_resids = np.empty(nobs)
            self._arrays = (lnsigma2, abs_std_resids, std_resids)

        egarch_recursion(parameters, resids, sigma2, p, o, q, nobs, backcast, var_bounds,
                         lnsigma2, std_resids, abs_std_resids)

        return sigma2

    def backcast_transform(self, backcast):
        backcast = super(EGARCH, self).backcast_transform(backcast)
        return np.log(backcast)

    def backcast(self, resids):
        return np.log(super(EGARCH, self).backcast(resids))

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        p, o, q = self.p, self.o, self.q
        errors = rng(nobs + burn)

        if initial_value is None:
            if q > 0:
                beta_sum = np.sum(parameters[p + o + 1:])
            else:
                beta_sum = 0.0

            if beta_sum < 1:
                initial_value = parameters[0] / (1.0 - beta_sum)
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)
        lnsigma2 = np.zeros(nobs + burn)
        abserrors = np.abs(errors)

        norm_const = np.sqrt(2 / np.pi)
        max_lag = np.max([p, o, q])
        lnsigma2[:max_lag] = initial_value
        sigma2[:max_lag] = np.exp(initial_value)
        data[:max_lag] = errors[:max_lag] * np.sqrt(sigma2[:max_lag])

        for t in range(max_lag, nobs + burn):
            loc = 0
            lnsigma2[t] = parameters[loc]
            loc += 1
            for j in range(p):
                lnsigma2[t] += parameters[loc] * (abserrors[t - 1 - j] - norm_const)
                loc += 1
            for j in range(o):
                lnsigma2[t] += parameters[loc] * errors[t - 1 - j]
                loc += 1
            for j in range(q):
                lnsigma2[t] += parameters[loc] * lnsigma2[t - 1 - j]
                loc += 1

        sigma2 = np.exp(lnsigma2)
        data = errors * np.sqrt(sigma2)

        return data[burn:], sigma2[burn:]

    def starting_values(self, resids):
        p, o, q = self.p, self.o, self.q
        alphas = [.01, .05, .1, .2]
        gammas = [-.1, 0.0, .1]
        betas = [.5, .7, .9, .98]
        agbs = list(itertools.product(*[alphas, gammas, betas]))

        target = np.log(np.mean(resids ** 2))

        svs = []
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = np.zeros(len(agbs))
        for i, values in enumerate(agbs):
            alpha, gamma, beta = values
            sv = (1.0 - beta) * target * np.ones(p + o + q + 1)
            if p > 0:
                sv[1:1 + p] = alpha / p
            if o > 0:
                sv[1 + p:1 + p + o] = gamma / o
            if q > 0:
                sv[1 + p + o:1 + p + o + q] = beta / q
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)

        return svs[int(loc)]

    def parameter_names(self):
        return _common_names(self.p, self.o, self.q)

    def _check_forecasting_method(self, method, horizon):
        if method == 'analytic' and horizon > 1:
            raise ValueError('Analytic forecasts not available for horizon > 1')
        return

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):

        _, forecasts = self._one_step_forecast(parameters, resids, backcast, var_bounds, horizon)
        forecasts[:start] = np.nan

        return VarianceForecast(forecasts)

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        sigma2, forecasts = self._one_step_forecast(parameters, resids, backcast, var_bounds,
                                                    horizon)
        t = resids.shape[0]
        p, o, q = self.p, self.o, self.q
        m = np.max([p, o, q])

        lnsigma2 = np.log(sigma2)
        e = resids / np.sqrt(sigma2)

        lnsigma2_mat = np.full((t, m), np.log(backcast))
        e_mat = np.zeros((t, m))
        abs_e_mat = np.full((t, m), np.sqrt(2 / np.pi))

        for i in range(m):
            lnsigma2_mat[m - i - 1:, i] = lnsigma2[:(t - (m - 1) + i)]
            e_mat[m - i - 1:, i] = e[:(t - (m - 1) + i)]
            abs_e_mat[m - i - 1:, i] = np.abs(e[:(t - (m - 1) + i)])

        paths = np.full((t, simulations, horizon), np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)

        sqrt2pi = np.sqrt(2 / np.pi)
        _lnsigma2 = np.empty((simulations, m + horizon))
        _e = np.empty((simulations, m + horizon))
        _abs_e = np.empty((simulations, m + horizon))
        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            _lnsigma2[:, :m] = lnsigma2_mat[i, :]
            _e[:, :m] = e_mat[i, :]
            _e[:, m:] = std_shocks
            _abs_e[:, :m] = abs_e_mat[i, :]
            _abs_e[:, m:] = np.abs(std_shocks)
            for j in range(horizon):
                loc = 0
                _lnsigma2[:, m + j] = parameters[loc]
                loc += 1
                for k in range(p):
                    _lnsigma2[:, m + j] += parameters[loc] * (_abs_e[:, m + j - 1 - k] - sqrt2pi)
                    loc += 1

                for k in range(o):
                    _lnsigma2[:, m + j] += parameters[loc] * _e[:, m + j - 1 - k]
                    loc += 1

                for k in range(q):
                    _lnsigma2[:, m + j] += parameters[loc] * _lnsigma2[:, m + j - 1 - k]
                    loc += 1
            paths[i, :, :] = np.exp(_lnsigma2[:, m:])
            shocks[i, :, :] = np.sqrt(paths[i, :, :]) * std_shocks

        return VarianceForecast(paths.mean(1), paths, shocks)


class FixedVariance(VolatilityProcess):
    """
    Fixed volatility process

    Parameters
    ----------
    variance : {array, Series}
        Array containing the variances to use.  Should have the same shape as the data used in the
        model.
    unit_scale : bool, optional
        Flag whether to enforce a unit scale.  If False, a scale parameter will be estimated so
        that the model variance will be proportional to ``variance``.  If True, the model variance
        is set of ``variance``

    Notes
    -----
    Allows a fixed set of variances to be used when estimating a mean model, allowing GLS
    estimation.
    """

    def __init__(self, variance, unit_scale=False):
        super(FixedVariance, self).__init__()
        self.num_params = 0 if unit_scale else 1
        self._unit_scale = unit_scale
        self.name = 'Fixed Variance'
        self.name += ' (Unit Scale)' if unit_scale else ''
        self._variance_series = ensure1d(variance, 'variance', True)
        self._variance = np.asarray(variance)

    def compute_variance(self, parameters, resids, sigma2, backcast, var_bounds):
        if self._stop - self._start != sigma2.shape[0]:
            raise ValueError('start and stop do not have the correct values.')
        sigma2[:] = self._variance[self._start:self._stop]
        if not self._unit_scale:
            sigma2 *= parameters[0]
        return sigma2

    def starting_values(self, resids):
        if not self._unit_scale:
            _resids = resids / np.sqrt(self._variance[self._start:self._stop])
            return np.array([_resids.var()])
        return np.empty(0)

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        raise NotImplementedError('Fixed Variance processes do not support simulation')

    def constraints(self):
        if not self._unit_scale:
            return np.ones((1, 1)), np.zeros(1)
        else:
            return np.ones((0, 0)), np.zeros(0)

    def backcast(self, resids):
        return 1.0

    def bounds(self, resids):
        if not self._unit_scale:
            v = self.starting_values(resids)
            _resids = resids / np.sqrt(self._variance[self._start:self._stop])
            mu = _resids.mean()
            return [(v / 100000.0, 10.0 * (v + mu ** 2.0))]
        return []

    def parameter_names(self):
        if not self._unit_scale:
            return ['scale']
        return []

    def _check_forecasting_method(self, method, horizon):
        return

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        t = resids.shape[0]
        forecasts = np.full((t, horizon), np.nan)

        return VarianceForecast(forecasts)

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        t = resids.shape[0]
        forecasts = np.full((t, horizon), np.nan)
        forecast_paths = np.empty((t, simulations, horizon))
        forecast_paths.fill(np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)

        return VarianceForecast(forecasts, forecast_paths, shocks)


class FIGARCH(VolatilityProcess):
    r"""
    FIGARCH model

    Parameters
    ----------
    p : {0, 1}
        Order of the symmetric innovation
    q : {0, 1}
        Order of the lagged (transformed) conditional variance
    power : float, optional
        Power to use with the innovations, abs(e) ** power.  Default is 2.0,
        which produces FIGARCH and related models. Using 1.0 produces
        FIAVARCH and related models.  Other powers can be specified, although
        these should be strictly positive, and usually larger than 0.25.
    truncation : int, optional
        Truncation point to use in ARCH(:math:`\infty`) representation.
        Default is 1000.

    Attributes
    ----------
    num_params : int
        The number of parameters in the model

    Examples
    --------
    >>> from arch.univariate import FIGARCH

    Standard FIGARCH

    >>> figarch = FIGARCH()

    FIARCH

    >>> fiarch = FIGARCH(p=0)

    FIAVGARCH process

    >>> fiavarch = FIGARCH(power=1.0)

    Notes
    -----
    In this class of processes, the variance dynamics are

    .. math::

        h_t = \omega + [1-\beta L - \phi L  (1-L)^d] \epsilon_t^2 + \beta h_{t-1}

    where ``L`` is the lag operator and ``d`` is the fractional differencing
    parameter. The model is estimated using the ARCH(:math:`\infty`)
    representation,

    .. math::

        h_t = (1-\beta)^{-1}  \omega + \sum_{i=1}^\infty \lambda_i \epsilon_{t-i}^2

    The weights are constructed using

    .. math::

        \delta_1 = d \\
        \lambda_1 = d - \beta + \phi

    and the recursive equations

    .. math::

        \delta_j = \frac{j - 1 - d}{j}  \delta_{j-1} \\
        \lambda_j = \beta \lambda_{j-1} + \delta_j - \phi \delta_{j-1}.

    When `power` is not 2, the ARCH(:math:`\infty`) representation is still used
    where :math:`\epsilon_t^2` is replaced by :math:`|\epsilon_t|^p` and
    ``p`` is the power.
    """

    def __init__(self, p=1, q=1, power=2.0, truncation=1000):
        super(FIGARCH, self).__init__()
        self.p = int(p)
        self.q = int(q)
        self.power = power
        self.num_params = 2 + p + q
        self._truncation = int(truncation)
        if p < 0 or q < 0 or p > 1 or q > 1:
            raise ValueError('p and q must be either 0 or 1.')
        if self._truncation <= 0:
            raise ValueError('truncation must be a positive integer')
        if power <= 0.0:
            raise ValueError('power must be strictly positive, usually larger than 0.25')
        self.name = self._name()

    @property
    def truncation(self):
        """Truncation lag for the ARCH-infinity approximation"""
        return self._truncation

    def __str__(self):
        descr = self.name

        if self.power != 1.0 and self.power != 2.0:
            descr = descr[:-1] + ', '
        else:
            descr += '('
        for k, v in (('p', self.p), ('q', self.q)):
            descr += k + ': ' + str(v) + ', '
        descr = descr[:-2] + ')'

        return descr

    def variance_bounds(self, resids, power=2.0):
        return super(FIGARCH, self).variance_bounds(resids, self.power)

    def _name(self):
        q, power = self.q, self.power
        if power == 2.0:
            if q == 0:
                return 'FIARCH'
            else:
                return 'FIGARCH'
        elif power == 1.0:
            if q == 0:
                return 'FIAVARCH'
            else:
                return 'FIAVGARCH'
        else:
            if q == 0:
                return 'Power FIARCH (power: {0:0.1f})'.format(self.power)
            else:
                return 'Power FIGARCH (power: {0:0.1f})'.format(self.power)

    def bounds(self, resids):
        v = np.mean(abs(resids) ** self.power)

        bounds = [(0.0, 10.0 * v)]
        bounds.extend([(0.0, 0.5)] * self.p)  # phi
        bounds.extend([(0.0, 1.0)])  # d
        bounds.extend([(0.0, 1.0)] * self.q)  # beta

        return bounds

    def constraints(self):

        # omega > 0 <- 1
        # 0 <= d <= 1 <- 2
        # 0 <= phi <= (1 - d) / 2 <- 2
        # 0 <= beta <= d + phi <- 2
        a = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, -2, -1, 0],
                      [0, 0, 1, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1],
                      [0, 1, 1, -1]])
        b = np.array([0, 0, -1, 0, -1, 0, 0])
        if not self.q:
            a = a[:-2, :-1]
            b = b[:-2]
        if not self.p:
            # Drop column 1 and rows 1 and 2
            a = np.delete(a, (1,), axis=1)
            a = np.delete(a, (1, 2), axis=0)
            b = np.delete(b, (1, 2))

        return a, b

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        # fresids is abs(resids) ** power
        power = self.power
        fresids = np.abs(resids) ** power

        p, q, truncation = self.p, self.q, self.truncation

        nobs = resids.shape[0]
        figarch_recursion(parameters, fresids, sigma2, p, q, nobs, truncation, backcast,
                          var_bounds)
        inv_power = 2.0 / power
        sigma2 **= inv_power

        return sigma2

    def backcast_transform(self, backcast):
        backcast = super(FIGARCH, self).backcast_transform(backcast)
        return np.sqrt(backcast) ** self.power

    def backcast(self, resids):
        power = self.power
        tau = min(75, resids.shape[0])
        w = (0.94 ** np.arange(tau))
        w = w / sum(w)
        backcast = np.sum((abs(resids[:tau]) ** power) * w)

        return backcast

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        truncation = self.truncation
        p, q, power = self.p, self.q, self.power
        lam = figarch_weights(parameters[1:], p, q, truncation)
        lam_rev = lam[::-1]
        errors = rng(truncation + nobs + burn)

        if initial_value is None:
            persistence = np.sum(lam)
            beta = parameters[-1] if q else 0.0

            initial_value = parameters[0]
            if beta < 1:
                initial_value /= (1 - beta)
            if persistence < 1:
                initial_value /= (1 - persistence)
            if persistence >= 1.0 or beta >= 1.0:
                warn(initial_value_warning, InitialValueWarning)

        sigma2 = np.empty(truncation + nobs + burn)
        data = np.empty(truncation + nobs + burn)
        fsigma = np.empty(truncation + nobs + burn)
        fdata = np.empty(truncation + nobs + burn)

        fsigma[:truncation] = initial_value
        sigma2[:truncation] = initial_value ** (2.0 / power)
        data[:truncation] = np.sqrt(sigma2[:truncation]) * errors[:truncation]
        fdata[:truncation] = abs(data[:truncation]) ** power
        omega = parameters[0]
        beta = parameters[-1] if q else 0
        omega_tilde = omega / (1 - beta)
        for t in range(truncation, truncation + nobs + burn):
            fsigma[t] = omega_tilde + lam_rev.dot(fdata[t - truncation:t])
            sigma2[t] = fsigma[t] ** (2.0 / power)
            data[t] = errors[t] * np.sqrt(sigma2[t])
            fdata[t] = abs(data[t]) ** power

        return data[truncation + burn:], sigma2[truncation + burn:]

    def starting_values(self, resids):
        truncation = self.truncation
        ds = [.2, .5, .7]
        phi_ratio = [.2, .5, .8] if self.p else [0]
        beta_ratio = [.1, .5, .9] if self.q else [0]

        power = self.power
        target = np.mean(abs(resids) ** power)
        scale = np.mean(resids ** 2) / (target ** (2.0 / power))
        target *= (scale ** (power / 2))

        svs = []
        for d in ds:
            for pr in phi_ratio:
                phi = (1 - d) / 2 * pr
                for br in beta_ratio:
                    beta = (d + phi) * br
                    temp = [phi, d, beta]
                    lam = figarch_weights(np.array(temp), 1, 1, truncation)
                    omega = (1 - beta) * target * (1 - np.sum(lam))
                    svs.append((omega, phi, d, beta))
        svs = set(svs)
        svs = [list(sv) for sv in svs]
        svs = np.array(svs)
        if not self.q:
            svs = svs[:, :-1]
        if not self.p:
            svs = np.c_[svs[:, [0]], svs[:, 2:]]

        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = np.zeros(len(svs))
        for i, sv in enumerate(svs):
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)

        return svs[int(loc)]

    def parameter_names(self):
        names = ['omega']
        if self.p:
            names += ['phi']
        names += ['d']
        if self.q:
            names += ['beta']
        return names

    def _check_forecasting_method(self, method, horizon):
        if horizon == 1:
            return

        if method == 'analytic' and self.power != 2.0:
            raise ValueError('Analytic forecasts not available for horizon > 1 when power != 2')
        return

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        sigma2, forecasts = self._one_step_forecast(parameters, resids, backcast,
                                                    var_bounds, horizon)
        if horizon == 1:
            forecasts[:start] = np.nan
            return VarianceForecast(forecasts)

        truncation = self.truncation
        p, q = self.p, self.q
        lam = figarch_weights(parameters[1:], p, q, truncation)
        lam_rev = lam[::-1]
        t = resids.shape[0]
        omega = parameters[0]
        beta = parameters[-1] if q else 0.0
        omega_tilde = omega / (1 - beta)
        temp_forecasts = np.empty(truncation + horizon)
        resids2 = resids ** 2
        for i in range(start, t):
            available = i + 1 - max(0, i - truncation + 1)
            temp_forecasts[truncation - available:truncation] = resids2[
                                                                max(0, i - truncation + 1):i + 1]
            if available < truncation:
                temp_forecasts[:truncation - available] = backcast
            for h in range(horizon):
                lagged_forecasts = temp_forecasts[h:truncation + h]
                temp_forecasts[truncation + h] = omega_tilde + lam_rev.dot(lagged_forecasts)
            forecasts[i, :] = temp_forecasts[truncation:]

        forecasts[:start] = np.nan
        return VarianceForecast(forecasts)

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        sigma2, forecasts = self._one_step_forecast(parameters, resids, backcast,
                                                    var_bounds, horizon)
        t = resids.shape[0]
        paths = np.full((t, simulations, horizon), np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)

        power = self.power

        truncation = self.truncation
        p, q = self.p, self.q
        lam = figarch_weights(parameters[1:], p, q, truncation)
        lam_rev = lam[::-1]
        t = resids.shape[0]
        omega = parameters[0]
        beta = parameters[-1] if q else 0.0
        omega_tilde = omega / (1 - beta)
        fpath = np.empty((simulations, truncation + horizon))
        fresids = np.abs(resids) ** power

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            available = i + 1 - max(0, i - truncation + 1)
            fpath[:, truncation - available:truncation] = fresids[max(0, i + 1 - truncation):i + 1]
            if available < truncation:
                fpath[:, :(truncation - available)] = backcast
            for h in range(horizon):
                # 1. Forecast transformed variance
                lagged_forecasts = fpath[:, h:truncation + h]
                temp = omega_tilde + lagged_forecasts.dot(lam_rev)
                # 2. Transform variance
                sigma2 = temp ** (2.0 / power)
                # 3. Simulate new residual
                shocks[i, :, h] = std_shocks[:, h] * np.sqrt(sigma2)
                paths[i, :, h] = sigma2
                forecasts[i, h] = sigma2.mean()
                # 4. Transform new residual
                fpath[:, truncation + h] = np.abs(shocks[i, :, h]) ** power

        forecasts[:start] = np.nan
        return VarianceForecast(forecasts, paths, shocks)


class VarianceTargetingGARCH(GARCH):
    r"""
    GARCH and related model estimation

    The following models can be specified using GARCH:
        * ARCH(p)
        * GARCH(p,q)
        * GJR-GARCH(p,o,q)
        * AVARCH(p)
        * AVGARCH(p,q)
        * TARCH(p,o,q)
        * Models with arbitrary, pre-specified powers

    Parameters
    ----------
    p : int
        Order of the symmetric innovation
    o : int
        Order of the asymmetric innovation
    q: int
        Order of the lagged (transformed) conditional variance

    Attributes
    ----------
    num_params : int
        The number of parameters in the model

    Examples
    --------
    >>> from arch.univariate import VarianceTargetingGARCH

    Standard GARCH(1,1) with targeting

    >>> vt = VarianceTargetingGARCH(p=1, q=1)

    Asymmetric GJR-GARCH process with targeting

    >>> vt = VarianceTargetingGARCH(p=1, o=1, q=1)

    Notes
    -----
    In this class of processes, the variance dynamics are

    .. math::

        \sigma_{t}^{\lambda}=
        bar{\omega}(1-\sum_{i=1}^{p}\alpha_{i}
            - \frac{1}{2}\sum_{j=1}^{o}\gamma_{j}
            - \sum_{k=1}^{q}\beta_{k})
        + \sum_{i=1}^{p}\alpha_{i}\left|\epsilon_{t-i}\right|^{\lambda}
        +\sum_{j=1}^{o}\gamma_{j}\left|\epsilon_{t-j}\right|^{\lambda}
        I\left[\epsilon_{t-j}<0\right]+\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{\lambda}
    """

    def __init__(self, p=1, o=0, q=1):
        super(VarianceTargetingGARCH, self).__init__()
        self.p = int(p)
        self.o = int(o)
        self.q = int(q)
        self.num_params = p + o + q
        if p < 0 or o < 0 or q < 0:
            raise ValueError('All lags lengths must be non-negative')
        if p == 0 and o == 0:
            raise ValueError('One of p or o must be strictly positive')
        self.name = 'Variance Targeting ' + self._name()

    def bounds(self, resids):
        bounds = super(VarianceTargetingGARCH, self).bounds(resids)
        return bounds[1:]

    def constraints(self):
        a, b = super(VarianceTargetingGARCH, self).constraints()
        a = a[1:, 1:]
        b = b[1:]
        return a, b

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):

        # Add target
        target = (resids ** 2).mean()
        abar = parameters[:self.p].sum()
        gbar = parameters[self.p:self.p + self.o].sum()
        bbar = parameters[self.p + self.o:].sum()
        omega = target * (1 - abar - 0.5 * gbar - bbar)
        omega = max(omega, np.finfo(np.double).eps)
        parameters = np.r_[omega, parameters]

        fresids = np.abs(resids) ** 2.0
        sresids = np.sign(resids)

        p, o, q = self.p, self.o, self.q
        nobs = resids.shape[0]

        garch_recursion(parameters, fresids, sresids, sigma2, p, o, q, nobs,
                        backcast, var_bounds)
        return sigma2

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        if initial_value is None:
            initial_value = parameters[0]

        parameters = self._targeting_to_stangard_garch(parameters)
        return super(VarianceTargetingGARCH, self).simulate(parameters, nobs, rng, burn=burn,
                                                            initial_value=initial_value)

    def _targeting_to_stangard_garch(self, parameters):
        p, o = self.p, self.o
        abar = parameters[:p].sum()
        gbar = parameters[p:p + o].sum()
        bbar = parameters[p + o:].sum()
        const = parameters[0](1 - abar - 0.5 * gbar - bbar)
        return np.r_[const, parameters]

    def parameter_names(self):
        return _common_names(self.p, self.o, self.q)[1:]

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        parameters = self._targeting_to_stangard_garch(parameters)
        return super(VarianceTargetingGARCH, self)._analytic_forecast(parameters, resids,
                                                                      backcast, var_bounds,
                                                                      start, horizon)

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        parameters = self._targeting_to_stangard_garch(parameters)
        return super(VarianceTargetingGARCH, self)._simulation_forecast(parameters, resids,
                                                                        backcast, var_bounds,
                                                                        start, horizon,
                                                                        simulations, rng)

    def starting_values(self, resids):
        p, o, q = self.p, self.o, self.q
        alphas = [.01, .05, .1, .2]
        gammas = alphas
        abg = [.5, .7, .9, .98]
        abgs = list(itertools.product(*[alphas, gammas, abg]))

        svs = []
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = np.zeros(len(abgs))
        for i, values in enumerate(abgs):
            alpha, gamma, agb = values
            sv = np.ones(p + o + q)
            if p > 0:
                sv[:p] = alpha / p
                agb -= alpha
            if o > 0:
                sv[p: p + o] = gamma / o
                agb -= gamma / 2.0
            if q > 0:
                sv[p + o:] = agb / q
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)

        return svs[int(loc)]
