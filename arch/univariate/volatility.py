"""
Volatility processes for ARCH model estimation.  All volatility processes must
inherit from :class:`VolatilityProcess` and provide the same methods with the
same inputs.
"""
from __future__ import division, absolute_import

import itertools

import numpy as np
from numpy import (sqrt, ones, zeros, isscalar, sign, ones_like, arange, empty, abs, array, finfo,
                   float64, log, exp, floor)

from .distribution import Normal
from ..compat.python import add_metaclass, range
from ..utility.array import ensure1d, DocStringInheritor

try:
    from .recursions import garch_recursion, harch_recursion, egarch_recursion
except ImportError:  # pragma: no cover
    from .recursions_python import (garch_recursion, harch_recursion, egarch_recursion)

__all__ = ['GARCH', 'ARCH', 'HARCH', 'ConstantVariance', 'EWMAVariance', 'RiskMetrics2006',
           'EGARCH', 'FixedVariance']


class BootstrapRng(object):
    """
    Simple fake RNG used to transform bootstrap-based forecasting into a standard
    simulation forecasting problem

    Parameters
    ----------
    std_resid : array
        Array containing standardized residuals
    start : int
        Location of first forecast
    """
    def __init__(self, std_resid, start):
        if start <= 0 or start > std_resid.shape[0]:
            raise ValueError('start must be > 0 and <= len(std_resid).')

        self.std_resid = std_resid
        self.start = start
        self._index = start

    def rng(self):
        def _rng(size):
            if self._index >= self.std_resid.shape[0]:
                raise IndexError('not enough data points.')
            index = np.random.random_sample(size)
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
    resids : array
        Residuals to use in the recursion
    sigma2 : array
        Conditional variances with same shape as resids
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    """

    # Throw away bounds
    var_bounds = ones((nobs, 1)) * np.array([-1.0, 1.7e308])

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


@add_metaclass(DocStringInheritor)
class VolatilityProcess(object):
    """
    Abstract base class for ARCH models.  Allows the conditional mean model to be specified
    separately from the conditional variance, even though parameters are estimated jointly.
    """

    __metaclass__ = DocStringInheritor

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
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def _one_step_forecast(self, parameters, resids, backcast, var_bounds, horizon):
        """
        One-step ahead forecast

        Parameters
        ----------
        parameters : array
            Parameters required to forecast the volatility model
        resids : array
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : array
            Array containing columns of lower and upper bounds
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].

        Returns
        -------
        sigma2 : array
            t element array containing the one-step ahead forecasts
        forecsts : array
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

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        """
        Analytic multi-step volatility forecasts from the model

        Parameters
        ----------
        parameters : array
            Parameters required to forecast the volatility model
        resids : array
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : array
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

        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        """
        Simulation-based volatility forecasts from the model

        Parameters
        ----------
        parameters : array
            Parameters required to forecast the volatility model
        resids : array
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : array
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
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def _bootstrap_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                            simulations):
        """
        Simulation-based volatility forecasts using model residuals

        Parameters
        ----------
        parameters : array
            Parameters required to forecast the volatility model
        resids : array
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : array
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
        sigma2 = np.empty_like(resids)
        self.compute_variance(parameters, resids, sigma2, backcast, var_bounds)
        std_resid = resids / np.sqrt(sigma2)
        if start < self._min_bootstrap_obs:
            raise ValueError('start must include more than {0} '
                             'observations'.format(self._min_bootstrap_obs))
        rng = BootstrapRng(std_resid, start).rng()
        return self._simulation_forecast(parameters, resids, backcast, var_bounds,
                                         start, horizon, simulations, rng)

    def variance_bounds(self, resids, power=2.0):
        """
        Parameters
        ----------
        resids : array
            Approximate residuals to use to compute the lower and upper bounds on the conditional
            variance
        power : float, optional
            Power used in the model. 2.0, the default corresponds to standard ARCH models that
            evolve in squares.

        Returns
        -------
        var_bounds : array
            Array containing columns of lower and upper bounds with the same number of elements as
            resids
        """
        nobs = resids.shape[0]

        tau = min(75, nobs)
        w = 0.94 ** arange(tau)
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

        return var_bounds

    def starting_values(self, resids):
        """
        Returns starting values for the ARCH model

        Parameters
        ----------
        resids : array
            Array of (approximate) residuals to use when computing starting values

        Returns
        -------
        sv : array
            Array of starting values
        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def backcast(self, resids):
        """
        Construct values for backcasting to start the recursion

        Parameters
        ----------
        resids : array
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

    def bounds(self, resids):
        """
        Returns bounds for parameters

        Parameters
        ----------
        resids : array
            Vector of (approximate) residuals

        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        """
        Compute the variance for the ARCH model

        Parameters
        ----------
        resids : array
            Vector of mean zero residuals
        sigma2 : array
            Array with same size as resids to store the conditional variance
        backcast : float
            Value to use when initializing ARCH recursion
        var_bounds : array
            Array containing columns of lower and upper bounds
        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def constraints(self):
        """
        Construct parameter constraints arrays for parameter estimation

        Returns
        -------
        A : array
            Parameters loadings in constraint. Shape is number of constraints by number of
            parameters
        b : array
            Constraint values, one for each constraint

        Notes
        -----
        Values returned are used in constructing linear inequality constraints of the form
        A.dot(parameters) - b >= 0
        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def forecast(self, parameters, resids, backcast, var_bounds, start=None, horizon=1,
                 method='analytic', simulations=1000, rng=None):
        """
        Forecast volatility from the model

        Parameters
        ----------
        parameters : array
            Parameters required to forecast the volatility model
        resids : array
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : array, 2-d
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
                                            horizon, simulations)

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        """
        Simulate data from the model

        Parameters
        ----------
        parameters : array
            Parameters required to simulate the volatility model
        nobs : int
            Number of data points to simulate
        rng : callable
            Callable function that takes a single integer input and returns a vector of random
            numbers
        burn : int, optional
            Number of additional observations to generate when initializing the simulation
        initial_value : array, optional
            Array of initial values to use when initializing the

        Returns
        -------
        simulated_data : array
            The simulated data
        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def _gaussian_loglikelihood(self, parameters, resids, backcast,
                                var_bounds):
        """
        Private implementation of a Gaussian log-likelihood for use in constructing starting
        values or other quantities that do not depend on the distribution used by the model.
        """
        sigma2 = np.zeros_like(resids)
        self.compute_variance(parameters, resids, sigma2, backcast, var_bounds)
        return self._normal.loglikelihoood([], resids, sigma2)

    def parameter_names(self):
        """
        Names for model parameters

        Returns
        -------
         names : list (str)
            Variables names
        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover


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
        forecasts = np.empty((t, horizon))
        forecasts.fill(np.nan)

        forecasts[start:, :] = parameters[0]
        forecast_paths = None
        return VarianceForecast(forecasts, forecast_paths)

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        t = resids.shape[0]
        forecasts = np.empty((t, horizon))
        forecasts.fill(np.nan)
        forecast_paths = np.empty((t, simulations, horizon))
        forecast_paths.fill(np.nan)
        shocks = np.empty((t, simulations, horizon))
        shocks.fill(np.nan)

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
    q: int
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
        a = zeros((k_arch + 2, k_arch + 1))
        for i in range(k_arch + 1):
            a[i, i] = 1.0
        for i in range(o):
            if i < p:
                a[i + p + 1, i + 1] = 1.0

        a[k_arch + 1, 1:] = -1.0
        a[k_arch + 1, p + 1:p + o + 1] = -0.5
        b = zeros(k_arch + 2)
        b[k_arch + 1] = -1.0
        return a, b

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        # fresids is abs(resids) ** power
        # sresids is I(resids<0)
        power = self.power
        fresids = abs(resids) ** power
        sresids = sign(resids)

        p, o, q = self.p, self.o, self.q
        nobs = resids.shape[0]

        garch_recursion(parameters, fresids, sresids, sigma2, p, o, q, nobs,
                        backcast, var_bounds)
        inv_power = 2.0 / power
        sigma2 **= inv_power

        return sigma2

    def backcast(self, resids):
        """
        Construct values for backcasting to start the recursion
        """
        power = self.power
        tau = min(75, resids.shape[0])
        w = (0.94 ** arange(tau))
        w = w / sum(w)
        backcast = np.sum((abs(resids[:tau]) ** power) * w)

        return backcast

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        p, o, q, power = self.p, self.o, self.q, self.power
        errors = rng(nobs + burn)

        if initial_value is None:
            scale = ones_like(parameters)
            scale[p + 1:p + o + 1] = 0.5

            persistence = np.sum(parameters[1:] * scale[1:])
            if (1.0 - persistence) > 0:
                initial_value = parameters[0] / (1.0 - persistence)
            else:
                from warnings import warn

                warn('Parameters are not consistent with a stationary model. Using the intercept '
                     'to initialize the model.')
                initial_value = parameters[0]

        sigma2 = zeros(nobs + burn)
        data = zeros(nobs + burn)
        fsigma = zeros(nobs + burn)
        fdata = zeros(nobs + burn)

        max_lag = np.max([p, o, q])
        fsigma[:max_lag] = initial_value
        sigma2[:max_lag] = initial_value ** (2.0 / power)
        data[:max_lag] = sqrt(sigma2[:max_lag]) * errors[:max_lag]
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
            data[t] = errors[t] * sqrt(sigma2[t])
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
        target *= (scale ** power)

        svs = []
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = zeros(len(abgs))
        for i, values in enumerate(abgs):
            alpha, gamma, agb = values
            sv = (1.0 - agb) * target * ones(p + o + q + 1)
            if p > 0:
                sv[1:1 + p] = alpha / p
                agb -= alpha
            if o > 0:
                sv[1 + p:1 + p + o] = gamma / o
                agb -= gamma / 2.0
            if q > 0:
                sv[1 + p + o:1 + p + o + q] = agb / q
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast,
                                                   var_bounds)
        loc = np.argmax(llfs)

        return svs[loc]

    def parameter_names(self):
        names = ['omega']
        names.extend(['alpha[' + str(i + 1) + ']' for i in range(self.p)])
        names.extend(['gamma[' + str(i + 1) + ']' for i in range(self.o)])
        names.extend(['beta[' + str(i + 1) + ']' for i in range(self.q)])
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
        shock = np.empty_like(scaled_forecast_paths)
        shock.fill(np.nan)

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
        paths = np.zeros((t, simulations, horizon))
        shocks = np.zeros((t, simulations, horizon))

        power = self.power
        m = np.max([self.p, self.o, self.q])
        scaled_forecast_paths = zeros((simulations, m + horizon))
        scaled_shock = zeros((simulations, m + horizon))
        asym_scaled_shock = zeros((simulations, m + horizon))

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

        paths[:start] = np.nan
        shocks[:start] = np.nan
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

        \sigma^{2}=\omega + \sum_{i=1}^{m}\alpha_{l_{i}}
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
        if isscalar(lags):
            lags = arange(1, lags + 1)
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
        a = zeros((k_arch + 2, k_arch + 1))
        for i in range(k_arch + 1):
            a[i, i] = 1.0
        a[k_arch + 1, 1:] = -1.0
        b = zeros(k_arch + 2)
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
                from warnings import warn

                warn('Parameters are not consistent with a stationary model. Using the intercept '
                     'to initialize the model.')
                initial_value = parameters[0]

        sigma2 = empty(nobs + burn)
        data = empty(nobs + burn)
        max_lag = np.max(lags)
        sigma2[:max_lag] = initial_value
        data[:max_lag] = sqrt(initial_value)
        for t in range(max_lag, nobs + burn):
            sigma2[t] = parameters[0]
            for i in range(lags.shape[0]):
                param = parameters[1 + i] / lags[i]
                for j in range(lags[i]):
                    sigma2[t] += param * data[t - 1 - j] ** 2.0
            data[t] = errors[t] * sqrt(sigma2[t])

        return data[burn:], sigma2[burn:]

    def starting_values(self, resids):
        k_arch = self._num_lags

        alpha = 0.9
        sv = (1.0 - alpha) * resids.var() * ones((k_arch + 1))
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

        shocks = np.empty((t, simulations, horizon))
        shocks.fill(np.nan)

        paths = np.empty((t, simulations, horizon))
        paths.fill(np.nan)

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

        alphas = arange(.1, .95, .05)
        svs = []
        backcast = self.backcast(resids)
        llfs = alphas.copy()
        var_bounds = self.variance_bounds(resids)
        for i, alpha in enumerate(alphas):
            sv = (1.0 - alpha) * resids.var() * ones((p + 1))
            sv[1:] = alpha / p
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)
        return svs[loc]


class EWMAVariance(VolatilityProcess):
    r"""
    Exponentially Weighted Moving-Average (RiskMetrics) Variance process

    Parameters
    ----------
    lam : float, optional.
        Smoothing parameter. Default is 0.94

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

    This model has no parameters since the smoothing parameter is fixed.
    """

    def __init__(self, lam=0.94):
        super(EWMAVariance, self).__init__()
        self.lam = lam
        self.num_params = 0
        if not 0.0 < lam < 1.0:
            raise ValueError('lam must be strictly between 0 and 1')
        self.name = 'EWMA/RiskMetrics'

    def __str__(self):
        descr = self.name + '(lam: ' + '{0:0.2f}'.format(self.lam) + ')'
        return descr

    def starting_values(self, resids):
        return np.empty((0,))

    def parameter_names(self):
        return []

    def variance_bounds(self, resids, power=2.0):
        return ones((resids.shape[0], 1)) * array([-1.0, finfo(float64).max])

    def bounds(self, resids):
        return []

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        return ewma_recursion(self.lam, resids, sigma2, resids.shape[0], backcast)

    def constraints(self):
        return np.empty((0, 0)), np.empty((0,))

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        errors = rng(nobs + burn)

        if initial_value is None:
            initial_value = 1.0

        sigma2 = zeros(nobs + burn)
        data = zeros(nobs + burn)

        sigma2[0] = initial_value
        data[0] = sqrt(sigma2[0])
        lam, one_m_lam = self.lam, 1.0 - self.lam
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
        paths = np.empty((t, simulations, horizon))
        paths.fill(np.nan)
        shocks = np.empty((t, simulations, horizon))
        shocks.fill(np.nan)
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
        Length of long cycle
    tau1 : int, optional
        Length of short cycle
    kmax : int, optional
        Number of components
    rho : float, optional
        Relative scale of adjacent cycles

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

    def __init__(self, tau0=1560, tau1=4, kmax=14, rho=sqrt(2)):
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
        weights: array
            Combination weights for EWMA components
        """
        tau0, tau1, kmax, rho = self.tau0, self.tau1, self.kmax, self.rho
        taus = tau1 * (rho ** np.arange(kmax))
        w = 1 - log(taus) / log(tau0)
        w = w / w.sum()

        return w

    def _ewma_smoothing_parameters(self):
        tau1, kmax, rho = self.tau1, self.kmax, self.rho
        taus = tau1 * (rho ** np.arange(kmax))
        mus = exp(-1.0 / taus)
        return mus

    def backcast(self, resids):
        """
        Construct values for backcasting to start the recursion

        Parameters
        ----------
        resids : array
            Vector of (approximate) residuals

        Returns
        -------
        backcast : array
            Backcast values for each EWMA component
        """

        nobs = resids.shape[0]
        mus = self._ewma_smoothing_parameters()

        resids2 = resids ** 2.0
        backcast = zeros(mus.shape[0])
        for k in range(int(self.kmax)):
            mu = mus[k]
            end_point = int(max(min(floor(log(.01) / log(mu)), nobs), k))
            weights = mu ** np.arange(end_point)
            weights = weights / weights.sum()
            backcast[k] = weights.dot(resids2[:end_point])

        return backcast

    def starting_values(self, resids):
        return np.empty((0,))

    def parameter_names(self):
        return []

    def variance_bounds(self, resids, power=2.0):
        return ones((resids.shape[0], 1)) * array([-1.0, finfo(float64).max])

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
        sigma2 = zeros(nobs + burn)
        data = zeros(nobs + burn)
        data[0] = sqrt(initial_value)
        sigma2[0] = w.dot(sigma2s[0])
        for t in range(1, nobs + burn):
            sigma2s[t] = mus * sigma2s[t - 1] + (1 - mus) * data[t - 1] ** 2.0
            sigma2[t] = w.dot(sigma2s[t])
            data[t] = sqrt(sigma2[t]) * errors[t]

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

        kmax = self.kmax
        w = self._ewma_combination_weights()
        mus = self._ewma_smoothing_parameters()

        t = resids.shape[0]
        paths = np.empty((t, simulations, horizon))
        paths.fill(np.nan)
        shocks = np.empty((t, simulations, horizon))
        shocks.fill(np.nan)

        temp_paths = np.empty((kmax, simulations, horizon))
        component_one_step = np.empty((t + 1, kmax))
        _resids = np.empty((t + 1))
        _resids[:-1] = resids
        for k in range(kmax):
            mu = mus[k]
            ewma_recursion(mu, _resids, component_one_step[:, k], t + 1, backcast[k])

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            for k in range(kmax):
                temp_paths[k, :, 0] = component_one_step[i, k]
            paths[i, :, 0] = w.dot(temp_paths[:, :, 0])
            shocks[i, :, 0] = std_shocks[:, 0] * np.sqrt(paths[i, :, 0])
            for j in range(1, horizon):
                for k in range(kmax):
                    mu = mus[k]
                    temp_paths[k, :, j] = mu * temp_paths[k, :, j - 1] + \
                        (1 - mu) * shocks[i, :, j - 1] ** 2.0
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
    q: int
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
        +\sum_{j=1}^{o}\gamma_{j}\left|e_{t-j}\right|
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
        log_const = log(10000.0)
        lnv = log(v)
        bounds = [(lnv - log_const, lnv + log_const)]
        bounds.extend([(-np.inf, np.inf)] * (self.p + self.o))
        bounds.extend([(0.0, float(self.q))] * self.q)

        return bounds

    def constraints(self):
        p, o, q = self.p, self.o, self.q
        k_arch = p + o + q
        a = zeros((1, k_arch + 1))
        a[0, p + o + 1:] = -1.0
        b = zeros((1,))
        b[0] = -1.0
        return a, b

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        p, o, q = self.p, self.o, self.q
        nobs = resids.shape[0]
        if (self._arrays is not None) and (self._arrays[0].shape[0] == nobs):
            lnsigma2, std_resids, abs_std_resids = self._arrays
        else:
            lnsigma2 = empty(nobs)
            abs_std_resids = empty(nobs)
            std_resids = empty(nobs)
            self._arrays = (lnsigma2, abs_std_resids, std_resids)

        egarch_recursion(parameters, resids, sigma2, p, o, q, nobs, backcast, var_bounds,
                         lnsigma2, std_resids, abs_std_resids)

        return sigma2

    def backcast(self, resids):
        """
        Construct values for backcasting to start the recursion
        """
        return log(super(EGARCH, self).backcast(resids))

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
                from warnings import warn

                warn('Parameters are not consistent with a stationary model. Using the intercept '
                     'to initialize the model.')
                initial_value = parameters[0]

        sigma2 = zeros(nobs + burn)
        data = zeros(nobs + burn)
        lnsigma2 = zeros(nobs + burn)
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
        data = errors * sqrt(sigma2)

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
        llfs = zeros(len(agbs))
        for i, values in enumerate(agbs):
            alpha, gamma, beta = values
            sv = (1.0 - beta) * target * ones(p + o + q + 1)
            if p > 0:
                sv[1:1 + p] = alpha / p
            if o > 0:
                sv[1 + p:1 + p + o] = gamma / o
            if q > 0:
                sv[1 + p + o:1 + p + o + q] = beta / q
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)

        return svs[loc]

    def parameter_names(self):
        names = ['omega']
        names.extend(['alpha[' + str(i + 1) + ']' for i in range(self.p)])
        names.extend(['gamma[' + str(i + 1) + ']' for i in range(self.o)])
        names.extend(['beta[' + str(i + 1) + ']' for i in range(self.q)])
        return names

    def _check_forecasting_method(self, method, horizon):
        if method == 'analytic' and horizon > 1:
            raise ValueError('Analytic forecasts not available for horizon > 1 when power != 2')
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

        lnsigma2_mat = np.zeros((t, m))
        lnsigma2_mat.fill(np.log(backcast))
        e_mat = np.zeros((t, m))
        abs_e_mat = np.empty((t, m))
        abs_e_mat.fill(np.sqrt(2 / np.pi))

        for i in range(m):
            lnsigma2_mat[m - i - 1:, i] = lnsigma2[:(t - (m - 1) + i)]
            e_mat[m - i - 1:, i] = e[:(t - (m - 1) + i)]
            abs_e_mat[m - i - 1:, i] = np.abs(e[:(t - (m - 1) + i)])

        paths = np.empty((t, simulations, horizon))
        paths.fill(np.nan)
        shocks = np.empty((t, simulations, horizon))
        shocks.fill(np.nan)

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
    variance : {array, pd.Series}
        Array containing the variances to use.  Shoule have the same shape as the data used in the
        model.
    unit_scale : bool, optional
        Flag whether to enfore a unit scale.  If False, a scale parameter will be estimated so
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
        forecasts = np.empty((t, horizon))
        forecasts.fill(np.nan)

        return VarianceForecast(forecasts)

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        t = resids.shape[0]
        forecasts = np.empty((t, horizon))
        forecasts.fill(np.nan)
        forecast_paths = np.empty((t, simulations, horizon))
        forecast_paths.fill(np.nan)
        shocks = np.empty((t, simulations, horizon))
        shocks.fill(np.nan)

        return VarianceForecast(forecasts, forecast_paths, shocks)
