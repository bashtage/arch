"""
Volatility processes for ARCH model estimation.  All volatility processes must
inherit from :class:`VolatilityProcess` and provide the same methods with the
same inputs.
"""
from __future__ import division, absolute_import

import itertools

import numpy as np
from numpy import sqrt, ones, zeros, isscalar, sign, ones_like, arange, \
    empty, abs, array, finfo, float64, log, exp, floor

from .distribution import Normal
from ..compat.python import add_metaclass, range
from ..utility.array import ensure1d, DocStringInheritor

try:
    from .recursions import garch_recursion, harch_recursion, egarch_recursion
except ImportError:  # pragma: no cover
    from .recursions_python import (garch_recursion, harch_recursion,
                                    egarch_recursion)

__all__ = ['GARCH', 'ARCH', 'HARCH', 'ConstantVariance', 'EWMAVariance',
           'RiskMetrics2006', 'EGARCH']


def ewma_recursion(lam, resids, sigma2, nobs, backcast):
    """
    Compute variance recursion for EWMA/RiskMetrics Variance

    Parameters
    ----------
    lam : float64
        Smoothing parameter
    resids : 1-d array, float64
        Residuals to use in the recursion
    sigma2 : 1-d array, float64
        Conditional variances with same shape as resids
    nobs : int
        Length of resids
    backcast : float64
        Value to use when initializing the recursion
    """

    # Throw away bounds
    var_bounds = ones((nobs, 1)) * np.array([-1.0, 1.7e308])

    garch_recursion(np.array([0.0, 1.0 - lam, lam]), resids ** 2.0,
                    resids, sigma2, 1, 0, 1, nobs, backcast, var_bounds)
    return sigma2


class VarianceForecast(object):
    _forecasts = None
    _forecast_paths = None

    def __init__(self, forecasts, forecast_paths=None):
        self._forecasts = forecasts
        self._forecast_paths = forecast_paths

    @property
    def forecasts(self):
        return self._forecasts

    @property
    def forecast_paths(self):
        return self._forecast_paths


@add_metaclass(DocStringInheritor)
class VolatilityProcess(object):
    """
    Abstract base class for ARCH models.  Allows the conditional mean model
    to be specified separately from the conditional variance, even though
    parameters are estimated jointly.
    """

    __metaclass__ = DocStringInheritor

    def __init__(self):
        self.num_params = 0
        self.name = ''
        self._normal = Normal()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__() + ', id: ' + hex(id(self))

    def variance_bounds(self, resids, power=2.0):
        """
        Parameters
        ----------
        resids : 1-d array
            Approximate residuals to use to compute the lower and upper bounds
            on the conditional variance
        power : float, optional
            Power used in the model. 2.0, the default corresponds to standard
            ARCH models that evolve in squares.

        Returns
        var_bounds : 2-d array
            Array containing columns of lower and upper bounds with the same
            number of elements as resids
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
        resids : 1-d array
            Array of (approximate) residuals to use when computing starting
            values

        Returns
        -------
        sv : 1-d array
            Array of starting values
        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def backcast(self, resids):
        """
        Construct values for backcasting to start the recursion

        Parameters
        ----------
        resids : 1-d array
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
        resids : 1-d array
            Vector of (approximate) residuals

        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        """
        Compute the variance for the ARCH model

        Parameters
        ----------
        resids : 1-d array
            Vector of mean zero residuals
        sigma2 : 1-d array
            Array with same size as resids to store the conditional variance
        backcast : float
            Value to use when initializing ARCH recursion
        var_bounds : 2-d array
            Array containing columns of lower and upper bounds
        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def constraints(self):
        """
        Construct parameter constraints arrays for parameter estimation

        Returns
        -------
        A : 2-d array
            Parameters loadings in constraint. Shape is number of constraints
            by number of parameters
        b : 1-d array
            Constraint values, one for each constraint

        Notes
        -----
        Values returned are used in constructing linear inequality constraints
        of the form A.dot(parameters) - b >= 0

        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def forecast(self, parameters, data, first_obs=None, horizon=1,
                 method='analytic', simulations=1000):
        """
        Construct forecasts
        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        """
        Simulate data from the model

        Parameters
        ----------
        parameters : 1-d array
            Parameters required to simulate the volatility model
        nobs : int
            Number of data points to simulate
        rng : callable
            Callable function that takes a single integer input and returns a
            vector of random numbers
        burn : int, optional
            Number of additional observations to generate when initializing the
            simulation
        initial_value : 1-d array, optional
            Array of initial values to use when initializing the

        Returns
        -------
        simulated_data : 1-d array
            The simulated data
        """
        raise NotImplementedError('Must be overridden')  # pragma: no cover

    def _gaussian_loglikelihood(self, parameters, resids, backcast,
                                var_bounds):
        """
        Private implementation of a Gaussian log-likelihood for use in
        constructing starting values or other quantities that do not depend on
        the distribution used by the model.
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
    """
    Constant volatility process

    Notes
    -----
    Model has
    """

    def __init__(self):
        super(ConstantVariance, self).__init__()
        self.num_params = 1
        self.name = 'Constant Variance'

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
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

    def forecast(self, parameters, resids, first_obs=None, horizon=1,
                 method='analytic', simulations=1000, simulator=None):
        t = resids.shape
        forecasts = np.empty((t, horizon))
        forecasts.fill(np.nan)

        if method == 'analytic':
            forecasts[first_obs:, :] = parameters[0]
            forecast_paths = None
            return VarianceForecast(forecasts, forecast_paths)

        if method == 'bootstrap':
            bs_resids = resids if first_obs is None else resids[:first_obs]
            sim_resids = np.random.choice(bs_resids, (simulations, horizon))
        elif method == 'simulate':
            sim_resids = simulator(simulations * horizon)
            sim_resids = sim_resids.reshape((simulations, horizon))
        forecast_paths = sim_resids * parameters[0]
        forecast_paths = np.tile(forecast_paths, (t, 1, 1))
        forecast_paths[:first_obs, :, :] = np.nan

        forecasts = np.squeeze(forecast_paths.mean(0))
        return VarianceForecast(forecasts, forecast_paths)


class GARCH(VolatilityProcess):
    """
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
        Power to use with the innovations, abs(e) ** power.  Default is 2.0,
        which produces ARCH and related models. Using 1.0 produces AVARCH and
        related models.  Other powers can be specified, although these should
        be strictly positive, and usually larger than 0.25.

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

        \\sigma_{t}^{\\lambda}=\\omega
        +\\sum_{i=1}^{p}\\alpha_{i}\\left|\\epsilon_{t-i}\\right|^{\\lambda}
        +\\sum_{j=1}^{o}\\gamma_{j}\\left|\\epsilon_{t-j}\\right|^{\\lambda}
        I\\left[\\epsilon_{t-j}<0\\right]
        +\\sum_{k=1}^{q}\\beta_{k}\\sigma_{t-k}^{\\lambda}
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
            raise ValueError('power must be strictly positive, usually larger '
                             'than 0.25')
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

                warn('Parameters are not consistent with a stationary model. '
                     'Using the intercept to initialize the model.')
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


class HARCH(VolatilityProcess):
    """
    Heterogeneous ARCH process

    Parameters
    ----------
    lags : list or 1-d array, int or int
        List of lags to include in the model, or if scalar, includes all lags
        up the value

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

        \\sigma^{2}=\\omega
        +\\sum_{i=1}^{m}\\alpha_{l_{i}}
        \\left(l_{i}^{-1}\\sum_{j=1}^{l_{i}}\\epsilon_{t-j}^{2}\\right)

    In the common case where lags=[1,5,22], the model is

    .. math::

        \\sigma_{t}^{2}=\\omega+\\alpha_{1}\\epsilon_{t-1}^{2}
        +\\alpha_{5}
        \\left(\\frac{1}{5}\\sum_{j=1}^{5}\\epsilon_{t-j}^{2}\\right)
        +\\alpha_{22}
        \\left(\\frac{1}{22}\\sum_{j=1}^{22}\\epsilon_{t-j}^{2}\\right)

    A HARCH process is a special case of an ARCH process where parameters in
    the more general ARCH process have been restricted.

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

        harch_recursion(parameters, resids, sigma2,
                        lags, nobs, backcast, var_bounds)
        return sigma2

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        lags = self.lags
        errors = rng(nobs + burn)

        if initial_value is None:
            if (1.0 - np.sum(parameters[1:])) > 0:
                initial_value = parameters[0] / (1.0 - np.sum(parameters[1:]))
            else:
                from warnings import warn

                warn('Parameters are not consistent with a stationary model. '
                     'Using the intercept to initialize the model.')
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
        names.extend(
            ['alpha[' + str(lags[i]) + ']' for i in range(self._num_lags)])
        return names


class ARCH(GARCH):
    """
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

        \\sigma_t^{2}=\\omega+\\sum_{i=1}^{p}\\alpha_{i}\\epsilon_{t-i}^{2}

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
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast,
                                                   var_bounds)
        loc = np.argmax(llfs)
        return svs[loc]


class EWMAVariance(VolatilityProcess):
    """
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

        \\sigma_t^{2}=\\lambda\\sigma_{t-1}^2 + (1-\\lambda)\\epsilon^2_{t-1}

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
        return ewma_recursion(self.lam, resids, sigma2, resids.shape[0],
                              backcast)

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
    The variance dynamics of the model are given as a weighted average of kmax
    EWMA variance proceses where the smoothing parameters and weights are
    determined by tau0, tau1 and rho.

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
        resids : 1-d array
            Vector of (approximate) residuals

        Returns
        -------
        backcast : 1-d array
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


class EGARCH(VolatilityProcess):
    """
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

        \\ln\\sigma_{t}^{2}=\\omega
        +\\sum_{i=1}^{p}\\alpha_{i}
        \\left(\\left|e_{t-i}\\right|-\\sqrt{2/\\pi}\\right)
        +\\sum_{j=1}^{o}\\gamma_{j}\\left|e_{t-j}\\right|
        +\\sum_{k=1}^{q}\\beta_{k}\\ln\\sigma_{t-k}^{2}

    where :math:`e_{t}=\\epsilon_{t}/\\sigma_{t}`.
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

        egarch_recursion(parameters, resids, sigma2, p, o, q, nobs, backcast,
                         var_bounds, lnsigma2, std_resids, abs_std_resids)

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

                warn('Parameters are not consistent with a stationary model. '
                     'Using the intercept to initialize the model.')
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
                lnsigma2[t] += parameters[loc] * \
                               (abserrors[t - 1 - j] - norm_const)
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
