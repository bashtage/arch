import copy
from collections import MutableMapping, Sequence
from collections import OrderedDict

import numpy as np
import pandas as pd
from cached_property import cached_property
from scipy.optimize import OptimizeResult
from statsmodels.tsa.tsatools import lagmat

from arch.multivariate.base import MultivariateARCHModel, MultivariateARCHModelResult
from arch.multivariate.data import TimeSeries
from arch.multivariate.utility import vech

COV_TYPES = {'white': 'White\'s Heteroskedasticity Consistent Estimator',
             'classic_ols': 'Homoskedastic (Classic)',
             'robust': 'Bollerslev-Wooldridge (Robust) Estimator',
             'mle': 'ML Estimator'}


class MultivariateSimulation(object):
    __slots__ = ('data', 'covariance', 'errors')

    def __init__(self, data, covariance, errors):
        self.data = data
        self.covariance = covariance
        self.errors = errors


class VARX(MultivariateARCHModel):
    r"""
    Vector Autoregression (VAR), with optional exogenous regressors,
    model estimation and simulation

    Parameters
    ----------
    y : {ndarray, Series}
        nobs element vector containing the dependent variable
    x : {ndarray, DataFrame}, optional
        nobs by k element array containing exogenous regressors
    lags : {scalar, list, ndarray}, optional
        Description of lag structure of the VAR.  Scalar includes all lags
        between 1 and the value.  A 1-d array includes only the lags listed.
    constant : bool, optional
        Flag whether the model should include a constant
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model
    distribution : Distribution, optional
        Error distribution to use in the model

    Examples
    --------
    >>> import numpy as np
    >>> from arch.multivariate import VARX
    >>> y = np.random.randn((100, 2))
    >>> var = VARX(y, lags=2)
    >>> res = var.fit()

    >>> from pandas import Series, date_range
    >>> index = date_range('2000-01-01', freq='M', periods=y.shape[0])
    >>> y = DataFrame(y, columns=['y1', 'y2'], index=index)
    >>> var = VARX(y, lags=[1, 4], hold_back=10)

    Notes
    -----
    The VAR-X model is described by

    .. math::

        Y_{t} = \Phi_{0} + \Phi_{1}Y_{t-1} + \ldots + \Phi_{p}Y_{t-p}
                + \Gamma X_{t} + \epsilon_{t}

    where :math:`\Phi_0` is the constant or intercept, :math:`\Phi_0` are
    parameter arrays on the lagged values, and :math:`\Gamma` contains the
    coefficient on the exogenous regressors, if any.
    """

    def __init__(self, y=None, x=None, lags=None, constant=True, hold_back=0, volatility=None,
                 distribution=None, nvar=None):
        super(VARX, self).__init__(y, volatility, distribution, hold_back, nvar)
        self._x = x
        self._lags = lags
        self._constant = constant
        self._max_lag = 0
        self._common_regressor = False
        self._rhs = None
        self._reg_names = []
        self._reformat_lags()
        self._hold_back = max(self._hold_back, self._max_lag)
        self._has_exogenous = False
        self._check_x()
        if y is not None:
            self._construct_regressors()

    @property
    def constant(self):
        """Lags included in the model"""
        return self._constant

    @property
    def lags(self):
        """Lags included in the model"""
        return self._lags

    def _reformat_lags(self):
        """Reformat lags to the common list format"""
        if self._lags is None:
            self._max_lag = 0
            self._lags = []
            return
        lags = self._lags
        if isinstance(lags, int) or np.isscalar(lags):
            lags = int(lags)
            if lags <= 0:
                raise ValueError('lags must be a positive scalar integer')
            lags = self._lags = list(range(1, lags + 1))
        else:
            lags = sorted(list(map(int, lags)))
            if min(lags) <= 0 or len(set(lags)) != len(lags):
                raise ValueError('lags must contain non-negative integers without repetition')
            self._lags = lags
        self._max_lag = max(lags)
        return

    def _check_single_x(self, x, key=None):
        """Check that a single x array confirms to requirements"""
        if key is not None and key not in self._y.frame:
            raise KeyError('Variable name {0} in x is not in y'.format(key))
        if x is not None and not isinstance(x, (np.ndarray, pd.DataFrame)):
            raise TypeError('The values in x must be DataFrames or ndarrays')
        if x is None:
            return None
        x = TimeSeries(x, name='x')
        if x.shape[0] != self._y.shape[0]:
            raise ValueError('x must have the same number of observations as y')
        return x

    def _check_x(self):
        """
        Check entire x input confirms to requirements.

        Notes
        -----
        Reformats to be a list corresponding to the columns of y
        """
        x = self._x
        self._x = [None] * self.nvar
        self._common_regressor = x is None
        if isinstance(x, (np.ndarray, pd.DataFrame)):
            tsx = self._check_single_x(x)
            self._x = [tsx] * self.nvar
            self._common_regressor = True
        elif isinstance(x, MutableMapping):
            keys = list(self._y.frame.columns)
            for key in x:
                self._x[keys.index(key)] = self._check_single_x(x[key], key)
        elif isinstance(x, Sequence):
            if len(x) != self._y.shape[1]:
                raise ValueError
            for i, single in enumerate(x):
                self._x[i] = self._check_single_x(single)
        elif x is not None:
            raise TypeError('x must be one of None, a ndarray, DataFrame, dict or list.')
        self._has_exogenous = any(map(lambda v: v is not None, self._x))
        return

    @staticmethod
    def _update_regressors(rhs, reg_names, x):
        if x is None:
            return rhs, reg_names
        rhs = np.hstack((rhs, x.array))
        reg_names = list(reg_names) + list(x.frame.columns)
        return rhs, reg_names

    def _construct_regressors(self):
        rhs = []
        reg_names = []
        nobs, nvar = self._y.shape
        max_lag = self._max_lag
        y = self._y.array
        for i, col in enumerate(self._y.frame):
            rhs.append(lagmat(y[:, i], self._max_lag))
            reg_names.extend(['{0}.L{1}'.format(col, j) for j in range(1, self._max_lag + 1)])

        if self._constant:
            rhs.insert(0, np.ones((nobs, 1)))
            reg_names.insert(0, 'Const')
        rhs = np.hstack(rhs)
        reg_names = np.array(reg_names)
        if max_lag > 1:
            reorder = np.arange(0, nvar * max_lag, max_lag)
            reorder = reorder + np.arange(max_lag)[:, None]
            reorder = reorder.ravel()
            if self._constant:
                reorder = np.hstack((0, reorder + 1))
            rhs = rhs[:, reorder]
            reg_names = reg_names[reorder]
        if 1 < max_lag != len(self._lags):
            retain = np.zeros(rhs.shape[1], dtype=bool)
            offset = 0
            if self._constant:
                retain[0] = True
                offset = 1
            for lag in self._lags:
                retain[offset + (lag - 1) * nvar:offset + lag * nvar] = True
            reg_names = reg_names[retain]
            rhs = rhs[:, retain]
        if self._common_regressor:
            rhs, reg_names = self._update_regressors(rhs, reg_names, self._x[0])
        else:
            _rhs = []
            _reg_names = []
            for x in self._x:
                upd_rhs, upd_reg_names = self._update_regressors(rhs, reg_names, x)
                _rhs.append(upd_rhs)
                _reg_names.append(upd_reg_names)
            rhs = _rhs
            reg_names = _reg_names
        self._rhs = rhs
        self._reg_names = reg_names

    def resids(self, params, y=None, regressors=None):
        y = self._fit_y if y is None else y
        x = self._fit_regressors if regressors is None else regressors
        if self._common_regressor:
            nreg = x.shape[1]
            reg_params = params[:nreg * self.nvar]
            reg_params = reg_params.reshape((self.nvar, nreg))
            return y - x.dot(reg_params.T)
        resids = np.empty_like(y)
        loc = 0
        for i, _x in enumerate(x):
            nreg = _x.shape[1]
            reg_params = params[loc:loc + nreg]
            resids[:, i] = y[:, i] - _x.dot(reg_params)
        return resids

    def parameter_names(self):
        """List of parameters names

        Returns
        -------
        names : list (str)
            List of variable names for the mean model
        """
        names = []
        if self._common_regressor:
            for c in self._y.frame:
                names.extend(['{0}.{1}'.format(c, name) for name in self._reg_names])
            return names
        for c, reg_names in zip(self._y.frame, self._reg_names):
            names.extend(['{0}.{1}'.format(c, name) for name in reg_names])
        return names

    @cached_property
    def num_params(self):
        """
        Number of parameters in the model
        """
        count = (self._constant + len(self._lags) * self.nvar) * self.nvar
        if self._common_regressor and self._x[0] is not None:
            count += self.nvar * self._x[0].shape[1]
        else:
            count += sum([x.shape[1] for x in self._x if x is not None])
        return count

    def constraints(self):
        return np.empty((0, self.num_params)), np.empty(0)

    def bounds(self):
        return [(-np.inf, np.inf) for _ in range(self.num_params)]

    def _r2(self, params):
        """
        Computes the model r-square.  Optional to over-ride.  Must match
        signature.
        """
        resids = self.resids(params)
        center = self._y.array.mean(0) if self._constant else 0
        tss = ((self._y.array - center) ** 2).sum(0)
        rss = (resids ** 2).sum(0)

        return 1.0 - rss / tss

    @staticmethod
    def _empty_lstsq(x, y):
        if not x.shape[1]:
            return np.empty(0), 0.0
        params = np.linalg.lstsq(x, y, rcond=None)[0]
        fitted = x.dot(params)
        return params, fitted

    def _regress(self):
        x = self._fit_regressors
        y = self._fit_y
        nobs = y.shape[0]
        if self._common_regressor:
            params, fitted = self._empty_lstsq(x, y)
            resids = y - fitted
        else:
            params = []
            resids = np.empty_like(y)
            for i in range(self.nvar):
                _y = y[:, i:i + 1]
                _x = x[i]
                _params, fitted = self._empty_lstsq(_x, _y)
                resids[:, i:i + 1] = _y - fitted
                params.append(_params)
            params = np.hstack(params)
        sigma = resids.T.dot(resids) / nobs
        return params.flatten(), sigma, resids

    def _normal_mle_cov(self, rhs, nobs, sigma):
        nvar = self.nvar
        if self._common_regressor:
            rhs = [rhs] * nvar
        cov_dim = self.num_params + (nvar * (nvar + 1)) // 2
        a_inv = np.zeros((self.num_params, self.num_params))
        b = np.zeros((self.num_params, self.num_params))
        loc = 0
        for i in range(nvar):
            row = [rhs[i].T.dot(rhs[j]) * sigma[i, j] for j in range(nvar)]
            row = np.hstack(row)
            dim = row.shape[0]
            b[loc:loc + dim, :] = row
            a_inv[loc:loc + dim, loc:loc + dim] = np.linalg.inv(rhs[i].T.dot(rhs[i]))
            loc += dim
        cov = np.zeros((cov_dim, cov_dim))
        cov[:loc, :loc] = np.linalg.multi_dot((a_inv, b, a_inv))
        sigma_cov = 2 * np.kron(sigma, sigma) / nobs
        locs = vech(sigma, loc=True)
        sigma_cov = sigma_cov[np.ix_(locs, locs)]
        cov[loc:, loc:] = sigma_cov

        return cov

    def _normal_mle_robust_cov(self):
        raise NotImplementedError

    def _fit_no_arch_normal_errors(self, cov_type='mle'):
        """
        Estimates model parameters

        Parameters
        ----------
        cov_type : str, optional
            Covariance estimator to use when estimating parameter variances and
            covariances.  One of 'hetero' or 'heteroskedastic' for Whites's
            covariance estimator, or 'mle' for the classic
            OLS estimator appropriate for homoskedastic data.  'hetero' is the
            the default.

        Returns
        -------
        result : MultivariateARCHModelResult
            Results class containing parameter estimates, estimated parameter
            covariance and related estimates

        Notes
        -----
        See :class:`MultivariateARCHModelResult` for details on computed results
        """
        nobs = self._fit_y.shape[0]
        if self._common_regressor:
            required_obs = self._rhs.shape[1]
        else:
            required_obs = max(map(lambda a: a.shape[1], self._rhs))
        if nobs < required_obs:
            raise ValueError('Insufficient data, {0} regressors in largest model, {1} '
                             'data points available'.format(required_obs, nobs))
        x = self._fit_regressors
        y = self._fit_y
        nvar = self.nvar
        nobs = y.shape[0]

        # Fake convergence results, see GH #87
        opt = OptimizeResult({'status': 0, 'message': ''})

        reg_params, sigma, reg_resids = self._regress()
        params = np.hstack((reg_params, vech(sigma)))
        if cov_type in ('mle',):
            param_cov = self._normal_mle_cov(x, nobs, sigma)
            cov_type = COV_TYPES['classic_ols']
        elif cov_type in ('robust',):
            self._normal_mle_robust_cov()
            raise NotImplementedError
        else:
            raise ValueError('Unknown cov_type')

        r2 = self._r2(reg_params)

        first_obs, last_obs = self._fit_indices
        resids = np.full_like(self._y.array, np.nan)
        resids[first_obs:last_obs] = reg_resids
        cov = np.full((nobs, nvar, nvar), np.nan)
        cov[first_obs:last_obs] = sigma
        names = self._all_parameter_names()
        loglikelihood = self._static_gaussian_loglikelihood(reg_resids)

        # Throw away names in the case of starting values
        num_params = params.shape[0]
        if len(names) != num_params:
            names = ['p' + str(i) for i in range(num_params)]

        fit_start, fit_stop = self._fit_indices

        return MultivariateARCHModelResult(params, param_cov, r2, resids, cov, cov_type,
                                           self._y.frame, names, loglikelihood,
                                           self._y.pandas_input,  opt, fit_start, fit_stop,
                                           copy.deepcopy(self))

    def _adjust_sample(self, first_obs, last_obs):

        first_obs_index = self._y.index_loc(first_obs, 0)
        first_obs_index += self._hold_back
        last_obs_index = self._y.index_loc(last_obs, self._y.shape[0])
        if last_obs_index <= first_obs_index:
            raise ValueError('first_obs and last_obs produce in an '
                             'empty array.')
        self._fit_indices = [first_obs_index, last_obs_index]
        self._fit_y = self._y.array[first_obs_index:last_obs_index]
        if self._common_regressor:
            self._fit_regressors = self._rhs[first_obs_index:last_obs_index]
        else:
            self._fit_regressors = [rhs[first_obs_index:last_obs_index] for rhs in self._rhs]
        self.volatility.start, self.volatility.stop = self._fit_indices

    def _convert_standard_var(self, params, x=None):
        const = np.zeros((self.nvar, 1))
        var_params = np.zeros((self._max_lag, self.nvar, self.nvar))
        nvar = self.nvar
        offset = 0
        x_params = []
        var_count = len(self._lags) * nvar
        for i in range(self.nvar):
            if self._constant:
                const[i] = params[offset]
                offset += 1
            var_params[:, i].flat = params[offset:offset + var_count]
            offset += var_count
            if x is not None:
                if isinstance(x, (pd.DataFrame, np.ndarray)):
                    _x = x
                else:
                    _x = x[i]
                x_count = 0 if _x is None else _x.shape[1]
                x_params.append(params[offset:offset + x_count])
                offset += x_count
        return const, var_params, x_params

    def simulate(self, params, nobs, burn=500, initial_value=None, x=None,
                 initial_cov=None):
        if x is not None:
            if not isinstance(x, (np.ndarray, pd.DataFrame)):
                raise TypeError('x must be a ndarray or a DataFrame')
            if x.shape[0] != nobs + burn:
                raise ValueError('x must have nobs + burn observations')
        nvar = self.nvar
        mp, vp, dp = self._parse_parameters(params)
        const, var_params, x_params = self._convert_standard_var(mp, x)
        rng = self.distribution.simulate(dp)
        sim = self.volatility.simulate(vp, nobs+burn, rng, burn, initial_cov)
        resids = sim.resids
        initial_value = np.zeros((1, self.nvar)) if initial_value is None else initial_value
        data = np.zeros((nobs + burn, nvar))
        for i in range(nobs + burn):
            data[i:i + 1] = const.T
            for lag in self._lags:
                if (i - lag) < 0:
                    lagged_data = initial_value
                else:
                    lagged_data = data[i - lag:i - lag + 1]
                data[i:i + 1] += (var_params[lag-1].dot(lagged_data.T)).T
            if x is not None:
                data[:, i:i + 1] += x_params.dot(x[i:i + 1, :].T)
            data[i:i + 1] += resids[i:i + 1]
        data = data[burn:]
        cov = sim.covariance[burn:]
        resids = resids[burn:]
        return MultivariateSimulation(data, cov, resids)

    def _model_description(self, include_lags=True):
        """Generates the model description for use by __str__ and related
        functions"""
        lagstr = 'none'
        if include_lags and self.lags:
            lagstr = ', '.join(['{0}'.format(lag) for lag in self._lags])
        od = OrderedDict()
        od['constant'] = 'yes' if self.constant else 'no'
        if include_lags:
            od['lags'] = lagstr
        od['exogenous'] = 'yes' if self._has_exogenous else 'no'
        if self._has_exogenous:
            details = 'common' if self._common_regressor else 'heterogeneous'
            od['exogenous structure'] = details
            exog_count = list(map(lambda a: 0 if a is None else a.shape[1], self._x))
            min_count = min(exog_count)
            max_count = max(exog_count)
            if self._common_regressor:
                xstr = str(min_count)
            else:
                xstr = '{0}-{1}'.format(min_count, max_count)
            od['no. of exog'] = xstr
        od['volatility'] = self.volatility.__str__()
        od['distribution'] = self.distribution.__str__()
        return od


class ConstantMean(VARX):
    """
    Constant mean model

    Parameters
    ----------
    y : {ndarray, Series}
        nobs element vector containing the dependent variable
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model
    distribution : Distribution, optional
        Error distribution to use in the model

    Examples
    --------
    >>> import numpy as np
    >>> from arch.multivariate import ConstantMean
    >>> y = np.random.randn((100, 2))
    >>> cm = ConstantMean(y)

    Notes
    -----
    The constant mean model is described by

    .. math::

        Y_{t} = \mu + \epsilon_{t}

    The constant mean model is a special case of the VAR-X where all
    coefficients except the intercept are restricted to have 0 coefficients.

    """
    def __init__(self, y=None, volatility=None, distribution=None, hold_back=None, nvar=None):
        super(ConstantMean, self).__init__(y=y, constant=True, volatility=volatility,
                                           distribution=distribution, hold_back=hold_back,
                                           nvar=nvar)

    def simulate(self, params, nobs, burn=500, initial_value=None, initial_cov=None):
        return super(ConstantMean, self).simulate(params, nobs, 500, x=None,
                                                  initial_value=initial_value,
                                                  initial_cov=initial_cov)


class ZeroMean(VARX):
    """
    Zero mean model

    Parameters
    ----------
    y : {ndarray, Series}
        nobs element vector containing the dependent variable
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model
    distribution : Distribution, optional
        Error distribution to use in the model

    Examples
    --------
    >>> import numpy as np
    >>> from arch.multivariate import ZeroMean
    >>> y = np.random.randn((100, 2))
    >>> zm = ZeroMean(y)

    Notes
    -----
    The constant mean model is described by

    .. math::

        Y_{t} = \epsilon_{t}

    The zero mean model is a special case of the VAR-X where all
    coefficients are restricted to have 0 coefficients.

    This model is typically used when the input `y` are residuals produced by
    filtering the original data through a time series model.
    """

    def __init__(self, y=None, volatility=None, distribution=None, hold_back=None, nvar=None):
        super(ZeroMean, self).__init__(y=y, constant=False, volatility=volatility,
                                       distribution=distribution, hold_back=hold_back,
                                       nvar=nvar)

    def simulate(self, params, nobs, burn=500, initial_value=None, initial_cov=None):
        return super(ZeroMean, self).simulate(params, nobs, 500, x=None,
                                              initial_value=initial_value,
                                              initial_cov=initial_cov)


class VARZeroSetter(object):
    """
    Utility to set equation-variable-lag combination in a VAR

    Parameters
    ----------
    nvar : int
        Number of variables in the VAR
    lags : List[int]
        List of lags included in the model
    variable_names : List[str], optional.
        List of variable names in the model.If not provided the default
        values y0, y1, ... are used.

    Examples
    --------
    >>> vz = VARZeroSetter(3, [1, 3], ['gdp', 'inflation', 'int_rate'])
    >>> vz.diagonalize()
    >>> vz.include(1, 'gdp', 'inflation')
    >>> vz.include(1, 'gdp', 'int_rate')
    >>> print(vz)
    VARZeroSetter
       Lag: 1
    Variable   gdp inflation int_rate
    Eq.
    gdp         +         +        +
    inflation  ---        +       ---
    int_rate   ---       ---       +
       Lag: 3
    Variable   gdp inflation int_rate
    Eq.
    gdp         +        ---      ---
    inflation  ---        +       ---
    int_rate   ---       ---       +
    """

    def __init__(self, nvar, lags, variable_names=None):
        self._max_lag = max(lags) if lags else 0
        self._lags = lags
        self._np_lags = np.array(lags) - 1
        self._nvar = nvar
        self._ind = np.ones((self._max_lag, nvar, nvar), dtype=bool)
        self._int_index = pd.Index(np.arange(nvar))
        if variable_names is not None:
            self._index = pd.Index(variable_names)
        else:
            self._index = pd.Index(['y{0}'.format(i) for i in range(nvar)])

    def _loc(self, value):
        try:
            return self._index.get_loc(value)
        except KeyError:
            return self._int_index.get_loc(value)

    def exclude(self, lag, equation, variable):
        """
        Exclude a lag-equation-variable combination

        Parameters
        ----------
        lag : int
            Lag for the exclusion
        equation : {str, int}
            Equation to apply the exclusion to.  If a string, uses variable
            names to identify the equation index.  If an integer, uses 0-based
            indexing.
        variable : {str, int}
            Variable to exclude from equation at lag. If a string, uses variable
            names to identify the variable index.  If an integer, uses 0-based
            indexing.
        """
        if not np.isin(lag - 1, self._np_lags):
            raise KeyError('Lag {0} is not included in the model.'.format(lag))
        self._ind[lag - 1, self._loc(equation), self._loc(variable)] = False

    def include(self, lag, equation, variable):
        """
        Include a lag-equation-variable combination

        Parameters
        ----------
        lag : int
            Lag for the inclusion
        equation : {str, int}
            Equation to apply the inclusion to.  If a string, uses variable
            names to identify the equation index.  If an integer, uses 0-based
            indexing.
        variable : {str, int}
            Variable to include in equation at lag. If a string, uses variable
            names to identify the variable index.  If an integer, uses 0-based
            indexing.
        """
        self._ind[lag - 1, self._loc(equation), self._loc(variable)] = True

    def diagonalize(self):
        """
        Set the selection to only include own-lags
        """
        for i in range(self._max_lag):
            self._ind[i] = np.eye(self._nvar)

    def exclude_all(self):
        """
        Exclude all variables from all equations at all lags.
        """
        self._ind[:, :, :] = False

    def include_all(self):
        """
        Include all variables from all equations at all lags.
        """
        self._ind[:, :, :] = True

    @property
    def flat(self):
        """
        Return the indices as a flattened 2-d array

        Returns
        -------
        ind : ndarray
            Array with shape (nvar, nvar * nlags) where the columns are
            blocked in groups of size nvar so that block i corresponds
            to lags[i].
        """
        ind = self._ind[self._np_lags]
        return np.hstack([i for i in ind])

    @property
    def stacked(self):
        """
        Return the indices as a stacked 3-d array

        Returns
        -------
        ind : ndarray
            Array with shape (nlags, nvar, nvar) where panel i contains
            the selected elements for lags[i].
        """
        return self._ind[self._np_lags]

    def __str__(self):
        idx = self._index.copy()
        idx.name = 'Eq.'
        col = idx.copy()
        col.name = 'Variable'
        out = self.__class__.__name__ + '\n'
        for lag in self._lags:
            out += '   Lag: {lag}\n'.format(lag=lag)
            df = pd.DataFrame(self._ind[lag - 1], index=idx, columns=col, dtype='object')
            out += str(df.applymap(lambda v: ' + ' if v else '---'))
            out += '\n'

        return out

    def __repr__(self):
        return self.__str__() + 'id: ' + hex(id(self))
