import copy
from collections import MutableMapping, Sequence

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
    def __init__(self, y=None, x=None, lags=None, constant=True, volatility=None,
                 distribution=None, hold_back=None, nvar=None):
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
        if y is not None:
            self._check_x()
            self._construct_regressors()


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
        if self._common_regressor:
            return self.nvar * self._rhs.shape[1]
        return sum(map(lambda a: a.shape[1], self._rhs))

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
        pass

    def _fit_no_arch_normal_errors(self, cov_type='robust'):
        """
        Must be overridden with closed form estimator
        """
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
        result : ARCHModelResult
            Results class containing parameter estimates, estimated parameter
            covariance and related estimates

        Notes
        -----
        See :class:`ARCHModelResult` for details on computed results
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

    def _convert_standard_var(self, params):
        const = np.zeros((self.nvar,1))
        var_params = np.zeros((self._max_lag, self.nvar, self.nvar))
        nvar = self.nvar
        if self._common_regressor:
            nreg = self._rhs.shape[1]
            params = params.copy()
            params = np.reshape(params, (nvar, nreg))
            offset = 0
            if self._constant:
                const[:] = params[:, :1]
                offset = 1
            for i, lag in enumerate(self._lags):
                var_params[i] = params[:, offset:offset + nvar]
                offset += nvar
            x_params = params[:, offset:]
        else:
            # TODO
            raise NotImplementedError
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
        const, var_params, x_params = self._convert_standard_var(mp)
        rng = self.distribution.simulate(dp)
        sim = self.volatility.simulate(vp, nobs+burn, rng, burn, initial_cov)
        resids = sim.resids
        initial_value = np.zeros((1,self.nvar)) if initial_value is None else initial_value
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


class ConstantMean(VARX):
    """
    TODO: Description
    """
    def __init__(self, y=None, volatility=None, distribution=None, hold_back=None, nvar=None):
        super(ConstantMean, self).__init__(y=y, constant=True, volatility=volatility, distribution=distribution,
                                           hold_back=hold_back, nvar=nvar)
