from collections import MutableMapping, Sequence
from collections import OrderedDict

import numpy as np
import pandas as pd
from cached_property import cached_property
from statsmodels.tsa.tsatools import lagmat
from scipy.optimize import OptimizeResult

from arch.compat.python import iteritems
from arch.multivariate.base import MultivariateARCHModel, MultivariateARCHModelResult
from arch.multivariate.data import TimeSeries
from arch.multivariate.utility import vech
from arch.utility.array import ensure1d

COV_TYPES = {'white': 'White\'s Heteroskedasticity Consistent Estimator',
             'classic_ols': 'Homoskedastic (Classic)',
             'robust': 'Bollerslev-Wooldridge (Robust) Estimator',
             'mle': 'ML Estimator'}


class ConstantMean(MultivariateARCHModel):
    def __init__(self, y=None, volatility=None, distribution=None, hold_back=None, nvar=None):
        super(ConstantMean, self).__init__(y, volatility, distribution, hold_back, nvar)
        self.constant = True

    def parameter_names(self):
        return self._generate_variable_names()

    @staticmethod
    def _static_gaussian_loglikelihood(resids):
        nobs = resids.shape[0]
        sigma = resids.dot(resids) / nobs
        _, logdet = np.linalg.slogdet(sigma)
        loglikelihood = -0.5 * nobs * np.log(2 * np.pi)
        loglikelihood -= 0.5 * nobs * logdet
        loglikelihood -= 0.5 * nobs

        return loglikelihood

    def _model_description(self, include_lags=True):
        """Generates the model description for use by __str__ and related
        functions"""
        conststr = 'yes' if self.constant else 'no'
        od = OrderedDict()
        od['constant'] = conststr
        od['volatility'] = self.volatility.__str__()
        od['distribution'] = self.distribution.__str__()
        return od

    def __str__(self):
        descr = self._model_description()
        descr_str = self.name + '('
        for key, val in iteritems(descr):
            if val:
                if key:
                    descr_str += key + ': ' + val + ', '
        descr_str = descr_str[:-2]  # Strip final ', '
        descr_str += ')'

        return descr_str

    def __repr__(self):
        txt = self.__str__()
        txt.replace('\n', '')
        return txt + ', id: ' + hex(id(self))

    def _repr_html_(self):
        """HTML representation for IPython Notebook"""
        descr = self._model_description()
        html = '<strong>' + self.name + '</strong>('
        for key, val in iteritems(descr):
            html += '<strong>' + key + ': </strong>' + val + ',\n'
        html += '<strong>ID: </strong> ' + hex(id(self)) + ')'
        return html

    def resids(self, params, y=None, regressors=None):
        y = self._fit_y if y is None else y

        return y - params[:self.num_params]

    @cached_property
    def num_params(self):
        """
        Returns the number of parameters
        """
        return self.nvar

    def simulate(self, params, nobs, burn=500, initial_value=None, x=None,
                 initial_value_vol=None):
        """
        Simulates data from a linear regression, AR or HAR models

        Parameters
        ----------
        params : ndarray
            Parameters to use when simulating the model.  Parameter order is
            [mean volatility distribution] where the parameters of the mean
            model are ordered [constant lag[0] lag[1] ... lag[p] ex[0] ...
            ex[k-1]] where lag[j] indicates the coefficient on the jth lag in
            the model and ex[j] is the coefficient on the jth exogenous
            variable.
        nobs : int
            Length of series to simulate
        burn : int, optional
            Number of values to simulate to initialize the model and remove
            dependence on initial values.
        initial_value : {ndarray, float}, optional
            Either a scalar value or `max(lags)` array set of initial values to
            use when initializing the model.  If omitted, 0.0 is used.
        x : {ndarray, DataFrame}, optional
            nobs + burn by k array of exogenous variables to include in the
            simulation.
        initial_value_vol : {ndarray, float}, optional
            An array or scalar to use when initializing the volatility process.

        Returns
        -------
        simulated_data : MultivariateSimulation
            Class with attributes:

            * `data` containing the simulated values
            * `covariance` containing the conditional covariance
            * `errors` containing the errors used in the simulation

        Examples
        --------
        >>> import numpy as np
        >>> from arch.multivariate import ConstantMean, ConstantCovariance
        >>> from arch.multivariate.utility import vech
        >>> cm = ConstantMean()
        >>> cm.volatility = ConstantCovariance()
        >>> means = np.array([1, 0.2, 0.3, 0.4])
        >>> cov = np.eye(4) + np.diag(ones(4))
        >>> cov_params = ConstantCovariance.transform_params(cov)
        >>> params = np.concatenate((means, cov_params))
        >>> sim_data = cm.simulate(params, 1000)
        """

        nvar = self.nvar
        mc = int(self.constant) * nvar
        vc = self.volatility.num_params
        dc = self.distribution.num_params
        num_params = mc + vc + dc
        params = ensure1d(params, 'params', series=False)
        if params.shape[0] != num_params:
            raise ValueError('params has the wrong number of elements. '
                             'Expected ' + str(num_params) +
                             ', got ' + str(params.shape[0]))

        dist_params = [] if dc == 0 else params[-dc:]
        vol_params = params[mc:mc + vc]
        simulator = self.distribution.simulate(dist_params)
        sim_data = self.volatility.simulate(vol_params, nobs + burn, simulator, burn,
                                            initial_value_vol)
        errors = sim_data.resids
        cov = sim_data.covariance

        y = errors + ensure1d(params[:mc], 'mean')

        ms = MultivariateSimulation(data=y[burn:], covariance=cov[burn:], errors=errors[burn:])
        return ms

    def _generate_variable_names(self):
        """Generates variable names or use in summaries"""
        return ['mu.{0}'.format(i) for i in range(self.nvar)]

    def _check_specification(self):
        """Checks the specification for obvious errors """
        pass

    def _fit_no_arch_normal_errors(self, cov_type='robust'):
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
        nobs, nvar = self._fit_y.shape

        if nobs < self.num_params:
            raise ValueError('Insufficient data, {nreg} regressors, {nobs} '
                             'data points available'.format(nreg=self.num_params, nobs=nobs))
        y = self._fit_y

        mu = y.mean(0)
        e = y - mu
        sigma = e.T.dot(e) / nobs

        params = np.hstack((mu, vech(np.linalg.cholesky(sigma))))

        return MultivariateARCHModelResult(params)

    def constraints(self):
        return np.empty((0, self.nvar)), np.empty(0)

    def bounds(self):
        mu = abs(self._y.array.mean(0))
        return [(-10 * m, 10 * m) for m in mu]

    def _r2(self, params):
        return 0.0

    def _adjust_sample(self, first_obs, last_obs):

        _first_obs_index = self._y.index_loc(first_obs, 0)
        _first_obs_index += self._hold_back
        _last_obs_index = self._y.index_loc(last_obs, self._y.shape[0])
        if _last_obs_index <= _first_obs_index:
            raise ValueError('first_obs and last_obs produce in an '
                             'empty array.')
        self._fit_indices = [_first_obs_index, _last_obs_index]
        self._fit_y = self._y.array[_first_obs_index:_last_obs_index]
        self.volatility.start, self.volatility.stop = self._fit_indices


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
        self._lhs = None
        self._rhs = None
        self._reg_names = []
        self._reformat_lags()
        self._check_x()
        self._construct_regressors()

    def _reformat_lags(self):
        """Reformat lags to the common list format"""
        if self._lags is None:
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
            for single in x:
                self._x.append(self._check_single_x(single))
        elif x is not None:
            raise TypeError('x must be one of None, a ndarray, DataFrame, dict or list.')

        return

    def _construct_regressors(self):
        lhs = []
        rhs = []
        reg_names = []
        nvar = self.nvar
        max_lag = self._max_lag
        y = self._y.array
        for i, col in enumerate(self._y.frame):
            rh, lh = lagmat(y[:, i], self._max_lag, trim='both', original='sep')
            lhs.append(lh)
            rhs.append(rh)
            reg_names.extend(['{0}.L{1}'.format(col, j) for j in range(1, self._max_lag + 1)])
        lhs = np.hstack(lhs)
        nobs = lhs.shape[0]
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
        if max_lag > 1 and len(self._lags) != max_lag:
            retain = np.zeros(rhs.shape[1], dtype=bool)
            offset = 0
            if self._constant:
                retain[0] = True
                offset = 1
            for lag in self._lags:
                retain[offset + (lag - 1) * nvar:offset + lag * nvar] = True
            reg_names = reg_names[retain]
            rhs = rhs[:, retain]
        # TODO: these are irtually identical whether global or variable-by-variable to refactor
        x_len = self._y.shape[0] - max_lag
        if self._common_regressor:
            if self._x[0] is not None:
                x = self._x[0]
                rhs = np.hstack((rhs, x.array[:x_len]))
                reg_names = reg_names.tolist() + list(x.frame.columns)
        else:
            new_rhs = []
            new_reg_names = []
            for x in self._x:
                if x is None:
                    new_rhs.append(rhs)
                    new_reg_names.append(reg_names)
                else:
                    new_rhs.append(np.hstack((rhs, x.array[:x_len])))
                    new_reg_names.append(reg_names.tolist() + list(x.frame.columns))
            rhs = new_rhs
            reg_names = new_reg_names
        self._rhs = rhs
        self._lhs = lhs
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
            nreg = _x.shape[0]
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
        return np.empty((0,self.num_params)), np.empty(0)

    def bounds(self):
        return [(-np.inf, np.inf) for _ in range(self.num_params)]

    def _r2(self, params):
        """
        Computes the model r-square.  Optional to over-ride.  Must match
        signature.
        """
        resids = self.resids(params)
        center = self._y.array.mean(0) if self._constant else 0
        tss = ((self._y.array - center)**2).sum(0)
        rss = (resids ** 2).sum(0)

        return 1.0 - rss / tss

    def _regress(self):
        x = self._fit_regressors
        y = self._fit_y
        nobs = y.shape[0]
        if self._common_regressor:
            params = np.linalg.lstsq(x,y)[0]
            resids = y - x.dot(params.T)
        else:
            params = []
            resids = np.empty_like(y)
            for i in range(self.nvar):
                _y = y[:,i:i+1]
                _x = x[i]
                single_params = np.linalg.lstsq(_x, _y)[0]
                resids[:,i] = _y - _x.dot(single_params)
                params.append(single_params)
            params = np.hstack(params)
        sigma = resids.T.dot(resids) / nobs
        return params.flatten(), sigma


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
            required_obs =self._rhs.shape[1]
        else:
            required_obs =max(map(lambda a: a.shape[1], self._rhs))
        if nobs < self.num_params:
            raise ValueError('Insufficient data, {0} regressors in largest model, {1} '
                             'data points available'.format(required_obs, nobs))
        x = self._fit_regressors
        y = self._fit_y
        nvar= self.nvar
        nobs = y.shape[0]

        # Fake convergence results, see GH #87
        opt = OptimizeResult({'status': 0, 'message': ''})

        if self.num_params == 0:
            loglikelihood = self._static_gaussian_loglikelihood(y)
            names = self._all_parameter_names()
            sigma = y.T.dot(y) / nobs
            params = vech(sigma)
            sq_resid = []
            for i in range(nvar):
                for j in range(nvar):
                    sq_resid.append(y[:,[i]]*y[:,[j]])
            sq_resid = np.hstack(sq_resid)
            resid4 = sq_resid.T @ sq_resid
            param_cov = resid4 - np.kron(sigma, sigma)
            param_cov = param_cov / nobs
            idx = vech(sigma, loc=True)
            param_cov = param_cov[np.ix_(idx,idx)]

            # TODO: This could be wrong
            cov = np.tile(sigma, (nobs, 1,1))
            # Throw away names in the case of starting values
            # TODO: What does this do?
            num_params = params.shape[0]
            if len(names) != num_params:
                names = ['p' + str(i) for i in range(num_params)]

            fit_start, fit_stop = self._fit_indices
            return MultivariateARCHModelResult(params)

        reg_params, sigma = self._regress()
        params = np.hstack((reg_params, vech(sigma)))
        nparams = params.shape[0]
        hessian = np.zeros((nparams, nparams))
        hessian[:self.num_params, :self.num_params] = -xpxi
        hessian[-1, -1] = -1
        if cov_type in ('mle',):
            param_cov = sigma2 * -hessian
            param_cov[self.num_params, self.num_params] = 2 * sigma2 ** 2.0
            param_cov /= nobs
            cov_type = COV_TYPES['classic_ols']
        elif cov_type in ('robust',):
            scores = np.zeros((nobs, self.num_params + 1))
            scores[:, :self.num_params] = x * e[:, None]
            scores[:, -1] = e ** 2.0 - sigma2
            score_cov = scores.T.dot(scores) / nobs
            param_cov = hessian.dot(score_cov).dot(hessian) / nobs
            cov_type = COV_TYPES['white']
        else:
            raise ValueError('Unknown cov_type')

        r2 = self._r2(regression_params)

        first_obs, last_obs = self._fit_indices
        resids = np.empty_like(self._y, dtype=np.float64)
        resids.fill(np.nan)
        resids[first_obs:last_obs] = e
        vol = np.zeros_like(resids)
        vol.fill(np.nan)
        vol[first_obs:last_obs] = np.sqrt(sigma2)
        names = self._all_parameter_names()
        loglikelihood = self._static_gaussian_loglikelihood(e)

        # Throw away names in the case of starting values
        num_params = params.shape[0]
        if len(names) != num_params:
            names = ['p' + str(i) for i in range(num_params)]

        fit_start, fit_stop = self._fit_indices
        return ARCHModelResult(params, param_cov, r2, resids, vol, cov_type,
                               self._y_series, names, loglikelihood,
                               self._is_pandas, opt, fit_start, fit_stop,
                               copy.deepcopy(self))


    def _adjust_sample(self, first_obs, last_obs):
        raise NotImplementedError

    def simulate(self, params, nobs, burn=500, initial_value=None, x=None,
                 initial_value_vol=None):
        raise NotImplementedError
