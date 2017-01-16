"""
Mean models to use with ARCH processes.  All mean models must inherit from
:class:`ARCHModel` and provide the same methods with the same inputs.
"""
from __future__ import division, absolute_import

import copy
from collections import OrderedDict

import numpy as np
from numpy import zeros, empty, ones, isscalar, log
from pandas import DataFrame
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tsa.tsatools import lagmat

from .base import ARCHModel, implicit_constant, ARCHModelResult, ARCHModelForecast
from .distribution import Normal, StudentsT, SkewStudent
from .volatility import ARCH, GARCH, HARCH, ConstantVariance, EGARCH
from ..compat.python import range, iteritems
from ..utility.array import ensure1d, parse_dataframe, cutoff_to_index

__all__ = ['HARX', 'ConstantMean', 'ZeroMean', 'ARX', 'arch_model', 'LS']

COV_TYPES = {'white': 'White\'s Heteroskedasticity Consistent Estimator',
             'classic_ols': 'Homoskedastic (Classic)',
             'robust': 'Bollerslev-Wooldridge (Robust) Estimator',
             'mle': 'ML Estimator'}


def _forecast_pad(count, forecasts):
    shape = list(forecasts.shape)
    shape[0] = count
    fill = np.empty(tuple(shape))
    fill.fill(np.nan)
    return np.concatenate((fill, forecasts))


def _ar_forecast(y, horizon, start_index, constant, arp):
    """

    Parameters
    ----------
    y : array
    horizon : int
    start_index : int
    constant : float
    arp : array

    Returns
    -------
    forecasts : array
    """
    t = y.shape[0]
    p = arp.shape[0]
    fcasts = np.empty((t, p + horizon))
    for i in range(p):
        fcasts[p - 1:, i] = y[i:(-p + i + 1)] if i < p - 1 else y[i:]
    for i in range(p, horizon + p):
        fcasts[:, i] = constant + fcasts[:, i-p:i].dot(arp[::-1])
    fcasts[: start_index] = np.nan

    return fcasts[:, p:]


def _ar_to_impulse(steps, params):
    p = params.shape[0]
    impulse = np.zeros(steps)
    impulse[0] = 1
    if p == 0:
        return impulse

    for i in range(1, steps):
        k = min(p - 1, i - 1)
        st = max(i - p, 0)
        impulse[i] = impulse[st:i].dot(params[k::-1])

    return impulse


class HARX(ARCHModel):
    r"""
    Heterogeneous Autoregression (HAR), with optional exogenous regressors,
    model estimation and simulation

    Parameters
    ----------
    y : array or Series
        nobs element vector containing the dependent variable
    x : array or DataFrame, optional
        nobs by k element array containing exogenous regressors
    lags : {scalar, array}, optional
        Description of lag structure of the HAR.  Scalar included all lags
        between 1 and the value.  A 1-d array includes the HAR lags 1:lags[0],
        1:lags[1], ... A 2-d array includes the HAR lags of the form
        lags[0,j]:lags[1,j] for all columns of lags.
    constant : bool, optional
        Flag whether the model should include a constant
    use_rotated : bool, optional
        Flag indicating to use the alternative rotated form of the HAR where
        HAR lags do not overlap
    hold_back : int
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
    >>> from arch.univariate import HARX
    >>> y = np.random.randn(100)
    >>> harx = HARX(y, lags=[1, 5, 22])
    >>> res = harx.fit()

    >>> from pandas import Series, date_range
    >>> index = date_range('2000-01-01', freq='M', periods=y.shape[0])
    >>> y = Series(y, name='y', index=index)
    >>> har = HARX(y, lags=[1, 6], hold_back=10)

    Notes
    -----
    The HAR-X model is described by

    .. math::

        y_t = \mu + \sum_{i=1}^p \phi_{L_{i}} \bar{y}_{t-L_{i,0}:L_{i,1}}
        + \gamma' x_t + \epsilon_t

    where :math:`\bar{y}_{t-L_{i,0}:L_{i,1}}` is the average value of
    :math:`y_t` between :math:`t-L_{i,0}` and :math:`t - L_{i,1}`.
    """

    def __init__(self, y=None, x=None, lags=None, constant=True,
                 use_rotated=False, hold_back=None, volatility=None,
                 distribution=None):
        super(HARX, self).__init__(y, hold_back=hold_back,
                                   volatility=volatility,
                                   distribution=distribution)
        self._x = x
        self._x_names = None
        self._x_index = None
        self.lags = lags
        self._lags = None
        self.constant = constant
        self.use_rotated = use_rotated
        self.regressors = None

        self.name = 'HAR'
        if self._x is not None:
            self.name += '-X'
        if lags is not None:
            max_lags = np.max(np.asarray(lags, dtype=np.int32))
        else:
            max_lags = 0
        self._max_lags = max_lags

        self._hold_back = max_lags if hold_back is None else hold_back

        if self._hold_back < max_lags:
            from warnings import warn

            warn('hold_back is less then the minimum number given the lags '
                 'selected', RuntimeWarning)
            self._hold_back = max_lags

        self._init_model()

    @property
    def x(self):
        """Gets the value of the exogenous regressors in the model"""
        return self._x

    def parameter_names(self):
        return self._generate_variable_names()

    @staticmethod
    def _static_gaussian_loglikelihood(resids):
        nobs = resids.shape[0]
        sigma2 = resids.dot(resids) / nobs

        loglikelihood = -0.5 * nobs * log(2 * np.pi)
        loglikelihood -= 0.5 * nobs * log(sigma2)
        loglikelihood -= 0.5 * nobs

        return loglikelihood

    def _model_description(self, include_lags=True):
        """Generates the model description for use by __str__ and related
        functions"""
        if include_lags:
            if self.lags is not None:
                lagstr = ['[' + str(lag[0]) + ':' + str(lag[1]) + ']'
                          for lag in self._lags.T]
                lagstr = ', '.join(lagstr)
            else:
                lagstr = 'none'
        xstr = str(self._x.shape[1]) if self._x is not None else '0'
        conststr = 'yes' if self.constant else 'no'
        od = OrderedDict()
        od['constant'] = conststr
        if include_lags:
            od['lags'] = lagstr
        od['no. of exog'] = xstr
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
            if val:
                if key:
                    html += '<strong>' + key + ': </strong>' + val + ',\n'
                else:
                    html += '<strong>' + val + '</strong>,\n'
        html += '<strong>ID: </strong> ' + hex(id(self)) + ')'
        return html

    def resids(self, params, y=None, regressors=None):
        regressors = self._fit_regressors if y is None else regressors
        y = self._fit_y if y is None else y

        return y - regressors.dot(params)

    @cache_readonly
    def num_params(self):
        """
        Returns the number of parameters
        """
        return int(self.regressors.shape[1])

    def simulate(self, params, nobs, burn=500, initial_value=None, x=None,
                 initial_value_vol=None):
        """
        Simulates data from a linear regression, AR or HAR models

        Parameters
        ----------
        params : array
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
        initial_value : array or float, optional
            Either a scalar value or `max(lags)` array set of initial values to
            use when initializing the model.  If omitted, 0.0 is used.
        x : array, optional
            nobs + burn by k array of exogenous variables to include in the
            simulation.
        initial_value_vol : array or float, optional
            An array or scalar to use when initializing the volatility process.

        Returns
        -------
        simulated_data : DataFrame
            DataFrame with columns data containing the simulated values,
            volatility, containing the conditional volatility and errors
            containing the errors used in the simulation

        Examples
        --------
        >>> import numpy as np
        >>> from arch.univariate import HARX, GARCH
        >>> harx = HARX(lags=[1, 5, 22])
        >>> harx.volatility = GARCH()
        >>> harx_params = np.array([1, 0.2, 0.3, 0.4])
        >>> garch_params = np.array([0.01, 0.07, 0.92])
        >>> params = np.concatenate((harx_params, garch_params))
        >>> sim_data = harx.simulate(params, 1000)

        Simulating models with exogenous regressors requires the regressors
        to have nobs plus burn data points

        >>> nobs = 100
        >>> burn = 200
        >>> x = np.random.randn(nobs + burn, 2)
        >>> x_params = np.array([1.0, 2.0])
        >>> params = np.concatenate((harx_params, x_params, garch_params))
        >>> sim_data = harx.simulate(params, nobs=nobs, burn=burn, x=x)
        """

        k_x = 0
        if x is not None:
            k_x = x.shape[1]
            if x.shape[0] != nobs + burn:
                raise ValueError('x must have nobs + burn rows')

        mc = int(self.constant) + self._lags.shape[1] + k_x
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
        sim_data = self.volatility.simulate(vol_params,
                                            nobs + burn,
                                            simulator,
                                            burn,
                                            initial_value_vol)
        errors = sim_data[0]
        vol = np.sqrt(sim_data[1])

        max_lag = np.max(self._lags)
        y = zeros(nobs + burn)
        if initial_value is None:
            initial_value = 0.0
        elif not isscalar(initial_value):
            initial_value = ensure1d(initial_value, 'initial_value')
            if initial_value.shape[0] != max_lag:
                raise ValueError('initial_value has the wrong shape')
        y[:max_lag] = initial_value

        for t in range(max_lag, nobs + burn):
            ind = 0
            if self.constant:
                y[t] = params[ind]
                ind += 1
            for lag in self._lags.T:
                y[t] += params[ind] * y[t - lag[1]:t - lag[0]].mean()
                ind += 1
            for i in range(k_x):
                y[t] += params[ind] * x[t, i]
            y[t] += errors[t]

        df = dict(data=y[burn:], volatility=vol[burn:], errors=errors[burn:])
        df = DataFrame(df)
        return df

    def _generate_variable_names(self):
        """Generates variable names or use in summaries"""
        variable_names = []
        lags = self._lags
        if self.constant:
            variable_names.append('Const')
        if lags is not None:
            variable_names.extend(self._generate_lag_names())
        if self._x is not None:
            variable_names.extend(self._x_names)
        return variable_names

    def _generate_lag_names(self):
        """Generates lag names.  Overridden by other models"""
        lags = self._lags
        names = []
        var_name = self._y_series.name
        if len(var_name) > 10:
            var_name = var_name[:4] + '...' + var_name[-3:]
        for i in range(lags.shape[1]):
            names.append(
                var_name + '[' + str(lags[0, i]) + ':' + str(lags[1, i]) + ']')
        return names

    def _check_specification(self):
        """Checks the specification for obvious errors """
        if self._x is not None:
            if self._x.ndim != 2 or self._x.shape[0] != self._y.shape[0]:
                raise ValueError(
                    'x must be nobs by n, where nobs is the same as '
                    'the number of elements in y')
            def_names = ['x' + str(i) for i in range(self._x.shape[1])]
            self._x_names, self._x_index = parse_dataframe(self._x, def_names)
            self._x = np.asarray(self._x)

    def _reformat_lags(self):
        """
        Reformats the input lags to a 2 by m array, which simplifies other
        operations.  Output is stored in _lags
        """
        lags = self.lags
        if lags is None:
            self._lags = None
            return
        lags = np.asarray(lags)
        if np.any(lags < 0):
            raise ValueError("Input to lags must be non-negative")

        if lags.ndim == 0:
            lags = np.arange(1, lags + 1)

        if lags.ndim == 1:
            if np.any(lags <= 0):
                raise ValueError('When using the 1-d format of lags, values '
                                 'must be positive')
            lags = np.unique(lags)
            temp = np.array([lags, lags])
            if self.use_rotated:
                temp[0, 1:] = temp[0, 0:-1]
                temp[0, 0] = 0
            else:
                temp[0, :] = 0
            self._lags = temp
        elif lags.ndim == 2:
            if lags.shape[0] != 2:
                raise ValueError('When using a 2-d array, lags must by k by 2')
            if np.any(lags[0] < 0) or np.any(lags[1] <= 0):
                raise ValueError('Incorrect values in lags')

            ind = np.lexsort(np.flipud(lags))
            lags = lags[:, ind]
            test_mat = zeros((lags.shape[1], np.max(lags)))
            for i in range(lags.shape[1]):
                test_mat[i, lags[0, i]:lags[1, i]] = 1.0
            rank = np.linalg.matrix_rank(test_mat)
            if rank != lags.shape[1]:
                raise ValueError('lags contains redundant entries')

            self._lags = lags
            if self.use_rotated:
                from warnings import warn

                warn('Rotation is not available when using the '
                     '2-d lags input format')
        else:
            raise ValueError('Incorrect format for lags')

    def _har_to_ar(self, params):
        if self._max_lags == 0:
            return params
        har = params[int(self.constant):]
        ar = np.zeros(self._max_lags)
        for value, lag in zip(har, self._lags.T):
            ar[lag[0]:lag[1]] += value / (lag[1] - lag[0])
        if self.constant:
            ar = np.concatenate((params[:1], ar))
        return ar

    def _init_model(self):
        """Should be called whenever the model is initialized or changed"""
        self._reformat_lags()
        self._check_specification()

        nobs_orig = self._y.shape[0]
        if self.constant:
            reg_constant = ones((nobs_orig, 1), dtype=np.float64)
        else:
            reg_constant = ones((nobs_orig, 0), dtype=np.float64)

        if self.lags is not None and nobs_orig > 0:
            maxlag = np.max(self.lags)
            lag_array = lagmat(self._y, maxlag)
            reg_lags = empty((nobs_orig, self._lags.shape[1]),
                             dtype=np.float64)
            for i, lags in enumerate(self._lags.T):
                reg_lags[:, i] = np.mean(lag_array[:, lags[0]:lags[1]], 1)
        else:
            reg_lags = empty((nobs_orig, 0), dtype=np.float64)

        if self._x is not None:
            reg_x = self._x
        else:
            reg_x = empty((nobs_orig, 0), dtype=np.float64)

        self.regressors = np.hstack((reg_constant, reg_lags, reg_x))

    def _r2(self, params):
        y = self._fit_y
        x = self._fit_regressors
        constant = False
        if x is not None and x.shape[1] > 0:
            constant = self.constant or implicit_constant(x)
        e = self.resids(params)
        if constant:
            y = y - np.mean(y)

        return 1.0 - e.T.dot(e) / y.dot(y)

    def _adjust_sample(self, first_obs, last_obs):
        index = self._y_series.index
        _first_obs_index = cutoff_to_index(first_obs, index, 0)
        _first_obs_index += self._hold_back
        _last_obs_index = cutoff_to_index(last_obs, index, self._y.shape[0])
        if _last_obs_index <= _first_obs_index:
            raise ValueError('first_obs and last_obs produce in an '
                             'empty array.')
        self._fit_indices = [_first_obs_index, _last_obs_index]
        self._fit_y = self._y[_first_obs_index:_last_obs_index]
        reg = self.regressors
        self._fit_regressors = reg[_first_obs_index:_last_obs_index]
        self.volatility.start, self.volatility.stop = self._fit_indices

    def _fit_no_arch_normal_errors(self, cov_type='robust'):
        """
        Estimates model parameters

        Parameters
        ----------
        cov_type : str, optional
            Covariance estimator to use when estimating parameter variances and
            covariances.  One of 'hetero' or 'heteroskedastic' for Whites's
            covariance estimator, or 'mle' for the classic
            OLS estimator appropriate for homoskedatic data.  'hetero' is the
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
        if nobs == 0:
            from warnings import warn

            warn('Cannot estimate model with no data', RuntimeWarning)
            return None

        if nobs < self.num_params:
            raise ValueError(
                'Insufficient data, ' + str(
                    self.num_params) + ' regressors, ' + str(
                    nobs) + ' data points available')
        x = self._fit_regressors
        y = self._fit_y

        # Fake convergence results, see GH #87
        _xopt = ['', '', 0, '', '']

        if x.shape[1] == 0:
            loglikelihood = self._static_gaussian_loglikelihood(y)
            names = self._all_parameter_names()
            sigma2 = y.dot(y) / nobs
            params = np.array([sigma2])
            param_cov = np.array([[np.mean(y ** 2 - sigma2) / nobs]])
            vol = np.zeros_like(y) * np.sqrt(sigma2)
            # Throw away names in the case of starting values
            num_params = params.shape[0]
            if len(names) != num_params:
                names = ['p' + str(i) for i in range(num_params)]

            fit_start, fit_stop = self._fit_indices
            return ARCHModelResult(params, param_cov, 0.0, y, vol, cov_type,
                                   self._y_series, names, loglikelihood,
                                   self._is_pandas, _xopt, fit_start, fit_stop,
                                   copy.deepcopy(self))

        regression_params = np.linalg.pinv(x).dot(y)
        xpxi = np.linalg.inv(x.T.dot(x) / nobs)
        e = y - x.dot(regression_params)
        sigma2 = e.T.dot(e) / nobs

        params = np.hstack((regression_params, sigma2))
        hessian = np.zeros((self.num_params + 1, self.num_params + 1))
        hessian[:self.num_params, :self.num_params] = -xpxi
        hessian[-1, -1] = -1
        if cov_type in ('mle',):
            param_cov = sigma2 * -hessian
            param_cov[self.num_params, self.num_params] = 2 * sigma2 ** 2.0
            param_cov /= nobs
            cov_type = COV_TYPES['classic_ols']
        elif cov_type in ('robust',):
            scores = zeros((nobs, self.num_params + 1))
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
                               self._is_pandas, _xopt, fit_start, fit_stop,
                               copy.deepcopy(self))

    def forecast(self, parameters, horizon=1, start=None, align='origin',
                 method='analytic', simulations=1000):
        if self._x is not None:
            raise RuntimeError('Forecasts are not available when the model '
                               'contains exogenous regressors.')
        # Check start
        earliest, default_start = self._fit_indices
        default_start = max(0, default_start - 1)
        start_index = cutoff_to_index(start, self._y_series.index, default_start)
        if start_index < (earliest - 1):
            raise ValueError('Due ot backcasting and/or data availability start cannot be less '
                             'than the index of the largest value in the right-hand-side '
                             'variables used to fit the first observation.  In this model, '
                             'this value is {0}.'.format(max(0, earliest - 1)))
        # Parse params
        parameters = np.asarray(parameters)
        mp, vp, dp = self._parse_parameters(parameters)

        #####################################
        # Compute residual variance forecasts
        #####################################
        # Backcast should use only the sample used in fitting
        resids = self.resids(mp)
        backcast = self._volatility.backcast(resids)
        full_resids = self.resids(mp, self._y[earliest:], self.regressors[earliest:])
        vb = self._volatility.variance_bounds(full_resids, 2.0)
        rng = self._distribution.simulate(dp)
        variance_start = max(0, start_index - earliest)
        vfcast = self._volatility.forecast(vp, full_resids, backcast, vb,
                                           start=variance_start,
                                           horizon=horizon, method=method,
                                           simulations=simulations, rng=rng)
        var_fcasts = vfcast.forecasts
        var_fcasts = _forecast_pad(earliest, var_fcasts)

        arp = self._har_to_ar(mp)
        constant = arp[0] if self.constant else 0.0
        dynp = arp[int(self.constant):]
        mean_fcast = _ar_forecast(self._y, horizon, start_index, constant, dynp)
        # Compute total variance forecasts, which depend on model
        impulse = _ar_to_impulse(horizon, dynp)
        longrun_var_fcasts = var_fcasts.copy()
        for i in range(horizon):
            lrf = var_fcasts[:, :(i + 1)].dot(impulse[i::-1] ** 2)
            longrun_var_fcasts[:, i] = lrf

        if method.lower() in ('simulation', 'bootstrap'):
            # TODO: This is not tested, and probably wrong
            variance_paths = _forecast_pad(earliest, vfcast.forecast_paths)
            long_run_variance_paths = variance_paths.copy()
            shocks = _forecast_pad(earliest, vfcast.shocks)
            for i in range(horizon):
                _impulses = impulse[i::-1][:, None]
                lrvp = variance_paths[:, :, :(i + 1)].dot(_impulses ** 2)
                long_run_variance_paths[:, :, i] = np.squeeze(lrvp)
            t, m = self._y.shape[0], self._max_lags
            mean_paths = np.empty((t, simulations, m + horizon))
            mean_paths.fill(np.nan)
            dynp_rev = dynp[::-1]
            for i in range(start_index, t):
                mean_paths[i, :, :m] = self._y[i - m + 1:i + 1]

                for j in range(horizon):
                    mean_paths[i, :, m + j] = constant + \
                                              mean_paths[i, :, j:m + j].dot(dynp_rev) + \
                                              shocks[i, :, j]
            mean_paths = mean_paths[:, :, m:]
        else:
            variance_paths = mean_paths = shocks = long_run_variance_paths = None

        index = self._y_series.index
        return ARCHModelForecast(index, mean_fcast, longrun_var_fcasts,
                                 var_fcasts, align=align,
                                 simulated_paths=mean_paths,
                                 simulated_residuals=shocks,
                                 simulated_variances=long_run_variance_paths,
                                 simulated_residual_variances=variance_paths)


class ConstantMean(HARX):
    """
    Constant mean model estimation and simulation.

    Parameters
    ----------
    y : array or Series
        nobs element vector containing the dependent variable
    hold_back : int
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
    >>> from arch.univariate import ConstantMean
    >>> y = np.random.randn(100)
    >>> cm = ConstantMean(y)
    >>> res = cm.fit()

    Notes
    -----
    The constant mean model is described by

    .. math::

        y_t = \mu + \epsilon_t

    """

    def __init__(self, y=None, hold_back=None,
                 volatility=None, distribution=None):
        super(ConstantMean, self).__init__(y, hold_back=hold_back,
                                           volatility=volatility,
                                           distribution=distribution)
        self.name = 'Constant Mean'

    def parameter_names(self):
        return ['mu']

    @cache_readonly
    def num_params(self):
        return 1

    def _model_description(self, include_lags=False):
        return super(ConstantMean, self)._model_description(include_lags)

    def simulate(self, params, nobs, burn=500, initial_value_vol=None):
        """
        Simulated data from a constant mean model

        Parameters
        ----------
        params : array
            Parameters to use when simulating the model.  Parameter order is
            [mean volatility distribution]. There is one parameter in the mean
            model, mu.
        nobs : int
            Length of series to simulate
        burn : int, optional
            Number of values to simulate to initialize the model and remove
            dependence on initial values.
        initial_value_vol : array or float, optional
            An array or scalar to use when initializing the volatility process.

        Returns
        -------
        simulated_data : DataFrame
            DataFrame with columns data containing the simulated values,
            volatility, containing the conditional volatility and errors
            containing the errors used in the simulation

        Examples
        --------
        Basic data simulation with a constant mean and volatility

        >>> import numpy as np
        >>> from arch.univariate import ConstantMean, GARCH
        >>> cm = ConstantMean()
        >>> cm.volatility = GARCH()
        >>> cm_params = np.array([1])
        >>> garch_params = np.array([0.01, 0.07, 0.92])
        >>> params = np.concatenate((cm_params, garch_params))
        >>> sim_data = cm.simulate(params, 1000)
        """
        mp, vp, dp = self._parse_parameters(params)

        sim_values = self.volatility.simulate(vp,
                                              nobs + burn,
                                              self.distribution.simulate(dp),
                                              burn,
                                              initial_value_vol)
        errors = sim_values[0]
        y = errors + mp
        vol = np.sqrt(sim_values[1])
        df = dict(data=y[burn:], volatility=vol[burn:], errors=errors[burn:])
        df = DataFrame(df)
        return df

    def resids(self, params, y=None, regressors=None):
        y = self._fit_y if y is None else y
        return y - params


class ZeroMean(HARX):
    """
    Model with zero conditional mean estimation and simulation

    Parameters
    ----------
    y : array or Series
        nobs element vector containing the dependent variable
    hold_back : int
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
    >>> from arch.univariate import ZeroMean
    >>> y = np.random.randn(100)
    >>> zm = ZeroMean(y)
    >>> res = zm.fit()

    Notes
    -----
    The zero mean model is described by

    .. math::

        y_t = \epsilon_t

    """

    def __init__(self, y=None, hold_back=None,
                 volatility=None, distribution=None):
        super(ZeroMean, self).__init__(y,
                                       x=None,
                                       constant=False,
                                       hold_back=hold_back,
                                       volatility=volatility,
                                       distribution=distribution)
        self.name = 'Zero Mean'

    def parameter_names(self):
        return []

    @cache_readonly
    def num_params(self):
        return 0

    def _model_description(self, include_lags=False):
        return super(ZeroMean, self)._model_description(include_lags)

    def simulate(self, params, nobs, burn=500, initial_value_vol=None):
        """
        Simulated data from a zero mean model

        Parameters
        ----------
        params : array
            Parameters to use when simulating the model.  Parameter order is
            [volatility distribution]. There are no mean parameters.
        nobs : int
            Length of series to simulate
        burn : int, optional
            Number of values to simulate to initialize the model and remove
            dependence on initial values.
        initial_value_vol : array or float, optional
            An array or scalar to use when initializing the volatility process.

        Returns
        -------
        simulated_data : DataFrame
            DataFrame with columns data containing the simulated values,
            volatility, containing the conditional volatility and errors
            containing the errors used in the simulation

        Examples
        --------
        Basic data simulation with no mean and constant volatility

        >>> from arch.univariate import ZeroMean
        >>> zm = ZeroMean()
        >>> sim_data = zm.simulate([1.0], 1000)

        Simulating data with a non-trivial volatility process

        >>> from arch.univariate import GARCH
        >>> zm.volatility = GARCH(p=1, o=1, q=1)
        >>> sim_data = zm.simulate([0.05, 0.1, 0.1, 0.8], 300)
        """

        mp, vp, dp = self._parse_parameters(params)

        sim_values = self.volatility.simulate(vp,
                                              nobs + burn,
                                              self.distribution.simulate(dp),
                                              burn,
                                              initial_value_vol)
        errors = sim_values[0]
        y = errors
        vol = np.sqrt(sim_values[1])
        df = dict(data=y[burn:], volatility=vol[burn:], errors=errors[burn:])
        df = DataFrame(df)

        return df

    def resids(self, params, y=None, regressors=None):
        return self._fit_y if y is None else y


class ARX(HARX):
    """
    Autoregressive model with optional exogenous regressors estimation and
    simulation

    Parameters
    ----------
    y : array or Series
        nobs element vector containing the dependent variable
    x : array or DataFrame, optional
        nobs by k element array containing exogenous regressors
    lags : scalar, 1-d array, optional
        Description of lag structure of the HAR.  Scalar included all lags
        between 1 and the value.  A 1-d array includes the AR lags lags[0],
        lags[1], ...
    constant : bool, optional
        Flag whether the model should include a constant
    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import ARX
    >>> y = np.random.randn(100)
    >>> arx = ARX(y, lags=[1, 5, 22])
    >>> res = arx.fit()

    Estimating an AR with GARCH(1,1) errors
    >>> from arch.univariate import GARCH
    >>> arx.volatility = GARCH()
    >>> res = arx.fit(update_freq=0, disp='off')

    Notes
    -----
    The AR-X model is described by

    .. math::

        y_t = \mu + \sum_{i=1}^p \phi_{L_{i}} y_{t-L_{i}} + \gamma' x_t
        + \epsilon_t

    """

    def __init__(self, y=None, x=None, lags=None, constant=True,
                 hold_back=None, volatility=None, distribution=None):
        # Convert lags to 2-d format

        if lags is not None:
            lags = np.asarray(lags)
            if lags.ndim == 0:
                if lags < 0:
                    raise ValueError('lags must be a positive integer.')
                elif lags == 0:
                    lags = None
                else:
                    lags = np.arange(1, lags + 1)
            if lags is not None:
                if lags.ndim == 1:
                    lags = np.vstack((lags, lags))
                    lags[0, :] -= 1
                else:
                    raise ValueError('lags does not follow a supported format')
        super(ARX, self).__init__(y, x, lags, constant, False,
                                  hold_back, volatility=volatility,
                                  distribution=distribution)
        self.name = 'AR'
        if self._x is not None:
            self.name += '-X'

    def _model_description(self):
        """Generates the model description for use by __str__ and related
        functions"""
        if self.lags is not None:
            lagstr = [str(lag[1]) for lag in self._lags.T]
            lagstr = ', '.join(lagstr)
        else:
            lagstr = 'none'
        xstr = str(self._x.shape[1]) if self._x is not None else '0'
        conststr = 'yes' if self.constant else 'no'
        od = OrderedDict()
        od['constant'] = conststr
        od['lags'] = lagstr
        od['no. of exog'] = xstr
        od['volatility'] = self.volatility.__str__()
        od['distribution'] = self.distribution.__str__()
        return od

    def _generate_lag_names(self):
        lags = self._lags
        names = []
        var_name = self._y_series.name
        if len(var_name) > 10:
            var_name = var_name[:4] + '...' + var_name[-3:]
        for i in range(lags.shape[1]):
            names.append(var_name + '[' + str(lags[1, i]) + ']')
        return names


class LS(HARX):
    """
    Least squares model estimation and simulation

    Parameters
    ----------
    y : array or Series
        nobs element vector containing the dependent variable
    x : array or DataFrame, optional
        nobs by k element array containing exogenous regressors
    constant : bool, optional
        Flag whether the model should include a constant
    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import LS
    >>> y = np.random.randn(100)
    >>> x = np.random.randn(100,2)
    >>> ls = LS(y, x)
    >>> res = ls.fit()

    Notes
    -----
    The AR-X model is described by

    .. math::

        y_t = \mu + \gamma' x_t + \epsilon_t

    """

    def __init__(self, y=None, x=None, constant=True, hold_back=None):
        # Convert lags to 2-d format
        super(LS, self).__init__(y, x, None, constant, False, hold_back)
        self.name = 'Least Squares'

    def _model_description(self, include_lags=False):
        return super(LS, self)._model_description(include_lags)


def arch_model(y, x=None, mean='Constant', lags=0, vol='Garch', p=1, o=0, q=1,
               power=2.0, dist='Normal', hold_back=None):
    """
    Convenience function to simplify initialization of ARCH models

    Parameters
    ----------
    y : array
        The dependent variable
    x : array, optional
        Exogenous regressors.  Ignored if model does not permit exogenous
        regressors.
    mean : str, optional
        Name of the mean model.  Currently supported options are: 'Constant',
        'Zero', 'ARX' and  'HARX'
    lags : int or list (int), optional
        Either a scalar integer value indicating lag length or a list of
        integers specifying lag locations.
    vol : str, optional
        Name of the volatility model.  Currently supported options are:
        'GARCH' (default),  "EGARCH', 'ARCH' and 'HARCH'
    p : int, optional
        Lag order of the symmetric innovation
    o : int, optional
        Lag order of the asymmetric innovation
    q : int, optional
        Lag order of lagged volatility or equivalent
    power : float, optional
        Power to use with GARCH and related models
    dist : int, optional
        Name of the error distribution.  Currently supported options are:
            'Normal' (default)
            'StudentsT'
            'SkewStudents'
    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.

    Returns
    -------
    model : ARCHModel
        Configured ARCH model

    Examples
    --------
    >>> import datetime as dt
    >>> start = dt.datetime(1990, 1, 1)
    >>> end = dt.datetime(2014, 1, 1)
    >>> import pandas_datareader.data as web
    >>> sp500 = web.get_data_yahoo('^GSPC', start=start, end=end)
    >>> returns = 100 * sp500['Adj Close'].pct_change().dropna()

    A basic GARCH(1,1) with a constant mean can be constructed using only
    the return data

    >>> from arch.univariate import arch_model
    >>> am = arch_model(returns)

    Alternative mean and volatility processes can be directly specified

    >>> am = arch_model(returns, mean='AR', lags=2, vol='harch', p=[1, 5, 22])

    This example demonstrates the construction of a zero mean process
    with a TARCH volatility process and Student t error distribution

    >>> am = arch_model(returns, mean='zero', p=1, o=1, q=1,
    ...                 power=1.0, dist='StudentsT')

    Notes
    -----
    Input that are not relevant for a particular specification, such as `lags`
    when `mean='zero'`, are silently ignored.
    """
    known_mean = ('zero', 'constant', 'harx', 'har', 'ar', 'arx', 'ls')
    known_vol = ('arch', 'garch', 'harch', 'constant', 'egarch')
    known_dist = ('normal', 'gaussian', 'studentst', 't', 'skewstudent',
                  'skewt')
    mean = mean.lower()
    vol = vol.lower()
    dist = dist.lower()
    if mean not in known_mean:
        raise ValueError('Unknown model type in mean')
    if vol.lower() not in known_vol:
        raise ValueError('Unknown model type in vol')
    if dist.lower() not in known_dist:
        raise ValueError('Unknown model type in dist')

    if mean == 'zero':
        am = ZeroMean(y, hold_back=hold_back)
    elif mean == 'constant':
        am = ConstantMean(y, hold_back=hold_back)
    elif mean == 'harx':
        am = HARX(y, x, lags, hold_back=hold_back)
    elif mean == 'har':
        am = HARX(y, None, lags, hold_back=hold_back)
    elif mean == 'arx':
        am = ARX(y, x, lags, hold_back=hold_back)
    elif mean == 'ar':
        am = ARX(y, None, lags, hold_back=hold_back)
    else:
        am = LS(y, x, hold_back=hold_back)

    if vol == 'constant':
        v = ConstantVariance()
    elif vol == 'arch':
        v = ARCH(p=p)
    elif vol == 'garch':
        v = GARCH(p=p, o=o, q=q, power=power)
    elif vol == 'egarch':
        v = EGARCH(p=p, o=o, q=q)
    else:  # vol == 'harch'
        v = HARCH(lags=p)

    if dist in ('normal', 'gaussian'):
        d = Normal()
    elif dist in ('skewstudent', 'skewt'):
        d = SkewStudent()
    elif dist in ('studentst', 't'):
        d = StudentsT()

    am.volatility = v
    am.distribution = d

    return am
