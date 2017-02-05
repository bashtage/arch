"""
Core classes for ARCH models
"""
from __future__ import division, absolute_import
from ..compat.python import add_metaclass, range

from copy import deepcopy
from functools import partial
import datetime as dt
import warnings

import numpy as np
from numpy.linalg import matrix_rank
from numpy import ones, zeros, sqrt, diag, empty, ceil
from scipy.optimize import fmin_slsqp
import scipy.stats as stats
import pandas as pd
from statsmodels.tools.decorators import cache_readonly, resettable_cache
from statsmodels.iolib.summary import Summary, fmt_2cols, fmt_params
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.numdiff import approx_fprime, approx_hess

from .distribution import Distribution, Normal
from .volatility import VolatilityProcess, ConstantVariance
from ..utility.array import ensure1d, DocStringInheritor
from ..utility.exceptions import ConvergenceWarning, StartingValueWarning, \
    convergence_warning, starting_value_warning

__all__ = ['implicit_constant', 'ARCHModelResult', 'ARCHModel']

# Callback variables
_callback_iter, _callback_llf = 0, 0.0,
_callback_func_count, _callback_iter_display = 0, 1


def _callback(*args):
    """
    Callback for use in optimization

    Parameters
    ----------
    parameters : array
        Parameter value (not used by function)

    Notes
    -----
    Uses global values to track iteration, iteration display frequency,
    log likelihood and function count
    """
    global _callback_iter
    _callback_iter += 1
    disp = 'Iteration: {0:>6},   Func. Count: {1:>6.3g},   Neg. LLF: {2}'
    if _callback_iter % _callback_iter_display == 0:
        print(disp.format(_callback_iter, _callback_func_count, _callback_llf))

    return None


def linear_constraint(x, *args, **kwargs):
    """
    Used for constraints of the forms
    A.dot(x) - b >= 0
    """
    return kwargs['a'].dot(x) - kwargs['b']


def constraint(a, b):
    """
    Constructor allowing linear_constraint to be used with arbitrary inputs

    Parameters
    ----------
    a : array
        Parameter loadings
    b : array
        Constraint bounds

    Returns
    -------
    partial : callable
        Callable constraint which accepts the parameters as the first input and
        returns a.dot(parameters) - b

    Notes
    -----
    Parameter constraints satisfy a.dot(parameters) - b >= 0

    """
    return partial(linear_constraint, a=a, b=b)


def format_float_fixed(x, max_digits=10, decimal=4):
    """Formats a floating point number so that if it can be well expressed
    in using a string with digits len, then it is converted simply, otherwise
    it is expressed in scientific notation"""
    # basic_format = '{:0.' + str(digits) + 'g}'
    if x == 0:
        return ('{:0.' + str(decimal) + 'f}').format(0.0)
    scale = np.log10(np.abs(x))
    scale = np.sign(scale) * ceil(np.abs(scale))
    if scale > (max_digits - 2 - decimal) or scale < -(decimal - 2):
        formatted = (
            '{0:' + str(max_digits) + '.' + str(decimal) + 'e}').format(x)
    else:
        formatted = (
            '{0:' + str(max_digits) + '.' + str(decimal) + 'f}').format(x)
    return formatted


def implicit_constant(x):
    """
    Test a matrix for an implicit constant

    Parameters
    ----------
    x : array
        Array to be tested

    Returns
    -------
    constant : bool
        Flag indicating whether the array has an implicit constant - whether
        the array has a set of columns that adds to a constant value
    """
    nobs = x.shape[0]
    rank = matrix_rank(np.hstack((ones((nobs, 1)), x)))
    return rank == x.shape[1]


@add_metaclass(DocStringInheritor)
class ARCHModel(object):
    """
    Abstract base class for mean models in ARCH processes.  Specifies the
    conditional mean process.

    All public methods that raise NotImplementedError should be overridden by
    any subclass.  Private methods that raise NotImplementedError are optional
    to override but recommended where applicable.
    """

    def __init__(self, y=None, volatility=None, distribution=None,
                 hold_back=None):

        # Set on model fit
        self._fit_indices = None
        self._fit_y = None

        self._is_pandas = isinstance(y, (pd.DataFrame, pd.Series))
        if y is not None:
            self._y_series = ensure1d(y, 'y', series=True)
        else:
            self._y_series = ensure1d(empty((0,)), 'y', series=True)

        self._y = np.asarray(self._y_series)
        self._y_original = y

        self.hold_back = hold_back
        self._hold_back = 0 if hold_back is None else hold_back

        self._volatility = None
        self._distribution = None
        self._backcast = None
        self._var_bounds = None

        if volatility is not None:
            self.volatility = volatility
        else:
            self.volatility = ConstantVariance()

        if distribution is not None:
            self.distribution = distribution
        else:
            self.distribution = Normal()

    def constraints(self):
        """
        Construct linear constraint arrays  for use in non-linear optimization

        Returns
        -------
        a : array
            Number of constraints by number of parameters loading array
        b : array
            Number of constraints array of lower bounds

        Notes
        -----
        Parameters satisfy a.dot(parameters) - b >= 0
        """
        return empty((0, self.num_params)), empty(0)

    def bounds(self):
        """
        Construct bounds for parameters to use in non-linear optimization

        Returns
        -------
        bounds : list (2-tuple of float)
            Bounds for parameters to use in estimation.
        """
        num_params = self.num_params
        return [(-np.inf, np.inf)] * num_params

    @property
    def y(self):
        """Returns the dependent variable"""
        return self._y_original

    @property
    def volatility(self):
        """Set or gets the volatility process

        Volatility processes must be a subclass of VolatilityProcess
        """
        return self._volatility

    @volatility.setter
    def volatility(self, value):
        if not isinstance(value, VolatilityProcess):
            raise ValueError("Must subclass VolatilityProcess")
        self._volatility = value

    @property
    def distribution(self):
        """Set or gets the error distribution

        Distributions must be a subclass of Distribution
        """
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        if not isinstance(value, Distribution):
            raise ValueError("Must subclass Distribution")
        self._distribution = value

    def _r2(self, params):
        """
        Computes the model r-square.  Optional to over-ride.  Must match
        signature.
        """
        raise NotImplementedError("Subclasses optionally may provide.")

    def _fit_no_arch_normal_errors(self, cov_type='robust'):
        """
        Must be overridden with closed form estimator
        """
        raise NotImplementedError("Subclasses must implement")

    def _loglikelihood(self, parameters, sigma2, backcast, var_bounds,
                       individual=False):
        """
        Computes the log-likelihood using the entire model

        Parameters
        ----------
        parameters
        sigma2
        backcast
        individual : bool, optional

        Returns
        -------
        neg_llf : float
            Negative of model loglikelihood
        """
        # Parse parameters
        global _callback_func_count, _callback_llf
        _callback_func_count += 1

        # 1. Resids
        mp, vp, dp = self._parse_parameters(parameters)
        resids = self.resids(mp)

        # 2. Compute sigma2 using VolatilityModel
        sigma2 = self.volatility.compute_variance(vp, resids, sigma2, backcast,
                                                  var_bounds)
        # 3. Compute log likelihood using Distribution
        llf = self.distribution.loglikelihoood(dp, resids, sigma2, individual)

        _callback_llf = -1.0 * llf
        return -1.0 * llf

    def _all_parameter_names(self):
        """Returns a list containing all parameter names from the mean model,
        volatility model and distribution"""

        names = self.parameter_names()
        names.extend(self.volatility.parameter_names())
        names.extend(self.distribution.parameter_names())

        return names

    def _parse_parameters(self, x):
        """Return the parameters of each model in a tuple"""

        km, kv = self.num_params, self.volatility.num_params
        return x[:km], x[km:km + kv], x[km + kv:]

    def fix(self, params, first_obs=None, last_obs=None):
        """
        Allows an ARCHModelFixedResult to be constructed from fixed parameters.

        Parameters
        ----------
        params: ndarray-like
            User specified parameters to use when generating the result. Must
            have the correct number of parameters for a given choice of mean
            model, volatility model and distribution.
        first_obs : {int, str, datetime, Timestamp}
            First observation to use when fixing model
        last_obs : {int, str, datetime, Timestamp}
            Last observation to use when fixing model

        Returns
        -------
        results : ARCHModelFixedResult
            Object containing model results

        Notes
        -----
        Parameters are not checked against model-specific constraints.
        """
        v = self.volatility

        self._adjust_sample(first_obs, last_obs)
        resids = self.resids(self.starting_values())
        sigma2 = np.zeros_like(resids)
        backcast = v.backcast(resids)
        self._backcast = backcast

        var_bounds = v.variance_bounds(resids)

        params = np.asarray(params)
        loglikelihood = -1.0 * self._loglikelihood(params, sigma2, backcast,
                                                   var_bounds)

        mp, vp, dp = self._parse_parameters(params)

        resids = self.resids(mp)
        vol = np.zeros_like(resids)
        self.volatility.compute_variance(vp, resids, vol, backcast, var_bounds)
        vol = np.sqrt(vol)

        names = self._all_parameter_names()
        # Reshape resids and vol
        first_obs, last_obs = self._fit_indices
        resids_final = np.empty_like(self._y, dtype=np.float64)
        resids_final.fill(np.nan)
        resids_final[first_obs:last_obs] = resids
        vol_final = np.empty_like(self._y, dtype=np.float64)
        vol_final.fill(np.nan)
        vol_final[first_obs:last_obs] = vol

        model_copy = deepcopy(self)
        return ARCHModelFixedResult(params, resids, vol, self._y_series, names,
                                    loglikelihood, self._is_pandas, model_copy)

    def _adjust_sample(self, first_obs, last_obs):
        """
        Performs sample adjustment for estimation

        Parameters
        ----------
        first_obs : {int, str, datetime, datetime64, Timestamp}
            First observation to use when estimating model
        last_obs : {int, str, datetime, datetime64, Timestamp}
            Last observation to use when estimating model

        Notes
        -----
        Adjusted sample must follow Python semantics of first_obs:last_obs
        """
        raise NotImplementedError("Subclasses must implement")

    def fit(self, update_freq=1, disp='final', starting_values=None,
            cov_type='robust', show_warning=True, first_obs=None,
            last_obs=None):
        """
        Fits the model given a nobs by 1 vector of sigma2 values

        Parameters
        ----------
        update_freq : int, optional
            Frequency of iteration updates.  Output is generated every
            `update_freq` iterations. Set to 0 to disable iterative output.
        disp : str
            Either 'final' to print optimization result or 'off' to display
            nothing
        starting_values : array, optional
            Array of starting values to use.  If not provided, starting values
            are constructed by the model components.
        cov_type : str, optional
            Estimation method of parameter covariance.  Supported options are
            'robust', which does not assume the Information Matrix Equality
            holds and 'classic' which does.  In the ARCH literature, 'robust'
            corresponds to Bollerslev-Wooldridge covariance estimator.
        show_warning : bool, optional
            Flag indicating whether convergence warnings should be shown.
        first_obs : {int, str, datetime, Timestamp}
            First observation to use when estimating model
        last_obs : {int, str, datetime, Timestamp}
            Last observation to use when estimating model

        Returns
        -------
        results : ARCHModelResult
            Object containing model results

        Notes
        -----
        A ConvergenceWarning is raised if SciPy's optimizer indicates
        difficulty finding the optimum.

        """
        if self._y_original is None:
            raise RuntimeError('Cannot estimate model without data.')

        # 1. Check in ARCH or Non-normal dist.  If no ARCH and normal,
        # use closed form
        v, d = self.volatility, self.distribution
        offsets = np.array((self.num_params, v.num_params, d.num_params))
        total_params = sum(offsets)
        has_closed_form = (v.closed_form and d.num_params == 0) or total_params == 0

        self._adjust_sample(first_obs, last_obs)

        if has_closed_form:
            try:
                return self._fit_no_arch_normal_errors(cov_type=cov_type)
            except NotImplementedError:
                pass

        resids = self.resids(self.starting_values())
        sigma2 = np.zeros_like(resids)
        backcast = v.backcast(resids)
        self._backcast = backcast
        sv_volatility = v.starting_values(resids)
        self._var_bounds = var_bounds = v.variance_bounds(resids)
        v.compute_variance(sv_volatility, resids, sigma2, backcast, var_bounds)
        std_resids = resids / sqrt(sigma2)

        # 2. Construct constraint matrices from all models and distribution
        constraints = (self.constraints(),
                       self.volatility.constraints(),
                       self.distribution.constraints())

        num_constraints = [c[0].shape[0] for c in constraints]
        num_constraints = np.array(num_constraints)
        num_params = offsets.sum()
        a = zeros((num_constraints.sum(), num_params))
        b = zeros(num_constraints.sum())

        for i, c in enumerate(constraints):
            r_en = num_constraints[:i + 1].sum()
            c_en = offsets[:i + 1].sum()
            r_st = r_en - num_constraints[i]
            c_st = c_en - offsets[i]

            if r_en - r_st > 0:
                a[r_st:r_en, c_st:c_en] = c[0]
                b[r_st:r_en] = c[1]

        bounds = self.bounds()
        bounds.extend(v.bounds(resids))
        bounds.extend(d.bounds(std_resids))

        # 3. Construct starting values from all models
        sv = starting_values
        if starting_values is not None:
            sv = ensure1d(starting_values, 'starting_values')
            valid = (sv.shape[0] == num_params)
            if a.shape[0] > 0:
                satisfies_constraints = a.dot(sv) - b > 0
                valid = valid and satisfies_constraints.all()
            for i, bound in enumerate(bounds):
                valid = valid and bound[0] <= sv[i] <= bound[1]
            if not valid:
                warnings.warn(starting_value_warning, StartingValueWarning)
                starting_values = None

        if starting_values is None:
            sv = (self.starting_values(),
                  sv_volatility,
                  d.starting_values(std_resids))
            sv = np.hstack(sv)

        # 4. Estimate models using constrained optimization
        global _callback_func_count, _callback_iter, _callback_iter_display
        _callback_func_count, _callback_iter = 0, 0
        if update_freq <= 0 or disp == 'off':
            _callback_iter_display = 2 ** 31

        else:
            _callback_iter_display = update_freq
        disp = 1 if disp == 'final' else 0

        func = self._loglikelihood
        args = (sigma2, backcast, var_bounds)
        f_ieqcons = constraint(a, b)

        xopt = fmin_slsqp(func, sv, f_ieqcons=f_ieqcons, bounds=bounds,
                          args=args, iter=100, acc=1e-06, iprint=1,
                          full_output=True, epsilon=1.4901161193847656e-08,
                          callback=_callback, disp=disp)

        # Check convergence
        imode, smode = xopt[3:]
        if show_warning:
            warnings.filterwarnings('always', '', ConvergenceWarning)
        else:
            warnings.filterwarnings('ignore', '', ConvergenceWarning)

        if imode != 0 and show_warning:
            warnings.warn(convergence_warning.format(code=imode,
                                                     string_message=smode),
                          ConvergenceWarning)

        # 5. Return results
        params = xopt[0]
        loglikelihood = -1.0 * xopt[1]

        mp, vp, dp = self._parse_parameters(params)

        resids = self.resids(mp)
        vol = np.zeros_like(resids)
        self.volatility.compute_variance(vp, resids, vol, backcast, var_bounds)
        vol = np.sqrt(vol)

        try:
            r2 = self._r2(mp)
        except NotImplementedError:
            r2 = np.nan

        names = self._all_parameter_names()
        # Reshape resids and vol
        first_obs, last_obs = self._fit_indices
        resids_final = np.empty_like(self._y, dtype=np.float64)
        resids_final.fill(np.nan)
        resids_final[first_obs:last_obs] = resids
        vol_final = np.empty_like(self._y, dtype=np.float64)
        vol_final.fill(np.nan)
        vol_final[first_obs:last_obs] = vol

        fit_start, fit_stop = self._fit_indices
        model_copy = deepcopy(self)
        return ARCHModelResult(params, None, r2, resids_final, vol_final,
                               cov_type, self._y_series, names, loglikelihood,
                               self._is_pandas, xopt, fit_start, fit_stop, model_copy)

    def parameter_names(self):
        """List of parameters names

        Returns
        -------
        names : list (str)
            List of variable names for the mean model
        """
        raise NotImplementedError('Subclasses must implement')

    def starting_values(self):
        """
        Returns starting values for the mean model, often the same as the
        values returned from fit

        Returns
        -------
        sv : array
            Starting values
        """
        params = np.asarray(self._fit_no_arch_normal_errors().params)
        # Remove sigma2
        if params.shape[0] == 1:
            return np.empty(0)
        elif params.shape[0] > 1:
            return params[:-1]

    @cache_readonly
    def num_params(self):
        """
        Returns the number of parameters
        """
        raise NotImplementedError('Subclasses must implement')

    def simulate(self, params, nobs, burn=500, initial_value=None, x=None,
                 initial_value_vol=None):
        raise NotImplementedError('Subclasses must implement')

    def resids(self, params, y=None, regressors=None):
        """
        Compute model residuals

        Parameters
        ----------
        params : array
            Model parameters
        y : array, optional
            Alternative values to use when computing model residuals
        regressors : array, optional
            Alternative regressor values to use when computing model residuals

        Returns
        -------
        resids : array
            Model residuals
        """
        raise NotImplementedError('Subclasses must implement')

    def compute_param_cov(self, params, backcast=None, robust=True):
        """
        Computes parameter covariances using numerical derivatives.

        Parameters
        ----------
        params : array
            Model parameters
        backcast : float
            Value to use for pre-sample observations
        robust : bool, optional
            Flag indicating whether to use robust standard errors (True) or
            classic MLE (False)

        """
        resids = self.resids(self.starting_values())
        var_bounds = self.volatility.variance_bounds(resids)
        nobs = resids.shape[0]
        if backcast is None and self._backcast is None:
            backcast = self.volatility.backcast(resids)
            self._backcast = backcast
        elif backcast is None:
            backcast = self._backcast

        kwargs = {'sigma2': np.zeros_like(resids),
                  'backcast': backcast,
                  'var_bounds': var_bounds,
                  'individual': False}

        hess = approx_hess(params, self._loglikelihood, kwargs=kwargs)
        hess /= nobs
        inv_hess = np.linalg.inv(hess)
        if robust:
            kwargs['individual'] = True
            scores = approx_fprime(params, self._loglikelihood, kwargs=kwargs)
            score_cov = np.cov(scores.T)
            return inv_hess.dot(score_cov).dot(inv_hess) / nobs
        else:
            return inv_hess / nobs

    def forecast(self, params, horizon=1, start=None, align='origin', method='analytic',
                 simulations=1000):
        """
        Construct forecasts from estimated model

        Parameters
        ----------
        params : array, optional
            Alternative parameters to use.  If not provided, the parameters
            estimated when fitting the model are used.  Must be identical in
            shape to the parameters computed by fitting the model.
        horizon : int, optional
           Number of steps to forecast
        start : {int, datetime, Timestamp, str}, optional
            An integer, datetime or str indicating the first observation to
            produce the forecast for.  Datetimes can only be used with pandas
            inputs that have a datetime index. Strings must be convertible
            to a date time, such as in '1945-01-01'.
        align : str, optional
            Either 'origin' or 'target'.  When set of 'origin', the t-th row
            of forecasts contains the forecasts for t+1, t+2, ..., t+h. When
            set to 'target', the t-th row contains the 1-step ahead forecast
            from time t-1, the 2 step from time t-2, ..., and the h-step from
            time t-h.  'target' simplified computing forecast errors since the
            realization and h-step forecast are aligned.
        method : {'analytic', 'simulation', 'bootstrap'}
            Method to use when producing the forecast. The default is analytic.
            The method only affects the variance forecast generation.  Not all
            volatility models support all methods. In particular, volatility
            models that do not evolve in squares such as EGARCH or TARCH do not
            support the 'analytic' method for horizons > 1.
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.

        Returns
        -------
        forecasts : DataFrame
            t by h data frame containing the forecasts.  The alignment of the forecasts is
            controlled by `align`.

        Examples
        --------
        >>> import pandas as pd
        >>> from arch import arch_model
        >>> am = arch_model(None,mean='HAR',lags=[1,5,22],vol='Constant')
        >>> sim_data = am.simulate([0.1,0.4,0.3,0.2,1.0], 250)
        >>> sim_data.index = pd.date_range('2000-01-01',periods=250)
        >>> am = arch_model(sim_data['data'],mean='HAR',lags=[1,5,22],  vol='Constant')
        >>> res = am.fit()
        >>> fig = res.hedgehog_plot()

        Notes
        -----
        The most basic 1-step ahead forecast will return a vector with the same
        length as the original data, where the t-th value will be the time-t
        forecast for time t + 1.  When the horizon is > 1, and when using the
        default value for `align`, the forecast value in position [t, h] is the
        time-t, h+1 step ahead forecast.

        If model contains exogenous variables (`model.x is not None`), then
        only 1-step ahead forecasts are available.  Using horizon > 1 will
        produce a warning and all columns, except the first, will be
        nan-filled.

        If `align` is 'origin', forecast[t,h] contains the forecast made using
        y[:t] (that is, up to but not including t) for horizon h + 1.  For
        example, y[100,2] contains the 3-step ahead forecast using the first
        100 data points, which will correspond to the realization y[100 + 2].
        If `align` is 'target', then the same forecast is in location
        [102, 2], so that it is aligned with the observation to use when
        evaluating, but still in the same column.
        """
        raise NotImplementedError('Subclasses must implement')


class ARCHModelFixedResult(object):
    """
    Results for fixed parameters for an ARCHModel model

    Parameters
    ----------
    params : array
        Estimated parameters
    resid : array
        Residuals from model.  Residuals have same shape as original data and
        contain nan-values in locations not used in estimation
    volatility : array
        Conditional volatility from model
    dep_var: Series
        Dependent variable
    names: list (str)
        Model parameter names
    loglikelihood : float
        Loglikelihood at specified parameters
    is_pandas : bool
        Whether the original input was pandas
    model : ARCHModel
        The model object used to estimate the parameters

    Methods
    -------
    summary
        Produce a summary of the results
    plot
        Produce a plot of the volatility and standardized residuals
    forecast
        Construct forecasts from a model

    Attributes
    ----------
    loglikelihood : float
        Value of the log-likelihood
    aic : float
        Akaike information criteria
    bic : float
        Schwarz/Bayes information criteria
    conditional_volatility : {array, Series}
        nobs element array containing the conditional volatility (square root
        of conditional variance).  The values are aligned with the input data
        so that the value in the t-th position is the variance of t-th error,
        which is computed using time-(t-1) information.
    params : Series
        Estimated parameters
    nobs : int
        Number of observations used in the estimation
    num_params : int
        Number of parameters in the model
    resid : {array, Series}
        nobs element array containing model residuals
    model : ARCHModel
        Model instance used to produce the fit
    """

    def __init__(self, params, resid, volatility, dep_var, names,
                 loglikelihood, is_pandas, model):
        self._params = params
        self._resid = resid
        self._is_pandas = is_pandas
        self.model = model
        self._datetime = dt.datetime.now()
        self._cache = resettable_cache()
        self._dep_var = dep_var
        self._dep_name = dep_var.name
        self._names = names
        self._loglikelihood = loglikelihood
        self._nobs = self.model._fit_y.shape[0]
        self._index = dep_var.index
        self._volatility = volatility

    def summary(self):
        """
        Constructs a summary of the results from a fit model.

        Returns
        -------
        summary : Summary instance
            Object that contains tables and facilitated export to text, html or
            latex
        """
        # Summary layout
        # 1. Overall information
        # 2. Mean parameters
        # 3. Volatility parameters
        # 4. Distribution parameters
        # 5. Notes

        model = self.model
        model_name = model.name + ' - ' + model.volatility.name

        # Summary Header
        top_left = [('Dep. Variable:', self._dep_name),
                    ('Mean Model:', model.name),
                    ('Vol Model:', model.volatility.name),
                    ('Distribution:', model.distribution.name),
                    ('Method:', 'User-specified Parameters'),
                    ('', ''),
                    ('Date:', self._datetime.strftime('%a, %b %d %Y')),
                    ('Time:', self._datetime.strftime('%H:%M:%S'))]

        top_right = [('R-squared:', '--'),
                     ('Adj. R-squared:', '--'),
                     ('Log-Likelihood:', '%#10.6g' % self.loglikelihood),
                     ('AIC:', '%#10.6g' % self.aic),
                     ('BIC:', '%#10.6g' % self.bic),
                     ('No. Observations:', self._nobs),
                     ('', ''),
                     ('', ''), ]

        title = model_name + ' Model Results'
        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        # create summary table instance
        smry = Summary()
        # Top Table
        # Parameter table
        fmt = fmt_2cols
        fmt['data_fmts'][1] = '%18s'

        top_right = [('%-21s' % ('  ' + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)

        stubs = self._names
        header = ['coef']
        vals = (self.params,)
        formats = [(10, 4)]
        pos = 0
        param_table_data = []
        for _ in range(len(vals[0])):
            row = []
            for i, val in enumerate(vals):
                if isinstance(val[pos], np.float64):
                    converted = format_float_fixed(val[pos], *formats[i])
                else:
                    converted = val[pos]
                row.append(converted)
            pos += 1
            param_table_data.append(row)

        mc = self.model.num_params
        vc = self.model.volatility.num_params
        dc = self.model.distribution.num_params
        counts = (mc, vc, dc)
        titles = ('Mean Model', 'Volatility Model', 'Distribution')
        total = 0
        for title, count in zip(titles, counts):
            if count == 0:
                continue

            table_data = param_table_data[total:total + count]
            table_stubs = stubs[total:total + count]
            total += count
            table = SimpleTable(table_data,
                                stubs=table_stubs,
                                txt_fmt=fmt_params,
                                headers=header, title=title)
            smry.tables.append(table)

        extra_text = ('Results generated with user-specified parameters.',
                      'Since the model was not estimated, there are no std. '
                      'errors.')
        smry.add_extra_txt(extra_text)
        return smry

    @cache_readonly
    def loglikelihood(self):
        """Model loglikelihood"""
        return self._loglikelihood

    @cache_readonly
    def aic(self):
        """Akaike Information Criteria

        -2 * loglikelihood + 2 * num_params"""
        return -2 * self.loglikelihood + 2 * self.num_params

    @cache_readonly
    def num_params(self):
        """Number of parameters in model"""
        return len(self.params)

    @cache_readonly
    def bic(self):
        """
        Schwarz/Bayesian Information Criteria

        -2 * loglikelihod + log(nobs) * num_params
        """
        return -2 * self.loglikelihood + np.log(self.nobs) * self.num_params

    @cache_readonly
    def params(self):
        """Model Parameters"""
        return pd.Series(self._params, index=self._names, name='params')

    @cache_readonly
    def conditional_volatility(self):
        """
        Estimated conditional volatility
        """
        if self._is_pandas:
            return pd.Series(self._volatility,
                             name='cond_vol',
                             index=self._index)
        else:
            return self._volatility

    @cache_readonly
    def nobs(self):
        """
        Number of data points used ot estimate model
        """
        return self._nobs

    @cache_readonly
    def resid(self):
        """
        Model residuals
        """
        if self._is_pandas:
            return pd.Series(self._resid, name='resid', index=self._index)
        else:
            return self._resid

    def plot(self, annualize=None, scale=None):
        """
        Plot standardized residuals and conditional volatility

        Parameters
        ----------
        annualize : str, optional
            String containing frequency of data that indicates plot should
            contain annualized volatility.  Supported values are 'D' (daily),
            'W' (weekly) and 'M' (monthly), which scale variance by 252, 52,
            and 12, respectively.
        scale : float, optional
            Value to use when scaling returns to annualize.  If scale is
            provides, annualize is ignored and the value in scale is used.

        Returns
        -------
        fig : figure
            Handle to the figure

        Examples
        --------
        >>> from arch import arch_model
        >>> am = arch_model(None)
        >>> sim_data = am.simulate([0.0, 0.01, 0.07, 0.92], 2520)
        >>> am = arch_model(sim_data['data'])
        >>> res = am.fit(update_freq=0, disp='off')
        >>> fig = res.plot()

        Produce a plot with annualized volatility

        >>> fig = res.plot(annualize='D')

        Override the usual scale of 252 to use 360 for an asset that trades
        most days of the year

        >>> fig = res.plot(scale=360)
        """
        from matplotlib.pyplot import figure
        fig = figure()

        ax = fig.add_subplot(2, 1, 1)
        ax.plot(self._index, self.resid / self.conditional_volatility)
        ax.set_title('Standardized Residuals')
        ax.axes.xaxis.set_ticklabels([])

        ax = fig.add_subplot(2, 1, 2)
        vol = self.conditional_volatility
        title = 'Annualized Conditional Volatility'
        if scale is not None:
            vol = vol * np.sqrt(scale)
        elif annualize is not None:
            scales = {'D': 252, 'W': 52, 'M': 12}
            if annualize in scales:
                vol = vol * np.sqrt(scales[annualize])
            else:
                raise ValueError('annualize not recognized')
        else:
            title = 'Conditional Volatility'

        ax.plot(self._index, vol)
        ax.set_title(title)

        return fig

    def forecast(self, params=None, horizon=1, start=None, align='origin', method='analytic',
                 simulations=1000):
        """
        Construct forecasts from estimated model

        Parameters
        ----------
        params : array, optional
            Alternative parameters to use.  If not provided, the parameters
            estimated when fitting the model are used.  Must be identical in
            shape to the parameters computed by fitting the model.
        horizon : int, optional
           Number of steps to forecast
        start : {int, datetime, Timestamp, str}, optional
            An integer, datetime or str indicating the first observation to
            produce the forecast for.  Datetimes can only be used with pandas
            inputs that have a datetime index. Strings must be convertible
            to a date time, such as in '1945-01-01'.
        align : str, optional
            Either 'origin' or 'target'.  When set of 'origin', the t-th row
            of forecasts contains the forecasts for t+1, t+2, ..., t+h. When
            set to 'target', the t-th row contains the 1-step ahead forecast
            from time t-1, the 2 step from time t-2, ..., and the h-step from
            time t-h.  'target' simplified computing forecast errors since the
            realization and h-step forecast are aligned.
        method : {'analytic', 'simulation', 'bootstrap'}
            Method to use when producing the forecast. The default is analytic.
            The method only affects the variance forecast generation.  Not all
            volatility models support all methods. In particular, volatility
            models that do not evolve in squares such as EGARCH or TARCH do not
            support the 'analytic' method for horizons > 1.
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.

        Returns
        -------
        forecasts : DataFrame
            t by h data frame containing the forecasts.  The alignment of the forecasts is
            controlled by `align`.

        Notes
        -----
        The most basic 1-step ahead forecast will return a vector with the same
        length as the original data, where the t-th value will be the time-t
        forecast for time t + 1.  When the horizon is > 1, and when using the
        default value for `align`, the forecast value in position [t, h] is the
        time-t, h+1 step ahead forecast.

        If model contains exogenous variables (`model.x is not None`), then
        only 1-step ahead forecasts are available.  Using horizon > 1 will
        produce a warning and all columns, except the first, will be
        nan-filled.

        If `align` is 'origin', forecast[t,h] contains the forecast made using
        y[:t] (that is, up to but not including t) for horizon h + 1.  For
        example, y[100,2] contains the 3-step ahead forecast using the first
        100 data points, which will correspond to the realization y[100 + 2].
        If `align` is 'target', then the same forecast is in location
        [102, 2], so that it is aligned with the observation to use when
        evaluating, but still in the same column.
        """
        if params is None:
            params = self._params
        else:
            if (params.size != np.array(self._params).size or
                    params.ndim != self._params.ndim):
                raise ValueError('params have incorrect dimensions')
        return self.model.forecast(params, horizon, start, align, method, simulations)

    def hedgehog_plot(self, params=None, horizon=10, step=10, start=None,
                      type='volatility', method='analytic', simulations=1000):
        """
        Plot forecasts from estimated model

        Parameters
        ----------
        params : 1d array-like, optional
            Alternative parameters to use.  If not provided, the parameters
            computed by fitting the model are used.  Must be identical in shape
            to the parameters computed by fitting the model.
        horizon : int, optional
            Number of steps to forecast
        step : int, optional
            Non-negative number of forecasts to skip between spines
        start : int, datetime or str, optional
            An integer, datetime or str indicating the first observation to
            produce the forecast for.  Datetimes can only be used with pandas
            inputs that have a datetime index.  Strings must be convertible
            to a date time, such as in '1945-01-01'.  If not provided, the start
            is set to the earliest forecastable date.
        type : {'volatility', 'mean'}
            Quantity to plot, the forecast volatility or the forecast mean
        method : {'analytic', 'simulation', 'bootstrap'}
            Method to use when producing the forecast. The default is analytic.
            The method only affects the variance forecast generation.  Not all
            volatility models support all methods. In particular, volatility
            models that do not evolve in squares such as EGARCH or TARCH do not
            support the 'analytic' method for horizons > 1.
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.

        Returns
        -------
        fig : figure
            Handle to the figure

        Examples
        --------
        >>> import pandas as pd
        >>> from arch import arch_model
        >>> am = arch_model(None,mean='HAR',lags=[1,5,22],vol='Constant')
        >>> sim_data = am.simulate([0.1,0.4,0.3,0.2,1.0], 250)
        >>> sim_data.index = pd.date_range('2000-01-01',periods=250)
        >>> am = arch_model(sim_data['data'],mean='HAR',lags=[1,5,22],  vol='Constant')
        >>> res = am.fit()
        >>> fig = res.hedgehog_plot(type='mean')
        """
        import matplotlib.pyplot as plt

        plot_mean = type.lower() == 'mean'
        if start is None:
            invalid_start = True
            start = 0
            while invalid_start:
                try:
                    forecasts = self.forecast(params, horizon, start,
                                              method=method, simulations=simulations)
                    invalid_start = False
                except ValueError:
                    start += 1
        else:
            forecasts = self.forecast(params, horizon, start, method=method,
                                      simulations=simulations)

        fig, ax = plt.subplots(1, 1)
        use_date = isinstance(self._dep_var.index, pd.DatetimeIndex)
        plot_fn = ax.plot_date if use_date else ax.plot
        x_values = np.array(self._dep_var.index)
        if plot_mean:
            y_values = np.asarray(self._dep_var)
        else:
            y_values = np.asarray(self.conditional_volatility)

        plot_fn(x_values, y_values, linestyle='-', marker='')
        first_obs = np.min(np.where(np.logical_not(np.isnan(forecasts.mean)))[0])
        spines = []
        t = forecasts.mean.shape[0]
        for i in range(first_obs, t, step):
            if i + horizon + 1 > x_values.shape[0]:
                continue
            temp_x = x_values[i:i + horizon + 1]
            if plot_mean:
                spine_data = forecasts.mean.iloc[i]
            else:
                spine_data = np.sqrt(forecasts.variance.iloc[i])
            temp_y = np.hstack((y_values[i], spine_data))
            line = plot_fn(temp_x, temp_y, linewidth=3, linestyle='-',
                           marker='')
            spines.append(line)
        color = spines[0][0].get_color()
        for spine in spines[1:]:
            spine[0].set_color(color)
        plot_type = 'Mean' if plot_mean else 'Volatility'
        ax.set_title(self._dep_name + ' ' + plot_type + ' Forecast Hedgehog Plot')

        return fig


class ARCHModelResult(ARCHModelFixedResult):
    """
    Results from estimation of an ARCHModel model

    Parameters
    ----------
    params : array
        Estimated parameters
    param_cov : array or None
        Estimated variance-covariance matrix of params.  If none, calls method
        to compute variance from model when parameter covariance is first used
        from result
    r2 : float
        Model R-squared
    resid : array
        Residuals from model.  Residuals have same shape as original data and
        contain nan-values in locations not used in estimation
    volatility : array
        Conditional volatility from model
    cov_type : str
        String describing the covariance estimator used
    dep_var: Series
        Dependent variable
    names: list (str)
        Model parameter names
    loglikelihood : float
        Loglikelihood at estimated parameters
    is_pandas : bool
        Whether the original input was pandas
    fit_start : int
        Integer index of the first observation used to fit the model
    fit_stop : int
        Integer index of the last observation used to fit the model using Pythonic
        syntax fit_start:fit_stop
    model : ARCHModel
        The model object used to estimate the parameters

    Methods
    -------
    summary
        Produce a summary of the results
    plot
        Produce a plot of the volatility and standardized residuals
    conf_int
        Confidence intervals

    Attributes
    ----------
    loglikelihood : float
        Value of the log-likelihood
    aic : float
        Akaike information criteria
    bic : float
        Schwarz/Bayes information criteria
    conditional_volatility : {array, Series}
        nobs element array containing the conditional volatility (square root
        of conditional variance).  The values are aligned with the input data
        so that the value in the t-th position is the variance of t-th error,
        which is computed using time-(t-1) information.
    params : Series
        Estimated parameters
    param_cov : DataFrame
        Estimated variance-covariance of the parameters
    rsquared : float
        R-squared
    rsquared_adj : float
        Degree of freedom adjusted R-squared
    nobs : int
        Number of observations used in the estimation
    num_params : int
        Number of parameters in the model
    tvalues : Series
        Array of t-statistics for the null the coefficient is 0
    std_err : Series
        Array of parameter standard errors
    pvalues : Series
        Array of p-values for the t-statistics
    resid : {array, Series}
        nobs element array containing model residuals
    model : ARCHModel
        Model instance used to produce the fit
    """

    def __init__(self, params, param_cov, r2, resid, volatility, cov_type,
                 dep_var, names, loglikelihood, is_pandas, optim_output,
                 fit_start, fit_stop, model):
        super(ARCHModelResult, self).__init__(params, resid, volatility,
                                              dep_var, names, loglikelihood,
                                              is_pandas, model)

        self._fit_indices = (fit_start, fit_stop)
        self._param_cov = param_cov
        self._r2 = r2
        self.cov_type = cov_type
        self._optim_output = optim_output

    def conf_int(self, alpha=0.05):
        """
        Parameters
        ----------
        alpha : float, optional
            Size (prob.) to use when constructing the confidence interval.

        Returns
        -------
        ci : array
            Array where the ith row contains the confidence interval  for the
            ith parameter
        """
        cv = stats.norm.ppf(1.0 - alpha / 2.0)
        se = self.std_err
        params = self.params

        return pd.DataFrame(np.vstack((params - cv * se, params + cv * se)).T,
                            columns=['lower', 'upper'], index=self._names)

    def summary(self):
        """
        Constructs a summary of the results from a fit model.

        Returns
        -------
        summary : Summary instance
            Object that contains tables and facilitated export to text, html or
            latex
        """
        # Summary layout
        # 1. Overall information
        # 2. Mean parameters
        # 3. Volatility parameters
        # 4. Distribution parameters
        # 5. Notes

        model = self.model
        model_name = model.name + ' - ' + model.volatility.name

        # Summary Header
        top_left = [('Dep. Variable:', self._dep_name),
                    ('Mean Model:', model.name),
                    ('Vol Model:', model.volatility.name),
                    ('Distribution:', model.distribution.name),
                    ('Method:', 'Maximum Likelihood'),
                    ('', ''),
                    ('Date:', self._datetime.strftime('%a, %b %d %Y')),
                    ('Time:', self._datetime.strftime('%H:%M:%S'))]

        top_right = [('R-squared:', '%#8.3f' % self.rsquared),
                     ('Adj. R-squared:', '%#8.3f' % self.rsquared_adj),
                     ('Log-Likelihood:', '%#10.6g' % self.loglikelihood),
                     ('AIC:', '%#10.6g' % self.aic),
                     ('BIC:', '%#10.6g' % self.bic),
                     ('No. Observations:', self._nobs),
                     ('Df Residuals:', self.nobs - self.num_params),
                     ('Df Model:', self.num_params)]

        title = model_name + ' Model Results'
        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        # create summary table instance
        smry = Summary()
        # Top Table
        # Parameter table
        fmt = fmt_2cols
        fmt['data_fmts'][1] = '%18s'

        top_right = [('%-21s' % ('  ' + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)

        conf_int = np.asarray(self.conf_int())
        conf_int_str = []
        for c in conf_int:
            conf_int_str.append('[' + format_float_fixed(c[0], 7, 3) +
                                ',' + format_float_fixed(c[1], 7, 3) + ']')

        stubs = self._names
        header = ['coef', 'std err', 't', 'P>|t|', '95.0% Conf. Int.']
        vals = (self.params,
                self.std_err,
                self.tvalues,
                self.pvalues,
                conf_int_str)
        formats = [(10, 4), (9, 3), (9, 3), (9, 3), None]
        pos = 0
        param_table_data = []
        for _ in range(len(vals[0])):
            row = []
            for i, val in enumerate(vals):
                if isinstance(val[pos], np.float64):
                    converted = format_float_fixed(val[pos], *formats[i])
                else:
                    converted = val[pos]
                row.append(converted)
            pos += 1
            param_table_data.append(row)

        mc = self.model.num_params
        vc = self.model.volatility.num_params
        dc = self.model.distribution.num_params
        counts = (mc, vc, dc)
        titles = ('Mean Model', 'Volatility Model', 'Distribution')
        total = 0
        for title, count in zip(titles, counts):
            if count == 0:
                continue

            table_data = param_table_data[total:total + count]
            table_stubs = stubs[total:total + count]
            total += count
            table = SimpleTable(table_data,
                                stubs=table_stubs,
                                txt_fmt=fmt_params,
                                headers=header, title=title)
            smry.tables.append(table)

        extra_text = ['Covariance estimator: ' + self.cov_type]

        if self.convergence_flag:
            extra_text.append("""
WARNING: The optimizer did not indicate sucessful convergence. The message was
{string_message}. See convergence_flag.""".format(
                string_message=self._optim_output[-1]))

        smry.add_extra_txt(extra_text)
        return smry

    @cache_readonly
    def param_cov(self):
        """Parameter covariance"""
        if self._param_cov is not None:
            param_cov = self._param_cov
        else:
            params = np.asarray(self.params)
            if self.cov_type == 'robust':
                param_cov = self.model.compute_param_cov(params)
            else:
                param_cov = self.model.compute_param_cov(params,
                                                         robust=False)
        return pd.DataFrame(param_cov, columns=self._names, index=self._names)

    @cache_readonly
    def rsquared(self):
        """
        R-squared
        """
        return self._r2

    @cache_readonly
    def fit_start(self):
        return self._fit_indices[0]

    @cache_readonly
    def fit_stop(self):
        return self._fit_indices[1]

    @cache_readonly
    def rsquared_adj(self):
        """
        Degree of freedom adjusted R-squared
        """
        return 1 - (
            (1 - self.rsquared) * (self.nobs - 1) / (
                self.nobs - self.model.num_params))

    @cache_readonly
    def pvalues(self):
        """
        Array of p-values for the t-statistics
        """
        return pd.Series(stats.norm.sf(np.abs(self.tvalues)) * 2,
                         index=self._names, name='pvalues')

    @cache_readonly
    def std_err(self):
        """
        Parameter standard error
        """
        return pd.Series(sqrt(diag(self.param_cov)),
                         index=self._names, name='std_err')

    @cache_readonly
    def tvalues(self):
        """
        t-statistics for the null the coefficient is 0
        """
        tvalues = self.params / self.std_err
        tvalues.name = 'tvalues'
        return tvalues

    @cache_readonly
    def convergence_flag(self):
        """
        scipy.optimize.fmin_slsqp result flag
        """
        return self._optim_output[3]


def _align_forecast(f, align):
    if align == 'origin':
        return f
    elif align in ('target', 'horizon'):
        for i, col in enumerate(f):
            f[col] = f[col].shift(i + 1)
        return f
    else:
        raise ValueError('Unknown alignment')


def _format_forecasts(values, index):
    horizon = values.shape[1]
    format_str = '{0:>0' + str(int(np.ceil(np.log10(horizon + 0.5)))) + '}'
    columns = ['h.' + format_str.format(h + 1) for h in range(horizon)]
    forecasts = pd.DataFrame(values, index=index,
                             columns=columns, dtype=np.float64)
    return forecasts


class ARCHModelForecastSimulation(object):
    """
    Container for a simulation or bootstrap-based forecasts from an ARCH Model

    Parameters
    ----------
    values
    residuals
    variances
    residual_variances

    Attributes
    ----------
    values : DataFrame
        Simulated values of the process
    residuals : DataFrame
        Simulated residuals used to produce the values
    variances : DataFrame
        Simulated variances of the values
    residual_variances : DataFrame
        Simulated variance of the residuals
    """

    def __init__(self, values, residuals, variances, residual_variances):
        self._values = values
        self._residuals = residuals
        self._variances = variances
        self._residual_variances = residual_variances

    @property
    def values(self):
        return self._values

    @property
    def residuals(self):
        return self._residuals

    @property
    def variances(self):
        return self._variances

    @property
    def residual_variances(self):
        return self._residual_variances


class ARCHModelForecast(object):
    """
    Container for forecasts from an ARCH Model

    Parameters
    ----------
    index : {list, array}
    mean : array
    variance : array
    residual_variance : array
    simulated_paths : array, optional
    simulated_variances : array, optional
    simulated_residual_variances : array, optional
    simulated_residuals : array, optional
    align : {'origin', 'target'}

    Attributes
    ----------
    mean : DataFrame
        Forecast values for the conditional mean of the process
    variance : DataFrame
        Forecast values for the conditional variance of the process
    residual_variance : DataFrame
        Forecast values for the conditional variance of the residuals
    simulations : ARCHModelForecastSimulation
        Object containing detailed simulation results if using a simulation-based method
    """
    def __init__(self, index, mean, variance, residual_variance,
                 simulated_paths=None, simulated_variances=None,
                 simulated_residual_variances=None, simulated_residuals=None,
                 align='origin'):

        mean = _format_forecasts(mean, index)
        variance = _format_forecasts(variance, index)
        residual_variance = _format_forecasts(residual_variance, index)

        self._mean = _align_forecast(mean, align=align)
        self._variance = _align_forecast(variance, align=align)
        self._residual_variance = _align_forecast(residual_variance, align=align)

        self._sim = ARCHModelForecastSimulation(simulated_paths,
                                                simulated_residuals,
                                                simulated_variances,
                                                simulated_residual_variances)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def residual_variance(self):
        return self._residual_variance

    @property
    def simulations(self):
        return self._sim
