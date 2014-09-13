"""
Core classes for ARCH models
"""
from __future__ import division, absolute_import
from copy import deepcopy
from functools import partial
import datetime as dt
from distutils.version import LooseVersion

import numpy as np
from numpy.linalg import matrix_rank
from numpy import ones, zeros, sqrt, diag, empty, ceil
import scipy
from scipy.optimize import fmin_slsqp
import scipy.stats as stats
import pandas as pd

from statsmodels.tools.decorators import cache_readonly, resettable_cache
from statsmodels.iolib.summary import Summary, fmt_2cols, fmt_params
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.numdiff import approx_fprime, approx_hess

from .distribution import Distribution, Normal
from .volatility import VolatilityProcess, ConstantVariance
from ..utils import ensure1d, DocStringInheritor, date_to_index
from ..compat.python import add_metaclass, range

__all__ = ['implicit_constant', 'ARCHModelResult', 'ARCHModel']
SP14 = LooseVersion(scipy.version.short_version).version[1] >= 14

# Callback variables
_callback_iter, _callback_llf = 0, 0.0,
_callback_func_count, _callback_iter_display = 0, 1


def _callback(parameters):
    """
    Callback for use in optimization

    Parameters
    ----------
    parameters : 1-d array
        Parameter value (not used by function)

    Notes
    -----
    Uses global values to track iteration, iteration display frequency,
    log likelihood and function count
    """
    global _callback_iter, _callback_iter_display
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
    a : 2-d array
        Parameter loadings
    b : 1-d array
        Constraint bounds

    Returns
    -------
    partial : callable
        Callable constraint which accepts the parameters as the first input and returns
        a.dot(parameters) - b

    Notes
    -----
    Parameter constraints satisfy a.dot(parameters) - b >= 0

    """
    return partial(linear_constraint, a=a, b=b)


def format_float_fixed(x, max_digits=10, decimal=4):
    """Formats a floating point number so that if it can be well expressed
    in using a string with digits len, then it is converted simply, otherwise it
    is expressed in scientific notaiton"""
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
    x : 2-d array
        Array to be tested

    Returns
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
                 hold_back=None,
                 last_obs=None):
        self._is_pandas = isinstance(y, (pd.DataFrame, pd.Series))
        if y is not None:
            self._y_series = ensure1d(y, 'y', series=True)
        else:
            self._y_series = ensure1d(empty((0,)), 'y', series=True)

        self._y = np.asarray(self._y_series)

        self.hold_back = hold_back
        if isinstance(hold_back, (str, dt.datetime, np.datetime64)):
            date_index = self._y_series.index
            _first_obs_index = date_to_index(hold_back, date_index)
            self.first_obs = date_index[_first_obs_index]
        elif hold_back is None:
            self.first_obs = _first_obs_index = 0
        else:
            _first_obs_index = hold_back
            self.first_obs = self._y_series.index[_first_obs_index]

        self.last_obs = _last_obs_index = last_obs
        if isinstance(last_obs, (str, dt.datetime, np.datetime64)):
            date_index = self._y_series.index
            _last_obs_index = date_to_index(last_obs, date_index)
            self.last_obs = date_index[_last_obs_index]
        elif last_obs is None:
            self.last_obs = _last_obs_index = self._y.shape[0]
        else:
            self.last_obs = self._y_series.index[last_obs]

        self.nobs = _last_obs_index - _first_obs_index
        self._indices = (_first_obs_index, _last_obs_index)

        self._volatility = None
        self._distribution = None
        self._backcast = None

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
        a : 2-d array
            Number of constraints by number of parameters loading array
        b : 1-d array
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
        return self._y

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

    def num_params(self):
        """
        Number of parameters in mean model, excluding any variance components
        """
        return 0

    def _r2(self, params):
        """
        Computes the model r-square.  Optional to over-ride.  Must match
        signature.
        """
        raise NotImplementedError("Subcalses optionally may provide.")

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
        # 3. Compute loglikelihood using Distribution
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
        kd= self.distribution.num_params
        return x[:km], x[km:km + kv], x[km + kv:]

    def fit(self, iter=1, disp='final', starting_values=None,
            cov_type='robust'):
        """
        Fits the model given a nobs by 1 vector of sigma2 values

        Parameters
        ----------
        iter : int, optional
            Frequency of iteration updates.  Output is generated every `iter`
            iterations. Set to 0 to disable iterative output.
        disp : str
            Either 'final' to print optimization result or 'off' to display
            nothing
        starting_values : 1-d array, optional
            Array of starting values to use.  If not provided, starting values
            are constructed by the model components.
        cov_type : str, optional
            Estimation method of parameter covariance.  Supported options are
            'robust', which does not assume the Information Matrix Equality holds
            and 'classic' which does.  In the ARCH literature, 'robust'
            corresponds to Bollerslev-Wooldridge covariance estimation.

        Returns
        -------
        results : ARCHModelResult
            Object containing model results
        """
        # 1. Check in ARCH or Non-normal dist.  If no ARCH and normal,
        # use closed form
        v, d = self.volatility, self.distribution
        offsets = np.array((self.num_params, v.num_params, d.num_params))
        total_params = sum(offsets)
        has_closed_form = (v.num_params == 1 and d.num_params == 0) or \
                          total_params == 0
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
        var_bounds  = v.variance_bounds(resids)
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

        var_bounds = v.variance_bounds(resids)
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
                import warnings
                warnings.warn("Starting values do not satisfy the parameter "
                              "constraints in the model.  The provided starting "
                              "values will be ignored.")
                starting_values = None

        if starting_values is None:
            sv = (self.starting_values(),
                  sv_volatility,
                  d.starting_values(std_resids))
            sv = np.hstack(sv)

        # 4. Estimate models using constrained optimization
        global _callback_func_count, _callback_iter, _callback_iter_display
        _callback_func_count, _callback_iter = 0, 0
        if iter <= 0:
            _callback_iter_display = 2 ** 31
        else:
            _callback_iter_display = iter

        func = self._loglikelihood
        args = (sigma2, backcast, var_bounds)
        f_ieqcons = constraint(a, b)
        disp = 1 if disp == 'final' else 0
        if SP14:
            xopt = fmin_slsqp(func, sv, f_ieqcons=f_ieqcons, bounds=bounds,
                              args=args, iter=100, acc=1e-06, iprint=1,
                              full_output=1, epsilon=1.4901161193847656e-08,
                              callback=_callback, disp=disp)
        else:
            if iter > 0:   # Fix limit in SciPy < 0.14
                disp = 2
            xopt = fmin_slsqp(func, sv, f_ieqcons=f_ieqcons, bounds=bounds,
                              args=args, iter=100, acc=1e-06, iprint=1,
                              full_output=1, epsilon=1.4901161193847656e-08,
                              disp=disp)

        # 5. Return results
        params = xopt[0]
        loglikelihood = -1.0 * xopt[1]

        mp, vp, dp = self._parse_parameters(params)

        resids = self.resids(mp)
        vol = np.zeros_like(resids)
        self.volatility.compute_variance(vp, resids, vol, backcast, var_bounds)
        vol = np.sqrt(vol)
        nobs = resids.shape[0]

        try:
            r2 = self._r2(mp)
        except NotImplementedError:
            r2 = np.nan

        names = self._all_parameter_names()
        # Reshape resids and vol
        first_obs, last_obs = self._indices
        resids_final = np.empty_like(self._y, dtype=np.float64)
        resids_final.fill(np.nan)
        resids_final[first_obs:last_obs] = resids
        vol_final = np.empty_like(self._y, dtype=np.float64)
        vol_final.fill(np.nan)
        vol_final[first_obs:last_obs] = vol

        model_copy = deepcopy(self)
        return ARCHModelResult(params, None, r2, resids_final, vol_final,
                               cov_type, self._y_series, names, loglikelihood,
                               self._is_pandas, model_copy)

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
        Returns starting values for the mean model, often the same as the values
        returned from fit

        Returns
        -------
        sv : 1-d array
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
        raise NotImplementedError('Must be overridden')

    def simulate(self, params, nobs, burn=500, initial_value=None, x=None,
                 initial_value_vol=None):
        raise NotImplementedError('Must be overridden')

    def resids(self, params):
        """
        Compute model residuals

        Parameters
        ----------
        params : 1-d array
            Model parameters

        Returns
        resids : 1-d array
            Model residuals
        """
        raise NotImplementedError('Must be overridden')

    def compute_param_cov(self, params, backcast=None, robust=True):
        """
        Computes parameter covariances using numerical derivatives.

        Parameters
        ----------
        params : 1-d array
            Model parameters
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


class ARCHModelResult(object):
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
    conditional_volatility : 1-d array or Series
        nobs element array containing the conditional volatility (square root
        of conditional variance)
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
    resid : 1-d array or Series
        nobs element array containing model residuals

    """

    def __init__(self, params, param_cov, r2, resid, volatility, cov_type,
                 dep_var, names, loglikelihood, is_pandas, model):
        self._params = params
        self._param_cov = param_cov
        self._resid = resid
        self._is_pandas = is_pandas
        self._r2 = r2
        self._model = model
        self._datetime = dt.datetime.now()
        self._cache = resettable_cache()
        self._dep_name = dep_var.name
        self._names = names
        self._loglikelihood = loglikelihood
        self._nobs = dep_var.shape[0]
        self._index = dep_var.index
        self.cov_type = cov_type
        self._volatility = volatility

    def conf_int(self, alpha=0.05):
        """
        Parameters
        ----------
        alpha : float, optional
            Size (probability) to use when constructing the confidence interval.

        Returns
        -------
        ci : 2-d array
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

        model = self._model
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
            conf_int_str.append('[' + format_float_fixed(c[0], 7, 3)
                                + ',' + format_float_fixed(c[1], 7, 3) + ']')

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
        for j in range(len(vals[0])):
            row = []
            for i, val in enumerate(vals):
                if isinstance(val[pos], np.float64):
                    converted = format_float_fixed(val[pos], *formats[i])
                else:
                    converted = val[pos]
                row.append(converted)
            pos += 1
            param_table_data.append(row)

        mc = self._model.num_params
        vc = self._model.volatility.num_params
        dc = self._model.distribution.num_params
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

        extra_text = ('Covariance estimator: ' + self.cov_type,)
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
    def param_cov(self):
        """Parameter covariance"""
        if self._param_cov is not None:
            param_cov = self._param_cov
        else:
            params = np.asarray(self.params)
            if self.cov_type == 'robust':
                param_cov = self._model.compute_param_cov(params)
            else:
                param_cov = self._model.compute_param_cov(params,
                                                          robust=False)
        return pd.DataFrame(param_cov, columns=self._names, index=self._names)

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
    def rsquared(self):
        """
        R-squared
        """
        return self._r2

    @cache_readonly
    def rsquared_adj(self):
        """
        Degree of freedom adjusted R-squared
        """
        return 1 - (
            (1 - self.rsquared) * (self.nobs - 1) / (
                self.nobs - self._model.num_params))

    @cache_readonly
    def nobs(self):
        """
        Number of data points used ot estimate model
        """
        return self._nobs

    @cache_readonly
    def pvalues(self):
        """
        Array of p-values for the t-statistics
        """
        return pd.Series(stats.norm.sf(np.abs(self.tvalues)) * 2,
                         index=self._names, name='pvalues')

    @cache_readonly
    def resid(self):
        """
        Model residuals
        """
        if self._is_pandas:
            return pd.Series(self._resid, name='resid', index=self._index)
        else:
            return self._resid

    @cache_readonly
    def std_err(self):
        """
        Parameter standard error
        """
        return pd.Series(diag(self.param_cov),
                         index=self._names, name='std_err')

    @cache_readonly
    def tvalues(self):
        """
        t-statistics for the null the coefficient is 0
        """
        tvalues = self.params / self.std_err
        tvalues.name = 'tvalues'
        return tvalues

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
        >>> res = am.fit(iter=0, disp='off')
        >>> fig = res.plot()

        Produce a plot with annualized volatility

        >>> fig = res.plot(annualize='D')

        Override the usual scale of 252 to use 360 for an asset that trades
        most days of the year

        >>> fig = res.plot(scale=360)
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()

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


        # def forecast(self, horizon=1, start=None, align='origin'):
        #             # TODO: Forecast implementation
        #  """
        # Construct forecasts from a HAR-X
        #
        # Parameters
        #     ----------
        #     horizon : int, optional
        #         Forecast horizon.  All forecasts up to horizon are constructed.
        #     start : int, optional
        #         First observation from which to construct forecasts.  If omitted,
        #         forecasts will start at the first observation used in the estimation
        #         of the model
        #     align : str, optional
        #         Either 'origin' or 'target' (equiv. 'horizon').  Determines how the
        #         array containing forecasts is aligned.  If 'origin', forecast[t,h]
        #         contains the forecast made using y[:t] (that is, up to but not
        #         including t) for horizon h + 1.  For example, y[100,2] contains the
        #         3-step ahead forecast using the first 100 data points, which will
        #         correspond to the realization y[100 + 2].  If 'target', then
        #         the same forecast is in location y[102, 2], so that it is aligned
        #         with the observation to use when evaluating, but still in the same
        #         column.
        #
        #     Returns
        #     -------
        #     ytph : array
        #         nobs by horizon array of forecast values.  Values that are not
        #         forecast are set to `np.nan`.  If align is `target', returned array
        #         is nobs + horizon by horizon.
        #
        #     Notes
        #     -----
        #     If model contains exogenous variables (`model.x is not None`), then only
        #     1-step ahead forecasts are available.  Using horizon > 1 will produce
        #     a warning and all columns, except the first, will be nan-filled.
        #     """
        #
        #     model = self._model
        #     constant, lags, x = model.constant, model._lags, model.x
        #     params = self.params
        #     max_lag = np.max(lags)
        #     y, nobs = model.y, model.nobs
        #     h = horizon
        #     ytph = empty((nobs, h))
        #     ytph.fill(np.nan)
        #
        #     ytemp = zeros(nobs + h)
        #     if start is None:
        #         start = self._model.first_obs
        #     elif start < self._model.first_obs:
        #         from warnings import warn
        #
        #         warn('Cannot start forecasts before the first observation used in '
        #              'estimation.', SyntaxWarning)
        #         start = self._model.first_obs
        #     if x is not None and h > 1:
        #         from warnings import warn
        #
        #         warn('Cannot forecasts for horizons longer than 1 when model has '
        #              'exogenous regressors', SyntaxWarning)
        #         pass
        #
        #     for t in range(start, nobs):
        #         for h in range(horizon):
        #             # Only 1 step ahead for models with x!
        #             if h > 1 and x is not None:
        #                 ytph[t, h] = np.nan
        #                 break
        #
        #             count = 0
        #
        #             ytemp[t - max_lag:t] = y[t - max_lag:t]
        #             ytph[t, h] = 0.0
        #             if constant:
        #                 ytph[t, h] += params[0]
        #                 count += 1
        #             for lag in lags.nobs:
        #                 val = ytemp[t + h - lag[1]:t + h - lag[0]]
        #                 if lag[1] - lag[0] > 1:
        #                     val = val.mean()
        #                 ytph[t, h] += params[count] * val
        #                 count += 1
        #             if x is not None:
        #                 ytph[t, h] += params[count:].dot(x[t, :])
        #             # Copy for recursion
        #             ytemp[t + h] = ytph[t, h]
        #
        #     if align.lower() in ('horizon', 'target'):
        #         temp = empty((nobs + horizon - 1, horizon))
        #         temp.fill(np.nan)
        #         for t in range(start, nobs):
        #             for h in range(horizon):
        #                 temp[t + h, h] = ytph[t, h]
        #         ytph = temp
        #
        #     return ytph

