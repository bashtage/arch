import datetime as dt
import warnings
from abc import abstractmethod
from arch.compat.python import iteritems
from copy import deepcopy

import numpy as np
from statsmodels.tools.numdiff import approx_fprime, approx_hess

from arch.compat.python import add_metaclass
from arch.multivariate.data import TimeSeries
from arch.multivariate.distribution import MultivariateNormal, MultivariateDistribution
from arch.multivariate.utility import symmetric_matrix_invroot
from arch.multivariate.volatility import ConstantCovariance, MultivariateVolatilityProcess
from arch.univariate.base import constraint
from arch.utility.array import AbstractDocStringInheritor, ensure1d
from arch.utility.exceptions import convergence_warning, ConvergenceWarning, \
    starting_value_warning, StartingValueWarning
from arch.vendor.cached_property import cached_property
import pandas as pd
import scipy.stats as stats

# Callback variables
_callback_iter, _callback_llf = 0, 0.0,
_callback_func_count, _callback_iter_display = 0, 1


def _callback(*args):
    """
    Callback for use in optimization

    Parameters
    ----------
    parameters : : ndarray
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


@add_metaclass(AbstractDocStringInheritor)
class MultivariateARCHModel(object):
    """
    Abstract base class for mean models in ARCH processes.  Specifies the
    conditional mean process.

    All public methods be overridden in subclass.  Private methods that
    raise NotImplementedError are optional to override but recommended where
    applicable.
    """

    def __init__(self, y=None, volatility=None, distribution=None,
                 hold_back=None, nvar=None):
        self.name = self.__class__.__name__
        if y is None and nvar is None:
            raise ValueError('nvar must be provided when y is None.')
        self._y = TimeSeries(y, nvar=nvar, name='y')
        self._nvar = nvar if nvar is not None else self._y.shape[1]
        # Set on model fit
        self._fit_indices = None
        self._fit_y = None

        self.hold_back = hold_back
        self._hold_back = 0 if hold_back is None else hold_back

        self._volatility = None
        self._distribution = None
        self._backcast = None

        if volatility is not None:
            self.volatility = volatility
        else:
            self.volatility = ConstantCovariance()

        if distribution is not None:
            self.distribution = distribution
        else:
            self.distribution = MultivariateNormal()

        self.volatility.nvar = self._nvar
        self.distribution.nvar = self._nvar

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

    @abstractmethod
    def constraints(self):
        """
        Construct linear constraint arrays for use in non-linear optimization

        Returns
        -------
        a : ndarray
            Number of constraints by number of parameters loading array
        b : ndarray
            Number of constraints array of lower bounds

        Notes
        -----
        Parameters satisfy a.dot(parameters) - b >= 0
        """
        pass

    @abstractmethod
    def bounds(self):
        """
        Construct bounds for parameters to use in non-linear optimization

        Returns
        -------
        bounds : list (2-tuple of float)
            Bounds for parameters to use in estimation.
        """
        pass

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
        if not isinstance(value, MultivariateVolatilityProcess):
            raise ValueError("Must subclass MultivariateVolatilityProcess")
        self._volatility = value

    @property
    def distribution(self):
        """Set or gets the error distribution

        Distributions must be a subclass of Distribution
        """
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        if not isinstance(value, MultivariateDistribution):
            raise ValueError("Must subclass Distribution")
        self._distribution = value

    @property
    def nvar(self):
        return self._y.shape[1]

    @abstractmethod
    def _r2(self, params):
        """
        Computes the model r-square.  Optional to over-ride.  Must match
        signature.
        """
        pass

    @abstractmethod
    def _fit_no_arch_normal_errors(self, cov_type='robust'):
        """
        Must be overridden with closed form estimator
        """
        pass

    @staticmethod
    def _static_gaussian_loglikelihood(resids):
        """
        Gaussian log-likelihood with 0 mean

        Parameters
        ----------
        resids : ndarray
            (nobs, nvar) array of data
        """
        nobs, nvar = resids.shape

        sigma = resids.T.dot(resids) / nobs
        _, logdet = np.linalg.slogdet(sigma)
        return -nvar / 2 * nobs * (np.log(2 * np.pi) + logdet + nobs)

    def _loglikelihood(self, parameters, sigma, backcast, individual=False):
        """
        Computes the log-likelihood using the entire model

        Parameters
        ----------
        parameters
        sigma
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
        sigma = self.volatility.compute_covariance(vp, resids, sigma, backcast)
        # 3. Compute log likelihood using Distribution
        llf = self.distribution.loglikelihood(dp, resids, sigma, individual)

        _callback_llf = -1.0 * llf
        return -1.0 * llf

    def _all_parameter_names(self):
        """Returns a list containing all parameter names from the mean model,
        volatility model and distribution"""

        names = self.parameter_names()
        names.extend(self.volatility.parameter_names())
        names.extend(self.distribution.parameter_names())

        return names

    def _parse_parameters(self, params):
        """Return the parameters of each model in a tuple"""
        params = np.asarray(params)
        km, kv = int(self.num_params), int(self.volatility.num_params)
        return params[:km], params[km:km + kv], params[km + kv:]

    @abstractmethod
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
        pass

    def fit(self, update_freq=1, disp='final', starting_values=None,
            cov_type='robust', show_warning=True, first_obs=None,
            last_obs=None, tol=None, options=None):
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
        starting_values : ndarray, optional
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
        tol : float, optional
            Tolerance for termination.
        options : dict, optional
            Options to pass to `scipy.optimize.minimize`.  Valid entries
            include 'ftol', 'eps', 'disp', and 'maxiter'.

        Returns
        -------
        results : ARCHModelResult
            Object containing model results

        Notes
        -----
        A ConvergenceWarning is raised if SciPy's optimizer indicates
        difficulty finding the optimum.

        Parameters are optimized using SLSQP.
        """
        if self._y.original is None:
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
        nobs = resids.shape[0]
        nvar = self.nvar
        sigma = np.zeros((nobs, nvar, nvar))
        backcast = v.backcast(resids)
        self._backcast = backcast
        sv_volatility = v.starting_values(resids)
        v.compute_covariance(sv_volatility, resids, sigma, backcast)
        std_resids = np.empty_like(resids)
        for i in range(nobs):
            std_resids[i:i + 1] = resids[i:i + 1].dot(symmetric_matrix_invroot(sigma[i]))

        # 2. Construct constraint matrices from all models and distribution
        constraints = (self.constraints(),
                       self.volatility.constraints(),
                       self.distribution.constraints())

        num_constraints = [c[0].shape[0] for c in constraints]
        num_constraints = np.array(num_constraints)
        num_params = offsets.sum()
        a = np.zeros((num_constraints.sum(), num_params))
        b = np.zeros(num_constraints.sum())

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
            sv = ensure1d(sv, 'starting_values')
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
        disp = True if disp == 'final' else False

        func = self._loglikelihood
        args = (sigma, backcast)
        ineq_constraints = constraint(a, b)
        func(sv, *args)
        from scipy.optimize import minimize

        options = {} if options is None else options
        options.setdefault('disp', disp)
        opt = minimize(func, sv, args=args, method='SLSQP', bounds=bounds,
                       constraints=ineq_constraints, tol=tol, callback=_callback,
                       options=options)

        if show_warning:
            warnings.filterwarnings('always', '', ConvergenceWarning)
        else:
            warnings.filterwarnings('ignore', '', ConvergenceWarning)

        if opt.status != 0 and show_warning:
            warnings.warn(convergence_warning.format(code=opt.status,
                                                     string_message=opt.message),
                          ConvergenceWarning)

        # 5. Return results
        params = opt.x
        loglikelihood = -1.0 * opt.fun

        mp, vp, dp = self._parse_parameters(params)

        resids = self.resids(mp)
        cov = np.zeros((nobs, nvar, nvar))
        self.volatility.compute_covariance(vp, resids, cov, backcast)

        try:
            r2 = self._r2(mp)
        except NotImplementedError:
            r2 = np.full(self.nvar, np.nan)

        names = self._all_parameter_names()
        # Reshape resids and vol
        first_obs, last_obs = self._fit_indices
        resids_final = np.full((nobs, nvar), np.nan)
        resids_final[first_obs:last_obs] = resids

        cov_final = np.full((nobs, nvar, nvar), np.nan)
        cov_final[first_obs:last_obs] = cov

        fit_start, fit_stop = self._fit_indices
        model_copy = deepcopy(self)

        return MultivariateARCHModelResult(params, None, r2, resids_final, cov_final, cov_type,
                                           self._y.frame, names, loglikelihood,
                                           self._y.pandas_input, opt, fit_start, fit_stop,
                                           model_copy)

    @abstractmethod
    def parameter_names(self):
        """List of parameters names

        Returns
        -------
        names : list (str)
            List of variable names for the mean model
        """
        pass

    def starting_values(self):
        """
        Returns starting values for the mean model, often the same as the
        values returned from fit

        Returns
        -------
        sv : ndarray
            Starting values
        """
        nvar = self.nvar
        distinct_cov = (nvar * (nvar + 1)) // 2
        params = np.asarray(self._fit_no_arch_normal_errors().params)
        if params.shape == distinct_cov:
            return np.empty(0)
        elif params.shape[0] > 1:
            return params[:-distinct_cov]

    @abstractmethod
    @cached_property
    def num_params(self):
        """
        Number of parameters in the model
        """
        pass

    @abstractmethod
    def simulate(self, params, nobs, burn=500, initial_value=None, x=None,
                 initial_value_vol=None):
        pass

    @abstractmethod
    def resids(self, params, y=None, regressors=None):
        """
        Compute model residuals

        Parameters
        ----------
        params : ndarray
            Model parameters
        y : ndarray, optional
            Alternative values to use when computing model residuals
        regressors : ndarray, optional
            Alternative regressor values to use when computing model residuals

        Returns
        -------
        resids : ndarray
            Model residuals
        """
        pass

    def compute_param_cov(self, params, backcast=None, robust=True):
        """
        Computes parameter covariances using numerical derivatives.

        Parameters
        ----------
        params : ndarray
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
            scores = approx_fprime(params, self._loglikelihood, kwargs=kwargs)  # type: np.ndarray
            score_cov = np.cov(scores.T)
            return inv_hess.dot(score_cov).dot(inv_hess) / nobs
        else:
            return inv_hess / nobs

    @abstractmethod
    def _model_description(self, include_lags=True):
        """Model description for use by __str__ and related functions"""
        pass


class MultivariateARCHModelResult(object):
    """
    Results from estimation of a multivariate ARCH Model model

    Parameters
    ----------
    params : ndarray
        Estimated parameters
    param_cov : {ndarray, None}
        Estimated variance-covariance matrix of params.  If none, calls method
        to compute variance from model when parameter covariance is first used
        from result
    r2 : float
        Model R-squared
    resid : ndarray
        Residuals from model.  Residuals have same shape as original data and
        contain nan-values in locations not used in estimation
    covariance : ndarray
        Conditional covariance from model
    cov_type : str
        String describing the covariance estimator used
    dep_var : Series
        Dependent variable
    names : list (str)
        Model parameter names
    loglikelihood : float
        Loglikelihood at estimated parameters
    input_pandas : bool
        Whether the original input was pandas
    fit_start : int
        Integer index of the first observation used to fit the model
    fit_stop : int
        Integer index of the last observation used to fit the model using
        slice notation `fit_start:fit_stop`
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
    params : Series
        Estimated parameters
    param_cov : DataFrame
        Estimated variance-covariance of the parameters
    resid : {ndarray, Series}
        nobs element array containing model residuals
    model : ARCHModel
        Model instance used to produce the fit
    """

    def __init__(self, params, param_cov, r2, resid, covariance, cov_type,
                 dep_var, names, loglikelihood, input_pandas, optim_output,
                 fit_start, fit_stop, model):

        self._params = params
        self._resid = resid
        self._input_pandas = input_pandas
        self.model = model
        self._datetime = dt.datetime.now()
        self._dep_var = dep_var
        self._dep_name = dep_var.columns
        self._names = names
        self._loglikelihood = loglikelihood
        self._nobs, self._nvar = self.model._fit_y.shape
        self._index = dep_var.index
        self._covariance = covariance
        self._fit_indices = (fit_start, fit_stop)
        self._param_cov = param_cov
        self._r2 = r2
        self.cov_type = cov_type
        self._optim_output = optim_output

    @cached_property
    def params(self):
        return pd.Series(self._params, index=self._names, name='params')

    @cached_property
    def rsquared(self):
        return pd.Series(self._r2, index=self._dep_name, name='rsquared')

    @cached_property
    def resid(self):
        """
        Model residuals
        """
        if self._input_pandas:
            return pd.DataFrame(self._resid, columns=self._dep_name, index=self._index)
        else:
            return self._resid

    @cached_property
    def loglikelihood(self):
        """Model loglikelihood"""
        return self._loglikelihood

    @cached_property
    def aic(self):
        """Akaike Information Criteria

        -2 * loglikelihood + 2 * num_params"""
        return -2 * self.loglikelihood + 2 * self.num_params

    @cached_property
    def num_params(self):
        """Number of parameters in model"""
        return len(self.params)

    @cached_property
    def bic(self):
        """
        Schwarz/Bayesian Information Criteria

        -2 * loglikelihood + log(nobs) * num_params
        """
        return -2 * self.loglikelihood + np.log(self.nobs) * self.num_params

    @cached_property
    def nvar(self):
        """
        Number of data points used to estimate model
        """
        return self._nvar

    @cached_property
    def nobs(self):
        """
        Number of data points used to estimate model
        """
        return self._nobs

    @cached_property
    def fit_start(self):
        return self._fit_indices[0]

    @cached_property
    def fit_stop(self):
        return self._fit_indices[1]

    @cached_property
    def rsquared_adj(self):
        """
        Degree of freedom adjusted R-squared
        """
        return 1 - ((1 - self.rsquared) * (self.nobs - 1) / (self.nobs - self.model.nreg))

    @cached_property
    def pvalues(self):
        """
        Array of p-values for the t-statistics
        """
        return pd.Series(stats.norm.sf(np.abs(self.tvalues)) * 2,
                         index=self._names, name='pvalues')

    @cached_property
    def std_err(self):
        """
        Array of parameter standard errors
        """
        return pd.Series(np.sqrt(np.diag(self.param_cov)),
                         index=self._names, name='std_err')

    @cached_property
    def tvalues(self):
        """
        Array of t-statistics testing the null that the coefficient are 0
        """
        tvalues = self.params / self.std_err
        tvalues.name = 'tvalues'
        return tvalues

    @cached_property
    def convergence_flag(self):
        """
        scipy.optimize.minimize result flag
        """
        return self._optim_output.status

    @cached_property
    def param_cov(self):
        """Parameter covariance"""
        if self._param_cov is not None:
            param_cov = self._param_cov
        else:
            params = np.asarray(self.params)
            if self.cov_type == 'robust':
                param_cov = self.model.compute_param_cov(params)
            else:
                param_cov = self.model.compute_param_cov(params, robust=False)
        return pd.DataFrame(param_cov, columns=self._names, index=self._names)
