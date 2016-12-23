from __future__ import absolute_import, division
from ..compat.python import iteritems, itervalues, add_metaclass, range
from ..utility.array import DocStringInheritor

import copy

import numpy as np
from numpy.random import RandomState
import pandas as pd
import scipy.stats as stats

__all__ = ['IIDBootstrap', 'StationaryBootstrap', 'CircularBlockBootstrap',
           'MovingBlockBootstrap']

try:
    from ._samplers import stationary_bootstrap_sample
except ImportError:  # pragma: no cover
    from ._samplers_python import stationary_bootstrap_sample


def _loo_jackknife(func, nobs, args, kwargs):
    """
    Leave one out jackknife estimation

    Parameters
    ----------
    func : callable
        Function that computes parameters.  Called using func(*args, **kwargs)
    nobs : int
        Number of observation in the data
    args : list
        List of positional inputs (arrays, Series or DataFrames)
    kwargs: dict
        List of keyword inputs (arrays, Series or DataFrames)

    Returns
    -------
    results : array
        Array containing the jackknife results where row i corresponds to
        leaving observation i out of the sample
    """
    results = []
    for i in range(nobs):
        items = np.r_[0:i, i + 1:nobs]
        args_copy = []
        for arg in args:
            if isinstance(arg, (pd.Series, pd.DataFrame)):
                args_copy.append(arg.iloc[items])
            else:
                args_copy.append(arg[items])
        kwargs_copy = {}
        for k, v in iteritems(kwargs):
            if isinstance(v, (pd.Series, pd.DataFrame)):
                kwargs_copy[k] = v.iloc[items]
            else:
                kwargs_copy[k] = v[items]
        results.append(func(*args_copy, **kwargs_copy))
    return np.array(results)


def _add_extra_kwargs(kwargs, extra_kwargs=None):
    """
    Safely add additional keyword arguments to an existing dictionary

    Parameters
    ----------
    kwargs : dict
        Keyword argument dictionary
    extra_kwargs : dict, optional
        Keyword argument dictionary to add

    Returns
    -------
    augmented_kwargs : dict
        Keyword dictionary with added keyword arguments

    Notes
    -----
    There is no checking for duplicate keys
    """
    if extra_kwargs is None:
        return kwargs
    else:
        return dict(list(kwargs.items()) + list(extra_kwargs.items()))


@add_metaclass(DocStringInheritor)
class IIDBootstrap(object):
    """
    Bootstrap using uniform resampling

    Parameters
    ----------
    args
        Positional arguments to bootstrap
    kwargs
        Keyword arguments to bootstrap

    Attributes
    ----------
    index : array
        The current index of the bootstrap
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data)
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered)
    kw_data : dict
        Dictionary containing the keyword arguments
    random_state : RandomState
        RandomState instance used by bootstrap

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input date.

    Data entered using keyword arguments is directly accessibly as an
    attribute.

    Examples
    --------
    Data can be accessed in a number of ways.  Positional data is retained in
    the same order as it was entered when the bootstrap was initialized.
    Keyword data is available both as an attribute or using a dictionary syntax
    on kw_data.

    >>> from arch.bootstrap import IIDBootstrap
    >>> from numpy.random import standard_normal
    >>> y = standard_normal((500, 1))
    >>> x = standard_normal((500,2))
    >>> z = standard_normal(500)
    >>> bs = IIDBootstrap(x, y=y, z=z)
    >>> for data in bs.bootstrap(100):
    ...     bs_x = data[0][0]
    ...     bs_y = data[1]['y']
    ...     bs_z = bs.z
    """

    def __init__(self, *args, **kwargs):
        self.random_state = RandomState()
        self._initial_state = self.random_state.get_state()
        self._args = args
        self._kwargs = kwargs
        if args:
            self._num_items = len(args[0])
        elif kwargs:
            key = list(kwargs.keys())[0]
            self._num_items = len(kwargs[key])

        all_args = list(args)
        all_args.extend([v for v in itervalues(kwargs)])

        for arg in all_args:
            if len(arg) != self._num_items:
                raise ValueError("All inputs must have the same number of "
                                 "elements in axis 0")
        self._index = np.arange(self._num_items)

        self._parameters = []
        self._seed = None
        self.pos_data = args
        self.kw_data = kwargs
        self.data = (args, kwargs)

        self._base = None
        self._results = None
        self._studentized_results = None
        self._last_func = None
        self._name = 'IID Bootstrap'
        for key, value in iteritems(kwargs):
            attr = getattr(self, key, None)
            if attr is None:
                self.__setattr__(key, value)
            else:
                raise ValueError(key + ' is a reserved name')

    def __str__(self):
        txt = self._name
        txt += '(no. pos. inputs: ' + str(len(self.pos_data))
        txt += ', no. keyword inputs: ' + str(len(self.kw_data)) + ')'
        return txt

    def __repr__(self):
        return self.__str__()[:-1] + ', ID: ' + hex(id(self)) + ')'

    def _repr_html(self):
        html = '<strong>' + self._name + '</strong>('
        html += '<strong>no. pos. inputs</strong>: ' + str(len(self.pos_data))
        html += ', <strong>no. keyword inputs</strong>: ' + \
                str(len(self.kw_data))
        html += ', <strong>ID</strong>: ' + hex(id(self)) + ')'
        return html

    @property
    def index(self):
        """
        Returns the current index of the bootstrap
        """
        return self._index

    def get_state(self):
        """
        Gets the state of the bootstrap's random number generator

        Returns
        -------
        state : RandomState state vector
            Array containing the state
        """
        return self.random_state.get_state()

    def set_state(self, state):
        """
        Sets the state of the bootstrap's random number generator

        Parameters
        ----------
        state : RandomState state vector
            Array containing the state
        """

        return self.random_state.set_state(state)

    def seed(self, value):
        """
        Seeds the bootstrap's random number generator

        Parameters
        ----------
        value : int
            Integer to use as the seed
        """
        self._seed = value
        self.random_state.seed(value)
        return None

    def reset(self, use_seed=True):
        """
        Resets the bootstrap to either its initial state or the last seed.

        Parameters
        ----------
        use_seed : bool, optional
            Flag indicating whether to use the last seed if provided.  If
            False or if no seed has been set, the bootstrap will be reset
            to the initial state.  Default is True
        """
        self._index = np.arange(self._num_items)
        self._resample()
        self.random_state.set_state(self._initial_state)
        if use_seed and self._seed is not None:
            self.seed(self._seed)
        return None

    def bootstrap(self, reps):
        """
        Iterator for use when bootstrapping

        Parameters
        ----------
        reps : int
            Number of bootstrap replications

        Example
        -------
        The key steps are problem dependent and so this example shows the use
        as an iterator that does not produce any output

        >>> from arch.bootstrap import IIDBootstrap
        >>> import numpy as np
        >>> bs = IIDBootstrap(np.arange(100), x=np.random.randn(100))
        >>> for posdata, kwdata in bs.bootstrap(1000):
        ...     # Do something with the positional data and/or keyword data
        ...     pass

        .. note::

            Note this is a generic example and so the class used should be the
            name of the required bootstrap

        Notes
        -----
        The iterator returns a tuple containing the data entered in positional
        arguments as a tuple and the data entered using keywords as a
        dictionary
        """
        for _ in range(reps):
            indices = np.asarray(self.update_indices())
            self._index = indices
            yield self._resample()

    def conf_int(self, func, reps=1000, method='basic', size=0.95, tail='two',
                 extra_kwargs=None, reuse=False, sampling='nonparametric',
                 std_err_func=None, studentize_reps=1000):
        """
        Parameters
        ----------
        func : callable
            Function the computes parameter values.  See Notes for requirements
        reps : int, optional
            Number of bootstrap replications
        method : string, optional
            One of 'basic', 'percentile', 'studentized', 'norm' (identical to
            'var', 'cov'), 'bc' (identical to 'debiased', 'bias-corrected'), or
            'bca'
        size : float, optional
            Coverage of confidence interval
        tail : string, optional
            One of 'two', 'upper' or 'lower'.
        reuse : bool, optional
            Flag indicating whether to reuse previously computed bootstrap
            results.  This allows alternative methods to be compared without
            rerunning the bootstrap simulation.  Reuse is ignored if reps is
            not the same across multiple runs, func changes across calls, or
            method is 'studentized'.
        sampling : string, optional
            Type of sampling to use: 'nonparametric', 'semi-parametric' (or
            'semi') or 'parametric'.  The default is 'nonparametric'.  See
            notes about the changes to func required when using 'semi' or
            'parametric'.
        extra_kwargs : dict, optional
            Extra keyword arguments to use when calling func and std_err_func,
            when appropriate
        std_err_func : callable, optional
            Function to use when standardizing estimated parameters when using
            the studentized bootstrap.  Providing an analytical function
            eliminates the need for a nested bootstrap
        studentize_reps : int, optional
            Number of bootstraps to use in the innter component when using the
            studentized bootstrap.  Ignored when ``std_err_func`` is provided

        Returns
        -------
        intervals : 2-d array
            Computed confidence interval.  Row 0 contains the lower bounds, and
            row 1 contains the upper bounds.  Each column corresponds to a
            parameter. When tail is 'lower', all upper bounds are inf.
            Similarly, 'upper' sets all lower bounds to -inf.

        Examples
        --------
        >>> import numpy as np
        >>> def func(x):
        ...     return x.mean(0)
        >>> y = np.random.randn(1000, 2)
        >>> from arch.bootstrap import IIDBootstrap
        >>> bs = IIDBootstrap(y)
        >>> ci = bs.conf_int(func, 1000)

        Notes
        -----
        When there are no extra keyword arguments, the function is called

        .. code:: python

            func(*args, **kwargs)

        where args and kwargs are the bootstrap version of the data provided
        when setting up the bootstrap.  When extra keyword arguments are used,
        these are appended to kwargs before calling func.

        The standard error function, if provided, must return a vector of
        parameter standard errors and is called

        .. code:: python

            std_err_func(params, *args, **kwargs)

        where ``params`` is the vector of estimated parameters using the same
        bootstrap data as in args and kwargs.

        The bootstraps are:

        * 'basic' - Basic confidence using the estimated parameter and
          difference between the estimated parameter and the bootstrap
          parameters
        * 'percentile' - Direct use of bootstrap percentiles
        * 'norm' - Makes use of normal approximation and bootstrap covariance
          estimator
        * 'studentized' - Uses either a standard error function or a nested
          bootstrap to estimate percentiles and the bootstrap covariance for
          scale
        * 'bc' - Bias corrected using estimate bootstrap bias correction
        * 'bca' - Bias corrected and accelerated, adding acceleration parameter
          to 'bc' method

        """
        studentized = 'studentized'
        if not 0.0 < size < 1.0:
            raise ValueError('size must be strictly between 0 and 1')
        tail = tail.lower()
        if tail not in ('two', 'lower', 'upper'):
            raise ValueError('tail must be one of two-sided, lower or upper')
        studentize_reps = studentize_reps if method == studentized else 0

        _reuse = False
        if reuse:
            # check conditions for reuse
            _reuse = (self._results is not None and
                      len(self._results) == reps and
                      method != studentized and
                      self._last_func is func)

        if not _reuse:
            if reuse:
                import warnings

                warn = 'The conditions to reuse the previous bootstrap has ' \
                       'not been satisfied. A new bootstrap will be used.'
                warnings.warn(warn, RuntimeWarning)
            self._construct_bootstrap_estimates(func, reps, extra_kwargs,
                                                std_err_func=std_err_func,
                                                studentize_reps=studentize_reps,  # noqa
                                                sampling=sampling)

        base, results = self._base, self._results
        studentized_results = self._studentized_results

        std_err = []
        if method in ('norm', 'var', 'cov', studentized):
            errors = results - results.mean(axis=0)
            std_err = np.sqrt(np.diag(errors.T.dot(errors) / reps))

        if tail == 'two':
            alpha = (1.0 - size) / 2
        else:
            alpha = (1.0 - size)

        percentiles = [alpha, 1.0 - alpha]
        norm_quantiles = stats.norm.ppf(percentiles)

        if method in ('norm', 'var', 'cov'):
            lower = base + norm_quantiles[0] * std_err
            upper = base + norm_quantiles[1] * std_err

        elif method in ('percentile', 'basic', studentized,
                        'debiased', 'bc', 'bias-corrected', 'bca'):
            values = results
            if method == studentized:
                # studentized uses studentized parameter estimates
                values = studentized_results

            if method in ('debiased', 'bc', 'bias-corrected', 'bca'):
                # bias corrected uses modified percentiles, but is
                # otherwise identical to the percentile method
                p = (results < base).mean(axis=0)
                b = stats.norm.ppf(p)
                b = b[:, None]
                if method == 'bca':
                    nobs = self._num_items
                    jk_params = _loo_jackknife(func, nobs, self._args,
                                               self._kwargs)
                    u = (nobs - 1) * (jk_params - base)
                    numer = np.sum(u ** 3, 0)
                    denom = 6 * (np.sum(u ** 2, 0) ** (3.0 / 2.0))
                    small = denom < (np.abs(numer) * np.finfo(np.float64).eps)
                    if small.any():
                        message = 'Jackknife variance estimate {jk_var} is ' \
                                  'too small to use BCa'
                        raise RuntimeError(message.format(jk_var=denom))
                    a = numer / denom
                    a = a[:, None]
                else:
                    a = 0.0

                percentiles = stats.norm.cdf(b + (b + norm_quantiles) /
                                             (1.0 - a * (b + norm_quantiles)))
                percentiles = list(100 * percentiles)
            else:
                percentiles = [100 * p for p in percentiles]  # Rescale

            if method not in ('bc', 'debiased', 'bias-corrected', 'bca'):
                ci = np.asarray(np.percentile(values, percentiles, axis=0))
                lower = ci[0, :]
                upper = ci[1, :]
            else:
                k = values.shape[1]
                lower = np.zeros(k)
                upper = np.zeros(k)
                for i in range(k):
                    lower[i], upper[i] = np.percentile(values[:, i],
                                                       list(percentiles[i]))

            # Basic and studentized use the lower empirical quantile to
            # compute upper and vice versa.  Bias corrected and percentile use
            # upper to estimate the upper, and lower to estimate the lower
            if method == 'basic':
                lower_copy = lower + 0.0
                lower = 2.0 * base - upper
                upper = 2.0 * base - lower_copy
            elif method == studentized:
                lower_copy = lower + 0.0
                lower = base - upper * std_err
                upper = base - lower_copy * std_err

        else:
            raise ValueError('Unknown method')

        if tail == 'lower':
            upper = np.zeros_like(base)
            upper.fill(np.inf)
        elif tail == 'upper':
            lower = np.zeros_like(base)
            lower.fill(-1 * np.inf)

        return np.vstack((lower, upper))

    def clone(self, *args, **kwargs):
        """
        Clones the bootstrap using different data.

        Parameters
        ----------
        args
            Positional arguments to bootstrap
        kwargs
            Keyword arguments to bootstrap

        Returns
        -------
        bs
            Bootstrap instance
        """
        pos_arguments = copy.deepcopy(self._parameters)
        pos_arguments.extend(args)
        bs = self.__class__(*pos_arguments, **kwargs)
        if self._seed is not None:
            bs.seed(self._seed)
        return bs

    def apply(self, func, reps=1000, extra_kwargs=None):
        """
        Applies a function to bootstrap replicated data

        Parameters
        ----------
        func : callable
            Function the computes parameter values.  See Notes for requirements
        reps : int, optional
            Number of bootstrap replications
        extra_kwargs : dict, optional
            Extra keyword arguments to use when calling func.  Must not
            conflict with keyword arguments used to initialize bootstrap

        Returns
        -------
        results : array
            reps by nparam array of computed function values where each row
            corresponds to a bootstrap iteration

        Notes
        -----
        When there are no extra keyword arguments, the function is called

        .. code:: python

            func(params, *args, **kwargs)

        where args and kwargs are the bootstrap version of the data provided
        when setting up the bootstrap.  When extra keyword arguments are used,
        these are appended to kwargs before calling func

        Examples
        --------
        >>> import numpy as np
        >>> x = np.random.randn(1000,2)
        >>> from arch.bootstrap import IIDBootstrap
        >>> bs = IIDBootstrap(x)
        >>> def func(y):
        ...     return y.mean(0)
        >>> results = bs.apply(func, 100)
        """
        kwargs = _add_extra_kwargs(self._kwargs, extra_kwargs)
        base = func(*self._args, **kwargs)
        try:
            num_params = base.shape[0]
        except:
            num_params = 1
        results = np.zeros((reps, num_params))
        count = 0
        for pos_data, kw_data in self.bootstrap(reps):
            kwargs = _add_extra_kwargs(kw_data, extra_kwargs)
            results[count] = func(*pos_data, **kwargs)
            count += 1
        return results

    def _construct_bootstrap_estimates(self, func, reps, extra_kwargs=None,
                                       std_err_func=None, studentize_reps=0,
                                       sampling='nonparametric'):
        # Private, more complicated version of apply
        self._last_func = func
        semi = parametric = False
        if sampling == 'parametric':
            parametric = True
        elif sampling == 'semiparametric':
            semi = True

        if extra_kwargs is not None:
            if any(k in self._kwargs for k in extra_kwargs):
                raise ValueError('extra_kwargs contains keys used for variable'
                                 ' names in the bootstrap')
        kwargs = _add_extra_kwargs(self._kwargs, extra_kwargs)
        base = func(*self._args, **kwargs)

        num_params = 1 if np.isscalar(base) else base.shape[0]
        results = np.zeros((reps, num_params))
        studentized_results = np.zeros((reps, num_params))

        count = 0
        for pos_data, kw_data in self.bootstrap(reps):
            kwargs = _add_extra_kwargs(kw_data, extra_kwargs)
            if parametric:
                kwargs['state'] = self.random_state
                kwargs['params'] = base
            elif semi:
                kwargs['params'] = base
            results[count] = func(*pos_data, **kwargs)
            if std_err_func is not None:
                std_err = std_err_func(results[count], *pos_data, **kwargs)
                studentized_results[count] = (results[count] - base) / std_err
            elif studentize_reps > 0:
                # Need new bootstrap of same type
                nested_bs = self.clone(*pos_data, **kw_data)
                # Set the seed to ensure reproducability
                seed = self.random_state.randint(2 ** 31 - 1)
                nested_bs.seed(seed)
                cov = nested_bs.cov(func, studentize_reps,
                                    extra_kwargs=extra_kwargs)
                std_err = np.sqrt(np.diag(cov))
                studentized_results[count] = (results[count] - base) / std_err
            count += 1

        self._base = np.asarray(base)
        self._results = np.asarray(results)
        self._studentized_results = np.asarray(studentized_results)

    def cov(self, func, reps=1000, recenter=True, extra_kwargs=None):
        """
        Compute parameter covariance using bootstrap

        Parameters
        ----------
        func : callable
            Callable function that returns the statistic of interest as a
            1-d array
        reps : int, optional
            Number of bootstrap replications
        recenter : bool, optional
            Whether to center the bootstrap variance estimator on the average
            of the bootstrap samples (True) or to center on the original sample
            estimate (False).  Default is True.
        extra_kwargs: dict, optional
            Dictionary of extra keyword arguments to pass to func

        Returns
        -------
        cov: array
            Bootstrap covariance estimator

        Notes
        -----
        func must have the signature

        .. code:: python

            func(params, *args, **kwargs)

        where params are a 1-dimensional array, and `*args` and `**kwargs` are
        data used in the the bootstrap.  The first argument, params, will be
        none when called using the original data, and will contain the estimate
        computed using the original data in bootstrap replications.  This
        parameter is passed to allow parametric bootstrap simulation.

        Example
        -------
        Bootstrap covariance of the mean

        >>> from arch.bootstrap import IIDBootstrap
        >>> import numpy as np
        >>> def func(x):
        ...     return x.mean(axis=0)
        >>> y = np.random.randn(1000, 3)
        >>> bs = IIDBootstrap(y)
        >>> cov = bs.cov(func, 1000)

        Bootstrap covariance using a function that takes additional input

        >>> def func(x, stat='mean'):
        ...     if stat=='mean':
        ...         return x.mean(axis=0)
        ...     elif stat=='var':
        ...         return x.var(axis=0)
        >>> cov = bs.cov(func, 1000, extra_kwargs={'stat':'var'})

        .. note::

            Note this is a generic example and so the class used should be the
            name of the required bootstrap

        """
        self._construct_bootstrap_estimates(func, reps, extra_kwargs)
        base, results = self._base, self._results

        if recenter:
            errors = results - np.mean(results, 0)
        else:
            errors = results - base

        return errors.T.dot(errors) / reps

    def var(self, func, reps=1000, recenter=True, extra_kwargs=None):
        """
        Compute parameter variance using bootstrap

        Parameters
        ----------
        func : callable
            Callable function that returns the statistic of interest as a
            1-d array
        reps : int, optional
            Number of bootstrap replications
        recenter : bool, optional
            Whether to center the bootstrap variance estimator on the average
            of the bootstrap samples (True) or to center on the original sample
            estimate (False).  Default is True.
        extra_kwargs: dict, optional
            Dictionary of extra keyword arguments to pass to func

        Returns
        -------
        var : array
            Bootstrap variance estimator

        Notes
        -----
        func must have the signature

        .. code:: python

            func(params, *args, **kwargs)

        where params are a 1-dimensional array, and `*args` and `**kwargs` are
        data used in the the bootstrap.  The first argument, params, will be
        none when called using the original data, and will contain the estimate
        computed using the original data in bootstrap replications.  This
        parameter is passed to allow parametric bootstrap simulation.

        Example
        -------
        Bootstrap covariance of the mean

        >>> from arch.bootstrap import IIDBootstrap
        >>> import numpy as np
        >>> def func(x):
        ...     return x.mean(axis=0)
        >>> y = np.random.randn(1000, 3)
        >>> bs = IIDBootstrap(y)
        >>> variances = bs.var(func, 1000)

        Bootstrap covariance using a function that takes additional input

        >>> def func(x, stat='mean'):
        ...     if stat=='mean':
        ...         return x.mean(axis=0)
        ...     elif stat=='var':
        ...         return x.var(axis=0)
        >>> variances = bs.var(func, 1000, extra_kwargs={'stat': 'var'})

        .. note::

            Note this is a generic example and so the class used should be the
            name of the required bootstrap

        """
        self._construct_bootstrap_estimates(func, reps, extra_kwargs)
        base, results = self._base, self._results

        if recenter:
            errors = results - np.mean(results, 0)
        else:
            errors = results - base

        return (errors ** 2).sum(0) / reps

    def update_indices(self):
        """
        Update indices for the next iteration of the bootstrap.  This must
        be overridden when creating new bootstraps.
        """
        return self.random_state.randint(self._num_items,
                                         size=self._num_items)

    def _resample(self):
        """
        Resample all data using the values in _index
        """
        indices = self._index
        pos_data = []
        for values in self._args:
            if isinstance(values, (pd.Series, pd.DataFrame)):
                pos_data.append(values.iloc[indices])
            else:
                pos_data.append(values[indices])
        named_data = {}
        for key, values in iteritems(self._kwargs):
            if isinstance(values, (pd.Series, pd.DataFrame)):
                named_data[key] = values.iloc[indices]
            else:
                named_data[key] = values[indices]
            setattr(self, key, named_data[key])

        self.pos_data = pos_data
        self.kw_data = named_data
        self.data = (pos_data, named_data)
        return self.data


class CircularBlockBootstrap(IIDBootstrap):
    """
    Bootstrap based on blocks of the same length with end-to-start wrap around

    Parameters
    ----------
    block_size : int
        Size of block to use
    args
        Positional arguments to bootstrap
    kwargs
        Keyword arguments to bootstrap

    Attributes
    ----------
    index : array
        The current index of the bootstrap
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data)
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered)
    kw_data : dict
        Dictionary containing the keyword arguments
    random_state : RandomState
        RandomState instance used by bootstrap

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input date.

    Data entered using keyword arguments is directly accessibly as an
    attribute.

    Examples
    --------
    Data can be accessed in a number of ways.  Positional data is retained in
    the same order as it was entered when the bootstrap was initialized.
    Keyword data is available both as an attribute or using a dictionary syntax
    on kw_data.

    >>> from arch.bootstrap import CircularBlockBootstrap
    >>> from numpy.random import standard_normal
    >>> y = standard_normal((500, 1))
    >>> x = standard_normal((500, 2))
    >>> z = standard_normal(500)
    >>> bs = CircularBlockBootstrap(17, x, y=y, z=z)
    >>> for data in bs.bootstrap(100):
    ...     bs_x = data[0][0]
    ...     bs_y = data[1]['y']
    ...     bs_z = bs.z
    """

    def __init__(self, block_size, *args, **kwargs):
        super(CircularBlockBootstrap, self).__init__(*args, **kwargs)
        self.block_size = block_size
        self._parameters = [block_size]
        self._name = 'Circular Block Bootstrap'

    def __str__(self):
        txt = self._name
        txt += '(block size: ' + str(self.block_size)
        txt += ', no. pos. inputs: ' + str(len(self.pos_data))
        txt += ', no. keyword inputs: ' + str(len(self.kw_data)) + ')'
        return txt

    def _repr_html(self):
        html = '<strong>' + self._name + '</strong>('
        html += '<strong>block size</strong>: ' + str(self.block_size)
        html += ', <strong>no. pos. inputs</strong>: ' + \
                str(len(self.pos_data))
        html += ', <strong>no. keyword inputs</strong>: ' + \
                str(len(self.kw_data))
        html += ', <strong>ID</strong>: ' + hex(id(self)) + ')'
        return html

    def update_indices(self):
        num_blocks = self._num_items // self.block_size
        if num_blocks * self.block_size < self._num_items:
            num_blocks += 1
        indices = self.random_state.randint(self._num_items, size=num_blocks)
        indices = indices[:, None] + np.arange(self.block_size)
        indices = indices.flatten()
        indices %= self._num_items

        if indices.shape[0] > self._num_items:
            return indices[:self._num_items]
        else:
            return indices


class StationaryBootstrap(CircularBlockBootstrap):
    """
    Politis and Romano (1994) bootstrap with expon. distributed block sizes

    Parameters
    ----------
    block_size : int
        Average size of block to use
    args
        Positional arguments to bootstrap
    kwargs
        Keyword arguments to bootstrap

    Attributes
    ----------
    index : array
        The current index of the bootstrap
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data)
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered)
    kw_data : dict
        Dictionary containing the keyword arguments
    random_state : RandomState
        RandomState instance used by bootstrap

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input date.

    Data entered using keyword arguments is directly accessibly as an
    attribute.

    Examples
    --------
    Data can be accessed in a number of ways.  Positional data is retained in
    the same order as it was entered when the bootstrap was initialized.
    Keyword data is available both as an attribute or using a dictionary syntax
    on kw_data.

    >>> from arch.bootstrap import StationaryBootstrap
    >>> from numpy.random import standard_normal
    >>> y = standard_normal((500, 1))
    >>> x = standard_normal((500,2))
    >>> z = standard_normal(500)
    >>> bs = StationaryBootstrap(12, x, y=y, z=z)
    >>> for data in bs.bootstrap(100):
    ...     bs_x = data[0][0]
    ...     bs_y = data[1]['y']
    ...     bs_z = bs.z
    """

    def __init__(self, block_size, *args, **kwargs):
        super(StationaryBootstrap, self).__init__(block_size, *args, **kwargs)
        self._name = 'Stationary Bootstrap'
        self._p = 1.0 / block_size
        self._name = 'Stationary Bootstrap'

    def update_indices(self):
        indices = self.random_state.randint(self._num_items,
                                            size=self._num_items)
        indices = indices.astype(np.int64)
        u = self.random_state.random_sample(self._num_items)
        return stationary_bootstrap_sample(indices, u, self._p)


class MovingBlockBootstrap(CircularBlockBootstrap):
    """
    Bootstrap based on blocks of the same length without wrap around

    Parameters
    ----------
    block_size : int
        Size of block to use
    args
        Positional arguments to bootstrap
    kwargs
        Keyword arguments to bootstrap

    Attributes
    ----------
    index : array
        The current index of the bootstrap
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data)
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered)
    kw_data : dict
        Dictionary containing the keyword arguments
    random_state : RandomState
        RandomState instance used by bootstrap

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input date.

    Data entered using keyword arguments is directly accessibly as an
    attribute.

    Examples
    --------
    Data can be accessed in a number of ways.  Positional data is retained in
    the same order as it was entered when the bootstrap was initialized.
    Keyword data is available both as an attribute or using a dictionary syntax
    on kw_data.

    >>> from arch.bootstrap import MovingBlockBootstrap
    >>> from numpy.random import standard_normal
    >>> y = standard_normal((500, 1))
    >>> x = standard_normal((500,2))
    >>> z = standard_normal(500)
    >>> bs = MovingBlockBootstrap(7, x, y=y, z=z)
    >>> for data in bs.bootstrap(100):
    ...     bs_x = data[0][0]
    ...     bs_y = data[1]['y']
    ...     bs_z = bs.z
    """

    def __init__(self, block_size, *args, **kwargs):
        super(MovingBlockBootstrap, self).__init__(block_size, *args, **kwargs)
        self._name = 'Moving Block Bootstrap'

    def update_indices(self):
        num_blocks = self._num_items // self.block_size
        if num_blocks * self.block_size < self._num_items:
            num_blocks += 1
        max_index = self._num_items - self.block_size + 1
        indices = self.random_state.randint(max_index, size=num_blocks)
        indices = indices[:, None] + np.arange(self.block_size)
        indices = indices.flatten()

        if indices.shape[0] > self._num_items:
            return indices[:self._num_items]
        else:
            return indices


class MOONBootstrap(IIDBootstrap):  # pragma: no cover

    def __init__(self, block_size, *args, **kwargs):  # pragma: no cover
        super(MOONBootstrap, self).__init__(*args, **kwargs)
        self.block_size = block_size

    def update_indices(self):  # pragma: no cover
        raise NotImplementedError
