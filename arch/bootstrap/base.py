import copy
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.random import RandomState
import pandas as pd
import scipy.stats as stats

from arch.typing import ArrayLike, NDArray
from arch.utility.array import DocStringInheritor, ensure2d
from arch.utility.exceptions import (
    StudentizationError,
    arg_type_error,
    kwarg_type_error,
    studentization_error,
)

__all__ = [
    "IIDBootstrap",
    "StationaryBootstrap",
    "CircularBlockBootstrap",
    "MovingBlockBootstrap",
    "IndependentSamplesBootstrap",
    "optimal_block_length",
]

try:
    from arch.bootstrap._samplers import stationary_bootstrap_sample
except ImportError:  # pragma: no cover
    from arch.bootstrap._samplers_python import stationary_bootstrap_sample


def _single_optimal_block(x: NDArray) -> Tuple[float, float]:
    """
    Compute the optimal window length for a single series

    Parameters
    ----------
    x : ndarray
        The data to use in the optimal window estimation

    Returns
    -------
    stationary : float
        Estimated optimal window length for stationary bootstrap
    circular : float
        Estimated optimal window length for circular bootstrap
    """
    nobs = x.shape[0]
    eps = x - x.mean(0)
    b_max = np.ceil(min(3 * np.sqrt(nobs), nobs / 3))
    kn = max(5, int(np.log10(nobs)))
    m_max = int(np.ceil(np.sqrt(nobs))) + kn
    # Find first collection of kn autocorrelations that are insignificant
    cv = 2 * np.sqrt(np.log10(nobs) / nobs)
    acv = np.zeros(m_max + 1)
    abs_acorr = np.zeros(m_max + 1)
    opt_m: Optional[int] = None
    for i in range(m_max + 1):
        v1 = eps[i + 1 :] @ eps[i + 1 :]
        v2 = eps[: -(i + 1)] @ eps[: -(i + 1)]
        cross_prod = eps[i:] @ eps[: nobs - i]
        acv[i] = cross_prod / nobs
        abs_acorr[i] = np.abs(cross_prod) / np.sqrt(v1 * v2)
        if i >= kn:
            if np.all(abs_acorr[i - kn : i] < cv) and opt_m is None:
                opt_m = i - kn
    m = 2 * max(opt_m, 1) if opt_m is not None else m_max
    m = min(m, m_max)

    g = 0.0
    lr_acv = acv[0]
    for k in range(1, m + 1):
        lam = 1 if k / m <= 1 / 2 else 2 * (1 - k / m)
        g += 2 * lam * k * acv[k]
        lr_acv += 2 * lam * acv[k]
    d_sb = 2 * lr_acv ** 2
    d_cb = 4 / 3 * lr_acv ** 2
    b_sb = ((2 * g ** 2) / d_sb) ** (1 / 3) * nobs ** (1 / 3)
    b_cb = ((2 * g ** 2) / d_cb) ** (1 / 3) * nobs ** (1 / 3)
    b_sb = min(b_sb, b_max)
    b_cb = min(b_cb, b_max)
    return b_sb, b_cb


def optimal_block_length(x: ArrayLike) -> pd.DataFrame:
    r"""
    Estimate optimal window length for time-series bootstraps

    Parameters
    ----------
    x : array_like
        A one-dimensional or two-dimensional array-like.  Operates columns by
        column if 2-dimensional.

    Returns
    -------
    DataFrame
        A DataFrame with two columns `b_sb`, the estimated optimal block size
        for the Stationary Bootstrap and `b_cb`, the estimated optimal block
        size for the circular bootstrap.

    See Also
    --------
    arch.bootstrap.StationaryBootstrap
       Politis and Romano's bootstrap with exp. distributed block lengths
    arch.bootstrap.CircularBlockBootstrap
       Circular (wrap-around) bootstrap

    Notes
    -----
    Algorithm described in ([1]_) its correction ([2]_) depends on a tuning
    parameter m, which is chosen as the first value where k_n consecutive
    autocorrelations of x are all inside a conservative band of
    :math:`\pm 2\sqrt{\log_{10}(n)/n}` where n is the sample size. The maximum
    value of m is set to :math:`\lceil \sqrt{n} + k_n \rceil` where
    :math:`k_n=\max(5, \log_{10}(n))`. The block length is then computed as

    .. math::

       b^{OPT}_i = \left(\frac{2g^2}{d_{i}} n\right)^{\frac{1}{3}}

    where

    .. math::

       g & = \sum_{k=-m}^m h\left(\frac{k}{m}\right)|k|\hat{\gamma_{k}} \\
       h(x) & = \min(1, 2(1-|x|)) \\
       d_{i} & = c_{i} \left(\hat{\sigma}^2\right)^2 \\
       \hat{\sigma}^2 & = \sum_{k=-m}^m h\left(\frac{k}{m}\right)\hat{\gamma_{k}} \\
       \hat{\gamma_{i}} & = n^{-1} \sum_{k=i+1}^n
                          \left(x_k-\bar{x}\right)\left(x_{k-i}-\bar{x}\right) \\

    and the two remaining constants :math:`c_i` are 2 for the Stationary
    bootstrap and 4/3 for the Circular bootstrap.

    Some of the tuning parameters are taken from Andrew Patton's MATLAB
    program that computes the optimal block length.  The block lengths do
    not match this implementation since the autocovariances and
    autocorrelations are all computed using the maximum sample length
    rather than a common sampling length.

    References
    ----------
    .. [1] Dimitris N. Politis & Halbert White (2004) Automatic Block-Length
       Selection for the Dependent Bootstrap, Econometric Reviews, 23:1,
       53-70, DOI: 10.1081/ETC-120028836.
    .. [2] Andrew Patton , Dimitris N. Politis & Halbert White (2009)
       Correction to “Automatic Block-Length Selection for the Dependent
       Bootstrap” by D. Politis and H. White, Econometric Reviews, 28:4,
       372-375, DOI: 10.1080/07474930802459016.
    """
    x_arr = ensure2d(np.asarray(x), "x")
    opt = [_single_optimal_block(col) for col in x_arr.T]
    if isinstance(x, (pd.DataFrame, pd.Series)):
        idx = [x.name] if isinstance(x, pd.Series) else list(x.columns)
    else:
        idx = [i for i in range(x_arr.shape[1])]
    return pd.DataFrame(opt, index=idx, columns=["stationary", "circular"])


def _get_acceleration(jk_params: NDArray) -> float:
    """
    Estimates the BCa acceleration parameter using jackknife estimates
    of theta.

    Parameters
    ----------
    jk_params : ndarray
        Array containing the jackknife results where row i corresponds to
        leaving observation i out of the sample. Returned by _loo_jackknife.

    Returns
    -------
    float
        Value of the acceleration parameter "a" used in the BCa bootstrap.
    """
    u = jk_params.mean() - jk_params
    numer = np.sum(u ** 3, 0)
    denom = 6 * (np.sum(u ** 2, 0) ** (3.0 / 2.0))
    small = denom < (np.abs(numer) * np.finfo(np.float64).eps)
    if small.any():
        message = "Jackknife variance estimate {jk_var} is too small to use BCa"
        raise RuntimeError(message.format(jk_var=denom))
    a = numer / denom
    a = np.atleast_1d(a)
    return a[:, None]


def _loo_jackknife(
    func: Callable[..., NDArray],
    nobs: int,
    args: Sequence[ArrayLike],
    kwargs: Dict[str, ArrayLike],
    extra_kwargs: Optional[Dict[str, ArrayLike]] = None,
) -> NDArray:
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
    kwargs : dict
        List of keyword inputs (arrays, Series or DataFrames)

    Returns
    -------
    ndarray
        Array containing the jackknife results where row i corresponds to
        leaving observation i out of the sample
    """
    results = []
    for i in range(nobs):
        items = np.r_[0:i, i + 1 : nobs]
        args_copy = []
        for arg in args:
            if isinstance(arg, (pd.Series, pd.DataFrame)):
                args_copy.append(arg.iloc[items])
            else:
                args_copy.append(arg[items])
        kwargs_copy = {}
        for k, v in kwargs.items():
            if isinstance(v, (pd.Series, pd.DataFrame)):
                kwargs_copy[k] = v.iloc[items]
            else:
                kwargs_copy[k] = v[items]
        if extra_kwargs is not None:
            kwargs_copy.update(extra_kwargs)
        results.append(func(*args_copy, **kwargs_copy))
    return np.array(results)


def _add_extra_kwargs(
    kwargs: Dict[str, Any], extra_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Safely add additional keyword arguments to an existing dictionary

    Parameters
    ----------
    kwargs : dict
        Keyword argument dictionary
    extra_kwargs : dict, default None
        Keyword argument dictionary to add

    Returns
    -------
    dict
        Keyword dictionary with added keyword arguments

    Notes
    -----
    There is no checking for duplicate keys
    """
    if extra_kwargs is None:
        return kwargs
    else:
        kwargs_copy = kwargs.copy()
        kwargs_copy.update(extra_kwargs)
        return kwargs_copy


class IIDBootstrap(object, metaclass=DocStringInheritor):
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
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data)
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered)
    kw_data : dict
        Dictionary containing the keyword arguments

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input date.

    Data entered using keyword arguments is directly accessibly as an
    attribute.

    To ensure a reproducible bootstrap, you must set the ``random_state``
    attribute after the bootstrap has been created. See the example below.
    Note that ``random_state`` is a reserved keyword and any variable
    passed using this keyword must be an instance of ``RandomState``.

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

    Set the random_state if reproducibility is required

    >>> from numpy.random import RandomState
    >>> rs = RandomState(1234)
    >>> bs = IIDBootstrap(x, y=y, z=z, random_state=rs)

    See also
    --------
    arch.bootstrap.IndependentSamplesBootstrap
    """

    _name = "IID Bootstrap"
    _common_size_required = True

    def __init__(
        self, *args: ArrayLike, **kwargs: Union[RandomState, ArrayLike]
    ) -> None:
        self._args = list(args)
        self._kwargs = kwargs
        random_state = self._kwargs.pop("random_state", None)

        if isinstance(random_state, RandomState):
            self._random_state = random_state
        elif random_state is None:
            self._random_state = RandomState()
        else:
            raise TypeError(
                "random_state keyword argument must contain a RandomState instance when used."
            )

        self._initial_state = self._random_state.get_state()

        self._check_data()
        if args:
            self._num_items = len(args[0])
        elif kwargs:
            key = list(kwargs.keys())[0]
            self._num_items = len(kwargs[key])
        all_args = list(args)
        all_args.extend([v for v in kwargs.values()])
        if self._common_size_required:
            for arg in all_args:
                if len(arg) != self._num_items:
                    raise ValueError(
                        "All inputs must have the same number of elements in axis 0"
                    )
        self._index = np.arange(self._num_items)

        self._parameters: List[int] = []
        self._seed: Optional[Union[int, List[int], NDArray]] = None
        self.pos_data = args
        self.kw_data = kwargs
        self.data = (self.pos_data, self.kw_data)

        self._base: Optional[NDArray] = None
        self._results: Optional[NDArray] = None
        self._studentized_results = None
        self._last_func: Optional[Callable[..., ArrayLike]] = None
        for key, value in kwargs.items():
            attr = getattr(self, key, None)
            if attr is None:
                self.__setattr__(key, value)
            else:
                raise ValueError(key + " is a reserved name")

    def __str__(self) -> str:
        txt = self._name
        txt += "(no. pos. inputs: " + str(len(self.pos_data))
        txt += ", no. keyword inputs: " + str(len(self.kw_data)) + ")"
        return txt

    def __repr__(self) -> str:
        return self.__str__()[:-1] + ", ID: " + hex(id(self)) + ")"

    def _repr_html(self) -> str:
        html = "<strong>" + self._name + "</strong>("
        html += "<strong>no. pos. inputs</strong>: " + str(len(self.pos_data))
        html += ", <strong>no. keyword inputs</strong>: " + str(len(self.kw_data))
        html += ", <strong>ID</strong>: " + hex(id(self)) + ")"
        return html

    @property
    def random_state(self) -> np.random.RandomState:
        """
        Set or get the instance random state

        Parameters
        ----------
        random_state : RandomState
            RandomState instance used by bootstrap

        Returns
        -------
        RandomState
            RandomState instance used by bootstrap
        """
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: np.random.RandomState) -> None:
        if not isinstance(random_state, RandomState):
            raise TypeError("Value being set must be a RandomState")
        self._random_state = random_state

    @property
    def index(self) -> NDArray:
        """
        The current index of the bootstrap
        """
        return self._index

    def get_state(self) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        """
        Gets the state of the bootstrap's random number generator

        Returns
        -------
        {dict, tuple}
            Dictionary or tuple containing the state.
        """
        return self.random_state.get_state()

    def set_state(self, state: Union[Dict[str, Any], Tuple[Any, ...]]) -> None:
        """
        Sets the state of the bootstrap's random number generator

        Parameters
        ----------
        state : {dict, tuple}
            Dictionary or tuple containing the state.
        """
        self.random_state.set_state(state)

    def seed(self, value: Union[int, List[int], NDArray]) -> None:
        """
        Seeds the bootstrap's random number generator

        Parameters
        ----------
        value : {int, List[int], ndarray}
            Value to use as the seed.
        """
        self._seed = value
        self.random_state.seed(value)

    def reset(self, use_seed: bool = True) -> None:
        """
        Resets the bootstrap to either its initial state or the last seed.

        Parameters
        ----------
        use_seed : bool, default True
            Flag indicating whether to use the last seed if provided.  If
            False or if no seed has been set, the bootstrap will be reset
            to the initial state.  Default is True
        """
        self._index = np.arange(self._num_items)
        self._resample()
        self.random_state.set_state(self._initial_state)
        if use_seed and self._seed is not None:
            self.seed(self._seed)

    def bootstrap(
        self, reps: int
    ) -> Generator[Tuple[Tuple[ArrayLike, ...], Dict[str, ArrayLike]], None, None]:
        """
        Iterator for use when bootstrapping

        Parameters
        ----------
        reps : int
            Number of bootstrap replications

        Returns
        -------
        generator
            Generator to iterate over in bootstrap calculations

        Examples
        --------
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
            self._index = self.update_indices()
            yield self._resample()

    def conf_int(
        self,
        func: Callable[..., ArrayLike],
        reps: int = 1000,
        method: str = "basic",
        size: float = 0.95,
        tail: str = "two",
        extra_kwargs: Optional[Dict[str, Any]] = None,
        reuse: bool = False,
        sampling: str = "nonparametric",
        std_err_func: Optional[Callable[..., ArrayLike]] = None,
        studentize_reps: int = 1000,
    ) -> NDArray:
        """
        Parameters
        ----------
        func : callable
            Function the computes parameter values.  See Notes for requirements
        reps : int, default 1000
            Number of bootstrap replications
        method : string, default "basic"
            One of 'basic', 'percentile', 'studentized', 'norm' (identical to
            'var', 'cov'), 'bc' (identical to 'debiased', 'bias-corrected'), or
            'bca'
        size : float, default 0.95
            Coverage of confidence interval
        tail : string, default "two"
            One of 'two', 'upper' or 'lower'.
        reuse : bool, default False
            Flag indicating whether to reuse previously computed bootstrap
            results.  This allows alternative methods to be compared without
            rerunning the bootstrap simulation.  Reuse is ignored if reps is
            not the same across multiple runs, func changes across calls, or
            method is 'studentized'.
        sampling : string, default "nonparametric"
            Type of sampling to use: 'nonparametric', 'semi-parametric' (or
            'semi') or 'parametric'.  The default is 'nonparametric'.  See
            notes about the changes to func required when using 'semi' or
            'parametric'.
        extra_kwargs : dict, default None
            Extra keyword arguments to use when calling func and std_err_func,
            when appropriate
        std_err_func : callable, default None
            Function to use when standardizing estimated parameters when using
            the studentized bootstrap.  Providing an analytical function
            eliminates the need for a nested bootstrap
        studentize_reps : int, default 1000
            Number of bootstraps to use in the inner bootstrap when using the
            studentized bootstrap.  Ignored when ``std_err_func`` is provided

        Returns
        -------
        ndarray
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
        studentized = "studentized"
        if not 0.0 < size < 1.0:
            raise ValueError("size must be strictly between 0 and 1")
        tail = tail.lower()
        if tail not in ("two", "lower", "upper"):
            raise ValueError("tail must be one of two-sided, lower or upper")
        studentize_reps = studentize_reps if method == studentized else 0

        _reuse = False
        if reuse:
            # check conditions for reuse
            _reuse = (
                self._results is not None
                and len(self._results) == reps
                and method != studentized
                and self._last_func is func
            )

        if not _reuse:
            if reuse:
                import warnings

                warn = (
                    "The conditions to reuse the previous bootstrap has "
                    "not been satisfied. A new bootstrap will be used."
                )
                warnings.warn(warn, RuntimeWarning)
            self._construct_bootstrap_estimates(
                func,
                reps,
                extra_kwargs,
                std_err_func=std_err_func,
                studentize_reps=studentize_reps,  # noqa
                sampling=sampling,
            )

        base, results = self._base, self._results
        assert results is not None
        assert base is not None
        studentized_results = self._studentized_results

        std_err = []
        if method in ("norm", "var", "cov", studentized):
            errors = results - results.mean(axis=0)
            std_err = np.sqrt(np.diag(errors.T.dot(errors) / reps))

        if tail == "two":
            alpha = (1.0 - size) / 2
        else:
            alpha = 1.0 - size
        nreps = 1 if not base.shape else base.shape[0]
        percentiles = np.array([[alpha, 1.0 - alpha]] * nreps)
        norm_quantiles = stats.norm.ppf(percentiles)

        if method in ("norm", "var", "cov"):
            lower = base + norm_quantiles[:, 0] * std_err
            upper = base + norm_quantiles[:, 1] * std_err

        elif method in (
            "percentile",
            "basic",
            studentized,
            "debiased",
            "bc",
            "bias-corrected",
            "bca",
        ):
            values = results
            if method == studentized:
                # studentized uses studentized parameter estimates
                values = studentized_results

            if method in ("debiased", "bc", "bias-corrected", "bca"):
                # bias corrected uses modified percentiles, but is
                # otherwise identical to the percentile method
                b = self._bca_bias()
                if method == "bca":
                    lens = [len(arg) for arg in self._args] + [
                        len(kwarg) for kwarg in self._kwargs.values()
                    ]
                    if min(lens) != max(lens):
                        raise ValueError(
                            "BCa cannot be applied to statistics "
                            "computed from datasets with "
                            "different lengths"
                        )
                    a = self._bca_acceleration(func, extra_kwargs)
                else:
                    a = 0.0
                percentiles = stats.norm.cdf(
                    b + (b + norm_quantiles) / (1.0 - a * (b + norm_quantiles))
                )
                percentiles = list(100 * percentiles)
            else:
                percentiles = [100 * p for p in percentiles]  # Rescale

            k = values.shape[1]
            lower = np.zeros(k)
            upper = np.zeros(k)
            for i in range(k):
                lower[i], upper[i] = np.percentile(values[:, i], list(percentiles[i]))
            # Basic and studentized use the lower empirical quantile to
            # compute upper and vice versa.  Bias corrected and percentile use
            # upper to estimate the upper, and lower to estimate the lower
            if method == "basic":
                lower_copy = lower + 0.0
                lower = 2.0 * base - upper
                upper = 2.0 * base - lower_copy
            elif method == studentized:
                lower_copy = lower + 0.0
                lower = base - upper * std_err
                upper = base - lower_copy * std_err

        else:
            raise ValueError("Unknown method")

        if tail == "lower":
            upper = np.zeros_like(base)
            upper.fill(np.inf)
        elif tail == "upper":
            lower = np.zeros_like(base)
            lower.fill(-1 * np.inf)

        return np.vstack((lower, upper))

    def _check_data(self) -> None:
        supported = (np.ndarray, pd.DataFrame, pd.Series)
        for i, arg in enumerate(self._args):
            if not isinstance(arg, supported):
                raise TypeError(arg_type_error.format(i=i, arg_type=type(arg)))
        for key in self._kwargs:
            if not isinstance(self._kwargs[key], supported):
                arg_type = type(self._kwargs[key])
                raise TypeError(kwarg_type_error.format(key=key, arg_type=arg_type))

    def _bca_bias(self) -> NDArray:
        assert self._results is not None
        assert self._base is not None
        p = (self._results < self._base).mean(axis=0)
        b = stats.norm.ppf(p)
        return b[:, None]

    def _bca_acceleration(
        self, func: Callable[..., ArrayLike], extra_kwags: Optional[Dict[str, Any]]
    ) -> float:
        nobs = self._num_items
        jk_params = _loo_jackknife(func, nobs, self._args, self._kwargs, extra_kwags)
        return _get_acceleration(jk_params)

    def clone(self, *args: ArrayLike, **kwargs: ArrayLike) -> "IIDBootstrap":
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
        pos_arguments: List[Union[int, ArrayLike]] = copy.deepcopy(self._parameters)
        pos_arguments.extend(args)
        bs = self.__class__(*pos_arguments, **kwargs)
        if self._seed is not None:
            bs.seed(self._seed)
        return bs

    def apply(
        self,
        func: Callable[..., ArrayLike],
        reps: int = 1000,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> NDArray:
        """
        Applies a function to bootstrap replicated data

        Parameters
        ----------
        func : callable
            Function the computes parameter values.  See Notes for requirements
        reps : int, default 1000
            Number of bootstrap replications
        extra_kwargs : dict, default None
            Extra keyword arguments to use when calling func.  Must not
            conflict with keyword arguments used to initialize bootstrap

        Returns
        -------
        ndarray
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
        except (IndexError, AttributeError):
            num_params = 1
        results = np.zeros((reps, num_params))
        count = 0
        for pos_data, kw_data in self.bootstrap(reps):
            kwargs = _add_extra_kwargs(kw_data, extra_kwargs)
            results[count] = func(*pos_data, **kwargs)
            count += 1
        return results

    def _construct_bootstrap_estimates(
        self,
        func: Callable[..., ArrayLike],
        reps: int,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        std_err_func: Optional[Callable[..., ArrayLike]] = None,
        studentize_reps: int = 0,
        sampling: str = "nonparametric",
    ) -> None:
        eps = np.finfo(np.double).eps
        # Private, more complicated version of apply
        self._last_func = func
        semi = parametric = False
        if sampling == "parametric":
            parametric = True
        elif sampling == "semiparametric":
            semi = True

        if extra_kwargs is not None:
            if any(k in self._kwargs for k in extra_kwargs):
                raise ValueError(
                    "extra_kwargs contains keys used for variable"
                    " names in the bootstrap"
                )
        kwargs = _add_extra_kwargs(self._kwargs, extra_kwargs)
        base = func(*self._args, **kwargs)

        num_params = 1 if np.isscalar(base) else base.shape[0]
        results = np.zeros((reps, num_params))
        studentized_results = np.zeros((reps, num_params))

        count = 0
        for pos_data, kw_data in self.bootstrap(reps):
            kwargs = _add_extra_kwargs(kw_data, extra_kwargs)
            if parametric:
                kwargs["state"] = self.random_state
                kwargs["params"] = base
            elif semi:
                kwargs["params"] = base
            results[count] = func(*pos_data, **kwargs)
            if std_err_func is not None:
                std_err = std_err_func(results[count], *pos_data, **kwargs)
                studentized_results[count] = (results[count] - base) / std_err
            elif studentize_reps > 0:
                # Need new bootstrap of same type
                nested_bs = self.clone(*pos_data, **kw_data)
                # Set the seed to ensure reproducibility
                seed = self.random_state.randint(2 ** 31 - 1)
                nested_bs.seed(seed)
                cov = nested_bs.cov(func, studentize_reps, extra_kwargs=extra_kwargs)
                std_err = np.sqrt(np.diag(cov))
                err = results[count] - base
                if np.any(std_err <= (eps * np.abs(err))):
                    raise StudentizationError(studentization_error.format(cov=cov))
                studentized_results[count] = err / std_err
            count += 1

        self._base = np.asarray(base)
        self._results = np.asarray(results)
        self._studentized_results = np.asarray(studentized_results)

    def cov(
        self,
        func: Callable[..., ArrayLike],
        reps: int = 1000,
        recenter: bool = True,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[float, NDArray]:
        """
        Compute parameter covariance using bootstrap

        Parameters
        ----------
        func : callable
            Callable function that returns the statistic of interest as a
            1-d array
        reps : int, default 1000
            Number of bootstrap replications
        recenter : bool, default True
            Whether to center the bootstrap variance estimator on the average
            of the bootstrap samples (True) or to center on the original sample
            estimate (False).  Default is True.
        extra_kwargs : dict, default None
            Dictionary of extra keyword arguments to pass to func

        Returns
        -------
        ndarray
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

        Examples
        --------
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
        assert results is not None
        assert base is not None
        if recenter:
            errors = results - np.mean(results, 0)
        else:
            errors = results - base

        return errors.T.dot(errors) / reps

    def var(
        self,
        func: Callable[..., ArrayLike],
        reps: int = 1000,
        recenter: bool = True,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[float, NDArray]:
        """
        Compute parameter variance using bootstrap

        Parameters
        ----------
        func : callable
            Callable function that returns the statistic of interest as a
            1-d array
        reps : int, default 1000
            Number of bootstrap replications
        recenter : bool, default True
            Whether to center the bootstrap variance estimator on the average
            of the bootstrap samples (True) or to center on the original sample
            estimate (False).  Default is True.
        extra_kwargs: dict, default None
            Dictionary of extra keyword arguments to pass to func

        Returns
        -------
        ndarray
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

        Examples
        --------
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
        assert results is not None
        assert base is not None
        if recenter:
            errors = results - np.mean(results, 0)
        else:
            errors = results - base

        return (errors ** 2).sum(0) / reps

    def update_indices(self) -> NDArray:
        """
        Update indices for the next iteration of the bootstrap.  This must
        be overridden when creating new bootstraps.
        """
        return self._random_state.randint(self._num_items, size=self._num_items)

    def _resample(self) -> Tuple[Tuple[ArrayLike, ...], Dict[str, ArrayLike]]:
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
        for key, values in self._kwargs.items():
            if isinstance(values, (pd.Series, pd.DataFrame)):
                named_data[key] = values.iloc[indices]
            else:
                named_data[key] = values[indices]
            setattr(self, key, named_data[key])

        self.pos_data = tuple(pos_data)
        self.kw_data = named_data
        self.data = (self.pos_data, self.kw_data)
        return self.data


class IndependentSamplesBootstrap(IIDBootstrap):
    """
    Bootstrap where each input is independently resampled

    Parameters
    ----------
    args
        Positional arguments to bootstrap
    kwargs
        Keyword arguments to bootstrap

    Attributes
    ----------
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data)
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered)
    kw_data : dict
        Dictionary containing the keyword arguments

    Notes
    -----
    This bootstrap independently resamples each input and so is only
    appropriate when the inputs are independent. This structure allows
    bootstrapping statistics that depend on samples with unequal length, as
    is common in some experiments. If data have cross-sectional dependence, so
    that observation ``i`` is related across all inputs, this bootstrap is
    inappropriate.

    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input date.

    Data entered using keyword arguments is directly accessibly as an
    attribute.

    To ensure a reproducible bootstrap, you must set the ``random_state``
    attribute after the bootstrap has been created. See the example below.
    Note that ``random_state`` is a reserved keyword and any variable
    passed using this keyword must be an instance of ``RandomState``.

    Examples
    --------
    Data can be accessed in a number of ways.  Positional data is retained in
    the same order as it was entered when the bootstrap was initialized.
    Keyword data is available both as an attribute or using a dictionary syntax
    on kw_data.

    >>> from arch.bootstrap import IndependentSamplesBootstrap
    >>> from numpy.random import standard_normal
    >>> y = standard_normal(500)
    >>> x = standard_normal(200)
    >>> z = standard_normal(2000)
    >>> bs = IndependentSamplesBootstrap(x, y=y, z=z)
    >>> for data in bs.bootstrap(100):
    ...     bs_x = data[0][0]
    ...     bs_y = data[1]['y']
    ...     bs_z = bs.z

    Set the random_state if reproducibility is required

    >>> from numpy.random import RandomState
    >>> rs = RandomState(1234)
    >>> bs = IndependentSamplesBootstrap(x, y=y, z=z, random_state=rs)

    See also
    --------
    arch.bootstrap.IIDBootstrap
    """

    _common_size_required = False
    _name = "Heterogeneous IID Bootstrap"

    def __init__(
        self, *args: ArrayLike, **kwargs: Union[RandomState, ArrayLike]
    ) -> None:
        super().__init__(*args, **kwargs)

        self._num_args = len(args)
        self._num_arg_items = [len(arg) for arg in args]
        self._num_kw_items = {key: len(kwargs[key]) for key in self._kwargs}

    def update_indices(self) -> Tuple[List[NDArray], Dict[str, NDArray]]:
        """
        Update indices for the next iteration of the bootstrap.  This must
        be overridden when creating new bootstraps.
        """
        randint = self._random_state.randint
        pos_indices = [
            randint(self._num_arg_items[i], size=self._num_arg_items[i])
            for i in range(self._num_args)
        ]
        kw_indices = {
            key: randint(self._num_kw_items[key], size=self._num_kw_items[key])
            for key in self._kwargs
        }
        return pos_indices, kw_indices

    @property
    def index(self) -> Tuple[List[NDArray], Dict[str, NDArray]]:
        """
        Returns the current index of the bootstrap

        Returns
        -------
        tuple[list[ndarray], dict[str, ndarray]]
            2-element tuple containing a list and a dictionary. The list
            contains indices for each of the positional arguments.  The
            dictionary contains the indices of keyword arguments.
        """
        return self._index

    def reset(self, use_seed: bool = True) -> None:
        """
        Resets the bootstrap to either its initial state or the last seed.

        Parameters
        ----------
        use_seed : bool, default True
            Flag indicating whether to use the last seed if provided.  If
            False or if no seed has been set, the bootstrap will be reset
            to the initial state.  Default is True
        """
        pos_indices = [np.arange(self._num_arg_items[i]) for i in range(self._num_args)]
        kw_indices = {key: np.arange(self._num_kw_items[key]) for key in self._kwargs}
        self._index = pos_indices, kw_indices
        self._resample()
        self.random_state.set_state(self._initial_state)
        if use_seed and self._seed is not None:
            self.seed(self._seed)

    def _resample(self) -> Tuple[Tuple[ArrayLike, ...], Dict[str, ArrayLike]]:
        """
        Resample all data using the values in _index
        """
        pos_indices, kw_indices = self._index
        pos_data = []
        for i, values in enumerate(self._args):
            if isinstance(values, (pd.Series, pd.DataFrame)):
                pos_data.append(values.iloc[pos_indices[i]])
            else:
                pos_data.append(values[pos_indices[i]])
        named_data = {}
        for key, values in self._kwargs.items():
            if isinstance(values, (pd.Series, pd.DataFrame)):
                named_data[key] = values.iloc[kw_indices[key]]
            else:
                named_data[key] = values[kw_indices[key]]
            setattr(self, key, named_data[key])

        self.pos_data = tuple(pos_data)
        self.kw_data = named_data
        self.data = (self.pos_data, named_data)
        return self.data


class CircularBlockBootstrap(IIDBootstrap):
    """
    Bootstrap using blocks of the same length with end-to-start wrap around

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
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data)
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered)
    kw_data : dict
        Dictionary containing the keyword arguments

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input date.

    Data entered using keyword arguments is directly accessibly as an
    attribute.

    To ensure a reproducible bootstrap, you must set the ``random_state``
    attribute after the bootstrap has been created. See the example below.
    Note that ``random_state`` is a reserved keyword and any variable
    passed using this keyword must be an instance of ``RandomState``.

    See Also
    --------
    arch.bootstrap.optimal_block_length
       Optimal block length estimation
    arch.bootstrap.StationaryBootstrap
       Politis and Romano's bootstrap with exp. distributed block lengths

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

    Set the random_state if reproducibility is required

    >>> from numpy.random import RandomState
    >>> rs = RandomState(1234)
    >>> bs = CircularBlockBootstrap(17, x, y=y, z=z, random_state=rs)
    """

    _name = "Circular Block Bootstrap"

    def __init__(
        self, block_size: int, *args: ArrayLike, **kwargs: Union[RandomState, ArrayLike]
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self._parameters = [block_size]

    def __str__(self) -> str:
        txt = self._name
        txt += "(block size: " + str(self.block_size)
        txt += ", no. pos. inputs: " + str(len(self.pos_data))
        txt += ", no. keyword inputs: " + str(len(self.kw_data)) + ")"
        return txt

    def _repr_html(self) -> str:
        html = "<strong>" + self._name + "</strong>("
        html += "<strong>block size</strong>: " + str(self.block_size)
        html += ", <strong>no. pos. inputs</strong>: " + str(len(self.pos_data))
        html += ", <strong>no. keyword inputs</strong>: " + str(len(self.kw_data))
        html += ", <strong>ID</strong>: " + hex(id(self)) + ")"
        return html

    def update_indices(self) -> NDArray:
        num_blocks = self._num_items // self.block_size
        if num_blocks * self.block_size < self._num_items:
            num_blocks += 1
        indices = self.random_state.randint(self._num_items, size=num_blocks)
        indices = indices[:, None] + np.arange(self.block_size)
        indices = indices.flatten()
        indices %= self._num_items

        if indices.shape[0] > self._num_items:
            return indices[: self._num_items]
        else:
            return indices


class StationaryBootstrap(CircularBlockBootstrap):
    """
    Politis and Romano (1994) bootstrap with expon distributed block sizes

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
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data)
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered)
    kw_data : dict
        Dictionary containing the keyword arguments

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input date.

    Data entered using keyword arguments is directly accessibly as an
    attribute.

    To ensure a reproducible bootstrap, you must set the ``random_state``
    attribute after the bootstrap has been created. See the example below.
    Note that ``random_state`` is a reserved keyword and any variable
    passed using this keyword must be an instance of ``RandomState``.

    See Also
    --------
    arch.bootstrap.optimal_block_length
       Optimal block length estimation
    arch.bootstrap.CircularBlockBootstrap
       Circular (wrap-around) bootstrap

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

    Set the random_state if reproducibility is required

    >>> from numpy.random import RandomState
    >>> rs = RandomState(1234)
    >>> bs = StationaryBootstrap(12, x, y=y, z=z, random_state=rs)
    """

    _name = "Stationary Bootstrap"

    def __init__(
        self, block_size: int, *args: ArrayLike, **kwargs: Union[RandomState, ArrayLike]
    ) -> None:
        super().__init__(block_size, *args, **kwargs)
        self._p = 1.0 / block_size

    def update_indices(self) -> NDArray:
        indices = self.random_state.randint(self._num_items, size=self._num_items)
        indices = indices.astype(np.int64)
        u = self.random_state.random_sample(self._num_items)
        return stationary_bootstrap_sample(indices, u, self._p)


class MovingBlockBootstrap(CircularBlockBootstrap):
    """
    Bootstrap using blocks of the same length without wrap around

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
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data)
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered)
    kw_data : dict
        Dictionary containing the keyword arguments

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input date.

    Data entered using keyword arguments is directly accessibly as an
    attribute.

    To ensure a reproducible bootstrap, you must set the ``random_state``
    attribute after the bootstrap has been created. See the example below.
    Note that ``random_state`` is a reserved keyword and any variable
    passed using this keyword must be an instance of ``RandomState``.

    See Also
    --------
    arch.bootstrap.optimal_block_length
       Optimal block length estimation
    arch.bootstrap.StationaryBootstrap
       Politis and Romano's bootstrap with exp. distributed block lengths
    arch.bootstrap.CircularBlockBootstrap
       Circular (wrap-around) bootstrap

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

    Set the random_state if reproducibility is required

    >>> from numpy.random import RandomState
    >>> rs = RandomState(1234)
    >>> bs = MovingBlockBootstrap(7, x, y=y, z=z, random_state=rs)
    """

    _name = "Moving Block Bootstrap"

    def __init__(
        self, block_size: int, *args: ArrayLike, **kwargs: Union[RandomState, ArrayLike]
    ) -> None:
        super().__init__(block_size, *args, **kwargs)

    def update_indices(self) -> None:
        num_blocks = self._num_items // self.block_size
        if num_blocks * self.block_size < self._num_items:
            num_blocks += 1
        max_index = self._num_items - self.block_size + 1
        indices = self.random_state.randint(max_index, size=num_blocks)
        indices = indices[:, None] + np.arange(self.block_size)
        indices = indices.flatten()

        if indices.shape[0] > self._num_items:
            return indices[: self._num_items]
        else:
            return indices


class MOONBootstrap(IIDBootstrap):  # pragma: no cover
    def __init__(
        self, block_size: int, *args: ArrayLike, **kwargs: Union[RandomState, ArrayLike]
    ) -> None:  # pragma: no cover
        super().__init__(*args, **kwargs)
        self.block_size = block_size

    def update_indices(self) -> None:  # pragma: no cover
        raise NotImplementedError
