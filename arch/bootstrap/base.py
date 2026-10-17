from collections.abc import (
    Callable,
    Generator as PyGenerator,
    Hashable,
    Mapping,
    Sequence,
)
from typing import Any, Union, cast
import warnings

import numpy as np
from numpy.random import Generator, RandomState
import pandas as pd
from scipy import stats

from arch._typing import (
    AnyArray,
    ArrayLike,
    ArrayLike1D,
    ArrayLike2D,
    BootstrapIndexT,
    Float64Array,
    Float64Array2D,
    Int64Array,
    Int64Array1D,
    Literal,
    NDArray,
    RandomStateState,
)
from arch.utility._validation import (
    _describe,
    _fail,
    check_array,
    check_bool,
    check_callable,
    check_float,
    check_instance,
    check_int,
    check_option,
    check_seed,
)
from arch.utility.array import DocStringInheritor
from arch.utility.exceptions import StudentizationError, studentization_error

__all__ = [
    "CircularBlockBootstrap",
    "IIDBootstrap",
    "IndependentSamplesBootstrap",
    "MovingBlockBootstrap",
    "StationaryBootstrap",
    "optimal_block_length",
]

try:
    from arch.bootstrap._samplers import stationary_bootstrap_sample
except ImportError:  # pragma: no cover
    from arch.bootstrap._samplers_python import stationary_bootstrap_sample

# Lower case names of the confidence interval methods, including the aliases
_CONF_INT_METHODS = {
    "basic": "basic",
    "percentile": "percentile",
    "studentized": "studentized",
    "norm": "norm",
    "var": "norm",
    "cov": "norm",
    "bc": "bc",
    "debiased": "bc",
    "bias-corrected": "bc",
    "bca": "bca",
}


def _get_prng_state(
    prng: Generator | RandomState,
) -> RandomStateState | Mapping[str, Any]:
    if isinstance(prng, Generator):
        return prng.bit_generator.state
    else:
        assert isinstance(prng, RandomState)
        return prng.get_state()


def _get_random_integers(
    prng: Generator | RandomState, upper: int, *, size: int = 1
) -> Int64Array1D:
    if isinstance(prng, Generator):
        return cast("Int64Array1D", prng.integers(upper, size=size, dtype=np.int64))
    else:
        assert isinstance(prng, RandomState)
        return cast("Int64Array1D", prng.randint(upper, size=size, dtype=np.int64))


# The estimator uses the autocovariances up to m_max = ceil(sqrt(n)) +
# max(5, int(log10(n))), which needs n >= m_max + 2. This holds for n >= 11.
_MIN_OPTIMAL_BLOCK_OBS = 11


def _single_optimal_block(x: Float64Array) -> tuple[float, float]:
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
    opt_m: int | None = None
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
    d_sb = 2 * lr_acv**2
    d_cb = 4 / 3 * lr_acv**2
    b_sb = ((2 * g**2) / d_sb) ** (1 / 3) * nobs ** (1 / 3)
    b_cb = ((2 * g**2) / d_cb) ** (1 / 3) * nobs ** (1 / 3)
    b_sb = min(b_sb, b_max)
    b_cb = min(b_cb, b_max)
    return b_sb, b_cb


def optimal_block_length(x: ArrayLike1D | ArrayLike2D) -> pd.DataFrame:
    r"""
    Estimate optimal window length for time-series bootstraps.

    Parameters
    ----------
    x : array_like
        The data, which can be one-dimensional or two-dimensional.  Operates
        column by column if 2-dimensional. The data are converted to float64
        and must be finite and have at least 11 observations.

    Returns
    -------
    DataFrame
        A DataFrame with two columns `stationary`, the estimated optimal block
        size for the Stationary Bootstrap, and `circular`, the estimated
        optimal block size for the circular bootstrap. The index contains the
        column names of x if it is a DataFrame, the name of x if it is a Series
        and the column positions otherwise.

    Raises
    ------
    TypeError
        If ``x`` cannot be converted to float64 (strings, dates, complex
        numbers or None).
    ValueError
        If ``x`` has more than 2 dimensions, no columns or fewer than 11
        observations, or contains NaN or inf.

    See Also
    --------
    arch.bootstrap.StationaryBootstrap
        Politis and Romano's bootstrap with exp. distributed block lengths.
    arch.bootstrap.CircularBlockBootstrap
        Circular (wrap-around) bootstrap.

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
    rule = (
        "a 1-d or 2-d array of finite numbers with at least "
        f"{_MIN_OPTIMAL_BLOCK_OBS} rows and 1 column"
    )
    data = check_array("x", x)
    x_arr = data.reshape(-1, 1) if data.ndim < 2 else data
    if x_arr.ndim > 2 or x_arr.shape[0] < _MIN_OPTIMAL_BLOCK_OBS or x_arr.shape[1] < 1:
        _fail(ValueError, "x", rule, _describe(data))
    opt = [_single_optimal_block(col) for col in x_arr.T]
    if isinstance(x, pd.DataFrame):
        idx: list[Hashable] = list(x.columns)
    elif isinstance(x, pd.Series):
        idx = [x.name]
    else:
        idx = list(range(x_arr.shape[1]))
    return pd.DataFrame(opt, index=idx, columns=["stationary", "circular"])


def _get_acceleration(jk_params: Float64Array) -> Float64Array2D:
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
    eps = np.finfo(np.float64).eps
    if not np.all(np.isfinite(jk_params)):
        raise RuntimeError(
            "The jackknife estimates are not finite, so the acceleration used by "
            "BCa cannot be computed."
        )
    u = jk_params.mean(0) - jk_params
    numer = np.sum(u**3, 0)
    denom = 6 * (np.sum(u**2, 0) ** (3.0 / 2.0))
    # The jackknife estimates are all equal, up to rounding error, if the
    # spread is tiny relative to their size. The rounding error would
    # determine the acceleration (or 0/0 when they are exactly equal).
    spread = np.ptp(jk_params, axis=0)
    scale = np.abs(jk_params).max(axis=0)
    small = (denom <= (np.abs(numer) * eps)) | (spread <= 8 * eps * scale)
    if np.any(small):
        message = "Jackknife variance estimate {jk_var} is too small to use BCa"
        raise RuntimeError(message.format(jk_var=denom))
    a = numer / denom
    a = np.atleast_1d(a)
    return a[:, None]


def _loo_jackknife(
    func: Callable[..., Float64Array],
    nobs: int,
    args: Sequence[ArrayLike],
    kwargs: dict[str, ArrayLike],
    extra_kwargs: dict[str, ArrayLike] | None = None,
) -> Float64Array:
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
        args_copy: list[ArrayLike] = []
        for arg in args:
            if isinstance(arg, (pd.Series, pd.DataFrame)):
                args_copy.append(
                    cast("Union[pd.Series, pd.DataFrame]", arg.iloc[items])
                )
            else:
                args_copy.append(arg[items])
        kwargs_copy: dict[str, ArrayLike] = {}
        for k, v in kwargs.items():
            if isinstance(v, (pd.Series, pd.DataFrame)):
                kwargs_copy[k] = cast("Union[pd.Series, pd.DataFrame]", v.iloc[items])
            else:
                kwargs_copy[k] = v[items]
        if extra_kwargs is not None:
            kwargs_copy.update(extra_kwargs)
        results.append(func(*args_copy, **kwargs_copy))
    return np.array(results)


def _check_block_size(block_size: Any, integer: bool = True) -> int | float:
    """
    Check the block size of a block bootstrap

    Parameters
    ----------
    block_size : {int, float}
        The block size
    integer : bool, default True
        Whether the block size must be an integer. The block size of the
        stationary bootstrap is the mean block length, which can be any
        number that is at least 1.
    Returns
    -------
    int or float
        The block size. An integer is returned as a Python int.

    Raises
    ------
    TypeError
        If the block size is not an integer (or a number if integer is False)
    ValueError
        If the block size is less than 1 or not finite
    """
    if integer:
        return check_int("block_size", block_size, lower=1)
    checked = check_float("block_size", block_size, lower=1)
    if isinstance(block_size, (int, np.integer)):
        return int(block_size)
    return checked


def _check_extra_kwargs(
    kwargs: dict[str, Any], extra_kwargs: Mapping[str, Any] | None = None
) -> Mapping[str, Any] | None:
    """
    Check the extra keyword arguments and that they do not replace bootstrapped data

    Parameters
    ----------
    kwargs : dict
        Keyword argument dictionary containing the bootstrapped data
    extra_kwargs : dict, default None
        Extra keyword arguments

    Returns
    -------
    dict or None
        extra_kwargs, unchanged

    Raises
    ------
    TypeError
        If extra_kwargs is not a mapping or None, or has keys that are not str
    ValueError
        If extra_kwargs contains a key used for the data in kwargs
    """
    extra = check_instance("extra_kwargs", extra_kwargs, Mapping, allow_none=True)
    if extra is None:
        return None
    rule = "a dict with str keys that are not names of the bootstrapped data"
    for key in extra:
        if not isinstance(key, str):
            _fail(TypeError, "extra_kwargs", rule, f"the key {key!r} (dict)")
        if key in kwargs:
            _fail(ValueError, "extra_kwargs", rule, f"the key {key!r} (dict)")
    return extra


def _add_extra_kwargs(
    kwargs: dict[str, Any], extra_kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
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
        A new keyword dictionary with the added keyword arguments.  The input
        dictionary is never returned, so the caller can modify the result
        without changing the input.

    Notes
    -----
    There is no checking for duplicate keys
    """
    kwargs_copy = kwargs.copy()
    if extra_kwargs is not None:
        kwargs_copy.update(extra_kwargs)
    return kwargs_copy


class IIDBootstrap(metaclass=DocStringInheritor):
    """
    Bootstrap using uniform resampling.

    Parameters
    ----------
    *args : ndarray, Series or DataFrame
        Positional arguments to bootstrap. Each must have at least one row.
    seed : int, Generator or RandomState, optional
        Seed to use to ensure reproducible results. If an int, it must be
        non-negative and is passed to ``np.random.default_rng``. If None, a
        fresh Generator is constructed with system-provided entropy.
    **kwargs : ndarray, Series or DataFrame
        Keyword arguments to bootstrap. Each must have at least one row.

    Attributes
    ----------
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data).
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered).
    kw_data : dict
        Dictionary containing the keyword arguments.

    Raises
    ------
    TypeError
        If an input is not an ndarray, Series or DataFrame, or ``seed`` is not
        None, a non-negative int, a Generator or a RandomState.
    ValueError
        If there are no inputs, an input has no rows, the inputs
        do not have the same number of rows, ``seed`` is a negative
        int, or a keyword argument has the name of an attribute or method of
        the bootstrap.

    See Also
    --------
    arch.bootstrap.IndependentSamplesBootstrap
        Bootstrap where each input is independently resampled.

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input data.

    The data are used as given. They are not converted to float64 and are not
    checked for NaN or inf, since the function that is bootstrapped decides how
    to treat them (for example, using ``np.nanmean``).

    Data entered using keyword arguments is directly accessible as an
    attribute.

    To ensure a reproducible bootstrap, you must pass the ``seed`` keyword
    argument when the bootstrap is created. See the example below.
    Note that ``seed`` is a reserved keyword and any variable
    passed using this keyword must be an integer, a ``Generator`` or a
    ``RandomState``.

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

    Set the seed if reproducibility is required

    >>> from numpy.random import default_rng
    >>> seed = default_rng(1234)
    >>> bs = IIDBootstrap(x, y=y, z=z, seed=seed)

    This is equivalent to

    >>> bs = IIDBootstrap(x, y=y, z=z, seed=1234)
    """

    _name = "IID Bootstrap"
    _common_size_required = True

    def __init__(
        self,
        *args: ArrayLike,
        seed: int | Generator | RandomState | None = None,
        **kwargs: ArrayLike,
    ) -> None:
        _seed = check_seed("seed", seed)
        self._args = list(args)
        self._kwargs = kwargs
        self._generator: Generator | RandomState

        self._check_data()
        if args:
            self._num_items = len(args[0])
        else:
            key = next(iter(kwargs.keys()))
            self._num_items = len(kwargs[key])

        if isinstance(_seed, (RandomState, Generator)):
            self._generator = _seed
        elif _seed is None:
            self._generator = np.random.default_rng()
        else:
            self._generator = np.random.default_rng(_seed)

        self._initial_state = _get_prng_state(self._generator)
        self._index: BootstrapIndexT = np.arange(self._num_items, dtype=np.int64)

        self._parameters: list[int] = []
        self.pos_data: tuple[AnyArray | pd.Series | pd.DataFrame, ...] = args
        self.kw_data: dict[str, AnyArray | pd.Series | pd.DataFrame] = kwargs
        self.data: tuple[
            tuple[AnyArray | pd.Series | pd.DataFrame, ...],
            dict[str, AnyArray | pd.Series | pd.DataFrame],
        ] = (self.pos_data, self.kw_data)

        self._base: Float64Array | None = None
        self._results: Float64Array | None = None
        self._studentized_results: Float64Array | None = None
        self._last_func: Callable[..., ArrayLike] | None = None
        for key in kwargs:
            if getattr(self, key, None) is not None:
                _fail(
                    ValueError,
                    "kwargs",
                    "keywords that are not attributes or methods of the bootstrap",
                    f"{key!r} (str)",
                )
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __str__(self) -> str:
        txt = self._name
        txt += "(no. pos. inputs: " + str(len(self.pos_data))
        txt += ", no. keyword inputs: " + str(len(self.kw_data)) + ")"
        return txt

    def __repr__(self) -> str:
        return self.__str__()[:-1] + ", ID: " + hex(id(self)) + ")"

    def _repr_html_(self) -> str:
        html = "<strong>" + self._name + "</strong>("
        html += "<strong>no. pos. inputs</strong>: " + str(len(self.pos_data))
        html += ", <strong>no. keyword inputs</strong>: " + str(len(self.kw_data))
        html += ", <strong>ID</strong>: " + hex(id(self)) + ")"
        return html

    @property
    def generator(self) -> Generator | RandomState:
        """
        Set or get the instance PRNG.

        Returns
        -------
        Generator or RandomState
            The instance of the Generator or RandomState instance used
            by bootstrap.
        """
        return self._generator

    @generator.setter
    def generator(self, value: Generator | RandomState) -> None:
        self._generator = check_instance("generator", value, (Generator, RandomState))

    @property
    def index(self) -> BootstrapIndexT:
        """
        The current index of the bootstrap.
        """
        return self._index

    @property
    def state(self) -> RandomStateState | Mapping[str, Any]:
        """
        Set or get the generator's state.

        Returns
        -------
        tuple or dict
            A tuple or dictionary containing the generator's state.
            If using a ``RandomState``, the value returned is a tuple.
            Otherwise it is a dictionary.
        """
        if isinstance(self._generator, Generator):
            return self._generator.bit_generator.state
        else:
            assert isinstance(self._generator, RandomState)
            return self._generator.get_state()

    @state.setter
    def state(self, value: RandomStateState | Mapping[str, Any]) -> None:
        if isinstance(self._generator, Generator):
            state = check_instance("state", value, Mapping)
            self._generator.bit_generator.state = state
        else:
            assert isinstance(self._generator, RandomState)
            state = check_instance("state", value, (tuple, Mapping))
            self._generator.set_state(cast("RandomStateState", state))

    def reset(self, use_seed: bool = True) -> None:
        """
        Reset the bootstrap to its initial state.

        The generator is reset to the state it had when the bootstrap was
        created, which is the state determined by the seed if one was provided,
        and the data are restored to their original order.

        Parameters
        ----------
        use_seed : bool, optional
            This argument is not used. The generator is always reset to the
            state it had when the bootstrap was created.

        Raises
        ------
        TypeError
            If ``use_seed`` is not a bool.
        """
        check_bool("use_seed", use_seed)
        self._index = np.arange(self._num_items, dtype=np.int64)
        self._resample()
        self.state = self._initial_state

    def bootstrap(
        self, reps: int
    ) -> PyGenerator[tuple[tuple[ArrayLike, ...], dict[str, ArrayLike]], None, None]:
        """
        Iterate over the bootstrap replications.

        Parameters
        ----------
        reps : int
            Number of bootstrap replications. Must be at least 1.

        Returns
        -------
        generator
            An iterator over the replications. It yields a tuple with the data
            of one bootstrap replication. The first element is a tuple
            containing the data entered in positional arguments and the second
            element is a dict containing the data entered using keywords.

        Raises
        ------
        TypeError
            If ``reps`` is not an int.
        ValueError
            If ``reps`` is less than 1. A moving block bootstrap also raises if
            its block size is larger than the number of observations.

        Notes
        -----
        The iterator returns a tuple containing the data entered in positional
        arguments as a tuple and the data entered using keywords as a
        dictionary.

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
            name of the required bootstrap.
        """
        reps = check_int("reps", reps, lower=1)
        self._check_sampling()
        return self._iterate(reps)

    def _check_sampling(self) -> None:
        """
        Check that the bootstrap can draw samples from its data

        This is called when the iteration starts, once for all replications,
        and is overridden by bootstraps with a condition on the data and the
        parameters that is not a condition of creating the bootstrap.
        """

    def _iterate(
        self, reps: int
    ) -> PyGenerator[tuple[tuple[ArrayLike, ...], dict[str, ArrayLike]], None, None]:
        for _ in range(reps):
            self._index = self.update_indices()
            yield self._resample()

    def conf_int(
        self,
        func: Callable[..., Float64Array],
        reps: int = 1000,
        method: Literal[
            "basic", "percentile", "studentized", "norm", "bc", "bca"
        ] = "basic",
        size: float = 0.95,
        tail: Literal["two", "upper", "lower"] = "two",
        extra_kwargs: dict[str, Any] | None = None,
        reuse: bool = False,
        sampling: Literal[
            "nonparametric", "semi-parametric", "semi", "parametric", "semiparametric"
        ] = "nonparametric",
        std_err_func: Callable[..., ArrayLike] | None = None,
        studentize_reps: int = 1000,
    ) -> Float64Array:
        """
        Compute confidence intervals using the bootstrap.

        Parameters
        ----------
        func : callable
            Function that computes parameter values.  See Notes for
            requirements.
        reps : int, optional
            Number of bootstrap replications. Must be at least 1.
        method : {"basic", "percentile", "studentized", "norm", "bc", "bca"}, optional
            The method used to construct the interval.  "norm" is identical to
            "var" and "cov", and "bc" is identical to "debiased" and
            "bias-corrected".  See Notes for a description of each method. Not case
            sensitive.
        size : float, optional
            Coverage of confidence interval. Must be strictly between 0 and 1.
        tail : {"two", "upper", "lower"}, optional
            The tail of the interval.  When tail is "lower", all upper bounds
            are inf.  Similarly, "upper" sets all lower bounds to -inf. Not case
            sensitive.
        extra_kwargs : dict, optional
            Extra keyword arguments to use when calling func and std_err_func,
            when appropriate.
        reuse : bool, optional
            Flag indicating whether to reuse previously computed bootstrap
            results.  This allows alternative methods to be compared without
            rerunning the bootstrap simulation.  Reuse is ignored if reps is
            not the same across multiple runs, func changes across calls, or
            method is "studentized".  Otherwise the previous results are
            reused even if extra_kwargs or sampling have changed, so use a new
            call without reuse (or ``reset``) after changing them.
        sampling : {"nonparametric", "semi-parametric", "parametric"}, optional
            Type of sampling to use.  Not case sensitive.  "semi" and
            "semiparametric" are identical to "semi-parametric".  See
            :ref:`semiparametric-bootstraps` and :ref:`parametric-bootstraps`
            for the changes to func required when using "semi-parametric" or
            "parametric".
        std_err_func : callable, optional
            Function to use when standardizing estimated parameters when using
            the studentized bootstrap.  Providing an analytical function
            eliminates the need for a nested bootstrap.
        studentize_reps : int, optional
            Number of bootstraps to use in the inner bootstrap when using the
            studentized bootstrap.  Its value is ignored, but it must be an int,
            when ``std_err_func`` is provided or ``method`` is not
            "studentized". Otherwise it must be at least 1.

        Returns
        -------
        ndarray
            Computed confidence interval.  Row 0 contains the lower bounds, and
            row 1 contains the upper bounds.  Each column corresponds to a
            parameter. When tail is 'lower', all upper bounds are inf.
            Similarly, 'upper' sets all lower bounds to -inf.

        Raises
        ------
        TypeError
            If ``func`` or ``std_err_func`` is not callable, ``reps`` or
            ``studentize_reps`` is not an int, ``size`` is not a number,
            ``reuse`` is not a bool, ``method``, ``tail`` or ``sampling`` is not
            a str, or ``extra_kwargs`` is not a dict.
        ValueError
            If ``reps`` is less than 1, ``size`` is not in (0, 1),
            ``studentize_reps`` is less than 1 when it is used, ``method``,
            ``tail`` or ``sampling`` is not one of the options, or
            ``extra_kwargs`` has a key that is the name of bootstrapped data.

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
        bootstrap data as in args and kwargs.  When ``sampling`` is
        ``"semi-parametric"`` or ``"parametric"``, the keyword arguments
        ``params`` and ``state`` that are passed to ``func`` to simulate the
        data are not passed to ``std_err_func``, which only receives the
        data and ``extra_kwargs``.  The studentized bootstrap with a nested
        bootstrap simulates the data in the nested bootstrap in the same way
        as in the outer bootstrap.

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

        The acceleration parameter used by 'bca' is estimated using a
        leave-one-out jackknife of the observations in the original data. It is
        the iid jackknife for all bootstraps, including the block bootstraps,
        and so does not take account of serial dependence or the block size.

        When ``std_err_func`` is not provided, the standard error used by
        'studentized' is estimated in each bootstrap sample using a nested
        bootstrap of the same type as this bootstrap, which applies the same
        block size to the bootstrap sample. The block size of the nested
        bootstrap is not selected separately from the block size of this
        bootstrap. Provide ``std_err_func`` to use a different method, for
        example, a nested block bootstrap with a different block size.

        Examples
        --------
        >>> import numpy as np
        >>> def func(x):
        ...     return x.mean(0)
        >>> y = np.random.randn(1000, 2)
        >>> from arch.bootstrap import IIDBootstrap
        >>> bs = IIDBootstrap(y)
        >>> ci = bs.conf_int(func, 1000)
        """
        studentized = "studentized"
        check_callable("func", func)
        reps = check_int("reps", reps, lower=1)
        method = check_option("method", method, _CONF_INT_METHODS)
        size = check_float("size", size, lower=0.0, upper=1.0, inclusive="neither")
        tail_name = check_option("tail", tail, ("two", "lower", "upper"))
        _check_extra_kwargs(self._kwargs, extra_kwargs)
        reuse = check_bool("reuse", reuse)
        std_err_func = check_callable("std_err_func", std_err_func, allow_none=True)
        # The value is ignored, and so can be anything, unless it is used
        used = method == studentized and std_err_func is None
        studentize_reps = check_int(
            "studentize_reps", studentize_reps, lower=1 if used else None
        )
        studentize_reps = studentize_reps if method == studentized else 0
        sampling = check_option(
            "sampling",
            sampling,
            {
                "nonparametric": "nonparametric",
                "parametric": "parametric",
                "semi": "semiparametric",
                "semi-parametric": "semiparametric",
                "semiparametric": "semiparametric",
            },
        )
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
                warn = (
                    "The conditions to reuse the previous bootstrap has "
                    "not been satisfied. A new bootstrap will be used."
                )
                warnings.warn(warn, RuntimeWarning, stacklevel=2)
            self._construct_bootstrap_estimates(
                func,
                reps,
                extra_kwargs,
                std_err_func=std_err_func,
                studentize_reps=studentize_reps,
                sampling=sampling,
            )

        base, results = self._base, self._results
        assert results is not None
        assert base is not None
        studentized_results = self._studentized_results

        std_err = np.empty((0,))
        if method in ("norm", studentized):
            errors = results - results.mean(axis=0)
            std_err = np.asarray(np.sqrt(np.diag(errors.T.dot(errors) / reps)))

        if tail_name == "two":
            alpha = (1.0 - size) / 2
        else:
            alpha = 1.0 - size
        nreps = 1 if not base.shape else base.shape[0]
        percentiles = np.array([[alpha, 1.0 - alpha]] * nreps)
        norm_quantiles = stats.norm.ppf(percentiles)

        if method == "norm":
            lower = base + norm_quantiles[:, 0] * std_err
            upper = base + norm_quantiles[:, 1] * std_err

        else:
            values = results
            if method == studentized:
                # studentized uses studentized parameter estimates
                values = cast("Float64Array", studentized_results)

            if method in ("bc", "bca"):
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
                    a = np.zeros((1, 1))
                percentiles = stats.norm.cdf(
                    b + (b + norm_quantiles) / (1.0 - a * (b + norm_quantiles))
                )
                percentiles *= 100
            else:
                percentiles = np.array([100 * p for p in percentiles])  # Rescale

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

        if tail_name == "lower":
            upper = np.zeros_like(base)
            upper.fill(np.inf)
        elif tail_name == "upper":
            lower = np.zeros_like(base)
            lower.fill(-1 * np.inf)

        return np.vstack((lower, upper))

    def _check_data(self) -> None:
        supported = (np.ndarray, pd.DataFrame, pd.Series)
        inputs: list[tuple[str, Any]] = [
            (f"args[{i}]", arg) for i, arg in enumerate(self._args)
        ]
        inputs.extend(self._kwargs.items())
        if not inputs:
            _fail(
                ValueError,
                "args",
                "at least one NumPy array, Series or DataFrame, as a positional "
                "or a keyword argument",
                "no data (tuple)",
            )
        for name, arg in inputs:
            check_instance(name, arg, supported)
            if arg.ndim == 0 or len(arg) == 0:
                _fail(
                    ValueError,
                    name,
                    "an array, Series or DataFrame with at least 1 row",
                    _describe(arg),
                )
        if self._common_size_required:
            first_name, first = inputs[0]
            for name, arg in inputs[1:]:
                if len(arg) != len(first):
                    _fail(
                        ValueError,
                        name,
                        f"an array with {len(first)} rows, as {first_name}",
                        _describe(arg),
                    )

    def _bca_bias(self) -> Float64Array:
        assert self._results is not None
        assert self._base is not None
        p = (self._results < self._base).mean(axis=0)
        if p.min() <= 0.0 or p.max() >= 1.0:
            raise RuntimeError(
                "Empirical probability used in bias correction is 0 or 1, and so "
                "bias cannot be corrected. This may occur in extremum statistics "
                "that are not well approximated by a normal in a finite sample."
            )
        b = stats.norm.ppf(p)
        return b[:, None]

    def _bca_acceleration(
        self, func: Callable[..., Float64Array], extra_kwags: dict[str, Any] | None
    ) -> Float64Array2D:
        nobs = self._num_items
        jk_params = _loo_jackknife(func, nobs, self._args, self._kwargs, extra_kwags)
        return _get_acceleration(jk_params)

    def clone(
        self,
        *args: ArrayLike,
        seed: int | Generator | RandomState | None = None,
        **kwargs: ArrayLike,
    ) -> "IIDBootstrap":
        """
        Clone the bootstrap using different data with a fresh prng.

        Parameters
        ----------
        *args : ndarray, Series or DataFrame
            Positional arguments to bootstrap.
        seed : int, Generator or RandomState, optional
            The seed value to pass to the cloned generator.
        **kwargs : ndarray, Series or DataFrame
            Keyword arguments to bootstrap.

        Returns
        -------
        IIDBootstrap
            A new bootstrap instance of the same type as this instance.
        """
        bs = self.__class__(*args, seed=seed, **kwargs)
        return bs

    def apply(
        self,
        func: Callable[..., ArrayLike],
        reps: int = 1000,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Float64Array:
        """
        Apply a function to bootstrap replicated data.

        Parameters
        ----------
        func : callable
            Function that computes parameter values.  See Notes for
            requirements.
        reps : int, optional
            Number of bootstrap replications. Must be at least 1.
        extra_kwargs : dict, optional
            Extra keyword arguments to use when calling func.  Must not
            conflict with keyword arguments used to initialize bootstrap.

        Returns
        -------
        ndarray
            The reps by nparam array of computed function values where each row
            corresponds to a bootstrap iteration.

        Raises
        ------
        TypeError
            If ``func`` is not callable, ``reps`` is not an int, or
            ``extra_kwargs`` is not a dict.
        ValueError
            If ``reps`` is less than 1, or ``extra_kwargs`` has a key that is
            the name of bootstrapped data.

        Notes
        -----
        When there are no extra keyword arguments, the function is called

        .. code:: python

            func(*args, **kwargs)

        where args and kwargs are the bootstrap version of the data provided
        when setting up the bootstrap.  When extra keyword arguments are used,
        these are appended to kwargs before calling func.  The function can
        return a scalar, a 0-dimensional array or a 1-dimensional array-like.

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
        check_callable("func", func)
        reps = check_int("reps", reps, lower=1)
        _check_extra_kwargs(self._kwargs, extra_kwargs)
        kwargs = _add_extra_kwargs(self._kwargs, extra_kwargs)
        base = np.asarray(func(*self._args, **kwargs))
        num_params = 1 if base.ndim == 0 else base.shape[0]
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
        extra_kwargs: dict[str, Any] | None = None,
        std_err_func: Callable[..., ArrayLike] | None = None,
        studentize_reps: int = 0,
        sampling: Literal[
            "nonparametric", "semi-parametric", "semi", "parametric", "semiparametric"
        ] = "nonparametric",
    ) -> None:
        eps = np.finfo(np.double).eps
        # Private, more complicated version of apply
        self._last_func = func
        semi = parametric = False
        if sampling == "parametric":
            parametric = True
        elif sampling == "semiparametric":
            semi = True

        kwargs = _add_extra_kwargs(self._kwargs, extra_kwargs)
        # Allows lists, tuples, scalars and 0-d arrays
        base = np.asarray(func(*self._args, **kwargs))

        num_params = 1 if base.ndim == 0 else base.shape[0]
        results = np.zeros((reps, num_params))
        studentized_results = np.zeros((reps, num_params))

        count = 0
        for pos_data, kw_data in self.bootstrap(reps):
            # Data and extra keywords only, without the simulation keywords
            data_kwargs = _add_extra_kwargs(kw_data, extra_kwargs)
            kwargs = data_kwargs.copy()
            if parametric:
                kwargs["state"] = self._generator
                kwargs["params"] = base
            elif semi:
                kwargs["params"] = base
            results[count] = func(*pos_data, **kwargs)
            if std_err_func is not None:
                # params and state are only used to simulate data, and params
                # would conflict with the first argument of std_err_func
                std_err = std_err_func(results[count], *pos_data, **data_kwargs)
                studentized_results[count] = (results[count] - base) / std_err
            elif studentize_reps > 0:
                # Set the seed to ensure reproducibility
                seed = _get_random_integers(self._generator, 2**31 - 1, size=1)
                # Need new bootstrap of same type
                nested_bs = self.clone(*pos_data, **kw_data, seed=seed[0])
                # The nested draws are simulated like the outer draws
                cov = nested_bs._bootstrap_cov(
                    func, studentize_reps, extra_kwargs=extra_kwargs, sampling=sampling
                )
                std_err = np.sqrt(np.diag(cov))
                err = results[count] - base
                if np.any(std_err <= (eps * np.abs(err))):
                    raise StudentizationError(studentization_error.format(cov=cov))
                studentized_results[count] = err / std_err
            count += 1

        self._base = base
        self._results = np.asarray(results)
        self._studentized_results = np.asarray(studentized_results)

    def cov(
        self,
        func: Callable[..., ArrayLike],
        reps: int = 1000,
        recenter: bool = True,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> float | Float64Array:
        """
        Compute parameter covariance using bootstrap.

        Parameters
        ----------
        func : callable
            Callable function that returns the statistic of interest as a
            1-d array.
        reps : int, optional
            Number of bootstrap replications. Must be at least 1.
        recenter : bool, optional
            Whether to center the bootstrap variance estimator on the average
            of the bootstrap samples (True) or to center on the original sample
            estimate (False).
        extra_kwargs : dict, optional
            Dictionary of extra keyword arguments to pass to func.

        Returns
        -------
        ndarray
            Bootstrap covariance estimator.

        Raises
        ------
        TypeError
            If ``func`` is not callable, ``reps`` is not an int, ``recenter``
            is not a bool, or ``extra_kwargs`` is not a dict.
        ValueError
            If ``reps`` is less than 1, or ``extra_kwargs`` has a key that is
            the name of bootstrapped data.

        Notes
        -----
        func must have the signature

        .. code:: python

            func(*args, **kwargs)

        where `*args` and `**kwargs` are the bootstrap version of the data
        used in the bootstrap, and any ``extra_kwargs`` are appended to
        `**kwargs`. The function returns a scalar, a 0-dimensional array or a
        1-dimensional array-like. A ``params`` argument is not passed, so only
        the nonparametric bootstrap can be used.

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
            name of the required bootstrap.
        """
        check_callable("func", func)
        reps = check_int("reps", reps, lower=1)
        recenter = check_bool("recenter", recenter)
        _check_extra_kwargs(self._kwargs, extra_kwargs)
        return self._bootstrap_cov(func, reps, recenter, extra_kwargs)

    def _bootstrap_cov(
        self,
        func: Callable[..., ArrayLike],
        reps: int,
        recenter: bool = True,
        extra_kwargs: dict[str, Any] | None = None,
        sampling: Literal[
            "nonparametric", "semi-parametric", "semi", "parametric", "semiparametric"
        ] = "nonparametric",
    ) -> Float64Array:
        self._construct_bootstrap_estimates(func, reps, extra_kwargs, sampling=sampling)
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
        extra_kwargs: dict[str, Any] | None = None,
    ) -> float | Float64Array:
        """
        Compute parameter variance using bootstrap.

        Parameters
        ----------
        func : callable
            Callable function that returns the statistic of interest as a
            1-d array.
        reps : int, optional
            Number of bootstrap replications. Must be at least 1.
        recenter : bool, optional
            Whether to center the bootstrap variance estimator on the average
            of the bootstrap samples (True) or to center on the original sample
            estimate (False).
        extra_kwargs : dict, optional
            Dictionary of extra keyword arguments to pass to func.

        Returns
        -------
        ndarray
            Bootstrap variance estimator.

        Raises
        ------
        TypeError
            If ``func`` is not callable, ``reps`` is not an int, ``recenter``
            is not a bool, or ``extra_kwargs`` is not a dict.
        ValueError
            If ``reps`` is less than 1, or ``extra_kwargs`` has a key that is
            the name of bootstrapped data.

        Notes
        -----
        func must have the signature

        .. code:: python

            func(*args, **kwargs)

        where `*args` and `**kwargs` are the bootstrap version of the data
        used in the bootstrap, and any ``extra_kwargs`` are appended to
        `**kwargs`. The function returns a scalar, a 0-dimensional array or a
        1-dimensional array-like. A ``params`` argument is not passed, so only
        the nonparametric bootstrap can be used.

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
            name of the required bootstrap.
        """
        check_callable("func", func)
        reps = check_int("reps", reps, lower=1)
        recenter = check_bool("recenter", recenter)
        _check_extra_kwargs(self._kwargs, extra_kwargs)
        self._construct_bootstrap_estimates(func, reps, extra_kwargs)
        base, results = self._base, self._results
        assert results is not None
        assert base is not None
        if recenter:
            errors = results - np.mean(results, 0)
        else:
            errors = results - base

        return (errors**2).sum(0) / reps

    def update_indices(self) -> BootstrapIndexT:
        """
        Update indices for the next iteration of the bootstrap.

        This must be overridden when creating new bootstraps.

        Returns
        -------
        ndarray
            The indices of the observations in the next bootstrap sample.
        """
        return _get_random_integers(
            self._generator, self._num_items, size=self._num_items
        )

    def _resample(self) -> tuple[tuple[ArrayLike, ...], dict[str, ArrayLike]]:
        """
        Resample all data using the values in _index
        """
        indices = cast("Union[Int64Array, tuple[Int64Array, ...]]", self._index)
        pos_data: list[NDArray | pd.DataFrame | pd.Series] = []
        for values in self._args:
            if isinstance(values, (pd.Series, pd.DataFrame)):
                assert isinstance(indices, np.ndarray)
                pos_data.append(values.iloc[indices])
            else:
                assert isinstance(values, np.ndarray)
                pos_data.append(values[indices])
        named_data: dict[str, NDArray | pd.DataFrame | pd.Series] = {}
        for key, values in self._kwargs.items():
            if isinstance(values, (pd.Series, pd.DataFrame)):
                assert isinstance(indices, np.ndarray)
                named_data[key] = values.iloc[indices]
            else:
                assert isinstance(values, np.ndarray)
                named_data[key] = values[indices]
            setattr(self, key, named_data[key])

        self.pos_data = tuple(pos_data)
        self.kw_data = named_data
        self.data = (self.pos_data, self.kw_data)
        return self.data


class IndependentSamplesBootstrap(IIDBootstrap):
    """
    Bootstrap where each input is independently resampled.

    Parameters
    ----------
    *args : ndarray, Series or DataFrame
        Positional arguments to bootstrap. Each must have at least one row.
    seed : int, Generator or RandomState, optional
        Seed to use to ensure reproducible results. If an int, it must be
        non-negative and is passed to ``np.random.default_rng``. If None, a
        fresh Generator is constructed with system-provided entropy.
    **kwargs : ndarray, Series or DataFrame
        Keyword arguments to bootstrap. Each must have at least one row.

    Attributes
    ----------
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data).
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered).
    kw_data : dict
        Dictionary containing the keyword arguments.

    Raises
    ------
    TypeError
        If an input is not an ndarray, Series or DataFrame, or ``seed`` is not
        None, a non-negative int, a Generator or a RandomState.
    ValueError
        If there are no inputs, an input has no rows, ``seed`` is a negative
        int, or a keyword argument has the name of an attribute or method of
        the bootstrap.

    See Also
    --------
    arch.bootstrap.IIDBootstrap
        Bootstrap using uniform resampling.

    Notes
    -----
    This bootstrap independently resamples each input and so is only
    appropriate when the inputs are independent. This structure allows
    bootstrapping statistics that depend on samples with unequal length, as
    is common in some experiments. If data have cross-sectional dependence, so
    that observation ``i`` is related across all inputs, this bootstrap is
    inappropriate.

    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input data.

    The data are used as given. They are not converted to float64 and are not
    checked for NaN or inf, since the function that is bootstrapped decides how
    to treat them (for example, using ``np.nanmean``).

    Data entered using keyword arguments is directly accessible as an
    attribute.

    To ensure a reproducible bootstrap, you must pass the ``seed`` keyword
    argument when the bootstrap is created. See the example below.
    Note that ``seed`` is a reserved keyword and any variable
    passed using this keyword must be an integer, a ``Generator`` or a
    ``RandomState``.

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

    Set the seed if reproducibility is required

    >>> from numpy.random import default_rng
    >>> gen = default_rng(1234)
    >>> bs = IndependentSamplesBootstrap(x, y=y, z=z, seed=gen)
    """

    _common_size_required = False
    _name = "Heterogeneous IID Bootstrap"

    def __init__(
        self,
        *args: ArrayLike,
        seed: int | Generator | RandomState | None = None,
        **kwargs: ArrayLike,
    ) -> None:
        super().__init__(*args, seed=seed, **kwargs)

        self._num_args = len(args)
        self._num_arg_items = [len(arg) for arg in args]
        self._num_kw_items = {key: len(kwargs[key]) for key in self._kwargs}

    def update_indices(
        self,
    ) -> tuple[list[Int64Array1D], dict[str, Int64Array1D]]:
        """
        Update indices for the next iteration of the bootstrap.

        This must be overridden when creating new bootstraps.

        Returns
        -------
        tuple
            A two-element tuple. The first element is a list that contains the
            indices for each of the positional arguments. The second element
            is a dict that contains the indices of the keyword arguments.
        """
        pos_indices = [
            _get_random_integers(
                self._generator, self._num_arg_items[i], size=self._num_arg_items[i]
            )
            for i in range(self._num_args)
        ]
        kw_indices = {
            key: _get_random_integers(
                self._generator, self._num_kw_items[key], size=self._num_kw_items[key]
            )
            for key in self._kwargs
        }
        return pos_indices, kw_indices

    @property
    def index(self) -> BootstrapIndexT:
        """
        The current index of the bootstrap.

        Returns
        -------
        tuple
            A two-element tuple containing a list and a dictionary. The list
            contains indices for each of the positional arguments.  The
            dictionary contains the indices of keyword arguments.
        """
        return self._index

    def reset(self, use_seed: bool = True) -> None:
        """
        Reset the bootstrap to its initial state.

        The generator is reset to the state it had when the bootstrap was
        created, which is the state determined by the seed if one was provided,
        and the data are restored to their original order.

        Parameters
        ----------
        use_seed : bool, optional
            This argument is not used. The generator is always reset to the
            state it had when the bootstrap was created.

        Raises
        ------
        TypeError
            If ``use_seed`` is not a bool.
        """
        check_bool("use_seed", use_seed)
        pos_indices: list[Int64Array1D] = [
            np.arange(self._num_arg_items[i], dtype=np.int64)
            for i in range(self._num_args)
        ]
        kw_indices: dict[str, Int64Array1D] = {
            key: np.arange(self._num_kw_items[key], dtype=np.int64)
            for key in self._kwargs
        }
        self._index = pos_indices, kw_indices
        self._resample()
        self.state = self._initial_state

    def _resample(self) -> tuple[tuple[ArrayLike, ...], dict[str, ArrayLike]]:
        """
        Resample all data using the values in _index
        """
        pos_indices, kw_indices = cast(
            "tuple[list[Int64Array], dict[str, Int64Array]]", self._index
        )
        pos_data: list[NDArray | pd.DataFrame | pd.Series] = []
        for i, values in enumerate(self._args):
            if isinstance(values, (pd.Series, pd.DataFrame)):
                pos_data.append(values.iloc[pos_indices[i]])
            else:
                assert isinstance(values, np.ndarray)
                pos_data.append(values[pos_indices[i]])
        named_data: dict[str, AnyArray | pd.Series | pd.DataFrame] = {}
        for key, values in self._kwargs.items():
            idx = kw_indices[key]
            if isinstance(values, (pd.Series, pd.DataFrame)):
                named_data[key] = values.iloc[idx]
            else:
                assert isinstance(values, np.ndarray)
                named_data[key] = values[idx]
            setattr(self, key, named_data[key])

        self.pos_data = tuple(pos_data)
        self.kw_data = named_data
        self.data = (self.pos_data, named_data)
        return self.data


class CircularBlockBootstrap(IIDBootstrap):
    """
    Bootstrap using blocks of the same length with end-to-start wrap around.

    Parameters
    ----------
    block_size : int
        Size of block to use. Must be an integer that is at least 1.
    *args : ndarray, Series or DataFrame
        Positional arguments to bootstrap. Each must have at least one row.
    seed : int, Generator or RandomState, optional
        Seed to use to ensure reproducible results. If an int, it must be
        non-negative and is passed to ``np.random.default_rng``. If None, a
        fresh Generator is constructed with system-provided entropy.
    **kwargs : ndarray, Series or DataFrame
        Keyword arguments to bootstrap. Each must have at least one row.

    Attributes
    ----------
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data).
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered).
    kw_data : dict
        Dictionary containing the keyword arguments.

    Raises
    ------
    TypeError
        If an input is not an ndarray, Series or DataFrame, ``block_size`` is
        not an int, or ``seed`` is not None, a non-negative int, a Generator or a
        RandomState.
    ValueError
        If there are no inputs, an input has no rows, the inputs do not have
        the same number of rows, ``block_size`` is less than 1, ``seed`` is a
        negative int, or a keyword argument has the name of an attribute or
        method of the bootstrap.

    See Also
    --------
    arch.bootstrap.optimal_block_length
        Optimal block length estimation.
    arch.bootstrap.StationaryBootstrap
        Politis and Romano's bootstrap with exp. distributed block lengths.

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input data.

    The data are used as given. They are not converted to float64 and are not
    checked for NaN or inf, since the function that is bootstrapped decides how
    to treat them (for example, using ``np.nanmean``).

    Data entered using keyword arguments is directly accessible as an
    attribute.

    To ensure a reproducible bootstrap, you must pass the ``seed`` keyword
    argument when the bootstrap is created. See the example below.
    Note that ``seed`` is a reserved keyword and any variable
    passed using this keyword must be an integer, a ``Generator`` or a
    ``RandomState``.

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

    Set the seed if reproducibility is required

    >>> from numpy.random import default_rng
    >>> gen = default_rng(1234)
    >>> bs = CircularBlockBootstrap(17, x, y=y, z=z, seed=gen)
    """

    _name = "Circular Block Bootstrap"
    _integer_block_size = True

    def __init__(
        self,
        block_size: int,
        *args: ArrayLike,
        seed: int | Generator | RandomState | None = None,
        **kwargs: ArrayLike,
    ) -> None:
        checked = cast("int", _check_block_size(block_size, self._integer_block_size))
        super().__init__(*args, seed=seed, **kwargs)
        self._block_size: int = checked
        self._parameters = [checked]

    @property
    def block_size(self) -> int:
        """
        The block size of the bootstrap.

        Returns
        -------
        int
            The block size, which is the size of each block, or the average
            size of the blocks in the stationary bootstrap. It cannot be
            changed after the bootstrap has been created.
        """
        return self._block_size

    @block_size.setter
    def block_size(self, value: int) -> None:
        raise AttributeError(
            "block_size cannot be changed after the bootstrap has been created. "
            "Create a new bootstrap with the required block size, for example "
            "using clone."
        )

    def clone(
        self,
        *args: ArrayLike,
        seed: int | Generator | RandomState | None = None,
        **kwargs: ArrayLike,
    ) -> "CircularBlockBootstrap":
        """
        Clone the bootstrap using different data with a fresh prng.

        The block size is the block size of this bootstrap.

        Parameters
        ----------
        *args : ndarray, Series or DataFrame
            Positional arguments to bootstrap.
        seed : int, Generator or RandomState, optional
            The seed value to pass to the cloned generator.
        **kwargs : ndarray, Series or DataFrame
            Keyword arguments to bootstrap.

        Returns
        -------
        CircularBlockBootstrap
            A new bootstrap instance of the same type as this instance.
        """
        return self.__class__(self.block_size, *args, seed=seed, **kwargs)

    def __str__(self) -> str:
        txt = self._name
        txt += "(block size: " + str(self.block_size)
        txt += ", no. pos. inputs: " + str(len(self.pos_data))
        txt += ", no. keyword inputs: " + str(len(self.kw_data)) + ")"
        return txt

    def _repr_html_(self) -> str:
        html = "<strong>" + self._name + "</strong>("
        html += "<strong>block size</strong>: " + str(self.block_size)
        html += ", <strong>no. pos. inputs</strong>: " + str(len(self.pos_data))
        html += ", <strong>no. keyword inputs</strong>: " + str(len(self.kw_data))
        html += ", <strong>ID</strong>: " + hex(id(self)) + ")"
        return html

    def update_indices(self) -> Int64Array1D:
        num_blocks = self._num_items // self.block_size
        if num_blocks * self.block_size < self._num_items:
            num_blocks += 1
        indices = _get_random_integers(
            self._generator, self._num_items, size=num_blocks
        )
        _indices = indices[:, None] + np.arange(self.block_size, dtype=np.int64)
        indices = _indices.flatten()
        indices %= self._num_items

        if indices.shape[0] > self._num_items:
            return cast("Int64Array1D", indices[: self._num_items])
        else:
            return cast("Int64Array1D", indices)


class StationaryBootstrap(CircularBlockBootstrap):
    """
    Politis and Romano (1994) bootstrap with expon distributed block sizes.

    Parameters
    ----------
    block_size : int or float
        Average size of block to use. Can be any number that is at least 1,
        for example, the value estimated by ``optimal_block_length``.
    *args : ndarray, Series or DataFrame
        Positional arguments to bootstrap. Each must have at least one row.
    seed : int, Generator or RandomState, optional
        Seed to use to ensure reproducible results. If an int, it must be
        non-negative and is passed to ``np.random.default_rng``. If None, a
        fresh Generator is constructed with system-provided entropy.
    **kwargs : ndarray, Series or DataFrame
        Keyword arguments to bootstrap. Each must have at least one row.

    Attributes
    ----------
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data).
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered).
    kw_data : dict
        Dictionary containing the keyword arguments.

    Raises
    ------
    TypeError
        If an input is not an ndarray, Series or DataFrame, ``block_size`` is
        not an int or a float, or ``seed`` is not None, a non-negative int, a Generator or a
        RandomState.
    ValueError
        If there are no inputs, an input has no rows, the inputs do not have
        the same number of rows, ``block_size`` is less than 1 or not finite, ``seed`` is a
        negative int, or a keyword argument has the name of an attribute or
        method of the bootstrap.

    See Also
    --------
    arch.bootstrap.optimal_block_length
        Optimal block length estimation.
    arch.bootstrap.CircularBlockBootstrap
        Circular (wrap-around) bootstrap.

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input data.

    The data are used as given. They are not converted to float64 and are not
    checked for NaN or inf, since the function that is bootstrapped decides how
    to treat them (for example, using ``np.nanmean``).

    Data entered using keyword arguments is directly accessible as an
    attribute.

    To ensure a reproducible bootstrap, you must pass the ``seed`` keyword
    argument when the bootstrap is created. See the example below.
    Note that ``seed`` is a reserved keyword and any variable
    passed using this keyword must be an integer, a ``Generator`` or a
    ``RandomState``.

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

    Set the seed if reproducibility is required

    >>> from numpy.random import default_rng
    >>> gen = default_rng(1234)
    >>> bs = StationaryBootstrap(12, x, y=y, z=z, seed=gen)
    """

    _name = "Stationary Bootstrap"
    # The block size is the mean of the geometric block length
    _integer_block_size = False

    def __init__(
        self,
        block_size: int,
        *args: ArrayLike,
        seed: int | Generator | RandomState | None = None,
        **kwargs: ArrayLike,
    ) -> None:
        super().__init__(block_size, *args, seed=seed, **kwargs)
        self._p = 1.0 / self._block_size

    def update_indices(self) -> Int64Array1D:
        indices = _get_random_integers(
            self._generator, self._num_items, size=self._num_items
        )
        indices = indices.astype(np.int64)
        if isinstance(self._generator, Generator):
            u = self._generator.random(self._num_items)
        else:
            assert isinstance(self._generator, RandomState)
            u = self._generator.random_sample(self._num_items)
        return stationary_bootstrap_sample(indices, u, self._p)


class MovingBlockBootstrap(CircularBlockBootstrap):
    """
    Bootstrap using blocks of the same length without wrap around.

    Parameters
    ----------
    block_size : int
        Size of block to use. Must be an integer that is at least 1. Blocks do
        not wrap around, so the block size cannot be larger than the number of
        observations, which is checked when the iteration starts.
    *args : ndarray, Series or DataFrame
        Positional arguments to bootstrap. Each must have at least one row.
    seed : int, Generator or RandomState, optional
        Seed to use to ensure reproducible results. If an int, it must be
        non-negative and is passed to ``np.random.default_rng``. If None, a
        fresh Generator is constructed with system-provided entropy.
    **kwargs : ndarray, Series or DataFrame
        Keyword arguments to bootstrap. Each must have at least one row.

    Attributes
    ----------
    data : tuple
        Two-element tuple with the pos_data in the first position and kw_data
        in the second (pos_data, kw_data).
    pos_data : tuple
        Tuple containing the positional arguments (in the order entered).
    kw_data : dict
        Dictionary containing the keyword arguments.

    Raises
    ------
    TypeError
        If an input is not an ndarray, Series or DataFrame, ``block_size`` is
        not an int, or ``seed`` is not None, a non-negative int, a Generator or a
        RandomState.
    ValueError
        If there are no inputs, an input has no rows, the inputs do not have
        the same number of rows, ``block_size`` is less than 1 (or, when the
        iteration starts, larger than the number of observations), ``seed`` is a
        negative int, or a keyword argument has the name of an attribute or
        method of the bootstrap.

    See Also
    --------
    arch.bootstrap.optimal_block_length
        Optimal block length estimation.
    arch.bootstrap.StationaryBootstrap
        Politis and Romano's bootstrap with exp. distributed block lengths.
    arch.bootstrap.CircularBlockBootstrap
        Circular (wrap-around) bootstrap.

    Notes
    -----
    Supports numpy arrays and pandas Series and DataFrames.  Data returned has
    the same type as the input data.

    The data are used as given. They are not converted to float64 and are not
    checked for NaN or inf, since the function that is bootstrapped decides how
    to treat them (for example, using ``np.nanmean``).

    Data entered using keyword arguments is directly accessible as an
    attribute.

    To ensure a reproducible bootstrap, you must pass the ``seed`` keyword
    argument when the bootstrap is created. See the example below.
    Note that ``seed`` is a reserved keyword and any variable
    passed using this keyword must be an integer, a ``Generator`` or a
    ``RandomState``.

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

    Set the seed if reproducibility is required

    >>> from numpy.random import default_rng
    >>> gen = default_rng(1234)
    >>> bs = MovingBlockBootstrap(7, x, y=y, z=z, seed = gen)
    """

    _name = "Moving Block Bootstrap"

    def __init__(
        self,
        block_size: int,
        *args: ArrayLike,
        seed: int | Generator | RandomState | None = None,
        **kwargs: ArrayLike,
    ) -> None:
        super().__init__(block_size, *args, seed=seed, **kwargs)

    def _check_sampling(self) -> None:
        # The blocks do not wrap around, so a block is at most the whole sample
        check_int("block_size", self._block_size, lower=1, upper=self._num_items)

    def update_indices(self) -> Int64Array1D:
        num_blocks = self._num_items // self.block_size
        if num_blocks * self.block_size < self._num_items:
            num_blocks += 1
        max_index = self._num_items - self.block_size + 1
        indices = _get_random_integers(self._generator, max_index, size=num_blocks)
        indices = indices[:, None] + np.arange(self.block_size)
        indices = indices.flatten()

        if indices.shape[0] > self._num_items:
            return cast("Int64Array1D", indices[: self._num_items])
        else:
            return indices


class MOONBootstrap(IIDBootstrap):  # pragma: no cover
    def __init__(
        self,
        block_size: int,
        *args: ArrayLike,
        seed: int | Generator | RandomState | None = None,
        **kwargs: ArrayLike,
    ) -> None:
        super().__init__(*args, seed=seed, **kwargs)
        self.block_size: int = block_size

    def update_indices(
        self,
    ) -> (
        Int64Array1D | tuple[list[Int64Array1D], dict[str, Int64Array1D]]
    ):  # pragma: no cover
        raise NotImplementedError
