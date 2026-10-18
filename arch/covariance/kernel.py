from arch.compat.numba import DISABLE_NUMBA, HAS_NUMBA, jit

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
import re
from typing import TypeVar, cast

import numpy as np
from pandas import DataFrame, Index

from arch._typing import ArrayLike, Float64Array, Float64Array2D
from arch.utility._validation import (
    _describe,
    _fail,
    check_array,
    check_bool,
    check_float,
    check_int,
    check_option,
)
from arch.utility.array import AbstractDocStringInheritor, ensure2d
from arch.vendor._decorators import Substitution

__all__ = [
    "Andrews",
    "Bartlett",
    "CovarianceEstimate",
    "CovarianceEstimator",
    "Gallant",
    "NeweyWest",
    "Parzen",
    "ParzenCauchy",
    "ParzenGeometric",
    "ParzenRiesz",
    "QuadraticSpectral",
    "TukeyHamming",
    "TukeyHanning",
    "TukeyParzen",
    "ZeroLag",
]

# The canonical names of the kernels that are available by name to the
# long-run covariance consumers, which are the names of the classes. This is the
# only list of kernel names: KERNEL_ESTIMATORS is built from it, and the
# estimators in arch.unitroot and arch.covariance.var look up kernels using
# get_kernel_estimator. ZeroLag is deliberately excluded since it is only a
# building block for arch.covariance.var.PreWhitenedRecolored when kernel=None
# (VAR-HAC).
KERNELS = [
    "Bartlett",
    "Parzen",
    "ParzenCauchy",
    "ParzenGeometric",
    "ParzenRiesz",
    "TukeyHamming",
    "TukeyHanning",
    "TukeyParzen",
    "QuadraticSpectral",
    "Andrews",
    "Gallant",
    "NeweyWest",
]


def normalize_kernel_name(kernel: str) -> str:
    """
    Normalize the name of a kernel.

    Parameters
    ----------
    kernel : str
        The name of a kernel.

    Returns
    -------
    str
        The name in lower case without hyphens or underscores, so that
        "quadratic-spectral" and "quadratic_spectral" are both
        "quadraticspectral".
    """
    return kernel.replace("-", "").replace("_", "").lower()


def _kernel_display_name(name: str) -> str:
    """The lower case name of a kernel with hyphens between the words"""
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


class CovarianceEstimate:
    r"""
    Covariance estimate using a long-run covariance estimator.

    Parameters
    ----------
    short_run : ndarray
        The short-run covariance estimate.
    one_sided_strict : ndarray
        The one-sided strict covariance estimate.
    columns : Index or list of str, optional
        Column labels to use if covariance estimates are returned as
        DataFrames.
    long_run : ndarray, optional
        The long-run covariance estimate. If not provided, computed from
        short_run and one_sided_strict.
    one_sided : ndarray, optional
        The one-sided covariance estimate. If not provided, computed
        from short_run and one_sided_strict.

    Notes
    -----
    If :math:`\Gamma_0` is the short-run covariance and :math:`\Lambda_1` is
    the one-sided strict covariance, then the long-run covariance is defined

    .. math::

        \Omega = \Gamma_0 + \Lambda_1 + \Lambda_1^H

    and the one-sided covariance is

    .. math::

        \Lambda_0 = \Gamma_0 + \Lambda_1.

    :math:`^H` is the conjugate transpose, which is the transpose for real
    estimates. The estimates of complex data are complex Hermitian matrices.
    """

    def __init__(
        self,
        short_run: Float64Array,
        one_sided_strict: Float64Array,
        columns: Index | list[str] | None = None,
        long_run: Float64Array | None = None,
        one_sided: Float64Array | None = None,
    ) -> None:
        self._sr = short_run
        self._oss = one_sided_strict
        self._columns = columns
        self._long_run = long_run
        self._one_sided = one_sided

    def _wrap(self, value: Float64Array) -> Float64Array | DataFrame:
        if self._columns is not None:
            return DataFrame(value, columns=self._columns, index=self._columns)
        return value

    @cached_property
    def long_run(self) -> Float64Array | DataFrame:
        """
        The long-run covariance estimate.
        """
        if self._long_run is not None:
            long_run = self._long_run
        elif np.iscomplexobj(self._oss):
            # Added in this order the result is exactly Hermitian
            long_run = self._sr + (self._oss + self._oss.conj().T)
        else:
            long_run = self._sr + self._oss + self._oss.T
        return self._wrap(long_run)

    @cached_property
    def short_run(self) -> Float64Array | DataFrame:
        """
        The short-run covariance estimate.
        """
        return self._wrap(self._sr)

    @cached_property
    def one_sided(self) -> Float64Array | DataFrame:
        """
        The one-sided covariance estimate.
        """
        if self._one_sided is not None:
            one_sided = self._one_sided
        else:
            one_sided = self._sr + self._oss
        return self._wrap(one_sided)

    @cached_property
    def one_sided_strict(self) -> Float64Array | DataFrame:
        """
        The one-sided strict covariance estimate.
        """
        return self._wrap(self._oss)


def _cov_kernel(
    df: int, k: int, num_weights: int, w: np.ndarray, x: np.ndarray
) -> np.ndarray:
    oss = np.zeros((k, k))
    for i in range(1, num_weights):
        oss += w[i] * (x[i:].T @ x[:-i]) / df
    return oss


def _cov_kernel_complex(
    df: int, k: int, num_weights: int, w: np.ndarray, x: np.ndarray
) -> np.ndarray:
    # sum_t x_t x_{t-i}^H, so that the lag-i autocovariance is Hermitian
    oss = np.zeros((k, k), dtype=x.dtype)
    x_conj = np.conj(x)
    for i in range(1, num_weights):
        oss += w[i] * (x[i:].T @ x_conj[:-i]) / df
    return oss


if HAS_NUMBA and not DISABLE_NUMBA:
    _cov_jit = jit(_cov_kernel, nopython=True)
    _cov_jit_complex = jit(_cov_kernel_complex, nopython=True)
else:
    _cov_jit = _cov_kernel
    _cov_jit_complex = _cov_kernel_complex


def _kernel_covariance_template() -> None:
    # The docstring is the template of the docstrings of the kernel estimators, which
    # is filled using the name of the kernel and the formula of its weights. It is
    # written as a docstring so that it is processed in the same way as the
    # docstring of a class.
    r"""
    %(kernel_name)s kernel covariance estimation.

    Parameters
    ----------
    x : array_like
        The data to use in covariance estimation. Observations are in rows and
        variables in columns. Integer, Boolean and float32 data are converted to
        float64, and complex data to complex128. float16 data are not accepted.
        Must have at least 2 rows and 1 column, and must not contain NaN or
        infinite values.
    bandwidth : float, optional
        The kernel's bandwidth.  If None, optimal bandwidth is estimated. Must
        be finite and non-negative.
    df_adjust : int, optional
        Degrees of freedom to remove when adjusting the covariance. Uses the
        number of observations in x minus df_adjust when dividing
        inner-products. Must be at least 0 and smaller than the number of
        observations in x.
    center : bool, optional
        A flag indicating whether x should be demeaned before estimating the
        covariance.
    weights : array_like, optional
        An array of weights used to combine when estimating optimal bandwidth.
        If not provided, a vector of 1s is used. Must have nvar elements, be
        finite and non-negative, and at least one must be strictly positive.
    force_int : bool, optional
        Force bandwidth to be an integer.

    Raises
    ------
    TypeError
        If x cannot be converted to float64 or complex128 (it is float16, or
        contains strings or objects that are not numbers), if weights cannot be
        converted to float64, if bandwidth is not a real number, if df_adjust is
        not an integer (a float, including 5.0, or a Boolean is not accepted), or
        if center or force_int is not a bool.
    ValueError
        If x is not 2-d, has fewer than 2 rows or no columns or contains NaN or
        infinite values; if bandwidth is negative or not finite; if df_adjust
        is negative or not smaller than the number of observations; or if
        weights does not have nvar finite elements, has a negative element or
        has only zeros.

    Notes
    -----
    The kernel weights are computed using

    .. math::

       %(formula)s

    where :math:`z=\frac{h}{H}, h=0, 1, \ldots, H` where H is the bandwidth.

    If the bandwidth is not provided, it is estimated using the constants of
    [1]_ and the automatic lag selection of [2]_. See ``opt_bandwidth`` for
    details.

    **Complex data.** The estimators accept complex data, for which they are
    Hermitian. The lag-:math:`j` autocovariance is
    :math:`\hat{\Gamma}_j=T^{-1}\sum_t x_t x_{t-j}^H`, where :math:`^H` is the
    conjugate transpose, :math:`\hat{\Gamma}_{-j}=\hat{\Gamma}_j^H`, and the
    long-run covariance
    :math:`\hat{\Omega}=\hat{\Gamma}_0+\sum_{j\geq 1}w_j(\hat{\Gamma}_j+
    \hat{\Gamma}_j^H)` is a complex Hermitian matrix. If :math:`x_t=a_t+ib_t`,
    then :math:`\hat{\Omega}=\hat{\Omega}_{aa}+\hat{\Omega}_{bb}+
    i(\hat{\Omega}_{ba}-\hat{\Omega}_{ab})`, where :math:`\hat{\Omega}_{uv}` is
    the long-run covariance of the real data :math:`u` and :math:`v` with the same
    kernel and bandwidth. The automatic bandwidth combines the columns of x
    using ``weights`` and uses the real part of the autocovariances of this
    complex series. The pseudo-covariance :math:`E[x_t x_{t-j}^\prime]` is not
    estimated, and the estimates are complex128 arrays. Real data give float64
    estimates.

    References
    ----------
    .. [1] Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation
       consistent covariance matrix estimation. Econometrica, 59(3), 817-858.
    .. [2] Newey, W. K., & West, K. D. (1994). Automatic lag selection in
       covariance matrix estimation. Review of Economic Studies, 61(4), 631-653.
    """


_KERNEL_COVARIANCE_DOC = _kernel_covariance_template.__doc__
_KernelT = TypeVar("_KernelT", bound=type)


def _kernel_substitution(
    kernel_name: str, formula: str
) -> Callable[[_KernelT], _KernelT]:
    """Set the docstring of a kernel estimator using its name and weight formula"""
    substitution = Substitution(kernel_name=kernel_name, formula=formula)

    def decorator(cls: _KernelT) -> _KernelT:
        cls.__doc__ = _KERNEL_COVARIANCE_DOC
        return substitution(cls)

    return decorator


class CovarianceEstimator(ABC):
    r"""
    Base class of the kernel-based long-run covariance estimators.

    Parameters
    ----------
    x : array_like
        The data to use in covariance estimation. Observations are in rows and
        variables in columns. Integer, Boolean and float32 data are converted to
        float64, and complex data to complex128. float16 data are not accepted.
        Must have at least 2 rows and 1 column, and must not contain NaN or
        infinite values.
    bandwidth : float, optional
        The kernel's bandwidth.  If None, optimal bandwidth is estimated. Must
        be finite and non-negative.
    df_adjust : int, optional
        Degrees of freedom to remove when adjusting the covariance. Uses the
        number of observations in x minus df_adjust when dividing
        inner-products. Must be at least 0 and smaller than the number of
        observations in x.
    center : bool, optional
        A flag indicating whether x should be demeaned before estimating the
        covariance.
    weights : array_like, optional
        An array of weights used to combine when estimating optimal bandwidth.
        If not provided, a vector of 1s is used. Must have nvar elements, be
        finite and non-negative, and at least one must be strictly positive.
    force_int : bool, optional
        Force bandwidth to be an integer.

    Raises
    ------
    TypeError
        If x cannot be converted to float64 or complex128 (it is float16, or
        contains strings or objects that are not numbers), if weights cannot be
        converted to float64, if bandwidth is not a real number, if df_adjust is
        not an integer (a float, including 5.0, or a Boolean is not accepted), or
        if center or force_int is not a bool.
    ValueError
        If x is not 2-d, has fewer than 2 rows or no columns or contains NaN or
        infinite values; if bandwidth is negative or not finite; if df_adjust
        is negative or not smaller than the number of observations; or if
        weights does not have nvar finite elements, has a negative element or
        has only zeros.

    Notes
    -----
    The kernel weights are defined by the concrete estimators that inherit from
    this class. They are computed using :math:`z=\frac{h}{H}, h=0, 1, \ldots, H`
    where H is the bandwidth.

    If the bandwidth is not provided, it is estimated using the constants of
    [1]_ and the automatic lag selection of [2]_. See ``opt_bandwidth`` for
    details.

    **Complex data.** The estimators accept complex data, for which they are
    Hermitian. The lag-:math:`j` autocovariance is
    :math:`\hat{\Gamma}_j=T^{-1}\sum_t x_t x_{t-j}^H`, where :math:`^H` is the
    conjugate transpose, :math:`\hat{\Gamma}_{-j}=\hat{\Gamma}_j^H`, and the
    long-run covariance
    :math:`\hat{\Omega}=\hat{\Gamma}_0+\sum_{j\geq 1}w_j(\hat{\Gamma}_j+
    \hat{\Gamma}_j^H)` is a complex Hermitian matrix. If :math:`x_t=a_t+ib_t`,
    then :math:`\hat{\Omega}=\hat{\Omega}_{aa}+\hat{\Omega}_{bb}+
    i(\hat{\Omega}_{ba}-\hat{\Omega}_{ab})`, where :math:`\hat{\Omega}_{uv}` is
    the long-run covariance of the real data :math:`u` and :math:`v` with the same
    kernel and bandwidth. The automatic bandwidth combines the columns of x
    using ``weights`` and uses the real part of the autocovariances of this
    complex series. The pseudo-covariance :math:`E[x_t x_{t-j}^\prime]` is not
    estimated, and the estimates are complex128 arrays. Real data give float64
    estimates.

    References
    ----------
    .. [1] Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation
       consistent covariance matrix estimation. Econometrica, 59(3), 817-858.
    .. [2] Newey, W. K., & West, K. D. (1994). Automatic lag selection in
       covariance matrix estimation. Review of Economic Studies, 61(4), 631-653.
    """

    _name = ""
    # Whether complex data are accepted. The estimators that cannot be Hermitian, such
    # as the VAR prewhitening, set this to False
    _allow_complex = True

    def __init__(
        self,
        x: ArrayLike,
        bandwidth: float | None = None,
        df_adjust: int = 0,
        center: bool = True,
        weights: ArrayLike | None = None,
        force_int: bool = False,
    ):
        x_checked = check_array(
            "x", x, ndim=2, min_length=2, allow_complex=self._allow_complex
        )
        nobs, nvar = x_checked.shape
        if nvar < 1:
            _fail(
                ValueError,
                "x",
                "a 2-d array of finite numbers with at least 2 rows and 1 column",
                _describe(x),
            )
        bandwidth = check_float("bandwidth", bandwidth, lower=0.0, allow_none=True)
        df_adjust = check_int("df_adjust", df_adjust, lower=0, upper=nobs - 1)
        center = check_bool("center", center)
        weights_checked = check_array(
            "weights", weights, shape=(nvar,), allow_none=True
        )
        if weights_checked is not None and (
            np.any(weights_checked < 0) or np.all(weights_checked == 0)
        ):
            _fail(
                ValueError,
                "weights",
                "non-negative with at least one value that is strictly greater "
                "than 0",
                _describe(weights),
            )
        force_int = check_bool("force_int", force_int)

        self._x_orig = ensure2d(x, "x", allow_complex=self._allow_complex)
        self._x = np.asarray(self._x_orig)
        self._center = center
        if self._center:
            self._x = self._x - self._x.mean(0)
        self._bandwidth: float | None = bandwidth
        self._auto_bandwidth = bandwidth is None
        self._df_adjust = df_adjust
        self._df = nobs - self._df_adjust
        self._x_weights: Float64Array2D
        if weights_checked is None:
            self._x_weights = np.ones((nvar, 1))
        else:
            self._x_weights = cast(
                "Float64Array2D",
                np.array(weights_checked, dtype=float).ravel()[:, None],
            )
        self._force_int = force_int

    def __str__(self) -> str:
        out = (
            f"Kernel: {self.name}",
            f"Bandwidth: {self.bandwidth}",
            f"Degree of Freedom Adjustment: {self._df_adjust}",
            f"Centered: {self.centered}",
            f"Automatic Bandwidth: {self._auto_bandwidth}",
        )
        return "\n".join(out)

    def __repr__(self) -> str:
        return self.__str__() + f"\nID: {hex(id(self))}"

    @property
    def name(self) -> str:
        """
        The covariance estimator's name.

        Returns
        -------
        str
            The covariance estimator's name.
        """
        return self._name

    @property
    def bandwidth(self) -> float:
        """
        The bandwidth used by the covariance estimator.

        Returns
        -------
        float
            The user-provided or estimated optimal bandwidth.
        """
        if self._bandwidth is None:
            self._bandwidth = self.opt_bandwidth
        if self._force_int:
            return float(int(np.ceil(self._bandwidth)))
        assert self._bandwidth is not None
        return self._bandwidth

    @property
    def centered(self) -> bool:
        """
        Flag indicating whether the data are centered (demeaned).

        Returns
        -------
        bool
            A flag indicating whether the estimator is centered.
        """
        return self._center

    @property
    @abstractmethod
    def kernel_const(self) -> float:
        """
        The constant used in optimal bandwidth calculation.

        Returns
        -------
        float
            The constant value used in the optimal bandwidth calculation.
        """

    @property
    @abstractmethod
    def bandwidth_scale(self) -> float:
        """
        The power used in optimal bandwidth calculation.

        Returns
        -------
        float
            The power value used in the optimal bandwidth calculation.
        """

    @property
    @abstractmethod
    def rate(self) -> float:
        """
        The optimal rate used in bandwidth selection.

        Controls the number of lags used in the variance estimate that
        determines the estimate of the optimal bandwidth.

        Returns
        -------
        float
            The rate used in bandwidth selection.
        """

    def _alpha_q(self) -> float:
        q = self.bandwidth_scale
        v = self._x @ self._x_weights
        nobs = v.shape[0]
        n = int(np.ceil(4 * ((nobs / 100) ** self.rate)))
        f_0s = 0.0
        f_qs = 0.0
        # The autocovariance of a complex series is v_t times the conjugate of v_{t-j},
        # which is complex. The sums over -n, ..., n add j and -j, which are conjugates
        # (Hermitian), so only the real part of sigma_j is used.
        lagged = v.conj() if np.iscomplexobj(v) else v
        for j in range(n + 1):
            sig_j = float(np.real(np.squeeze(v[j:].T @ lagged[: (nobs - j)]))) / nobs
            scale = 1 + (j != 0)
            f_0s += scale * sig_j
            f_qs += scale * j**q * sig_j
        return (f_qs / f_0s) ** 2

    @cached_property
    def opt_bandwidth(self) -> float:
        r"""
        Estimate optimal bandwidth.

        Returns
        -------
        float
            The estimated optimal bandwidth.

        Notes
        -----
        Computed as

        .. math::

           \hat{b}_T = c_k \left[\hat{\alpha}\left(q\right) T \right]^{\frac{1}{2q+1}}

        where :math:`c_k` is a kernel-dependent constant, T is the sample size,
        q determines the optimal bandwidth rate for the kernel.

        The constants and the values of q are from [1]_. For example,
        :math:`c_k` is 1.1447 and q is 1 for the Bartlett kernel, 2.6614 and 2
        for the Parzen kernel, and 1.3221 and 2 for the quadratic-spectral
        kernel. :math:`\hat{\alpha}\left(q\right)` is not the parametric (AR(1))
        estimate that is suggested in [1]_. It is the nonparametric estimate
        of [2]_,

        .. math::

           \hat{\alpha}\left(q\right) = \left(
           \frac{\sum_{j=-n}^{n} \left|j\right|^q \hat{\sigma}_j}
           {\sum_{j=-n}^{n} \hat{\sigma}_j} \right)^2

        where :math:`\hat{\sigma}_j` is the lag-j autocovariance of the data
        combined using ``weights``, which is the real part of the autocovariance
        :math:`T^{-1}\sum_t v_t\bar{v}_{t-j}` of the combined series :math:`v_t`
        if x is complex, and
        :math:`n = \lceil 4 \left(T / 100\right)^{r} \rceil`. The rate r is
        2/9 for the Bartlett kernel, 4/25 for the Parzen kernel and 2/25 for
        the quadratic-spectral kernel, as in [2]_.

        References
        ----------
        .. [1] Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation
           consistent covariance matrix estimation. Econometrica, 59(3), 817-858.
        .. [2] Newey, W. K., & West, K. D. (1994). Automatic lag selection in
           covariance matrix estimation. Review of Economic Studies, 61(4),
           631-653.
        """
        c = self.kernel_const
        q = self.bandwidth_scale
        nobs = self._x.shape[0]
        alpha_q = self._alpha_q()
        bw = c * (alpha_q * nobs) ** (1 / (2 * q + 1))
        if self._force_int:
            bw = np.ceil(bw)
        return min(bw, nobs - 1.0)

    @abstractmethod
    def _weights(self) -> Float64Array:
        """
        Compute the kernel's weights
        """

    @cached_property
    def kernel_weights(self) -> Float64Array:
        """
        The weights used in covariance calculation.

        Returns
        -------
        ndarray
            The weight vector including 1 in position 0.
        """
        return self._weights()

    @cached_property
    def cov(self) -> CovarianceEstimate:
        """
        The estimated covariances.

        Returns
        -------
        CovarianceEstimate
            Covariance estimate instance containing 4 estimates:

            * long_run
            * short_run
            * one_sided
            * one_sided_strict

        See Also
        --------
        CovarianceEstimate : The container of the covariance estimates.
        """
        x = np.ascontiguousarray(self._x)
        k = x.shape[1]
        df = self._df
        w = self.kernel_weights
        num_weights = w.shape[0]

        if np.iscomplexobj(x):
            # Hermitian estimates: sum_t x_t x_{t-j}^H. The short-run covariance is
            # made exactly Hermitian, since the matrix product rounds each element
            # on its own.
            sr = x.T @ x.conj() / df
            sr = (sr + sr.conj().T) / 2
            oss = _cov_jit_complex(df, k, num_weights, w, x)
        else:
            sr = x.T @ x / df
            oss = _cov_jit(df, k, num_weights, w, x)

        labels = self._x_orig.columns if isinstance(self._x_orig, DataFrame) else None
        return CovarianceEstimate(sr, oss, labels)

    @property
    def force_int(self) -> bool:
        """
        Flag indicating whether the bandwidth is restricted to be an integer.
        """
        return self._force_int


_bartlett_formula = """\
w=\\begin{cases} 1-\\left|z\\right| & z\\leq1 \\\\ 0 & z>1 \\end{cases}
"""


@_kernel_substitution(kernel_name="Bartlett's (Newey-West)", formula=_bartlett_formula)
class Bartlett(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.1447

    @property
    def bandwidth_scale(self) -> float:
        return 1.0

    @property
    def rate(self) -> float:
        return 2 / 9

    def _weights(self) -> Float64Array:
        bw = self.bandwidth
        return ((bw + 1 - np.arange(int(bw + 1), dtype="double")) / (bw + 1)).astype(
            float
        )


_parzen_formula = """\
w=\\begin{cases}\
1-6z^{2}\\left(1-z\\right) & z\\leq\\frac{1}{2} \\\\ \
2\\left(1-z\\right)^{3} & \\frac{1}{2}<z\\leq1 \\\\ \
0 & z>1 \
\\end{cases}
"""


@_kernel_substitution(kernel_name="Parzen's", formula=_parzen_formula)
class Parzen(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 2.6614

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self) -> Float64Array:
        bw = self.bandwidth
        x = np.arange(int(bw + 1), dtype="double") / (bw + 1)
        w = np.empty_like(x)
        loc = x <= 0.5
        w[loc] = 1 - 6 * x[loc] ** 2 * (1 - x[loc])
        w[~loc] = 2 * (1 - x[~loc]) ** 3
        return w


_parzen_reisz_formula = """\
w=\\begin{cases} \
1-z^2 & z\\leq1 \\\\ \
0 & z>1 \
\\end{cases} \
"""


@_kernel_substitution(kernel_name="Parzen-Reisz", formula=_parzen_reisz_formula)
class ParzenRiesz(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.1340

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self) -> Float64Array:
        bw = self.bandwidth
        x = np.arange(int(bw + 1), dtype="double") / (bw + 1)
        return 1 - x**2


_parzen_geometric_formula = """\
w=\\begin{cases} \
\\frac{1}{1+z} & z\\leq1 \\\\ \
0 & z>1 \
\\end{cases} \
"""


@_kernel_substitution(
    kernel_name="Parzen's Geometric", formula=_parzen_geometric_formula
)
class ParzenGeometric(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.0000

    @property
    def bandwidth_scale(self) -> float:
        return 1

    @property
    def rate(self) -> float:
        return 2 / 9

    def _weights(self) -> Float64Array:
        bw = self.bandwidth
        x = np.arange(int(bw + 1), dtype="double") / (bw + 1)
        return 1 / (1 + x)


_parzen_cauchy_formula = """\
w=\\begin{cases} \
\\frac{1}{1+z^2} & z\\leq1 \\\\ \
0 & z>1 \
\\end{cases} \
"""


@_kernel_substitution(kernel_name="Parzen's Cauchy", formula=_parzen_cauchy_formula)
class ParzenCauchy(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.0924

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self) -> Float64Array:
        bw = self.bandwidth
        x = np.arange(int(bw + 1), dtype="double") / (bw + 1)
        return 1 / (1 + x**2)


_tukey_hamming_formula = """\
w=\\begin{cases} \
0.54 + 0.46 \\cos{\\pi z} & z\\leq1 \\\\ \
0 & z>1 \
\\end{cases} \
"""


@_kernel_substitution(kernel_name="Tukey-Hamming", formula=_tukey_hamming_formula)
class TukeyHamming(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.6694

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self) -> Float64Array:
        bw = self.bandwidth
        x = np.arange(int(bw + 1), dtype="double") / (bw + 1)
        return 0.54 + 0.46 * np.cos(np.pi * x)


_tukey_hanning_formula = """\
w=\\begin{cases} \
\\frac{1}{2} + \\frac{1}{2} \\cos{\\pi z} & z\\leq1 \\\\ \
0 & z>1 \
\\end{cases} \
"""


@_kernel_substitution(kernel_name="Tukey-Hanning", formula=_tukey_hanning_formula)
class TukeyHanning(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.7462

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self) -> Float64Array:
        bw = self.bandwidth
        x = np.arange(int(bw + 1), dtype="double") / (bw + 1)
        return 0.5 + 0.5 * np.cos(np.pi * x)


_tukey_parzen_formula = """\
w=\\begin{cases} \
0.436 + 0.564 \\cos{\\pi z} & z\\leq1 \\\\ \
0 & z>1 \
\\end{cases} \
"""


@_kernel_substitution(kernel_name="Tukey-Parzen", formula=_tukey_parzen_formula)
class TukeyParzen(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.8576

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self) -> Float64Array:
        bw = self.bandwidth
        x = np.arange(int(bw + 1), dtype="double") / (bw + 1)
        return 0.436 + 0.564 * np.cos(np.pi * x)


_qs_name = "Quadratic-Spectral (Andrews')"
_qs_formula = """\
w=\\begin{cases} \
1 & z=0\\\\ \
\\frac{3}{x^{2}}\\left(\\frac{\\sin x}{x}-\\cos x\\right),x=\\frac{6\\pi z}{5} & z>0 \
\\end{cases} \
"""


@_kernel_substitution(kernel_name=_qs_name, formula=_qs_formula)
class QuadraticSpectral(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.3221

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 2 / 25

    def _weights(self) -> Float64Array:
        bw = self.bandwidth
        nobs = self._x.shape[0]
        w = np.zeros(nobs)
        w[0] = 1.0
        if bw > 0:
            x = np.arange(1, nobs) / bw
            z = 6 * np.pi * x / 5
            w[1:] = 3 / z**2 * (np.sin(z) / z - np.cos(z))
        return w


class Gallant(Parzen):  # numpydoc ignore=PR02
    """
    Alternative name for Parzen covariance estimator.

    Parameters
    ----------
    x : array_like
        The data to use in covariance estimation. Observations are in rows and
        variables in columns. Integer, Boolean and float32 data are converted to
        float64, and complex data to complex128. float16 data are not accepted.
        Must have at least 2 rows and 1 column, and must not contain NaN or
        infinite values.
    bandwidth : float, optional
        The kernel's bandwidth.  If None, optimal bandwidth is estimated. Must
        be finite and non-negative.
    df_adjust : int, optional
        Degrees of freedom to remove when adjusting the covariance. Uses the
        number of observations in x minus df_adjust when dividing
        inner-products. Must be at least 0 and smaller than the number of
        observations in x.
    center : bool, optional
        A flag indicating whether x should be demeaned before estimating the
        covariance.
    weights : array_like, optional
        An array of weights used to combine when estimating optimal bandwidth.
        If not provided, a vector of 1s is used. Must have nvar elements, be
        finite and non-negative, and at least one must be strictly positive.
    force_int : bool, optional
        Force bandwidth to be an integer.

    Raises
    ------
    TypeError
        If x cannot be converted to float64 or complex128 (it is float16, or
        contains strings or objects that are not numbers), if weights cannot be
        converted to float64, if bandwidth is not a real number, if df_adjust is
        not an integer (a float, including 5.0, or a Boolean is not accepted), or
        if center or force_int is not a bool.
    ValueError
        If x is not 2-d, has fewer than 2 rows or no columns or contains NaN or
        infinite values; if bandwidth is negative or not finite; if df_adjust
        is negative or not smaller than the number of observations; or if
        weights does not have nvar finite elements, has a negative element or
        has only zeros.

    See Also
    --------
    Parzen : Kernel covariance estimator using the Parzen kernel.
    """


class Andrews(QuadraticSpectral):  # numpydoc ignore=PR02
    """
    Alternative name of the QuadraticSpectral covariance estimator.

    Parameters
    ----------
    x : array_like
        The data to use in covariance estimation. Observations are in rows and
        variables in columns. Integer, Boolean and float32 data are converted to
        float64, and complex data to complex128. float16 data are not accepted.
        Must have at least 2 rows and 1 column, and must not contain NaN or
        infinite values.
    bandwidth : float, optional
        The kernel's bandwidth.  If None, optimal bandwidth is estimated. Must
        be finite and non-negative.
    df_adjust : int, optional
        Degrees of freedom to remove when adjusting the covariance. Uses the
        number of observations in x minus df_adjust when dividing
        inner-products. Must be at least 0 and smaller than the number of
        observations in x.
    center : bool, optional
        A flag indicating whether x should be demeaned before estimating the
        covariance.
    weights : array_like, optional
        An array of weights used to combine when estimating optimal bandwidth.
        If not provided, a vector of 1s is used. Must have nvar elements, be
        finite and non-negative, and at least one must be strictly positive.
    force_int : bool, optional
        Force bandwidth to be an integer.

    Raises
    ------
    TypeError
        If x cannot be converted to float64 or complex128 (it is float16, or
        contains strings or objects that are not numbers), if weights cannot be
        converted to float64, if bandwidth is not a real number, if df_adjust is
        not an integer (a float, including 5.0, or a Boolean is not accepted), or
        if center or force_int is not a bool.
    ValueError
        If x is not 2-d, has fewer than 2 rows or no columns or contains NaN or
        infinite values; if bandwidth is negative or not finite; if df_adjust
        is negative or not smaller than the number of observations; or if
        weights does not have nvar finite elements, has a negative element or
        has only zeros.

    See Also
    --------
    QuadraticSpectral : Kernel covariance estimator using the quadratic spectral
        kernel.
    """


class NeweyWest(Bartlett):  # numpydoc ignore=PR02
    """
    Alternative name for Bartlett covariance estimator.

    Parameters
    ----------
    x : array_like
        The data to use in covariance estimation. Observations are in rows and
        variables in columns. Integer, Boolean and float32 data are converted to
        float64, and complex data to complex128. float16 data are not accepted.
        Must have at least 2 rows and 1 column, and must not contain NaN or
        infinite values.
    bandwidth : float, optional
        The kernel's bandwidth.  If None, optimal bandwidth is estimated. Must
        be finite and non-negative.
    df_adjust : int, optional
        Degrees of freedom to remove when adjusting the covariance. Uses the
        number of observations in x minus df_adjust when dividing
        inner-products. Must be at least 0 and smaller than the number of
        observations in x.
    center : bool, optional
        A flag indicating whether x should be demeaned before estimating the
        covariance.
    weights : array_like, optional
        An array of weights used to combine when estimating optimal bandwidth.
        If not provided, a vector of 1s is used. Must have nvar elements, be
        finite and non-negative, and at least one must be strictly positive.
    force_int : bool, optional
        Force bandwidth to be an integer.

    Raises
    ------
    TypeError
        If x cannot be converted to float64 or complex128 (it is float16, or
        contains strings or objects that are not numbers), if weights cannot be
        converted to float64, if bandwidth is not a real number, if df_adjust is
        not an integer (a float, including 5.0, or a Boolean is not accepted), or
        if center or force_int is not a bool.
    ValueError
        If x is not 2-d, has fewer than 2 rows or no columns or contains NaN or
        infinite values; if bandwidth is negative or not finite; if df_adjust
        is negative or not smaller than the number of observations; or if
        weights does not have nvar finite elements, has a negative element or
        has only zeros.

    See Also
    --------
    Bartlett : Kernel covariance estimator using the Bartlett kernel.
    """


_zero_lag_name = "Zero-lag (No autocorrelation)"
_zero_lag_formula = """\
w=\\begin{cases} \
1 & z=0\\\\ \
0 & z>0 \
\\end{cases} \
"""


@_kernel_substitution(kernel_name=_zero_lag_name, formula=_zero_lag_formula)
class ZeroLag(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.0

    @property
    def bandwidth_scale(self) -> float:
        return 0.0

    @property
    def rate(self) -> float:
        return 0.0

    @cached_property
    def opt_bandwidth(self) -> float:
        """
        The optimal bandwidth, which is always 0 since no lags are used.

        Returns
        -------
        float
            The bandwidth, always 0.
        """
        return 0.0

    def _weights(self) -> Float64Array:
        return np.ones(1)


KERNEL_ESTIMATORS: dict[str, type[CovarianceEstimator]] = {
    normalize_kernel_name(name): globals()[name] for name in KERNELS
}
"""The kernel estimators in KERNELS, keyed by their normalized names."""


def get_kernel_estimator(kernel: str) -> type[CovarianceEstimator]:
    """
    Find a kernel estimator using its name.

    Parameters
    ----------
    kernel : str
        The name of a kernel estimator, such as "bartlett" or
        "quadratic-spectral". The names are those of the classes in ``KERNELS``
        in lower case with hyphens between the words. The name is not case
        sensitive and hyphens and underscores are ignored, so that
        "quadratic-spectral", "quadratic_spectral" and "quadraticspectral" are
        all the same.

    Returns
    -------
    type
        The kernel estimator class, a subclass of CovarianceEstimator.

    Raises
    ------
    ValueError
        If the name is not one of the names in ``KERNELS``.
    """
    key = check_option(
        "kernel",
        kernel,
        tuple(KERNEL_ESTIMATORS),
        display=sorted(_kernel_display_name(name) for name in KERNELS),
        normalize=normalize_kernel_name,
    )
    return KERNEL_ESTIMATORS[key]
