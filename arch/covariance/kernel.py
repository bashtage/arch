from arch.compat.numba import jit

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional, SupportsInt, Union, cast

import numpy as np
from pandas import DataFrame, Index
from pandas.util._decorators import Substitution

from arch.typing import ArrayLike, Float64Array
from arch.utility.array import AbstractDocStringInheritor, ensure1d, ensure2d

__all__ = [
    "Bartlett",
    "Parzen",
    "ParzenCauchy",
    "ParzenGeometric",
    "ParzenRiesz",
    "TukeyHamming",
    "TukeyHanning",
    "TukeyParzen",
    "CovarianceEstimate",
    "CovarianceEstimator",
    "QuadraticSpectral",
    "Andrews",
    "Gallant",
    "NeweyWest",
]

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


class CovarianceEstimate:
    r"""
    Covariance estimate using a long-run covariance estimator

    Parameters
    ----------
    short_run : ndarray
        The short-run covariance estimate.
    one_sided_strict : ndarray
        The one-sided strict covariance estimate.
    columns : {None, list[str]}
        Column labels to use if covariance estimates are returned as
        DataFrames.
    long_run : ndarray, default None
        The long-run covariance estimate. If not provided, computed from
        short_run and one_sided_strict.
    one_sided : ndarray, default None
        The one-sided-strict covariance estimate. If not provided, computed
        from short_run and one_sided_strict.

    Notes
    -----
    If :math:`\Gamma_0` is the short-run covariance and :math:`\Lambda_1` is
    the one-sided strict covariance, then the long-run covariance is defined

    .. math::

        \Omega = \Gamma_0 + \Lambda_1 + \Lambda_1^\prime

    and the one-sided covariance is

    .. math::

        \Lambda_0 = \Gamma_0 + \Lambda_1.
    """

    def __init__(
        self,
        short_run: Float64Array,
        one_sided_strict: Float64Array,
        columns: Union[Index, list[str], None] = None,
        long_run: Optional[Float64Array] = None,
        one_sided: Optional[Float64Array] = None,
    ) -> None:
        self._sr = short_run
        self._oss = one_sided_strict
        self._columns = columns
        self._long_run = long_run
        self._one_sided = one_sided

    def _wrap(self, value: Float64Array) -> Union[Float64Array, DataFrame]:
        if self._columns is not None:
            return DataFrame(value, columns=self._columns, index=self._columns)
        return value

    @cached_property
    def long_run(self) -> Union[Float64Array, DataFrame]:
        """
        The long-run covariance estimate.
        """
        if self._long_run is not None:
            long_run = self._long_run
        else:
            long_run = self._sr + self._oss + self._oss.T
        return self._wrap(long_run)

    @cached_property
    def short_run(self) -> Union[Float64Array, DataFrame]:
        """
        The short-run covariance estimate.
        """
        return self._wrap(self._sr)

    @cached_property
    def one_sided(self) -> Union[Float64Array, DataFrame]:
        """
        The one-sided covariance estimate.
        """
        if self._one_sided is not None:
            one_sided = self._one_sided
        else:
            one_sided = self._sr + self._oss
        return self._wrap(one_sided)

    @cached_property
    def one_sided_strict(self) -> Union[Float64Array, DataFrame]:
        """
        The one-sided strict covariance estimate.
        """
        return self._wrap(self._oss)


@jit(nopython=True)
def _cov_jit(
    df: int, k: int, num_weights: int, w: np.ndarray, x: np.ndarray
) -> np.ndarray:
    oss = np.zeros((k, k))
    for i in range(1, num_weights):
        oss += w[i] * (x[i:].T @ x[:-i]) / df
    return oss


class CovarianceEstimator(ABC):
    r"""
    %(kernel_name)s kernel covariance estimation.

    Parameters
    ----------
    x : array_like
        The data to use in covariance estimation.
    bandwidth : float, default None
        The kernel's bandwidth.  If None, optimal bandwidth is estimated.
    df_adjust : int, default 0
        Degrees of freedom to remove when adjusting the covariance. Uses the
        number of observations in x minus df_adjust when dividing
        inner-products.
    center : bool, default True
        A flag indicating whether x should be demeaned before estimating the
        covariance.
    weights : array_like, default None
        An array of weights used to combine when estimating optimal bandwidth.
        If not provided, a vector of 1s is used. Must have nvar elements.
    force_int : bool, default False
        Force bandwidth to be an integer.

    Notes
    -----
    The kernel weights are computed using

    .. math::

       %(formula)s

    where :math:`z=\frac{h}{H}, h=0, 1, \ldots, H` where H is the bandwidth.
    """

    _name = ""

    def __init__(
        self,
        x: ArrayLike,
        bandwidth: Optional[float] = None,
        df_adjust: int = 0,
        center: bool = True,
        weights: Optional[ArrayLike] = None,
        force_int: bool = False,
    ):
        self._x_orig = ensure2d(x, "x")
        self._x = np.asarray(self._x_orig)
        self._center = center
        if self._center:
            self._x = self._x - self._x.mean(0)
        if bandwidth is not None:
            if not np.isscalar(bandwidth) or (cast(float, bandwidth) < 0.0):
                raise ValueError("bandwidth must be a non-negative scalar.")
        self._bandwidth = bandwidth
        self._auto_bandwidth = bandwidth is None
        if not np.isscalar(df_adjust) or (cast(int, df_adjust) < 0):
            raise ValueError("df_adjust must be a non-negative integer.")
        self._df_adjust = int(cast(SupportsInt, df_adjust))
        self._df = self._x.shape[0] - self._df_adjust
        if self._df <= 0:
            raise ValueError(
                "Degrees of freedom is <= 0 after adjusting the sample "
                "size of x using df_adjust. df_adjust must be less than"
                f" {self._x.shape[0]}"
            )
        if weights is None:
            xw = self._x_weights = np.ones((self._x.shape[1], 1))
        else:
            xw = ensure1d(np.asarray(weights), "weights", series=False)
            xw = self._x_weights = xw[:, None]
        if (
            xw.shape[0] != self._x.shape[1]
            or xw.shape[1] != 1
            or np.any(xw < 0)
            or np.all(xw == 0)
        ):
            raise ValueError(
                f"weights must be a 1 by {self._x.shape[1]} (x.shape[1]) "
                f"array with non-negative values where at least one value is "
                "strictly greater than 0."
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
        for j in range(n + 1):
            sig_j = float(np.squeeze(v[j:].T @ v[: (nobs - j)])) / nobs
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
        Weights used in covariance calculation.

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
        CovarianceEstimate
        """
        x = np.ascontiguousarray(self._x)
        k = x.shape[1]
        df = self._df
        sr = x.T @ x / df
        w = self.kernel_weights
        num_weights = w.shape[0]

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


@Substitution(kernel_name="Bartlett's (Newey-West)", formula=_bartlett_formula)
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
        return (bw + 1 - np.arange(int(bw + 1), dtype="double")) / (bw + 1)


_parzen_formula = """\
w=\\begin{cases}\
1-6z^{2}\\left(1-z\\right) & z\\leq\\frac{1}{2} \\\\ \
2\\left(1-z\\right)^{3} & \\frac{1}{2}<z\\leq1 \\\\ \
0 & z>1 \
\\end{cases}
"""


@Substitution(kernel_name="Parzen's", formula=_parzen_formula)
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


@Substitution(kernel_name="Parzen-Reisz", formula=_parzen_reisz_formula)
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


@Substitution(kernel_name="Parzen's Geometric", formula=_parzen_geometric_formula)
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


@Substitution(kernel_name="Parzen's Cauchy", formula=_parzen_cauchy_formula)
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


@Substitution(kernel_name="Tukey-Hamming", formula=_tukey_hamming_formula)
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


@Substitution(kernel_name="Tukey-Hanning", formula=_tukey_hanning_formula)
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


@Substitution(kernel_name="Tukey-Parzen", formula=_tukey_parzen_formula)
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


@Substitution(kernel_name=_qs_name, formula=_qs_formula)
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


class Gallant(Parzen):
    """
    Alternative name for Parzen covariance estimator.

    See Also
    --------
    Parzen
    """


class Andrews(QuadraticSpectral):
    """
    Alternative name of the QuadraticSpectral covariance estimator.

    See Also
    --------
    QuadraticSpectral
    """


class NeweyWest(Bartlett):
    """
    Alternative name for Bartlett covariance estimator.

    See Also
    --------
    Bartlett
    """
