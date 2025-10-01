"""
Distributions to use in ARCH models.  All distributions must inherit from
:class:`Distribution` and provide the same methods with the same inputs.
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from numpy import (
    abs,
    array,
    asarray,
    empty,
    exp,
    float64,
    int64,
    integer,
    isscalar,
    log,
    nan,
    ndarray,
    ones_like,
    pi,
    sign,
    sqrt,
    sum,
)
from numpy.random import Generator, RandomState, default_rng
from scipy import stats
from scipy.special import comb, gamma, gammainc, gammaincc, gammaln

from arch._typing import ArrayLike, ArrayLike1D, Float64Array, Float64Array1D
from arch.utility.array import AbstractDocStringInheritor, ensure1d, to_array_1d

__all__ = ["Distribution", "GeneralizedError", "Normal", "SkewStudent", "StudentsT"]


class Distribution(metaclass=ABCMeta):
    """
    Template for subclassing only
    """

    def __init__(
        self,
        *,
        seed: int | RandomState | Generator | None = None,
    ) -> None:
        self._name = "Distribution"
        self.num_params: int = 0
        self._parameters: Float64Array1D | None = None
        _seed = seed
        if _seed is None:
            self._generator: Generator | RandomState = default_rng()
        elif isinstance(_seed, (int, integer)):
            self._generator = default_rng(_seed)
        elif isinstance(_seed, (RandomState, Generator)):
            self._generator = _seed
        else:
            raise TypeError(
                "seed must by a NumPy Generator or RandomState or int, if not None."
            )

    @property
    def name(self) -> str:
        """The name of the distribution"""
        return self._name

    def _check_constraints(
        self, parameters: Sequence[float] | ArrayLike1D | None
    ) -> Float64Array1D:
        bounds = self.bounds(empty(0))
        if parameters is not None:
            params = asarray(ensure1d(parameters, "parameters", False))
            nparams = len(params)
        else:
            nparams = 0
            params = empty(0)
        if nparams != len(bounds):
            raise ValueError(f"parameters must have {len(bounds)} elements")
        if len(bounds) == 0:
            return empty(0)
        for p, n, b in zip(params, self.name, bounds, strict=False):
            if not (b[0] <= p <= b[1]):
                raise ValueError(
                    f"{n} does not satisfy the bounds requirement of ({b[0]}, {b[1]})"
                )
        return asarray(params)

    @property
    def generator(self) -> RandomState | Generator:
        """The NumPy Generator or RandomState attached to the distribution"""
        return self._generator

    @abstractmethod
    def _simulator(self, size: int | tuple[int, ...]) -> Float64Array:
        """
        Simulate i.i.d. draws from the distribution

        Parameters
        ----------
        size : int or tuple
            Shape of the draws from the distribution

        Returns
        -------
        rvs : ndarray
            Simulated pseudo random variables

        Notes
        -----
        Must call `simulate` before using `_simulator`
        """

    @abstractmethod
    def simulate(
        self, parameters: float | Sequence[float | int] | ArrayLike1D
    ) -> Callable[[int | tuple[int, ...]], Float64Array]:
        """
        Simulates i.i.d. draws from the distribution

        Parameters
        ----------
        parameters : ndarray
            Distribution parameters

        Returns
        -------
        simulator : callable
            Callable that take a single output size argument and returns i.i.d.
            draws from the distribution
        """

    @abstractmethod
    def constraints(self) -> tuple[Float64Array, Float64Array]:
        """
        Construct arrays to use in constrained optimization.

        Returns
        -------
        A : ndarray
            Constraint loadings
        b : ndarray
            Constraint values

        Notes
        -----
        Parameters satisfy the constraints A.dot(parameters)-b >= 0
        """

    @abstractmethod
    def bounds(self, resids: ArrayLike1D) -> list[tuple[float, float]]:
        """
        Parameter bounds for use in optimization.

        Parameters
        ----------
        resids : ndarray
             Residuals to use when computing the bounds

        Returns
        -------
        bounds : list
            List containing a single tuple with (lower, upper) bounds
        """

    @abstractmethod
    def loglikelihood(
        self,
        parameters: Sequence[float] | ArrayLike1D,
        resids: ArrayLike,
        sigma2: ArrayLike,
        individual: bool = False,
    ) -> float | Float64Array1D:
        """
        Loglikelihood evaluation.

        Parameters
        ----------
        parameters : ndarray
            Distribution shape parameters
        resids : ndarray
            nobs array of model residuals
        sigma2 : ndarray
            nobs array of conditional variances
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Notes
        -----
        Returns the loglikelihood where resids are the "data",
        and parameters and sigma2 are inputs.
        """

    @abstractmethod
    def starting_values(self, std_resid: ArrayLike1D) -> Float64Array1D:
        """
        Construct starting values for use in optimization.

        Parameters
        ----------
        std_resid : ndarray
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : ndarray
            The estimated shape parameters for the distribution

        Notes
        -----
        Size of sv depends on the distribution
        """

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """
        Names of distribution shape parameters

        Returns
        -------
        names : list (str)
            Parameter names
        """

    @abstractmethod
    def ppf(
        self,
        pits: float | Sequence[float] | ArrayLike1D,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> float | Float64Array:
        """
        Inverse cumulative density function (ICDF)

        Parameters
        ----------
        pits : {float, ndarray}
            Probability-integral-transformed values in the interval (0, 1).
        parameters : ndarray, optional
            Distribution parameters. Use ``None`` for parameterless
            distributions.

        Returns
        -------
        i : {float, ndarray}
            Inverse CDF values
        """

    @abstractmethod
    def cdf(
        self,
        resids: Sequence[float] | ArrayLike1D,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> Float64Array:
        """
        Cumulative distribution function

        Parameters
        ----------
        resids : ndarray
            Values at which to evaluate the cdf
        parameters : ndarray
            Distribution parameters. Use ``None`` for parameterless
            distributions.

        Returns
        -------
        f : ndarray
            CDF values
        """

    @abstractmethod
    def moment(
        self, n: int, parameters: Sequence[float] | ArrayLike1D | None = None
    ) -> float:
        """
        Moment of order n

        Parameters
        ----------
        n : int
            Order of moment
        parameters : ndarray, optional
            Distribution parameters. Use None for parameterless distributions.

        Returns
        -------
        float
            Calculated moment
        """

    @abstractmethod
    def partial_moment(
        self,
        n: int,
        z: float = 0.0,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> float:
        r"""
        Order n lower partial moment from -inf to z

        Parameters
        ----------
        n : int
            Order of partial moment
        z : float, optional
            Upper bound for partial moment integral
        parameters : ndarray, optional
            Distribution parameters.  Use None for parameterless distributions.

        Returns
        -------
        float
            Partial moment

        References
        ----------
        .. [1] Winkler et al. (1972) "The Determination of Partial Moments"
               *Management Science* Vol. 19 No. 3

        Notes
        -----
        The order n lower partial moment to z is

        .. math::

            \int_{-\infty}^{z}x^{n}f(x)dx

        See [1]_ for more details.
        """

    def __str__(self) -> str:
        return self._description()

    def __repr__(self) -> str:
        return self.__str__() + ", id: " + hex(id(self)) + ""

    def _description(self) -> str:
        return self.name + " distribution"


class Normal(Distribution, metaclass=AbstractDocStringInheritor):
    """
    Standard normal distribution for use with ARCH models

    Parameters
    ----------
    seed : {int, Generator, RandomState}, optional
        Random number generator instance or int to use. Set to ensure
        reproducibility. If using an int, the argument is passed to
        ``np.random.default_rng``.  If not provided, ``default_rng``
        is used with system-provided entropy.
    """

    def __init__(
        self,
        *,
        seed: None | int | RandomState | Generator = None,
    ) -> None:
        super().__init__(seed=seed)
        self._name = "Normal"

    def constraints(self) -> tuple[Float64Array, Float64Array]:
        return empty(0), empty(0)

    def bounds(self, resids: ArrayLike1D) -> list[tuple[float, float]]:
        return []

    def loglikelihood(
        self,
        parameters: Sequence[float] | ArrayLike1D,
        resids: ArrayLike,
        sigma2: ArrayLike,
        individual: bool = False,
    ) -> float | Float64Array1D:
        r"""Computes the log-likelihood of assuming residuals are normally
        distributed, conditional on the variance

        Parameters
        ----------
        parameters : ndarray
            The normal likelihood has no shape parameters. Empty since the
            standard normal has no shape parameters.
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma2 : ndarray
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln f\left(x\right)=-\frac{1}{2}\left(\ln2\pi+\ln\sigma^{2}
            +\frac{x^{2}}{\sigma^{2}}\right)

        """
        lls = -0.5 * (log(2 * pi) + log(sigma2) + resids**2.0 / sigma2)
        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid: ArrayLike1D) -> Float64Array1D:
        return empty(0)

    def _simulator(self, size: int | tuple[int, ...]) -> Float64Array:
        return self._generator.standard_normal(size)

    def simulate(
        self, parameters: float | Sequence[float | int] | ArrayLike1D
    ) -> Callable[[int | tuple[int, ...]], Float64Array]:
        return self._simulator

    def parameter_names(self) -> list[str]:
        return []

    def cdf(
        self,
        resids: Sequence[float] | ArrayLike1D,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> Float64Array:
        self._check_constraints(parameters)
        return stats.norm.cdf(asarray(resids))

    def ppf(
        self,
        pits: float | Sequence[float] | ArrayLike1D,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> Float64Array:
        self._check_constraints(parameters)
        scalar = isscalar(pits)
        if scalar:
            assert isinstance(pits, float)
            _pits = array([pits], dtype=np.float64)
        else:
            _pits = asarray(pits, dtype=float)
        ppf = stats.norm.ppf(_pits)
        if scalar:
            return ppf[0]
        else:
            return ppf

    def moment(
        self, n: int, parameters: Sequence[float] | ArrayLike1D | None = None
    ) -> float:
        if n < 0:
            return nan

        return stats.norm.moment(n)

    def partial_moment(
        self,
        n: int,
        z: float = 0.0,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> float:
        if n < 0:
            return nan
        elif n == 0:
            return stats.norm.cdf(z)
        elif n == 1:
            return -stats.norm.pdf(z)
        else:
            return -(z ** (n - 1)) * stats.norm.pdf(z) + (n - 1) * self.partial_moment(
                n - 2, z, parameters
            )


class StudentsT(Distribution, metaclass=AbstractDocStringInheritor):
    """
    Standardized Student's distribution for use with ARCH models

    Parameters
    ----------
    seed : {int, Generator, RandomState}, optional
        Random number generator instance or int to use. Set to ensure
        reproducibility. If using an int, the argument is passed to
        ``np.random.default_rng``.  If not provided, ``default_rng``
        is used with system-provided entropy.
    """

    def __init__(
        self,
        *,
        seed: None | int | RandomState | Generator = None,
    ) -> None:
        super().__init__(seed=seed)
        self._name = "Standardized Student's t"
        self.num_params: int = 1

    def constraints(self) -> tuple[Float64Array, Float64Array]:
        return array([[1], [-1]]), array([2.05, -500.0])

    def bounds(self, resids: ArrayLike1D) -> list[tuple[float, float]]:
        return [(2.05, 500.0)]

    def loglikelihood(
        self,
        parameters: Sequence[float] | ArrayLike1D,
        resids: ArrayLike,
        sigma2: ArrayLike,
        individual: bool = False,
    ) -> float | Float64Array1D:
        r"""Computes the log-likelihood of assuming residuals are have a
        standardized (to have unit variance) Student's t distribution,
        conditional on the variance.

        Parameters
        ----------
        parameters : ndarray
            Shape parameter of the t distribution
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma2 : ndarray
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln\Gamma\left(\frac{\nu+1}{2}\right)
            -\ln\Gamma\left(\frac{\nu}{2}\right)
            -\frac{1}{2}\ln(\pi\left(\nu-2\right)\sigma^{2})
            -\frac{\nu+1}{2}\ln(1+x^{2}/(\sigma^{2}(\nu-2)))

        where :math:`\Gamma` is the gamma function.
        """
        nu = parameters[0]
        lls = gammaln((nu + 1) / 2) - gammaln(nu / 2) - log(pi * (nu - 2)) / 2
        lls -= 0.5 * (log(sigma2))
        lls -= ((nu + 1) / 2) * (log(1 + (resids**2.0) / (sigma2 * (nu - 2))))

        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid: ArrayLike1D) -> Float64Array1D:
        """
        Construct starting values for use in optimization.

        Parameters
        ----------
        std_resid : ndarray
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : ndarray
            Array containing starting valuer for shape parameter

        Notes
        -----
        Uses relationship between kurtosis and degree of freedom parameter to
        produce a moment-based estimator for the starting values.
        """
        k = cast("float", stats.kurtosis(std_resid, fisher=False))
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)
        return array([sv])

    def _simulator(self, size: int | tuple[int, ...]) -> Float64Array:
        assert self._parameters is not None
        parameters = self._parameters
        std_dev = sqrt(parameters[0] / (parameters[0] - 2))
        return self._generator.standard_t(self._parameters[0], size=size) / std_dev

    def simulate(
        self, parameters: float | Sequence[float | int] | ArrayLike1D
    ) -> Callable[[int | tuple[int, ...]], Float64Array]:
        parameters = ensure1d(parameters, "parameters", False)
        if parameters[0] <= 2.0:
            raise ValueError("The shape parameter must be larger than 2")
        self._parameters = ensure1d(parameters, "parameters", series=False).astype(
            float
        )
        return self._simulator

    def parameter_names(self) -> list[str]:
        return ["nu"]

    def cdf(
        self,
        resids: Sequence[float] | ArrayLike1D,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> Float64Array:
        parameters = self._check_constraints(parameters)
        nu = parameters[0]
        var = nu / (nu - 2)
        return stats.t(nu, scale=1.0 / sqrt(var)).cdf(asarray(resids))

    def ppf(
        self,
        pits: float | Sequence[float] | ArrayLike1D,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> Float64Array:
        parameters = self._check_constraints(parameters)
        _pits = asarray(pits, dtype=float)
        nu = parameters[0]
        var = nu / (nu - 2)
        return stats.t(nu, scale=1.0 / sqrt(var)).ppf(_pits)

    def moment(
        self, n: int, parameters: Sequence[float] | ArrayLike1D | None = None
    ) -> float:
        if n < 0:
            return nan
        parameters = self._check_constraints(parameters)
        nu = parameters[0]
        var = nu / (nu - 2)
        return stats.t.moment(n, nu, scale=1.0 / sqrt(var))

    def partial_moment(
        self,
        n: int,
        z: float = 0.0,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> float:
        parameters = self._check_constraints(parameters)
        nu = parameters[0]
        var = nu / (nu - 2)
        scale = 1.0 / sqrt(var)
        moment = (scale**n) * self._ord_t_partial_moment(n, z / scale, nu)
        return moment

    @staticmethod
    def _ord_t_partial_moment(n: int, z: float, nu: float) -> float:
        r"""
        Partial moments for ordinary parameterization of Students t df=nu

        Parameters
        ----------
        n : int
            Order of partial moment
        z : float
            Upper bound for partial moment integral
        nu : float
            Degrees of freedom

        Returns
        -------
        float
            Calculated moment

        References
        ----------
        .. [1] Winkler et al. (1972) "The Determination of Partial Moments"
               *Management Science* Vol. 19 No. 3

        Notes
        -----
        The order n lower partial moment to z is

        .. math::

            \int_{-\infty}^{z}x^{n}f(x)dx

        See [1]_ for more details.
        """
        if n < 0 or n >= nu:
            return nan
        elif n == 0:
            moment = stats.t.cdf(z, nu)
        elif n == 1:
            c = gamma(0.5 * (nu + 1)) / (sqrt(nu * pi) * gamma(0.5 * nu))
            e = 0.5 * (nu + 1)
            moment = (0.5 * (c * nu) / (1 - e)) * ((1 + (z**2) / nu) ** (1 - e))
        else:
            t1 = (z ** (n - 1)) * (nu + z**2) * stats.t.pdf(z, nu)
            t2 = (n - 1) * nu * StudentsT._ord_t_partial_moment(n - 2, z, nu)
            moment = (1 / (n - nu)) * (t1 - t2)
        return moment


class SkewStudent(Distribution, metaclass=AbstractDocStringInheritor):
    r"""
    Standardized Skewed Student's distribution for use with ARCH models

    Parameters
    ----------
    seed : {int, Generator, RandomState}, optional
        Random number generator instance or int to use. Set to ensure
        reproducibility. If using an int, the argument is passed to
        ``np.random.default_rng``.  If not provided, ``default_rng``
        is used with system-provided entropy.

    Notes
    -----
    The Standardized Skewed Student's distribution ([1]_) takes two parameters,
    :math:`\eta` and :math:`\lambda`. :math:`\eta` controls the tail shape
    and is similar to the shape parameter in a Standardized Student's t.
    :math:`\lambda` controls the skewness. When :math:`\lambda=0` the
    distribution is identical to a standardized Student's t.

    References
    ----------
    .. [1] Hansen, B. E. (1994). Autoregressive conditional density estimation.
       *International Economic Review*, 35(3), 705-730.
       <https://www.ssc.wisc.edu/~bhansen/papers/ier_94.pdf>

    """

    def __init__(
        self,
        *,
        seed: None | int | RandomState | Generator = None,
    ) -> None:
        super().__init__(seed=seed)
        self._name = "Standardized Skew Student's t"
        self.num_params: int = 2

    def constraints(self) -> tuple[Float64Array, Float64Array]:
        return array([[1, 0], [-1, 0], [0, 1], [0, -1]]), array([2.05, -300.0, -1, -1])

    def bounds(self, resids: ArrayLike1D) -> list[tuple[float, float]]:
        return [(2.05, 300.0), (-1, 1)]

    def loglikelihood(
        self,
        parameters: Sequence[float] | ArrayLike1D,
        resids: ArrayLike,
        sigma2: ArrayLike,
        individual: bool = False,
    ) -> Float64Array1D:
        r"""
        Computes the log-likelihood of assuming residuals are have a
        standardized (to have unit variance) Skew Student's t distribution,
        conditional on the variance.

        Parameters
        ----------
        parameters : ndarray
            Shape parameter of the skew-t distribution
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma2 : ndarray
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln\left[\frac{bc}{\sigma}\left(1+\frac{1}{\eta-2}
                \left(\frac{a+bx/\sigma}
                {1+sgn(x/\sigma+a/b)\lambda}\right)^{2}\right)
                ^{-\left(\eta+1\right)/2}\right],

        where :math:`2<\eta<\infty`, and :math:`-1<\lambda<1`.
        The constants :math:`a`, :math:`b`, and :math:`c` are given by

        .. math::

            a=4\lambda c\frac{\eta-2}{\eta-1},
                \quad b^{2}=1+3\lambda^{2}-a^{2},
                \quad c=\frac{\Gamma\left(\frac{\eta+1}{2}\right)}
                {\sqrt{\pi\left(\eta-2\right)}
                \Gamma\left(\frac{\eta}{2}\right)},

        and :math:`\Gamma` is the gamma function.
        """
        parameters_arr: Float64Array1D
        parameters_arr = np.ravel(asarray(parameters, dtype=float))
        eta, lam = parameters_arr

        const_c = self.__const_c(parameters_arr)
        const_a = self.__const_a(parameters_arr)
        const_b = self.__const_b(parameters_arr)

        resids = resids / sigma2**0.5
        lls = log(const_b) + const_c - log(sigma2) / 2
        if abs(lam) >= 1.0:
            lam = sign(lam) * (1.0 - 1e-6)
        llf_resid = (
            (const_b * resids + const_a) / (1 + sign(resids + const_a / const_b) * lam)
        ) ** 2
        lls -= (eta + 1) / 2 * log(1 + llf_resid / (eta - 2))

        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid: ArrayLike1D) -> Float64Array1D:
        """
        Construct starting values for use in optimization.

        Parameters
        ----------
        std_resid : ndarray
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : ndarray
            Array containing starting valuer for shape parameter

        Notes
        -----
        Uses relationship between kurtosis and degree of freedom parameter to
        produce a moment-based estimator for the starting values.
        """
        k = cast("float", stats.kurtosis(std_resid, fisher=False))
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)
        return array([sv, 0.0])

    def _simulator(self, size: int | tuple[int, ...]) -> Float64Array:
        # No need to normalize since it is already done in parameterization
        assert self._parameters is not None
        if isinstance(self._generator, Generator):
            uniforms = self._generator.random(size=size)
        else:
            uniforms = self._generator.random_sample(size=size)
        ppf = self.ppf(uniforms, self._parameters)
        assert isinstance(ppf, ndarray)
        return ppf

    def simulate(
        self, parameters: float | Sequence[float | int] | ArrayLike1D
    ) -> Callable[[int | tuple[int, ...]], Float64Array]:
        parameters = ensure1d(parameters, "parameters", False)
        if parameters[0] <= 2.0:
            raise ValueError("The shape parameter must be larger than 2")
        if abs(parameters[1]) > 1.0:
            raise ValueError(
                "The skew parameter must be smaller than 1 in absolute value"
            )
        self._parameters = ensure1d(parameters, "parameters", series=False).astype(
            float
        )
        return self._simulator

    def parameter_names(self) -> list[str]:
        return ["eta", "lambda"]

    def __const_a(self, parameters: Float64Array1D) -> float:
        """
        Compute a constant.

        Parameters
        ----------
        parameters : ndarray
            Shape parameters of the skew-t distribution

        Returns
        -------
        a : float
            Constant used in the distribution

        """
        eta, lam = parameters
        c = self.__const_c(parameters)
        return float(4 * lam * exp(c) * (eta - 2) / (eta - 1))

    def __const_b(self, parameters: Float64Array1D) -> float:
        """
        Compute b constant.

        Parameters
        ----------
        parameters : ndarray
            Shape parameters of the skew-t distribution

        Returns
        -------
        b : float
            Constant used in the distribution
        """
        lam = float(parameters[1])
        a = self.__const_a(parameters)
        return (1 + 3 * lam**2 - a**2) ** 0.5

    @staticmethod
    def __const_c(parameters: Float64Array1D) -> float:
        """
        Compute c constant.

        Parameters
        ----------
        parameters : ndarray
            Shape parameters of the skew-t distribution

        Returns
        -------
        c : float
            Log of the constant used in loglikelihood
        """
        eta = parameters[0]
        # return gamma((eta+1)/2) / ((pi*(eta-2))**.5 * gamma(eta/2))
        return float(
            gammaln((eta + 1) / 2) - gammaln(eta / 2) - log(pi * (eta - 2)) / 2
        )

    def cdf(
        self,
        resids: Sequence[float] | ArrayLike1D,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> Float64Array:
        parameters = self._check_constraints(parameters)
        scalar = isscalar(resids)
        _resids = ensure1d(resids, "resids").astype(float)
        eta, lam = parameters

        a = self.__const_a(parameters)
        b = self.__const_b(parameters)

        var = eta / (eta - 2)
        y1 = (b * _resids + a) / (1 - lam) * sqrt(var)
        y2 = (b * _resids + a) / (1 + lam) * sqrt(var)
        tcdf = stats.t(eta).cdf
        p = (1 - lam) * tcdf(y1) * (_resids < (-a / b))
        p += (_resids >= (-a / b)) * ((1 - lam) / 2 + (1 + lam) * (tcdf(y2) - 0.5))
        if scalar:
            p = p[0]
        return p

    def ppf(
        self,
        pits: float | Sequence[float] | Float64Array | ArrayLike,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> float | Float64Array:
        parameters = self._check_constraints(parameters)
        scalar = isscalar(pits)
        if scalar:
            pits = array([pits])
        pits = asarray(pits, dtype=float)
        eta, lam = parameters

        a = self.__const_a(parameters)
        b = self.__const_b(parameters)

        cond = pits < (1 - lam) / 2

        icdf1 = stats.t.ppf(pits[cond] / (1 - lam), eta)
        icdf2 = stats.t.ppf(0.5 + (pits[~cond] - (1 - lam) / 2) / (1 + lam), eta)
        icdf = -999.99 * ones_like(pits)
        assert isinstance(icdf, ndarray)
        icdf[cond] = icdf1
        icdf[~cond] = icdf2
        icdf = icdf * (1 + sign(pits - (1 - lam) / 2) * lam) * (1 - 2 / eta) ** 0.5 - a
        icdf = icdf / float64(b)

        if scalar:
            return float(icdf[0])
        assert isinstance(icdf, ndarray)
        return icdf

    def moment(
        self, n: int, parameters: Sequence[float] | ArrayLike1D | None = None
    ) -> float:
        parameters = self._check_constraints(parameters)
        eta, lam = parameters

        if n < 0 or n >= eta:
            return nan

        a = self.__const_a(parameters)
        b = self.__const_b(parameters)

        loc = -a / b
        lscale = sqrt(1 - 2 / eta) * (1 - lam) / b
        rscale = sqrt(1 - 2 / eta) * (1 + lam) / b

        moment = 0.0
        for k in range(n + 1):  # binomial expansion around loc
            # 0->inf right partial moment for ordinary t(eta)
            r_pmom = (
                0.5
                * (gamma(0.5 * (k + 1)) * gamma(0.5 * (eta - k)) * eta ** (0.5 * k))
                / (sqrt(pi) * gamma(0.5 * eta))
            )
            l_pmom = ((-1) ** k) * r_pmom

            lhs = (1 - lam) * (lscale**k) * (loc ** (n - k)) * l_pmom
            rhs = (1 + lam) * (rscale**k) * (loc ** (n - k)) * r_pmom
            moment += comb(n, k) * (lhs + rhs)

        return moment

    def partial_moment(
        self,
        n: int,
        z: float = 0.0,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> float:
        parameters_arr = self._check_constraints(parameters)
        eta, lam = parameters_arr

        if n < 0 or n >= eta:
            return nan

        a = self.__const_a(parameters_arr)
        b = self.__const_b(parameters_arr)

        loc = -a / b
        lscale = sqrt(1 - 2 / eta) * (1 - lam) / b
        rscale = sqrt(1 - 2 / eta) * (1 + lam) / b

        moment = 0.0
        for k in range(n + 1):  # binomial expansion around loc
            lbound = min(z, loc)
            lhs = (
                (1 - lam)
                * (loc ** (n - k))
                * (lscale**k)
                * StudentsT._ord_t_partial_moment(
                    k, z=(lbound - loc) / lscale, nu=float(eta)
                )
            )

            if z > loc:
                rhs = (
                    (1 + lam)
                    * (loc ** (n - k))
                    * (rscale**k)
                    * (
                        StudentsT._ord_t_partial_moment(
                            k, z=(z - loc) / rscale, nu=float(eta)
                        )
                        - StudentsT._ord_t_partial_moment(k, z=0.0, nu=float(eta))
                    )
                )
            else:
                rhs = 0.0

            moment += comb(n, k) * (lhs + rhs)

        return moment


class GeneralizedError(Distribution, metaclass=AbstractDocStringInheritor):
    """
    Generalized Error distribution for use with ARCH models

    Parameters
    ----------
    seed : {int, Generator, RandomState}, optional
        Random number generator instance or int to use. Set to ensure
        reproducibility. If using an int, the argument is passed to
        ``np.random.default_rng``.  If not provided, ``default_rng``
        is used with system-provided entropy.
    """

    def __init__(
        self,
        *,
        seed: None | int | RandomState | Generator = None,
    ) -> None:
        super().__init__(seed=seed)
        self._name = "Generalized Error Distribution"
        self.num_params: int = 1

    def constraints(self) -> tuple[Float64Array, Float64Array]:
        return array([[1], [-1]]), array([1.01, -500.0])

    def bounds(self, resids: ArrayLike1D) -> list[tuple[float, float]]:
        return [(1.01, 500.0)]

    def loglikelihood(
        self,
        parameters: Sequence[float] | ArrayLike1D,
        resids: ArrayLike,
        sigma2: ArrayLike,
        individual: bool = False,
    ) -> Float64Array1D:
        r"""
        Computes the log-likelihood of assuming residuals are have a
        Generalized Error Distribution, conditional on the variance.

        Parameters
        ----------
        parameters : ndarray
            Shape parameter of the GED distribution
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma2 : ndarray
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln\nu-\ln c-\ln\Gamma(\frac{1}{\nu})-(1+\frac{1}{\nu})\ln2
            -\frac{1}{2}\ln\sigma^{2}
            -\frac{1}{2}\left|\frac{x}{c\sigma}\right|^{\nu}

        where :math:`\Gamma` is the gamma function and :math:`\ln c` is

        .. math::

            \ln c=\frac{1}{2}\left(\frac{-2}{\nu}\ln2+\ln\Gamma(\frac{1}{\nu})
            -\ln\Gamma(\frac{3}{\nu})\right).
        """
        nu = parameters[0]
        log_c = 0.5 * (-2 / nu * log(2) + gammaln(1 / nu) - gammaln(3 / nu))
        c = exp(log_c)
        lls = log(nu) - log_c - gammaln(1 / nu) - (1 + 1 / nu) * log(2)
        lls -= 0.5 * log(sigma2)
        lls -= 0.5 * abs(resids / (sqrt(sigma2) * c)) ** nu

        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid: ArrayLike1D) -> Float64Array1D:
        """
        Construct starting values for use in optimization.

        Parameters
        ----------
        std_resid : ndarray
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : ndarray
            Array containing starting valuer for shape parameter

        Notes
        -----
        Defaults to 1.5 which is implies heavier tails than a normal
        """
        return to_array_1d(array([1.5]))

    def _simulator(self, size: int | tuple[int, ...]) -> Float64Array:
        assert self._parameters is not None
        parameters = self._parameters
        nu = parameters[0]
        randoms = self._generator.standard_gamma(1 / nu, size) ** (1.0 / nu)
        if isinstance(self._generator, Generator):
            random_ints = self._generator.integers(0, 2, size, dtype=int64)
        else:
            random_ints = self._generator.randint(0, 2, size, dtype=int64)
        randoms *= 2.0 * random_ints - 1.0
        scale = sqrt(gamma(3.0 / nu) / gamma(1.0 / nu))

        return randoms / scale

    def simulate(
        self, parameters: float | Sequence[float | int] | ArrayLike1D
    ) -> Callable[[int | tuple[int, ...]], Float64Array]:
        parameters = ensure1d(parameters, "parameters", False)
        if parameters[0] <= 1.0:
            raise ValueError("The shape parameter must be larger than 1")
        self._parameters = ensure1d(parameters, "parameters", series=False).astype(
            float
        )
        return self._simulator

    def parameter_names(self) -> list[str]:
        return ["nu"]

    def ppf(
        self,
        pits: float | Sequence[float] | ArrayLike1D,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> Float64Array:
        parameters = self._check_constraints(parameters)
        _pits = asarray(pits, dtype=float)
        nu = parameters[0]
        var = stats.gennorm(nu).var()
        return stats.gennorm(nu, scale=1.0 / sqrt(var)).ppf(_pits)

    def cdf(
        self,
        resids: Sequence[float] | ArrayLike1D,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> Float64Array:
        parameters = self._check_constraints(parameters)
        nu = parameters[0]
        var = stats.gennorm(nu).var()
        _resids = ensure1d(resids, "resids", series=False).astype(float)
        return stats.gennorm(nu, scale=1.0 / sqrt(var)).cdf(_resids)

    def moment(
        self, n: int, parameters: Sequence[float] | ArrayLike1D | None = None
    ) -> float:
        if n < 0:
            return nan

        parameters = self._check_constraints(parameters)
        nu = parameters[0]
        var = stats.gennorm(nu).var()
        return stats.gennorm.moment(n, nu, scale=1.0 / sqrt(var))

    def partial_moment(
        self,
        n: int,
        z: float = 0.0,
        parameters: Sequence[float] | ArrayLike1D | None = None,
    ) -> float:
        parameters = self._check_constraints(parameters)
        nu = parameters[0]
        scale = 1.0 / sqrt(stats.gennorm(nu).var())
        moment = (scale**n) * self._ord_gennorm_partial_moment(n, z / scale, nu)
        return moment

    @staticmethod
    def _ord_gennorm_partial_moment(n: int, z: int, beta: float) -> float:
        r"""
        Partial moment for ordinary generalized normal parameterization.

        Parameters
        ----------
        n : int
            Order of partial moment
        z : float
            Upper bound for partial moment integral
        beta : float
            Parameter of generalized normal

        Returns
        -------
        float
            Partial moment

        Notes
        -----
        The standard parameterization follows:

        .. math::

        f(x)=\frac{\beta}{2\Gamma(\beta^{-1})}\text{exp}\left(-|x|^{\beta}\right)
        """
        if n < 0:
            return nan

        w = 0.5 * beta / gamma(1 / beta)

        # integral over (-inf, min(z,0))
        lz = abs(min(z, 0)) ** beta
        lterm = (
            w
            * ((-1) ** n)
            * (1 / beta)
            * gamma((n + 1) / beta)
            * gammaincc((n + 1) / beta, lz)
        )

        # remaining integral
        rz = max(0, z) ** beta
        rterm = w * (1 / beta) * gamma((n + 1) / beta) * gammainc((n + 1) / beta, rz)

        moment = lterm + rterm

        return moment
