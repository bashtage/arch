from functools import cached_property
from numbers import Integral, Real
from typing import NamedTuple, cast
import warnings

import numpy as np
from numpy.linalg import lstsq
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat

from arch._typing import ArrayLike, Float64Array
from arch.covariance import kernel as lrcov
from arch.covariance.kernel import CovarianceEstimate, CovarianceEstimator
from arch.vendor._decorators import Appender

__all__ = ["PreWhitenedRecolored"]

# Kernel lookup keyed by normalized name. Built locally rather than imported
# from arch.unitroot to avoid a circular import. ZeroLag is added so that
# kernel=None (VAR-HAC) can reuse the kernel machinery.
_KERNEL_ESTIMATORS: dict[str, type[CovarianceEstimator]] = {
    name.lower(): getattr(lrcov, name) for name in lrcov.KERNELS
}
_KERNEL_ESTIMATORS["zerolag"] = lrcov.ZeroLag
_KNOWN_KERNELS = "\n".join(sorted(_KERNEL_ESTIMATORS))
_KERNEL_ERR = (
    f"kernel is not a known kernel estimator. Must be one of:\n {_KNOWN_KERNELS}"
)


def _is_non_negative_integer(value: object) -> bool:
    """Test for a real, finite, non-negative value that is a whole number"""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    if isinstance(value, Integral):
        return cast("int", value) >= 0
    number = cast("float", value)
    return bool(np.isfinite(number)) and number >= 0 and int(number) == number


def _normalize_kernel_name(name: str) -> str:
    """
    Normalize a kernel name by removing - and _ and converting to lower case.

    Matches the normalization used in arch.unitroot._shared._check_kernel.
    """
    return name.replace("-", "").replace("_", "").lower()


class VARModel(NamedTuple):
    resids: Float64Array
    params: Float64Array
    var_order: int
    intercept: bool


class PreWhitenedRecolored(CovarianceEstimator):
    r"""
    VAR-HAC and Pre-Whitened-Recolored Long-run covariance estimation.

    Andrews & Monahan [1]_ PWRC and DenHaan-Levin's VAR-HAC [2]_ covariance
    estimators.

    Parameters
    ----------
    x : array_like
        The data to use in covariance estimation. Must not contain NaN or
        infinite values.
    lags : int, default None
        The number of lags to include in the VAR. If None, a specification
        search is used to select the order. Must be small enough that each
        equation of the VAR has more observations than parameters, see Notes.
    method : {"aic", "hqc", "bic"}, default "aic"
        The information criteria to use in the model specification search.
        Input is not case sensitive.
    diagonal : bool, default True
        Flag indicating whether the specification search also considers
        models where the coefficient matrices on the final lags are
        diagonal. A diagonal coefficient matrix restricts all off-diagonal
        coefficients to be zero. Only used when lags is None and x has more
        than one column.
    max_lag : int, default None
        The maximum lag to use in the model specification search. If None,
        then int(nobs**(1/3)) is used. The value used is limited, see Notes.
    sample_autocov : bool, default False
        Whether to use the sample autocovariance of x or the autocovariance
        implied by the estimated VAR when computing the short-run and
        one-sided covariances. Does not affect the long-run covariance.
    kernel : {str, None}, default "bartlett"
        The name of the kernel to use. Can be any available kernel. Input
        is normalised using lower casing and any underscores or hyphens
        are removed, so that "QuadraticSpectral", "quadratic-spectral" and
        "quadratic_spectral" are all the same. Use None to compute the
        VAR-HAC estimator, which recolors the residual covariance without
        applying a kernel to the residuals.
    bandwidth : float, default None
        The kernel's bandwidth.  If None, optimal bandwidth is estimated
        from the VAR residuals. Must be None or 0 when kernel is None.
    df_adjust : int, default 0
        Degrees of freedom to remove when adjusting the covariance. Uses the
        number of observations in x minus df_adjust when dividing
        inner-products, see Notes.
    center : bool, default True
        A flag indicating whether x should be demeaned before estimating the
        covariance.
    weights : array_like, default None
        An array of weights used to combine when estimating optimal bandwidth.
        If not provided, a vector of 1s is used. Must have nvar elements.
    force_int : bool, default False
        Force bandwidth to be an integer.

    See Also
    --------
    arch.covariance.kernel
        Kernel-based long-run covariance estimators

    Notes
    -----
    The estimator is computed in three steps.

    **Prewhitening.** A VAR is estimated by least squares, equation by
    equation,

    .. math::

       x_t = c + A_1 x_{t-1} + \ldots + A_P x_{t-P} + \epsilon_t

    where the constant :math:`c` is included only when ``center`` is True.
    When ``lags`` is provided, a VAR(``lags``) with unrestricted coefficient
    matrices is used. Otherwise the order is selected by minimizing

    .. math::

       IC = \ln|\hat{\Sigma}| + \lambda \frac{k}{T}

    where :math:`\hat{\Sigma}=T^{-1}\sum_t \hat{\epsilon}_t\hat{\epsilon}_t^\prime`,
    :math:`k` is the total number of estimated parameters, :math:`T` is the
    number of observations used in the regressions, which is common to all
    candidate models, and :math:`\lambda` is 2 (``"aic"``),
    :math:`2\ln\ln T` (``"hqc"``) or :math:`\ln T` (``"bic"``). Models with
    0, 1, ..., ``max_lag`` lags are considered. When ``diagonal`` is True
    and x has more than one column, the search also considers models with
    unrestricted coefficient matrices on the first :math:`p` lags and
    diagonal coefficient matrices on lags :math:`p+1, \ldots, q` for
    :math:`p < q \leq` ``max_lag``, so that each series only uses its own
    values at the additional lags. ``max_lag`` is limited to
    ``(nobs - nvar) // nvar`` and to the largest order for which each equation
    of the VAR has more observations than parameters. ``lags`` must also
    satisfy this restriction.

    **Kernel estimation.** The long-run covariance of the VAR residuals,
    :math:`\hat{\Omega}_\epsilon`, is estimated using the selected kernel
    applied to the residuals without centering. When ``kernel`` is None, a
    zero-lag kernel is used so that
    :math:`\hat{\Omega}_\epsilon=\hat{\Sigma}`, which is the VAR-HAC
    estimator of den Haan & Levin. This is also the case when the bandwidth is
    0. The products of the residuals are divided by :math:`T-` ``df_adjust``,
    where :math:`T` is the number of observations in x, rather than by the
    number of residuals, which is smaller when the order is positive. This
    matches the ``sandwich`` package in R: the estimator is identical to
    ``sandwich::meatHAC(prewhite=P, adjust=FALSE)`` when ``df_adjust`` is 0
    and to ``sandwich::meatHAC(prewhite=P, adjust=TRUE)`` when ``df_adjust`` is
    the number of columns in x, if the kernel weights are the same, x has been
    demeaned and ``center`` is False, since the VAR in R has no intercept.

    **Recoloring.** The long-run covariance of x is

    .. math::

       \hat{\Omega} = \hat{D}\hat{\Omega}_\epsilon\hat{D}^\prime,
       \quad \hat{D} = \left(I_N - \sum_{i=1}^P \hat{A}_i\right)^{-1}

    where :math:`N` is the number of columns in x. When the selected order
    is 0, no VAR is estimated and all returned values are those of the
    kernel estimator applied to x (demeaned when ``center`` is True).

    When the VAR order is positive, the returned
    :class:`~arch.covariance.kernel.CovarianceEstimate` contains

    * ``long_run``: :math:`\hat{\Omega}`.
    * ``short_run``: the upper-left :math:`N` by :math:`N` block of
      :math:`\Gamma_0`, the covariance of the stacked vector
      :math:`[x_t^\prime, \ldots, x_{t-P+1}^\prime]^\prime`. By default
      :math:`\Gamma_0` is implied by the estimated VAR and the covariance of
      its residuals, :math:`\hat{\Sigma}`, so that ``short_run`` is the
      variance of x implied by the VAR. When ``sample_autocov`` is True,
      :math:`\Gamma_0` is computed from the sample autocovariances of x and
      ``short_run`` is the sample variance of x. It is not the covariance of
      the VAR residuals, which is available as :attr:`resid_cov`.
    * ``one_sided_strict``: the upper-left :math:`N` by :math:`N` block of
      :math:`F(I-F)^{-1}\Gamma_0` where :math:`F` is the companion-form
      coefficient matrix of the VAR. This is
      :math:`\sum_{j\geq 1}\Gamma_j` where :math:`\Gamma_j=E[x_tx_{t-j}^\prime]`.
    * ``one_sided``: ``short_run`` plus ``one_sided_strict``.

    The one-sided covariances do not use the kernel. The identities in
    :class:`~arch.covariance.kernel.CovarianceEstimate` that define the
    one-sided covariances hold, so that
    ``one_sided = short_run + one_sided_strict``. The long-run covariance
    satisfies ``long_run = short_run + one_sided_strict + one_sided_strict.T``
    when the autocovariances are those of the VAR and the kernel is not used,
    that is when ``sample_autocov`` is False and ``kernel`` is None or the
    bandwidth is 0. Otherwise ``long_run`` differs from this sum.

    **Inspecting the VAR.** The prewhitening step determines the estimate, so
    its results are available. :attr:`order` is the order of the VAR, which is
    the only way to learn which order was chosen when ``lags`` is None.
    :attr:`resid` are the residuals of the VAR, which are the series that the
    kernel is applied to and can be tested to check that the VAR removed the
    serial correlation in x. :attr:`resid_cov` is their covariance, which is
    the matrix that is recolored into the long-run covariance of x. None of
    these require the VAR to be covariance stationary, so they are available
    even when :attr:`cov` raises an error because it is not.

    Examples
    --------
    >>> import numpy as np
    >>> from arch.covariance.var import PreWhitenedRecolored
    >>> rs = np.random.default_rng(0)
    >>> e = rs.standard_normal((1001, 2))
    >>> x = e[1:] + 0.5 * e[:-1]

    Prewhiten with a VAR(1) and use a Bartlett kernel with a bandwidth of 5

    >>> pwrc = PreWhitenedRecolored(x, lags=1, kernel="bartlett", bandwidth=5.0)
    >>> lrcov = pwrc.cov.long_run

    The order, residuals and residual covariance of the VAR are available

    >>> pwrc.order
    (1, 1)
    >>> resid = pwrc.resid
    >>> resid_cov = pwrc.resid_cov

    VAR-HAC with the VAR order selected using BIC

    >>> var_hac = PreWhitenedRecolored(x, kernel=None, method="bic")
    >>> lrcov = var_hac.cov.long_run

    References
    ----------
    .. [1] Andrews, D. W., & Monahan, J. C. (1992). An improved
       heteroskedasticity and autocorrelation consistent covariance matrix
       estimator. Econometrica: Journal of the Econometric Society, 953-966.
    .. [2] Haan, W. J. D., & Levin, A. T. (2000). Robust covariance matrix
       estimation with data-dependent VAR prewhitening order (No. 255).
       National Bureau of Economic Research.
    """

    def __init__(
        self,
        x: ArrayLike,
        *,
        lags: int | None = None,
        method: str = "aic",
        diagonal: bool = True,
        max_lag: int | None = None,
        sample_autocov: bool = False,
        kernel: str | None = "bartlett",
        bandwidth: float | None = None,
        df_adjust: int = 0,
        center: bool = True,
        weights: ArrayLike | None = None,
        force_int: bool = False,
    ) -> None:
        if not np.all(np.isfinite(np.asarray(x, dtype=float))):
            raise ValueError("x must not contain NaN or infinite values.")
        super().__init__(
            x,
            bandwidth=bandwidth,
            df_adjust=df_adjust,
            center=center,
            weights=weights,
            force_int=force_int,
        )
        self._kernel_name = kernel
        self._lags = 0
        self._diagonal_lags = 0
        if not isinstance(method, str) or method.lower() not in ("aic", "hqc", "bic"):
            raise ValueError("method must be one of 'aic', 'hqc' or 'bic'")
        self._method = method.lower()
        self._diagonal = diagonal
        if max_lag is not None and not _is_non_negative_integer(max_lag):
            raise ValueError("max_lag must be a non-negative integer.")
        self._max_lag = None if max_lag is None else int(max_lag)
        self._auto_lag_selection = True
        self._format_lags(lags)
        self._sample_autocov = sample_autocov
        if kernel is not None:
            kernel = _normalize_kernel_name(kernel)
        else:
            if self._bandwidth not in (0, None):
                raise ValueError("bandwidth must be None or 0 when kernel is None")
            self._bandwidth = 0.0
            self._auto_bandwidth = False
            kernel = "zerolag"
        if kernel not in _KERNEL_ESTIMATORS:
            raise ValueError(_KERNEL_ERR)

        self._kernel = _KERNEL_ESTIMATORS[kernel]
        self._kernel_instance: CovarianceEstimator | None = None
        self._var_model: VARModel | None = None

        # Attach for testing only
        self._ics: dict[tuple[int, int], float] = {}
        self._order = (0, 0)

    def _format_lags(self, lags: int | None) -> None:
        """
        Check lag inputs and standard values for lags and diagonal lags
        """
        if lags is None:
            return

        self._auto_lag_selection = False
        if not _is_non_negative_integer(lags):
            raise ValueError("lags must be a non-negative integer.")
        self._lags = int(cast("float", lags))
        largest = self._largest_lag()
        if self._lags > largest:
            nobs, nvar = self._x.shape
            raise ValueError(
                f"lags must be at most {largest} when x has {nobs} observations "
                f"and {nvar} series so that each equation of the VAR has more "
                "observations than parameters."
            )
        self._diagonal_lags = self._lags
        return

    def _largest_lag(self) -> int:
        """
        Largest VAR order that can be estimated.

        A VAR(P) uses nobs - P observations and has P * nvar + center
        parameters in each equation, so that a VAR is only estimable if
        nobs - P > P * nvar + center.
        """
        nobs, nvar = self._x.shape
        return max(0, (nobs - int(self._center) - 1) // (nvar + 1))

    def _ic(self, sigma: Float64Array, nparam: int, nobs: int) -> float:
        _, ld = np.linalg.slogdet(sigma)
        if self._method == "aic":
            return ld + 2 * nparam / nobs
        elif self._method == "hqc":
            return ld + 2 * np.log(np.log(nobs)) * nparam / nobs
        else:  # bic
            return ld + np.log(nobs) * nparam / nobs

    def _setup_model_data(
        self, max_lag: int
    ) -> tuple[Float64Array, Float64Array, Float64Array]:
        nobs, nvar = self._x.shape
        lhs = np.empty((nobs - max_lag, nvar))
        rhs = np.empty((nobs - max_lag, nvar * max_lag))
        rhs_locs = np.arange(0, nvar * max_lag, nvar)
        indiv_lags = np.empty((nvar, nobs - max_lag, max_lag))
        for i in range(nvar):
            lags, lead = lagmat(self._x[:, i], max_lag, trim="both", original="sep")
            lhs[:, [i]] = lead
            indiv_lags[i] = lags
            rhs[:, rhs_locs + i] = lags
        if self._center:
            rhs = add_constant(rhs, True)
        return lhs, rhs, indiv_lags

    @staticmethod
    def _fit_diagonal(
        x: Float64Array, diag_lag: int, lags: Float64Array
    ) -> Float64Array:
        nvar = x.shape[1]
        for i in range(nvar):
            lhs = lags[i, :, :diag_lag]
            x[:, i] -= lhs @ lstsq(lhs, x[:, i], rcond=None)[0]
        return x

    def _ic_from_vars(
        self,
        lhs: Float64Array,
        rhs: Float64Array,
        indiv_lags: Float64Array,
        full_order: int,
        max_lag: int,
    ) -> dict[tuple[int, int], float]:
        c = int(self._center)
        nobs, nvar = lhs.shape
        _rhs = rhs[:, : (c + full_order * nvar)]
        if _rhs.shape[1] > 0 and lhs.shape[1] > 0:
            params = lstsq(_rhs, lhs, rcond=None)[0]
            resids0 = lhs - _rhs @ params
        else:
            # Branch is a workaround of NumPy 1.15
            # TODO: Remove after NumPy 1.15 dropped
            resids0 = lhs
        sigma = resids0.T @ resids0 / nobs
        nparam = (c + full_order * nvar) * nvar
        ics: dict[tuple[int, int], float] = {
            (full_order, full_order): self._ic(sigma, nparam, nobs)
        }
        if not self._diagonal or self._x.shape[1] == 1:
            return ics

        purged_indiv_lags = np.empty((nvar, nobs, max_lag - full_order))
        for i in range(nvar):
            single = indiv_lags[i, :, full_order:]
            if single.shape[1] > 0 and _rhs.shape[1] > 0:
                params = lstsq(_rhs, single, rcond=None)[0]
                purged_indiv_lags[i] = single - _rhs @ params
            else:
                # Branch is a workaround of NumPy 1.15
                # TODO: Remove after NumPy 1.15 dropped
                purged_indiv_lags[i] = single

        for diag_lag in range(1, max_lag - full_order + 1):
            resids = self._fit_diagonal(resids0.copy(), diag_lag, purged_indiv_lags)
            sigma = resids.T @ resids / nobs
            nparam = (c + full_order * nvar) * nvar + nvar * diag_lag
            ics[(full_order, full_order + diag_lag)] = self._ic(sigma, nparam, nobs)
        return ics

    def _select_lags(self) -> tuple[int, int]:
        """Select lags if needed"""
        if not self._auto_lag_selection:
            return self._lags, self._diagonal_lags

        nobs, nvar = self._x.shape
        # Use rule-of-thumb is not provided
        max_lag = int(nobs ** (1 / 3)) if self._max_lag is None else self._max_lag
        # Ensure at least nvar obs left over and that the VAR can be estimated
        max_lag = min(max_lag, (nobs - nvar) // nvar, self._largest_lag())
        if max_lag == 0 and self._max_lag is None:
            warnings.warn(
                "The maximum number of lags is 0 since the number of time series "
                f"observations {nobs} is small relative to the number of time "
                f"series {nvar}.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._max_lag = max_lag
        lhs, rhs, indiv_lags = self._setup_model_data(max_lag)

        for full_order in range(max_lag + 1):
            _ics = self._ic_from_vars(lhs, rhs, indiv_lags, full_order, max_lag)
            self._ics.update(_ics)
        ic = np.array(list(self._ics.values()))
        models = list(self._ics.keys())
        return models[ic.argmin()]

    def _estimate_var(self, full_order: int, diag_order: int) -> VARModel:
        nvar = self._x.shape[1]
        center = int(self._center)
        max_lag = max(full_order, diag_order)
        lhs, rhs, extra_lags = self._setup_model_data(max_lag)
        c = int(self._center)
        rhs = rhs[:, : c + full_order * nvar]
        extra_lags = extra_lags[:, :, full_order:diag_order]

        params = np.zeros((nvar, nvar * max_lag + center))
        resids = np.empty_like(lhs)
        ncommon = rhs.shape[1]
        for i in range(nvar):
            full_rhs = np.hstack([rhs, extra_lags[i]])
            if full_rhs.shape[1] > 0:
                single_params = lstsq(full_rhs, lhs[:, i], rcond=None)[0]
                params[i, :ncommon] = single_params[:ncommon]
                locs = ncommon + i + nvar * np.arange(extra_lags[i].shape[1])
                params[i, locs] = single_params[ncommon:]
                resids[:, i] = lhs[:, i] - full_rhs @ single_params
            else:
                # Branch is a workaround of NumPy 1.15
                # TODO: Remove after NumPy 1.15 dropped
                resids[:, i] = lhs[:, i]

        return VARModel(resids, params, max_lag, self._center)

    def _estimate_sample_cov(self, nvar: int, nlag: int) -> Float64Array:
        """
        Sample covariance of the stacked vector [x_t', ..., x_{t-nlag+1}']'.

        With Gamma_j = E[x_t x_{t-j}'], block (r, c) of the covariance is
        E[x_{t-r} x_{t-c}'] = Gamma_{c-r}, so that the blocks above the
        diagonal are Gamma_1, Gamma_2, ... and those below are their transposes

            [Gamma0  Gamma1  Gamma2, ... ]
            [Gamma1' Gamma0  Gamma1, ... ]
            [Gamma2' Gamma1' Gamma0, ... ]

        Parameters
        ----------
        nvar : int
            The number of series in x.
        nlag : int
            The number of lags stacked.

        Returns
        -------
        ndarray
            The nvar * nlag by nvar * nlag sample covariance.
        """
        x = self._x
        if self._center:
            x = x - x.mean(0)
        nobs = x.shape[0]
        var_cov = np.zeros((nvar * nlag, nvar * nlag))
        gamma = np.zeros((nlag, nvar, nvar))
        for i in range(nlag):
            gamma[i] = (x[i:].T @ x[: (nobs - i)]) / self._df
        for r in range(nlag):
            for c in range(nlag):
                g = gamma[np.abs(r - c)]
                if r > c:
                    g = g.T
                var_cov[r * nvar : (r + 1) * nvar, c * nvar : (c + 1) * nvar] = g
        return var_cov

    @staticmethod
    def _estimate_model_cov(
        nvar: int, nlag: int, coeffs: Float64Array, short_run: Float64Array
    ) -> Float64Array:
        """
        Covariance of the stacked vector implied by the VAR in companion form

        Solves Gamma = F Gamma F' + Sigma where F is the companion-form
        coefficient matrix and Sigma has the residual covariance in its
        upper-left block and 0 elsewhere.
        """
        sigma = np.zeros((nvar * nlag, nvar * nlag))
        sigma[:nvar, :nvar] = short_run
        var_cov = solve_discrete_lyapunov(coeffs, sigma)
        return (var_cov + var_cov.T) / 2

    @staticmethod
    def _companion_coefs(var_model: VARModel) -> Float64Array:
        """Coefficient matrix of the VAR(1) in companion form"""
        nvar = var_model.resids.shape[1]
        nlag = var_model.var_order
        coeffs = np.zeros((nvar * nlag, nvar * nlag))
        coeffs[:nvar] = var_model.params[:, var_model.intercept :]
        for i in range(nlag - 1):
            coeffs[(i + 1) * nvar : (i + 2) * nvar, i * nvar : (i + 1) * nvar] = np.eye(
                nvar
            )
        return coeffs

    def _setup(self) -> tuple[VARModel, CovarianceEstimator]:
        """
        Select the VAR order, estimate the VAR and set up the residual kernel.

        The kernel estimator is applied to the VAR residuals, so its bandwidth
        is selected using the residuals and not using x. Only the VAR is
        needed, and so the bandwidth is available even if the VAR is not
        covariance stationary.
        """
        if self._var_model is None or self._kernel_instance is None:
            common, individual = self._select_lags()
            self._order = (common, individual)
            self._var_model = self._estimate_var(common, individual)
            self._kernel_instance = self._kernel(
                self._var_model.resids,
                bandwidth=self._bandwidth,
                df_adjust=0,
                center=False,
                weights=self._x_weights,
                force_int=self._force_int,
            )
        return self._var_model, self._kernel_instance

    @cached_property
    @Appender(CovarianceEstimator.cov.__doc__)
    def cov(self) -> CovarianceEstimate:
        var_mod, kernel_instance = self._setup()
        common, individual = self._order
        resids = var_mod.resids
        nobs, nvar = resids.shape
        kern_cov = kernel_instance.cov
        # The kernel divides by the number of residuals. Divide by the number
        # of observations in x less df_adjust, as the other estimators do.
        scale = nobs / self._df
        x_orig = self._x_orig
        columns = x_orig.columns if isinstance(x_orig, pd.DataFrame) else None
        if var_mod.var_order == 0:
            # Special case VAR(0): no recoloring
            short_run = scale * np.asarray(kern_cov.short_run)
            oss = scale * np.asarray(kern_cov.one_sided_strict)
            return CovarianceEstimate(short_run, oss, columns)
        comp_coefs = self._companion_coefs(var_mod)
        max_eig = np.abs(np.linalg.eigvals(comp_coefs)).max()
        if max_eig >= 1:
            raise ValueError(f"""\
The parameters of the estimated VAR model are not compatible with covariance \
stationarity, and the long-run covariance cannot be computed. The model estimated is \
a VAR({max(common, individual)}) where the final {max(0, individual - common)} lags \
have diagonal coefficient matrices. The maximum eigenvalue of the companion-form \
VAR(1) coefficient matrix is {max_eig}.""")
        coeff_sum = np.zeros((nvar, nvar))
        params = var_mod.params[:, var_mod.intercept :]
        for i in range(var_mod.var_order):
            coeff_sum += params[:, i * nvar : (i + 1) * nvar]
        d = np.linalg.inv(np.eye(nvar) - coeff_sum)
        # Recolor the kernel long-run covariance of the VAR residuals
        # (Andrews & Monahan 1992). With a zero bandwidth or kernel=None, the
        # kernel long run equals the residual covariance.
        resid_long_run = scale * np.asarray(kern_cov.long_run)
        long_run = d @ resid_long_run @ d.T

        # Covariance of the stacked [x_t, ..., x_{t-P+1}], which has Gamma_0
        # of x as its upper-left block
        if self._sample_autocov:
            comp_var_cov = self._estimate_sample_cov(nvar, var_mod.var_order)
        else:
            resid_cov = scale * np.asarray(kern_cov.short_run)
            comp_var_cov = self._estimate_model_cov(
                nvar, var_mod.var_order, comp_coefs, resid_cov
            )
        # F (I - F)^-1 Gamma_0 = F Gamma_0 + F^2 Gamma_0 + ..., and its
        # upper-left block is the sum of the autocovariances of x at lags 1, 2,
        # ... The one-sided covariance is then short_run + one_sided_strict.
        comp_nvar = comp_coefs.shape[0]
        i_minus_coefs = np.eye(comp_nvar) - comp_coefs
        comp_oss = comp_coefs @ np.linalg.solve(i_minus_coefs, comp_var_cov)
        short_run = comp_var_cov[:nvar, :nvar]
        one_sided_strict = comp_oss[:nvar, :nvar]

        return CovarianceEstimate(
            short_run, one_sided_strict, columns=columns, long_run=long_run
        )

    @property
    def bandwidth(self) -> float:
        """
        The bandwidth used by the kernel estimator.

        Returns
        -------
        float
            The user-provided bandwidth or the estimated optimal bandwidth.
            The kernel is applied to the residuals of the VAR, and so the
            estimated bandwidth is the optimal bandwidth for the residuals
            and not for x. It is 0 when kernel is None.
        """
        return self._setup()[1].bandwidth

    @cached_property
    def opt_bandwidth(self) -> float:
        """
        Estimate optimal bandwidth.

        Returns
        -------
        float
            The estimated optimal bandwidth of the residuals of the VAR.
            This is the bandwidth used when bandwidth is not provided.
        """
        return self._setup()[1].opt_bandwidth

    @property
    def bandwidth_scale(self) -> float:
        return self._setup()[1].bandwidth_scale

    @property
    def kernel_const(self) -> float:
        return self._setup()[1].kernel_const

    def _weights(self) -> Float64Array:
        return self._setup()[1]._weights()

    @property
    def rate(self) -> float:
        return self._setup()[1].rate

    def _wrap_matrix(self, value: Float64Array) -> Float64Array | pd.DataFrame:
        """Label a covariance matrix when x is a DataFrame or Series"""
        x_orig = self._x_orig
        if isinstance(x_orig, pd.DataFrame):
            return pd.DataFrame(value, columns=x_orig.columns, index=x_orig.columns)
        return value

    @property
    def order(self) -> tuple[int, int]:
        """
        The order of the VAR used to prewhiten x.

        Returns
        -------
        tuple of int
            A tuple ``(p, q)`` where the VAR has unrestricted coefficient
            matrices on lags 1, ..., ``p`` and, when ``q > p``, diagonal
            coefficient matrices on lags ``p + 1``, ..., ``q`` so that each
            series only depends on its own values at these lags. ``p`` equals
            ``q`` when the lags are provided, when ``diagonal`` is False or
            when x has a single column. ``q`` is the number of observations
            lost to the VAR.

        Notes
        -----
        When ``lags`` is None the order is selected using an information
        criterion, and this is the only way to find the selected order. An
        order of ``(0, 0)`` means that no VAR is used, so that the estimates
        are those of the kernel estimator applied to x. Larger orders mean that
        the kernel is applied to residuals with less serial dependence than x,
        which is the purpose of prewhitening. The order does not require the
        VAR to be covariance stationary.
        """
        self._setup()
        return self._order

    @property
    def resid(self) -> Float64Array | pd.DataFrame:
        """
        The residuals of the VAR used to prewhiten x.

        Returns
        -------
        ndarray or DataFrame
            The ``nobs - q`` by ``nvar`` residuals, where ``q`` is the second
            element of :attr:`order`, so that the first ``q`` observations of x
            are lost. A DataFrame with the dates or index of the final
            observations of x and the same columns is returned if x is a
            DataFrame or a Series, and an ndarray otherwise.

        Notes
        -----
        The kernel estimator is applied to these residuals and not to x, with
        the autocovariances computed without demeaning them. They can be used
        to check that the prewhitening was adequate. If the residuals are still
        serially correlated, a larger ``lags`` or ``max_lag`` should be used.
        When the order is 0 and ``center`` is True the residuals are x less its
        mean, and when the order is 0 and ``center`` is False they are x. If
        ``center`` is True the VAR includes a constant, so the residuals have
        mean zero. The residuals do not require the VAR to be covariance
        stationary.
        """
        var_mod, _ = self._setup()
        resids = var_mod.resids.copy()
        x_orig = self._x_orig
        if isinstance(x_orig, pd.DataFrame):
            return pd.DataFrame(
                resids, index=x_orig.index[var_mod.var_order :], columns=x_orig.columns
            )
        return resids

    @property
    def resid_cov(self) -> Float64Array | pd.DataFrame:
        r"""
        The covariance of the residuals of the VAR used to prewhiten x.

        Returns
        -------
        ndarray or DataFrame
            The ``nvar`` by ``nvar`` covariance of the residuals,
            :math:`\hat{\Sigma}`. A DataFrame with the columns of x as its
            index and columns is returned if x is a DataFrame or a Series,
            and an ndarray otherwise.

        Notes
        -----
        This is the innovation covariance of the VAR and is not the
        short-run covariance in :attr:`cov`, which is the covariance of x
        implied by the VAR. For a VAR(1) with coefficient matrix :math:`A` and
        ``sample_autocov`` False, ``cov.short_run`` equals
        ``A @ cov.short_run @ A.T + resid_cov``.

        Like all covariances computed by this estimator, the sum of the
        products of the residuals is divided by ``nobs - df_adjust`` where
        ``nobs`` is the number of observations in x, and not by the number of
        residuals. This is the covariance that is recolored to estimate the
        long-run covariance of x. Without a kernel, that is when ``kernel`` is
        None or the bandwidth is 0, ``cov.long_run`` equals
        :math:`\hat{D}\hat{\Sigma}\hat{D}^\prime`, which is the VAR-HAC
        estimator. When a kernel is used, the long-run covariance of the
        residuals is :math:`\hat{\Omega}_\epsilon` instead of
        :math:`\hat{\Sigma}`. The residual covariance does not require the VAR
        to be covariance stationary.
        """
        var_mod, _ = self._setup()
        resids = var_mod.resids
        return self._wrap_matrix(resids.T @ resids / self._df)
