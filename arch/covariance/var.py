from functools import cached_property
from typing import NamedTuple, cast
import warnings

import numpy as np
from numpy.linalg import lstsq
import pandas as pd
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat

from arch._typing import ArrayLike, Float64Array
import arch.covariance.kernel as lrcov
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
        The data to use in covariance estimation.
    lags : int, default None
        The number of lags to include in the VAR. If None, a specification
        search is used to select the order.
    method : {"aic", "hqc", "bic"}, default "aic"
        The information criteria to use in the model specification search.
    diagonal : bool, default True
        Flag indicating whether the specification search also considers
        models where the coefficient matrices on the final lags are
        diagonal. A diagonal coefficient matrix restricts all off-diagonal
        coefficients to be zero. Only used when lags is None and x has more
        than one column.
    max_lag : int, default None
        The maximum lag to use in the model specification search. If None,
        then int(nobs**(1/3)) is used.
    sample_autocov : bool, default False
        Whether to use the sample autocovariance of x or the autocovariance
        implied by the estimated VAR when computing the one-sided
        covariances. Does not affect the long-run covariance.
    kernel : {str, None}, default "bartlett".
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
        Degrees of freedom to remove when adjusting the covariance. Currently
        not used by this estimator, see Notes for the scaling applied.
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
    ``(nobs - nvar) // nvar``.

    **Kernel estimation.** The long-run covariance of the VAR residuals,
    :math:`\hat{\Omega}_\epsilon`, is estimated using the selected kernel
    applied to the residuals without centering and without a degree of
    freedom adjustment. When ``kernel`` is None, a zero-lag kernel is used
    so that :math:`\hat{\Omega}_\epsilon=\hat{\Sigma}`, which is the
    VAR-HAC estimator of den Haan & Levin. This is also the case when the
    bandwidth is 0.

    **Recoloring.** The long-run covariance of x is

    .. math::

       \hat{\Omega} = \frac{T}{T-N} \hat{D}\hat{\Omega}_\epsilon\hat{D}^\prime,
       \quad \hat{D} = \left(I_N - \sum_{i=1}^P \hat{A}_i\right)^{-1}

    where :math:`N` is the number of columns in x and :math:`T` is the number
    of VAR residuals. When the selected order
    is 0, no VAR is estimated and all returned values are those of the
    kernel estimator applied to x (demeaned when ``center`` is True)
    without the scale :math:`T/(T-N)`.

    When the VAR order is positive, the returned
    :class:`~arch.covariance.kernel.CovarianceEstimate` contains

    * ``long_run``: :math:`\hat{\Omega}`.
    * ``short_run``: :math:`\hat{\Sigma}`, the covariance of the VAR
      residuals, not the variance of x.
    * ``one_sided``: the upper-left :math:`N` by :math:`N` block of
      :math:`\frac{T}{T-N}(I-F)^{-1}\Gamma_0` where :math:`F` is the
      companion-form coefficient matrix of the VAR and :math:`\Gamma_0` is
      the covariance of the stacked vector
      :math:`[x_t^\prime, \ldots, x_{t-P+1}^\prime]^\prime`, either implied
      by the estimated VAR and :math:`\hat{\Sigma}` or, when
      ``sample_autocov`` is True, computed from the sample autocovariances
      of x.
    * ``one_sided_strict``: the upper-left :math:`N` by :math:`N` block of
      :math:`\frac{T}{T-N}F(I-F)^{-1}\Gamma_0`.

    The one-sided covariances do not use the kernel. Since ``short_run`` is
    the residual covariance, the identities in
    :class:`~arch.covariance.kernel.CovarianceEstimate` relating the
    short-run, one-sided and long-run covariances do not hold. When
    ``sample_autocov`` is False and the kernel is not used (``kernel`` is
    None or the bandwidth is 0), ``long_run`` equals
    ``one_sided + one_sided.T - (one_sided - one_sided_strict)``.

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
        self._diagonal_lags = (0,) * self._x.shape[0]
        self._method = method
        self._diagonal = diagonal
        self._max_lag = max_lag
        self._auto_lag_selection = True
        self._format_lags(lags)
        self._sample_autocov = sample_autocov
        if kernel is not None:
            kernel = _normalize_kernel_name(kernel)
        else:
            if self._bandwidth not in (0, None):
                raise ValueError("bandwidth must be None when kernel is None")
            self._bandwidth = None
            kernel = "zerolag"
        if kernel not in _KERNEL_ESTIMATORS:
            raise ValueError(_KERNEL_ERR)

        self._kernel = _KERNEL_ESTIMATORS[kernel]
        self._kernel_instance: CovarianceEstimator | None = None

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
        if (
            not np.isscalar(lags)
            or cast("float", lags) < 0
            or int(cast("float", lags)) != lags
        ):
            raise ValueError("lags must be a non-negative integer.")
        self._lags = int(cast("float", lags))
        self._diagonal_lags = self._lags
        return

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
        # Ensure at least nvar obs left over
        max_lag = min(max_lag, (nobs - nvar) // nvar)
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
        #  [Gamma0  Gamma1  Gamma2, ... ]
        #  [Gamma1' Gamma0  Gamma1, ... ]
        #  [Gamma2' Gamma1' Gamma0, ... ]

        :param nvar:
        :param nlag:
        :return:
        """
        x = self._x
        if self._center:
            x = x - x.mean(0)
        nobs = x.shape[0]
        var_cov = np.zeros((nvar * nlag, nvar * nlag))
        gamma = np.zeros((nlag, nvar, nvar))
        for i in range(nlag):
            gamma[i] = (x[i:].T @ x[: (nobs - i)]) / nobs
        for r in range(nlag):
            for c in range(nlag):
                g = gamma[np.abs(r - c)]
                if c > r:
                    g = g.T
                var_cov[r * nvar : (r + 1) * nvar, c * nvar : (c + 1) * nvar] = g
        return var_cov

    @staticmethod
    def _estimate_model_cov(
        nvar: int, nlag: int, coeffs: Float64Array, short_run: Float64Array
    ) -> Float64Array:
        sigma = np.zeros((nvar * nlag, nvar * nlag))
        sigma[:nvar, :nvar] = short_run
        multiplier = np.linalg.inv(np.eye(coeffs.size) - np.kron(coeffs, coeffs))
        vec_sigma = sigma.ravel()[:, None]
        vec_var_cov = multiplier @ vec_sigma
        var_cov = vec_var_cov.reshape((nvar * nlag, nvar * nlag)).T
        return var_cov

    def _companion_form(
        self, var_model: VARModel, short_run: Float64Array
    ) -> tuple[Float64Array, Float64Array]:
        nvar = var_model.resids.shape[1]
        nlag = var_model.var_order
        coeffs = np.zeros((nvar * nlag, nvar * nlag))
        coeffs[:nvar] = var_model.params[:, var_model.intercept :]
        for i in range(nlag - 1):
            coeffs[(i + 1) * nvar : (i + 2) * nvar, i * nvar : (i + 1) * nvar] = np.eye(
                nvar
            )
        if self._sample_autocov:
            var_cov = self._estimate_sample_cov(nvar, nlag)
        else:
            var_cov = self._estimate_model_cov(nvar, nlag, coeffs, short_run)
        return coeffs, var_cov

    @cached_property
    @Appender(CovarianceEstimator.cov.__doc__)
    def cov(self) -> CovarianceEstimate:
        common, individual = self._select_lags()
        self._order = (common, individual)
        var_mod = self._estimate_var(common, individual)
        resids = var_mod.resids
        nobs, nvar = resids.shape
        self._kernel_instance = self._kernel(
            resids,
            bandwidth=self._bandwidth,
            df_adjust=0,
            center=False,
            weights=self._x_weights,
            force_int=self._force_int,
        )
        kern_cov = self._kernel_instance.cov
        short_run = np.asarray(kern_cov.short_run)
        x_orig = self._x_orig
        columns = x_orig.columns if isinstance(x_orig, pd.DataFrame) else None
        if var_mod.var_order == 0:
            # Special case VAR(0): no recoloring and no T/(T-N) scale, see Notes
            oss = np.asarray(kern_cov.one_sided_strict)
            return CovarianceEstimate(short_run, oss, columns)
        comp_coefs, comp_var_cov = self._companion_form(var_mod, short_run)
        max_eig = np.abs(np.linalg.eigvals(comp_coefs)).max()
        if max_eig >= 1:
            raise ValueError(f"""\
The parameters of the estimated VAR model are not compatible with covariance \
stationarity, and the long-run covariance cannot be computed. The model estimated is \
a VAR({max(common, individual)}) where the final {max(0, individual-common)} lags \
have diagonal coefficient matrices. The maximum eigenvalue of the companion-form \
VAR(1) coefficient matrix is {max_eig}.""")
        coeff_sum = np.zeros((nvar, nvar))
        params = var_mod.params[:, var_mod.intercept :]
        for i in range(var_mod.var_order):
            coeff_sum += params[:, i * nvar : (i + 1) * nvar]
        d = np.linalg.inv(np.eye(nvar) - coeff_sum)
        scale = nobs / (nobs - nvar)
        # Recolor the kernel long-run covariance of the VAR residuals
        # (Andrews & Monahan 1992). With a zero bandwidth or kernel=None, the
        # kernel long run equals the residual short run.
        resid_long_run = np.asarray(kern_cov.long_run)
        long_run = scale * (d @ resid_long_run @ d.T)

        comp_nvar = comp_coefs.shape[0]
        i_minus_coefs_inv = np.linalg.inv(np.eye(comp_nvar) - comp_coefs)

        one_sided = scale * i_minus_coefs_inv @ comp_var_cov
        one_sided_strict = comp_coefs @ one_sided

        one_sided = one_sided[:nvar, :nvar]
        one_sided_strict = one_sided_strict[:nvar, :nvar]

        return CovarianceEstimate(
            short_run,
            one_sided_strict,
            columns=columns,
            long_run=long_run,
            one_sided=one_sided,
        )

    def _ensure_kernel_instantized(self) -> None:
        if self._kernel_instance is None:
            _ = self.cov

    @property
    def bandwidth_scale(self) -> float:
        self._ensure_kernel_instantized()
        assert self._kernel_instance is not None
        return self._kernel_instance.bandwidth_scale

    @property
    def kernel_const(self) -> float:
        self._ensure_kernel_instantized()
        assert self._kernel_instance is not None
        return self._kernel_instance.kernel_const

    def _weights(self) -> Float64Array:
        self._ensure_kernel_instantized()
        assert self._kernel_instance is not None
        return self._kernel_instance._weights()

    @property
    def rate(self) -> float:
        self._ensure_kernel_instantized()
        assert self._kernel_instance is not None
        return self._kernel_instance.rate
