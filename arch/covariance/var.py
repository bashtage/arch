from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
from numpy.linalg import lstsq
import pandas as pd
from pandas.util._decorators import Appender
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat

from arch.covariance import KERNEL_ERR, KERNEL_ESTIMATORS
from arch.covariance.kernel import (
    CovarianceEstimate,
    CovarianceEstimator,
    normalize_kernel_name,
)
from arch.typing import ArrayLike, NDArray
from arch.vendor import cached_property

__all__ = ["PreWhitenedRecolored"]


class VARModel(NamedTuple):
    resids: NDArray
    params: NDArray
    var_order: int
    intercept: bool


class PreWhitenedRecolored(CovarianceEstimator):
    """
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
        Flag indicating to consider both diagonal parameter coefficient
        matrices on lags. A diagonal coefficient matrix restricts all
        off-diagonal coefficient to be zero.
    max_lag : int, default None
        The maximum lag to use in the model specification search. If None,
        then nobs**(1/3) is used.
    sample_autocov : bool, default False
        Whether to the the same autocovariance or the theoretical
        autocovariance implied by the estimated VAR when computing
        the long-run covairance.
    kernel : {str, None}, default "bartlett".
        The name of the kernel to use. Can be any available kernel. Input
        is normalised using lower casing and any underscores or hyphens
        are removed, so that "QuadraticSpectral", "quadratic-spectral" and
        "quadratic_spectral" are all the same. Use None to prevent recoloring.
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

    See Also
    --------
    arch.covariance.kernel
        Kernel-based long-run covariance estimators

    Notes
    -----
    TODO: Add detailed notes

    Examples
    --------

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
        lags: Optional[int] = None,
        method: str = "aic",
        diagonal: bool = True,
        max_lag: Optional[int] = None,
        sample_autocov: bool = False,
        kernel: Optional[str] = "bartlett",
        bandwidth: Optional[float] = None,
        df_adjust: int = 0,
        center: bool = True,
        weights: Optional[ArrayLike] = None,
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
            kernel = normalize_kernel_name(kernel)
        else:
            if self._bandwidth not in (0, None):
                raise ValueError("bandwidth must be None when kernel is None")
            self._bandwidth = None
            kernel = "zerolag"
        if kernel not in KERNEL_ESTIMATORS:
            raise ValueError(KERNEL_ERR)

        self._kernel = KERNEL_ESTIMATORS[kernel]
        self._kernel_instance: Optional[CovarianceEstimator] = None

        # Attach for testing only
        self._ics: Dict[Tuple[int, int], float] = {}
        self._order = (0, 0)

    def _format_lags(self, lags: Optional[int]) -> None:
        """
        Check lag inputs and standard values for lags and diagonal lags
        """
        if lags is None:
            return

        self._auto_lag_selection = False
        if not np.isscalar(lags) or lags < 0 or int(lags) != lags:
            raise ValueError("lags must be a non-negative integer.")
        self._lags = int(lags)
        self._diagonal_lags = self._lags
        return

    def _ic(self, sigma: NDArray, nparam: int, nobs: int) -> float:
        _, ld = np.linalg.slogdet(sigma)
        if self._method == "aic":
            return ld + 2 * nparam / nobs
        elif self._method == "hqc":
            return ld + 2 * np.log(np.log(nobs)) * nparam / nobs
        else:  # bic
            return ld + np.log(nobs) * nparam / nobs

    def _setup_model_data(self, max_lag: int) -> Tuple[NDArray, NDArray, NDArray]:
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
    def _fit_diagonal(x: NDArray, diag_lag: int, lags: NDArray) -> NDArray:
        nvar = x.shape[1]
        for i in range(nvar):
            lhs = lags[i, :, :diag_lag]
            x[:, i] -= lhs @ lstsq(lhs, x[:, i], rcond=None)[0]
        return x

    def _ic_from_vars(
        self,
        lhs: NDArray,
        rhs: NDArray,
        indiv_lags: NDArray,
        full_order: int,
        max_lag: int,
    ) -> Dict[Tuple[int, int], float]:
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
        ics: Dict[Tuple[int, int], float] = {
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

    def _select_lags(self) -> Tuple[int, int]:
        """Select lags if needed"""
        if not self._auto_lag_selection:
            return self._lags, self._diagonal_lags

        nobs, nvar = self._x.shape
        # Use rule-of-thumb is not provided
        max_lag = int(nobs ** (1 / 3)) if self._max_lag is None else self._max_lag
        # Ensure at least nvar obs left over
        max_lag = min(max_lag, (nobs - nvar) // nvar)
        if max_lag == 0 and self._max_lag is None:
            import warnings

            warnings.warn(
                "The maximum number of lags is 0 since the number of time series "
                f"observations {nobs} is small relative to the number of time "
                f"series {nvar}.",
                RuntimeWarning,
            )
        self._max_lag = max_lag
        lhs, rhs, indiv_lags = self._setup_model_data(max_lag)

        for full_order in range(max_lag + 1):
            _ics = self._ic_from_vars(lhs, rhs, indiv_lags, full_order, max_lag)
            self._ics.update(_ics)
        ic = np.array([crit for crit in self._ics.values()])
        models = [key for key in self._ics.keys()]
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

    def _estimate_sample_cov(self, nvar: int, nlag: int) -> NDArray:
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
        nvar: int, nlag: int, coeffs: NDArray, short_run: NDArray
    ) -> NDArray:
        sigma = np.zeros((nvar * nlag, nvar * nlag))
        sigma[:nvar, :nvar] = short_run
        multiplier = np.linalg.inv(np.eye(coeffs.size) - np.kron(coeffs, coeffs))
        vec_sigma = sigma.ravel()[:, None]
        vec_var_cov = multiplier @ vec_sigma
        var_cov = vec_var_cov.reshape((nvar * nlag, nvar * nlag)).T
        return var_cov

    def _companion_form(
        self, var_model: VARModel, short_run: NDArray
    ) -> Tuple[NDArray, NDArray]:
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
        short_run = kern_cov.short_run
        x_orig = self._x_orig
        columns = x_orig.columns if isinstance(x_orig, pd.DataFrame) else None
        if var_mod.var_order == 0:
            # Special case VAR(0)
            # TODO: Docs should reflect different DoF adjustment
            oss = kern_cov.one_sided_strict
            return CovarianceEstimate(short_run, oss, columns)
        comp_coefs, comp_var_cov = self._companion_form(var_mod, short_run)
        max_eig = np.abs(np.linalg.eigvals(comp_coefs)).max()
        if max_eig >= 1:
            raise ValueError(
                f"""\
The parameters of the estimated VAR model are not compatible with covariance \
stationarity, and the long-run covariance cannot be computed. The model estimated is \
a VAR({max(common, individual)}) where the final {max(0, individual-common)} lags \
have diagonal coefficient matrices. The maximum eigenvalue of the companion-form \
VAR(1) coefficient matrix is {max_eig}."""
            )
        coeff_sum = np.zeros((nvar, nvar))
        params = var_mod.params[:, var_mod.intercept :]
        for i in range(var_mod.var_order):
            coeff_sum += params[:, i * nvar : (i + 1) * nvar]
        d = np.linalg.inv(np.eye(nvar) - coeff_sum)
        scale = nobs / (nobs - nvar)
        long_run = scale * (d @ short_run @ d.T)

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
            # Workaround to avoid linting noise
            getattr(self, "cov")

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

    def _weights(self) -> NDArray:
        self._ensure_kernel_instantized()
        assert self._kernel_instance is not None
        return self._kernel_instance._weights()

    @property
    def rate(self) -> float:
        self._ensure_kernel_instantized()
        assert self._kernel_instance is not None
        return self._kernel_instance.rate
