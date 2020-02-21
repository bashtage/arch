from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
from numpy import ones, zeros
from numpy.linalg import lstsq
import pandas as pd
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat

from arch.covariance.kernel import CovarianceEstimate, CovarianceEstimator
from arch.typing import ArrayLike, NDArray


class VARModel(NamedTuple):
    resids: NDArray
    params: NDArray
    var_order: int
    intercept: bool


class PreWhitenRecoloredCovariance(CovarianceEstimator):
    def __init__(
        self,
        x: ArrayLike,
        lags: Optional[int] = None,
        method: str = "aic",
        diagonal: bool = True,
        max_lag: Optional[int] = None,
        kernel: str = "bartlett",
        bandwidth: Optional[float] = None,
        df_adjust: int = 0,
        center: bool = True,
        weights: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__(
            x, bandwidth=bandwidth, df_adjust=df_adjust, center=center, weights=weights
        )
        self._kernel = kernel
        self._lags = 0
        self._diagonal_lags = (0,) * self._x.shape[0]
        self._method = method
        self._diagonal = diagonal
        self._max_lag = max_lag
        self._auto_lag_selection = True
        self._format_lags(lags)

        # Attach for testing only
        self._ics: Dict[Tuple[int, int], float] = {}
        self._order = (0, 0)

    def _format_lags(self, lags) -> None:
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

    def _ic(self, sigma, nparam, nobs):
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
    def _fit_diagonal(x, diag_lag, lags):
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
        params = lstsq(_rhs, lhs, rcond=None)[0]
        resids0 = lhs - _rhs @ params
        sigma = resids0.T @ resids0 / nobs
        nparam = (c + full_order * nvar) * nvar
        ics: Dict[Tuple[int, int], float] = {
            (full_order, full_order): self._ic(sigma, nparam, nobs)
        }
        if not self._diagonal:
            return ics

        purged_indiv_lags = np.empty((nvar, nobs, max_lag - full_order))
        for i in range(nvar):
            single = indiv_lags[i, :, full_order:]
            params = lstsq(_rhs, single, rcond=None)[0]
            purged_indiv_lags[i] = single - _rhs @ params
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
        nobs, nvar = self._x.shape
        center = int(self._center)
        max_lag = max(full_order, diag_order)
        lhs, rhs, extra_lags = self._setup_model_data(max_lag)
        c = int(self._center)
        rhs = rhs[:, : c + full_order * nvar]
        extra_lags = extra_lags[:, :, full_order:diag_order]

        params = zeros((nvar, nvar * max_lag + center))
        resids = np.empty_like(lhs)
        ncommon = rhs.shape[1]
        for i in range(nvar):
            full_rhs = np.hstack([rhs, extra_lags[i]])
            single_params = lstsq(full_rhs, lhs[:, i], rcond=None)[0]
            params[i, :ncommon] = single_params[:ncommon]
            locs = ncommon + i + nvar * np.arange(extra_lags[i].shape[1])
            params[i, locs] = single_params[ncommon:]
            resids[:, i] = lhs[:, i] - full_rhs @ single_params
        return VARModel(resids, params, max_lag, self._center)

    @staticmethod
    def _companion_form(
        var_model: VARModel, short_run: NDArray
    ) -> Tuple[NDArray, NDArray]:
        nvar = var_model.resids.shape[1]
        nlag = var_model.var_order
        coeffs = zeros((nvar * nlag, nvar * nlag))
        coeffs[:nvar] = var_model.params[:, var_model.intercept :]
        for i in range(nlag - 1):
            coeffs[(i + 1) * nvar : (i + 2) * nvar, i * nvar : (i + 1) * nvar] = np.eye(
                nvar
            )
        sigma = zeros((nvar * nlag, nvar * nlag))
        sigma[:nvar, :nvar] = short_run
        multiplier = np.linalg.inv(np.eye(coeffs.size) - np.kron(coeffs, coeffs))
        vec_sigma = sigma.ravel()[:, None]
        vec_var_cov = multiplier @ vec_sigma
        var_cov = vec_var_cov.reshape((nvar * nlag, nvar * nlag)).T
        return coeffs, var_cov

    @property
    def cov(self) -> CovarianceEstimate:
        x = self._x
        common, individual = self._select_lags()
        self._order = (common, individual)
        var_mod = self._estimate_var(common, individual)
        resids = var_mod.resids
        nobs, nvar = resids.shape
        short_run = resids.T @ resids / nobs
        if var_mod.var_order == 0:
            # Special case VAR(0)
            # TODO: Docs should reflect different DoF adjustment
            return CovarianceEstimate(short_run, np.zeros((nvar, nvar)))
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
        # TODO: Eigenvalue check
        coeff_sum = zeros((nvar, nvar))
        params = var_mod.params[:, var_mod.intercept :]
        for i in range(var_mod.var_order):
            coeff_sum += params[:, i * nvar : (i + 1) * nvar]
        d = np.linalg.inv(np.eye(nvar) - coeff_sum)
        scale = nobs / (nobs - nvar)
        long_run = scale * (d @ short_run @ d.T)

        # TODO: Consider an alternative using sample moments like
        #  [Gamma0  Gamma1  Gamma2, ... ]
        #  [Gamma1' Gamma0  Gamma1, ... ]
        #  [Gamma2' Gamma1' Gamma0, ... ]
        comp_nvar = comp_coefs.shape[0]
        i_minus_coefs_inv = np.linalg.inv(np.eye(comp_nvar) - comp_coefs)

        one_sided = scale * i_minus_coefs_inv @ comp_var_cov
        one_sided_strict = comp_coefs @ one_sided

        one_sided = one_sided[:nvar, :nvar]
        one_sided_strict = one_sided_strict[:nvar, :nvar]
        columns = x.columns if isinstance(x, pd.DataFrame) else None

        return CovarianceEstimate(
            short_run,
            one_sided_strict,
            columns=columns,
            long_run=long_run,
            one_sided=one_sided,
        )

    def bandwidth_scale(self) -> float:
        return 1.0

    def kernel_const(self) -> float:
        return 1.0

    def _weights(self) -> NDArray:
        return ones(0)

    def rate(self) -> float:
        return 2 / 9
