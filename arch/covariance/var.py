from typing import Dict, NamedTuple, Optional, Sequence, Tuple, List

import numpy as np
from numpy import column_stack, ones, zeros
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
        diagonal_lags: Optional[int] = None,
        method: str = "aic",
        max_lag: Optional[int] = None,
        diagonal: bool = True,
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
        self._lags = lags
        self._diagonal_lags = diagonal_lags
        self._method = method
        self._diagonal = diagonal
        self._max_lag = max_lag

    def _ic_single(
        self,
        idx: int,
        resid: NDArray,
        regressors: NDArray,
        lags: NDArray,
        lag: int,
        nparam: int,
    ) -> Tuple[int, NDArray]:
        add_lags = lags[:, lag:]
        params = np.linalg.lstsq(regressors, add_lags, rcond=None)[0]
        add_lags_resid = add_lags - regressors @ params
        curr_resid = resid[:, [idx]].copy()
        nobs = resid.shape[0]
        ic = np.full(add_lags.shape[1] + 1, np.inf)
        best_resids = resid
        for i in range(add_lags.shape[1] + 1):
            if i > 0:
                params = np.linalg.lstsq(add_lags_resid[:, :i], curr_resid, rcond=None)
                new_resid = curr_resid - add_lags_resid[:, :i] @ params[0]
                resid[:, [idx]] = new_resid
            sigma = resid.T @ resid / nobs
            _, ld = np.linalg.slogdet(sigma)
            if self._method == "aic":
                ic[i] = ld + 2 * (nparam + i) / nobs
            elif self._method == "hqc":
                ic[i] = ld + np.log(np.log(nobs)) * (nparam + i) / nobs
            else:  # bic
                ic[i] = ld + np.log(nobs) * (nparam + i) / nobs
            if ic[i] == ic.min():
                best_resids = resid.copy()
        return int(np.argmin(ic)), best_resids

    def _select_lags(self) -> Tuple[int, Tuple[int, ...]]:
        nobs, nvar = self._x.shape
        max_lag = int(nobs ** (1 / 3)) if self._max_lag is None else self._max_lag
        # Ensure at least nvar obs left over
        max_lag = min(max_lag, (nobs - nvar) // nvar)
        if max_lag == 0:
            import warnings

            warnings.warn(
                "The maximum number of lags is 0 since the number of time series "
                f"observations {nobs} is small relative to the number of time "
                f"series {nvar}.",
                RuntimeWarning,
            )

        lhs_data = []
        rhs_data:List[List[NDArray]] = [[] for _ in range(max_lag)]
        indiv_lags = []
        for i in range(nvar):
            lags, lead = lagmat(self._x[:, i], max_lag, trim="both", original="sep")
            lhs_data.append(lead)
            indiv_lags.append(lags)
            for k in range(max_lag):
                rhs_data[k].append(lags[:, [k]])

        lhs = column_stack(lhs_data)
        rhs = column_stack([column_stack(r) for r in rhs_data])

        if self._center:
            rhs = add_constant(rhs, True)
        c = int(self._center)
        lhs_obs = lhs.shape[0]
        ics: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        for i in range(max_lag + 1):
            indiv_lag_len = []
            x = rhs[:, : (i * nvar) + c]
            nparam = 0
            resid = lhs
            if i > 0 or self._center:
                params = lstsq(x, lhs, rcond=None)[0]
                resid = lhs - x @ params
                nparam = params.size
            for idx in range(nvar):
                lag_len, resid = self._ic_single(
                    idx, resid, x, indiv_lags[idx], i, nparam
                )
                indiv_lag_len.append(lag_len + i)

            sigma = resid.T @ resid / lhs_obs
            _, ld = np.linalg.slogdet(sigma)
            if self._method == "aic":
                ic = ld + 2 * nparam / lhs_obs
            elif self._method == "hqc":
                ic = ld + np.log(np.log(lhs_obs)) * nparam / lhs_obs
            else:  # bic
                ic = ld + np.log(lhs_obs) * nparam / lhs_obs
            ics[(i, tuple(indiv_lag_len))] = ic
        ic = np.array([crit for crit in ics.values()])
        models = [key for key in ics.keys()]
        return models[ic.argmin()]

    def _estimate_var(self, common: int, individual: Sequence[int]) -> VARModel:
        nobs, nvar = self._x.shape
        center = int(self._center)
        max_lag = max(common, max(individual))
        lhs = np.empty((nobs - max_lag, nvar))
        extra_lags = []
        rhs = np.empty((nobs - max_lag, nvar * common + center))
        if self._center:
            rhs[:, 0] = 1
        for i in range(nvar):
            lags, lead = lagmat(self._x[:, i], max_lag, trim="both", original="sep")
            lhs[:, i : i + 1] = lead
            extra_lags.append(lags[:, common : individual[i]])
            for k in range(common):
                rhs[:, center + i + nvar * k] = lags[:, k]
        params = zeros((nvar, nvar * max_lag + center))
        resids = np.empty_like(lhs)
        ncommon = rhs.shape[1]
        for i in range(nvar):
            full_rhs = np.hstack([rhs, extra_lags[i]])

            single_params = np.linalg.lstsq(full_rhs, lhs[:, i], rcond=None)[0]
            params[i, :ncommon] = single_params[:ncommon]
            locs = ncommon + nvar * np.arange(extra_lags[1].shape[1])
            params[i, locs] = single_params[ncommon:]
            resids[:, i] = lhs[:, i] - full_rhs @ single_params
        return VARModel(resids, params, max_lag, self._center)

    def _setup_lags(self) -> Tuple[int, Tuple[int, ...]]:
        nvar = self._x.shape[1]
        common = 0
        indiv = (0,) * nvar
        if self._lags is None and self._diagonal_lags is None:
            return self._select_lags()
        if self._lags is not None:
            common = self._lags
        if self._diagonal_lags is not None:
            indiv = (self._diagonal_lags,) * nvar
        return common, indiv

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
        return coeffs, sigma

    def cov(self) -> CovarianceEstimate:
        x = self._x
        common, individual = self._select_lags()
        var_mod = self._estimate_var(common, individual)
        resids = var_mod.resids
        nobs, nvar = resids.shape
        short_run = resids.T @ resids / nobs
        coeff_sum = zeros((nvar, nvar))
        params = var_mod.params[:, var_mod.intercept :]
        for i in range(var_mod.var_order):
            coeff_sum += params[:, i * nvar : (i + 1) * nvar]
        d = np.linalg.inv(np.eye(nvar) - coeff_sum)
        scale = nobs / (nobs - nvar)
        long_run = scale * (d @ short_run @ d)
        comp_coefs, comp_sigma = self._companion_form(var_mod, short_run)
        comp_nvar = comp_coefs.shape[0]
        i_minus_coefs_inv = np.linalg.inv(np.eye(comp_nvar) - comp_coefs)
        one_sided = scale * i_minus_coefs_inv @ comp_sigma
        one_sided = one_sided[:nvar, :nvar]
        one_sided_strict = scale * comp_coefs @ i_minus_coefs_inv @ comp_sigma
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
