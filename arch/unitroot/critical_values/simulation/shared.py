from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS


def format_dict(d: dict[Any, Any]) -> str:
    return (
        str(d)
        .replace(" ", "")
        .replace("],", "],\n")
        .replace(":", ":\n")
        .replace("},", "},\n")
    )


class PvalueResult(NamedTuple):
    large_p: list[float]
    small_p: list[float]
    tau_max: float
    tau_star: float
    tau_min: float


def estimate_cv_regression(
    results: pd.DataFrame, critical_values: Sequence[float]
) -> dict[float, list[float]]:
    """
    Parameters
    ----------
    results : DataFrame
        A dataframe with rows contaoning the quantiles and columns containign the
        number of observations
    critical_values : Sequence[float]
        The critical values to use
    """
    # For percentiles 1, 5 and 10, regress on a constant, and powers of 1/T
    out = {}
    quantiles = np.asarray(results.index)
    tau = np.array(results.columns).reshape((1, -1)).T
    rhs = (1.0 / tau) ** np.arange(4)
    for cv in critical_values:
        loc = np.argmin(np.abs(100 * quantiles - cv))
        lhs = np.squeeze(np.asarray(results.iloc[loc]))
        res = OLS(lhs, rhs).fit()
        params = res.params.copy()
        params[res.pvalues > 0.05] = 0.0
        out[cv] = [round(val, 5) for val in params]
    return out


def fit_pval_model(
    quantiles: pd.Series | pd.DataFrame,
    small_order: int = 3,
    use_log: bool = False,
    drop_insignif: bool = True,
) -> PvalueResult:
    if small_order not in (3, 4):
        raise ValueError("Small order must be 3 or 4")
    quantiles = quantiles.sort_index(ascending=False)
    percentiles = quantiles.index.to_numpy()
    lhs = stats.norm.ppf(percentiles)
    data = np.asarray(quantiles)
    avg_test_stats = data.mean(1)
    avg_test_std = data.std(1)
    avg_test_stats = avg_test_stats[:, None]

    rhs = avg_test_stats ** np.arange(4)
    rhs_large = rhs
    rhs_log = np.log(np.abs(avg_test_stats)) ** np.arange(4)
    lhs_large = lhs
    res_large = WLS(lhs_large, rhs, weights=1.0 / avg_test_std).fit()
    temp = res_large.params.copy()
    if drop_insignif:
        temp[res_large.pvalues > 0.05] = 0.0
    large_p = temp

    # Compute tau_max, by finding the func maximum
    p = res_large.params
    poly_roots = np.roots(np.array([3, 2, 1.0]) * p[:0:-1])
    if np.isreal(poly_roots[0]):
        tau_max = float(np.squeeze(np.real(np.max(poly_roots))))
    else:
        tau_max = np.inf

    # Small p regression using only p<=15%
    cutoff = np.where(percentiles <= 0.150)[0]
    lhs_small = lhs[cutoff]
    if use_log:
        avg_test_stats = np.log(np.abs(avg_test_stats[cutoff]))
        avg_test_std = np.log(np.abs(data[cutoff])).std(1)
        assert np.all(np.isfinite(avg_test_std))
        rhs = avg_test_stats ** np.arange(small_order)
    else:
        avg_test_stats = avg_test_stats[cutoff]
        avg_test_std = avg_test_std[cutoff]
        rhs = avg_test_stats ** np.arange(small_order)

    res_small = WLS(lhs_small, rhs, weights=1.0 / avg_test_std).fit()
    temp = res_small.params
    if drop_insignif:
        temp[res_small.pvalues > 0.05] = 0.0
    small_p = temp

    # Compute tau star
    err_large = lhs_large - rhs_large.dot(large_p)
    params = small_p.copy()
    if small_order == 3:
        # Missing 1 parameter here, replace with 0
        params = np.append(params, 0.0)
    if use_log:
        pred_small = rhs_log.dot(params)
    else:
        pred_small = rhs_large.dot(params)
    err_small = lhs_large - pred_small
    # Find the location that minimizes the total absolute error
    m = lhs_large.shape[0]
    abs_err = np.zeros((m, 1))
    for j in range(m):
        abs_err[j] = np.abs(err_large[:j]).sum() + np.abs(err_small[j:]).sum()
    loc = np.argmin(abs_err)
    tau_star = rhs_large[loc, 1]
    if use_log:
        assert tau_star < 0
    # Compute tau min
    tau_min = -params[1] / (2 * params[2])
    if use_log:
        assert small_order == 4
        assert params[2] * params[3] < 0
        tau_min = -np.inf
    large_p = [round(val, 5) for val in large_p]
    small_p = [round(val, 5) for val in small_p]
    tau_max = round(tau_max, 5)
    tau_star = round(tau_star, 5)
    tau_min = round(tau_min, 5)
    return PvalueResult(large_p, small_p, tau_max, tau_star, tau_min)
