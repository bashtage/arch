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


def _junction_constraints(
    tau: float, small_order: int = 4, use_log: bool = True
) -> np.ndarray:
    """
    Linear constraints that join the two p-value polynomials at ``tau``.

    The parameter vector is (a_0, ..., a_{k-1}, b_0, ..., b_3), where k is
    small_order and the small p polynomial a is in log|stat| if use_log is True and
    in stat otherwise. The large p polynomial b is in stat. The two rows require
    that the fitted values and the first derivatives with respect to the statistic
    agree at tau, which must be negative if use_log is True.
    """
    k = np.arange(small_order)
    x = np.log(abs(tau)) if use_log else tau
    value_a = x**k
    slope_a = np.zeros(small_order)
    slope_a[1:] = k[1:] * x ** (k[1:] - 1)
    if use_log:
        # d log|stat| / d stat = 1 / stat
        slope_a /= tau
    k = np.arange(4)
    value_b = tau**k
    slope_b = np.zeros(4)
    slope_b[1:] = k[1:] * tau ** (k[1:] - 1)
    return np.vstack([np.r_[value_a, -value_b], np.r_[slope_a, -slope_b]])


def _tail_end(coefs: np.ndarray, x0: float, direction: float) -> float | None:
    """
    Find where a polynomial stops falling on the ray x0 + direction * t, t >= 0.

    Parameters
    ----------
    coefs : ndarray
        The coefficients of the polynomial, lowest power first.
    x0 : float
        The start of the ray.
    direction : {1.0, -1.0}
        The direction of the ray.

    Returns
    -------
    float or None
        None if the polynomial rises at x0. Otherwise the first turning point of
        the polynomial on the ray, or direction * inf if there is none, in which
        case the polynomial falls to -inf along the whole ray.
    """
    deriv = np.polynomial.polynomial.polyder(coefs)
    if direction * np.polynomial.polynomial.polyval(x0, deriv) > 0:
        return None
    roots = np.polynomial.polynomial.polyroots(deriv) if deriv.shape[0] > 1 else []
    ahead = [
        float(root.real)
        for root in np.atleast_1d(roots)
        if abs(root.imag) <= 1e-10 * max(1.0, abs(root))
        and direction * (root.real - x0) > 0
    ]
    if not ahead:
        return direction * np.inf
    return min(ahead, key=lambda root: direction * (root - x0))


def _monotone_bounds(
    small_p: np.ndarray,
    large_p: np.ndarray,
    tau: float,
    use_log: bool,
    max_tail_step: float | None,
) -> tuple[float, float] | None:
    """
    Check that both pieces are increasing in the statistic on their domain.

    The small p polynomial must be increasing in the statistic on (-inf, tau] and
    the large p polynomial on [tau, inf). If a polynomial turns, the p-value is set
    to 0 below (or 1 above) its turning point, which is allowed if the p-value
    steps by no more than max_tail_step there.

    Returns
    -------
    tuple[float, float] or None
        tau_min and tau_max, where the p-value is set to 0 and 1, or None if the
        pieces are not monotone. Infinite bounds mean that the p-value tends to 0
        and 1 without a step.
    """
    # The small p piece is in u = log|stat|, which increases as the statistic
    # falls, or in the statistic itself
    if use_log:
        end = _tail_end(small_p, float(np.log(abs(tau))), 1.0)
    else:
        end = _tail_end(small_p, tau, -1.0)
    # The large p polynomial rises with the statistic when its negative falls
    tau_max = _tail_end(-large_p, tau, 1.0)
    if end is None or tau_max is None:
        return None
    if np.isfinite(end):
        if max_tail_step is None:
            return None
        step = stats.norm.cdf(np.polynomial.polynomial.polyval(end, small_p))
        if step > max_tail_step:
            return None
    # A turning point beyond log|stat| = 709.8 is never reached by a finite stat
    with np.errstate(over="ignore"):
        tau_min = -np.exp(end) if use_log else end
    if np.isfinite(tau_max):
        if max_tail_step is None:
            return None
        step = stats.norm.sf(np.polynomial.polynomial.polyval(tau_max, large_p))
        if step > max_tail_step:
            return None
    return float(tau_min), float(tau_max)


def _round_toward(value: float, target: float, significant_digits: int) -> float:
    """Round to significant_digits, toward target if the rounding changes value"""
    if not np.isfinite(value):
        return float(value)
    ulp = 10.0 ** (np.floor(np.log10(abs(value))) - (significant_digits - 1))
    rounded = np.ceil(value / ulp) if target > value else np.floor(value / ulp)
    return float(f"{rounded * ulp:.{significant_digits - 1}e}")


def fit_smooth_pval_model(
    quantiles: pd.DataFrame,
    tail_cutoff: float = 0.15,
    tail_tolerance: float = 0.001,
    body_tolerance: float = 0.03,
    junction_range: tuple[float, float] = (0.02, 0.95),
    significant_digits: int = 10,
    small_order: int = 4,
    use_log: bool = True,
    max_tail_step: float | None = None,
) -> PvalueResult:
    """
    Fit p-value models that join smoothly at tau_star.

    The p-value is Phi(poly(log|stat|)) (or Phi(poly(stat)) if use_log is False)
    if stat <= tau_star and Phi(poly(stat)) otherwise, where the second polynomial
    is cubic. The two polynomials are fit jointly, using the points at or below
    tau_star for the first and the points at or above tau_star for the second,
    subject to the constraints that the two polynomials have the same value and
    the same first derivative at tau_star. The fitted p-value is then continuous
    and has a continuous first derivative. tau_star is chosen among the average
    simulated quantiles in `junction_range` to minimize the weighted squared error
    of the model that is used. Only junctions where both pieces are monotone are
    considered, so that the p-value is a monotone function of the statistic on the
    real line.

    Parameters
    ----------
    quantiles : DataFrame
        The simulated quantiles, indexed by the probability. Columns are
        independent simulations of the same quantiles.
    tail_cutoff : float, default 0.15
        Probabilities up to this value are weighted using `tail_tolerance`.
    tail_tolerance : float, default 0.001
        The relative error in p that is tolerated when the probability is no
        larger than `tail_cutoff`. A tighter tolerance gives the point more
        weight.
    body_tolerance : float, default 0.03
        The tolerance for all other probabilities. It is relative to p if p <= 0.5
        and relative to 1 - p, with a floor of 0.05, otherwise.
    junction_range : tuple[float, float], default (0.02, 0.95)
        The smallest and largest probability considered for tau_star.
    significant_digits : int, default 10
        The number of significant digits of the returned coefficients. The
        digits must be large enough that rounding does not make the p-value
        discontinuous.
    small_order : int, default 4
        The number of coefficients of the small p polynomial, 3 for a quadratic
        and 4 for a cubic.
    use_log : bool, default True
        Whether the small p polynomial is in log|stat|. If False, it is in stat.
    max_tail_step : float, optional
        If None, both polynomials must be monotone on their whole domain, so that
        the p-value tends to 0 and 1 in the tails. Otherwise a polynomial may turn
        in its tail outside the simulated quantiles, if the p-value is no larger
        than max_tail_step (or no smaller than 1 - max_tail_step) at the turning
        point. The p-value is then set to 0 below (or 1 above) the turning point,
        which is a step of at most max_tail_step.

    Returns
    -------
    PvalueResult
        The coefficients and the junction. tau_min and tau_max are the points
        below and above which the p-value is 0 and 1. They are -inf and inf when
        both tails are monotone and tend to 0 and 1.
    """
    if small_order not in (3, 4):
        raise ValueError("small_order must be 3 or 4")
    quantiles = quantiles.sort_index(ascending=False)
    probs = quantiles.index.to_numpy(dtype=float)
    data = np.asarray(quantiles, dtype=float)
    stats_mean = data.mean(1)
    lhs = stats.norm.ppf(probs)
    if np.any(np.diff(stats_mean) >= 0):
        raise ValueError("The average quantiles must increase with the probability")

    # Weights are the inverse squared tolerance for the error in the p-value,
    # which is phi(lhs) times the error in lhs
    rel_tol = np.where(probs <= tail_cutoff, tail_tolerance, body_tolerance)
    scale = np.minimum(probs, np.maximum(1 - probs, 0.05))
    sqrt_w = stats.norm.pdf(lhs) / (rel_tol * scale)

    if use_log:
        rhs_small = np.log(np.abs(stats_mean))[:, None] ** np.arange(small_order)
    else:
        rhs_small = stats_mean[:, None] ** np.arange(small_order)
    rhs_lin = stats_mean[:, None] ** np.arange(4)

    n = probs.shape[0]
    nparams = small_order + 4
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for i in np.where((probs >= junction_range[0]) & (probs <= junction_range[1]))[0]:
        tau = round(float(stats_mean[i]), 5)
        if use_log and tau >= 0:
            continue
        # The large p piece uses rows 0..i and the small p piece rows i..n-1, so the
        # junction point is in both
        n_large = i + 1
        design = np.zeros((n_large + n - i, nparams))
        design[:n_large, small_order:] = rhs_lin[:n_large]
        design[n_large:, :small_order] = rhs_small[i:]
        sw = np.r_[sqrt_w[:n_large], sqrt_w[i:]]
        design = design * sw[:, None]
        y = np.r_[lhs[:n_large], lhs[i:]] * sw
        # Scale the columns, since the powers span many orders of magnitude
        col_scale = np.linalg.norm(design, axis=0)
        constraints = _junction_constraints(tau, small_order, use_log) / col_scale
        null_basis = np.linalg.svd(constraints)[2][2:].T
        coefs = np.linalg.lstsq((design / col_scale) @ null_basis, y, rcond=None)[0]
        params = (null_basis @ coefs) / col_scale
        small_p, large_p = params[:small_order], params[small_order:]
        bounds = _monotone_bounds(small_p, large_p, tau, use_log, max_tail_step)
        # The p-value can only be set to 0 or 1 outside of the simulated quantiles
        if bounds is None or bounds[0] >= stats_mean[-1] or bounds[1] <= stats_mean[0]:
            continue
        fitted = np.where(np.arange(n) >= i, rhs_small @ small_p, rhs_lin @ large_p)
        loss = float((((lhs - fitted) * sqrt_w) ** 2).sum())
        if best is None or loss < best[0]:
            best = (loss, tau, small_p, large_p)
    if best is None:
        raise RuntimeError("No junction gives a monotone p-value")
    _, tau, small_p, large_p = best

    def sig_round(values: np.ndarray) -> list[float]:
        return [float(f"{val:.{significant_digits - 1}e}") for val in values]

    small = sig_round(small_p)
    large = sig_round(large_p)
    # Rounding moves the polynomials apart at tau by about 1e-9 in the normal
    # quantile scale. Move the constant of the large p polynomial up to the first
    # rounded value at which the large p polynomial is not below the small p
    # polynomial, so that rounding never makes the p-value step down at tau.
    x_tau = np.log(abs(tau)) if use_log else tau
    target = np.polyval(small[::-1], x_tau)
    target -= np.polyval(np.r_[0.0, large[1:]][::-1], tau)
    ulp = 10.0 ** (np.floor(np.log10(abs(target))) - (significant_digits - 1))
    large[0] = float(
        f"{np.ceil((target + 1e-12) / ulp) * ulp:.{significant_digits - 1}e}"
    )
    gap = np.polyval(large[::-1], tau) - np.polyval(small[::-1], x_tau)
    jump = np.abs(
        _junction_constraints(tau, small_order, use_log) @ np.r_[small, large]
    )
    if not (0 <= gap < 1e-8 and jump[1] < 1e-8):
        raise RuntimeError("Rounding the coefficients broke the continuity")
    # The bounds of the rounded polynomials, rounded toward tau so that the
    # p-value is monotone between them
    bounds = _monotone_bounds(
        np.array(small), np.array(large), tau, use_log, max_tail_step
    )
    if bounds is None:
        raise RuntimeError("Rounding the coefficients broke the monotonicity")
    tau_min = _round_toward(bounds[0], tau, significant_digits)
    tau_max = _round_toward(bounds[1], tau, significant_digits)
    return PvalueResult(large, small, tau_max, tau, tau_min)
