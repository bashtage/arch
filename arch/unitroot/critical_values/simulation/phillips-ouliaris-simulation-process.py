from collections import defaultdict
import glob
from itertools import product
import os
from typing import List, NamedTuple

from black import FileMode, TargetVersion, format_file_contents
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from statsmodels.regression.linear_model import OLS, WLS

from phillips_ouliaris import FILE_TYPES, ROOT, TRENDS
from shared import format_dict

META = {"z_a": "negative", "z_t": "negative", "p_u": "positive", "p_z": "positive"}
CRITICAL_VALUES = (1, 5, 10)
PLOT = False
WINS = defaultdict(lambda: 0)
# 1. Load data
# 2. Compute critical values


class PvalueResult(NamedTuple):
    large_p: List[float]
    small_p: List[float]
    tau_max: float
    tau_star: float
    tau_min: float


def xval(lhs, rhs, log=True, folds=5):
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    pcg = np.random.PCG64(849756746597530743027509)
    gen = np.random.Generator(pcg)
    nobs = lhs.shape[0]
    idx = gen.permutation(nobs)
    predictions = np.empty((nobs, 6))
    for fold in range(folds):
        right = int(fold * nobs / folds)
        left = int((fold + 1) * nobs / folds)
        locs = idx[np.r_[np.arange(0, right), np.arange(left, nobs)]]
        end = nobs if fold == folds - 1 else left
        pred_loc = idx[np.arange(right, end)]
        pred_rhs = rhs[pred_loc]
        sm = OLS(lhs[locs], rhs[locs, :3]).fit()
        predictions[pred_loc, 0] = pred_rhs[:, :3] @ sm.params
        lg = OLS(lhs[locs], rhs[locs]).fit()
        predictions[pred_loc, 1] = pred_rhs @ lg.params
        if log and np.all(np.sign(lhs) == np.sign(lhs)[0]):
            log_lhs = np.log(np.abs(lhs))
            sgn = np.sign(lhs[0])
            sm_log = OLS(log_lhs[locs], rhs[locs, :3]).fit()
            sigma2 = (sm_log.resid ** 2).mean()
            predictions[pred_loc, 2] = sgn * np.exp(pred_rhs[:, :3] @ sm_log.params)
            predictions[pred_loc, 3] = sgn * np.exp(
                pred_rhs[:, :3] @ sm_log.params + sigma2 / 2
            )

            lg_log = OLS(log_lhs[locs], rhs[locs]).fit()
            sigma2 = (lg_log.resid ** 2).mean()
            predictions[pred_loc, 4] = sgn * np.exp(pred_rhs @ lg_log.params)
            predictions[pred_loc, 5] = sgn * np.exp(
                pred_rhs @ lg_log.params + sigma2 / 2
            )
    errors = lhs[:, None] - predictions
    best = np.argmin(errors.var(0))
    WINS[best] += 1


def estimate_cv_regression(results, statistic):
    # For percentiles 1, 5 and 10, regress on a constant, and powers of 1/T
    out = {}
    quantiles = np.asarray(results.index)
    tau = np.array(results.columns).reshape((1, -1)).T
    rhs = (1.0 / tau) ** np.arange(4)
    for cv in CRITICAL_VALUES:
        if META[statistic] == "negative":
            loc = np.argmin(np.abs(100 * quantiles - cv))
        else:
            loc = np.argmin(np.abs(100 * quantiles - (100 - cv)))
        lhs = np.squeeze(np.asarray(results.iloc[loc]))
        xval(lhs, rhs)
        res = OLS(lhs, rhs).fit()
        params = res.params.copy()
        if res.pvalues[-1] > 0.05:
            params[-1] = 0.00
        out[cv] = [round(val, 5) for val in params]
    return out, tau.min()


def fit_pval_model(quantiles):
    percentiles = quantiles.index.to_numpy()
    lhs = stats.norm.ppf(percentiles)
    data = np.asarray(quantiles)
    avg_test_stats = data.mean(1)
    avg_test_std = data.std(1)
    avg_test_stats = avg_test_stats[:, None]

    rhs = avg_test_stats ** np.arange(4)
    rhs_large = rhs
    lhs_large = lhs
    res_large = WLS(lhs_large, rhs, weights=1.0 / avg_test_std).fit()
    large_p = res_large.params.tolist()

    # Compute tau_max, by finding the func maximum
    p = res_large.params
    poly_roots = np.roots(np.array([3, 2, 1.0]) * p[:0:-1])
    if np.isreal(poly_roots[0]):
        tau_max = float(np.squeeze(np.real(np.max(poly_roots))))
    else:
        tau_max = np.inf

    # Small p regression using only p<=15%
    cutoff = np.where(percentiles <= 0.150)[0]
    avg_test_stats = avg_test_stats[cutoff]
    avg_test_std = avg_test_std[cutoff]
    lhs_small = lhs[cutoff]
    rhs = avg_test_stats ** np.arange(3)
    res_small = WLS(lhs_small, rhs, weights=1.0 / avg_test_std).fit()
    small_p = res_small.params.tolist()

    # Compute tau star
    err_large = lhs_large - rhs_large.dot(res_large.params)
    # Missing 1 parameter here, replace with 0
    params = np.append(res_small.params, 0.0)
    err_small = lhs_large - rhs_large.dot(params)
    # Find the location that minimizes the total absolute error
    m = lhs_large.shape[0]
    abs_err = np.zeros((m, 1))
    for j in range(m):
        abs_err[j] = np.abs(err_large[:j]).sum() + np.abs(err_small[j:]).sum()
    loc = np.argmin(abs_err)
    tau_star = rhs_large[loc, 1]
    # Compute tau min
    tau_min = -params[1] / (2 * params[2])
    large_p = [round(val, 5) for val in large_p]
    small_p = [round(val, 5) for val in small_p]
    tau_max = round(tau_max, 5)
    tau_star = round(tau_star, 5)
    tau_min = round(tau_min, 5)
    return PvalueResult(large_p, small_p, tau_max, tau_star, tau_min)


results = defaultdict(list)
num_files = {}
for file_type in FILE_TYPES:
    for trend in TRENDS:
        pattern = f"*-statistic-{file_type}-trend-{trend}-*.hdf"
        result_files = glob.glob(os.path.join(ROOT, pattern))
        num_files[(file_type, trend)] = len(result_files)
        for rf in result_files:
            temp = pd.DataFrame(pd.read_hdf(rf, "results"))
            statistics = temp.columns.levels[2]
            for stat in statistics:
                single = temp.loc[:, pd.IndexSlice[:, :, stat]]
                single.columns = single.columns.droplevel(2)
                results[(stat, trend)].append(single)

assert len(num_files) > 0
# assert all([nf == num_files[0] for nf in num_files])
nsimulation = {k: 250_000 * v for k, v in num_files.items()}

joined = defaultdict(list)
for key in results:
    temp = results[key]
    stoch_trends = temp[0].columns.levels[1]
    for st in stoch_trends:
        for df in temp:
            single = df.loc[:, pd.IndexSlice[:, st]]
            single.columns = single.columns.droplevel(1)
            single = single.dropna(axis=1, how="all")
            joined[key + (st,)].append(single)

final = {key: pd.concat(joined[key], axis=1) for key in joined}
stat_names = {"p_z": "Pz", "p_u": "Pu", "z_t": "Zt", "z_a": "Za"}
cv_params = {}
cv_tau_min = {}
for key in final:
    final_key = (stat_names[key[0]],) + key[1:]
    cv_params[final_key], cv_tau_min[final_key] = estimate_cv_regression(
        final[key], key[0]
    )

print("Best methods")
for key in sorted(WINS):
    print(f"{key}: {WINS[key]}")

report = []
for key in nsimulation:
    s = key[0].upper()
    t = key[1]
    n = nsimulation[key]
    report.append(f"{s}-type statistics with trend {t} based on {n:,} simulations")

counts = "\n".join(report)

STATISTICS = set(key[0] for key in final)
TRENDS = set(key[1] for key in final)
NSTOCHASTICS = set(key[-1] for key in final)
quantiles_d = defaultdict(list)
pval_data = {}
for key in product(STATISTICS, TRENDS, NSTOCHASTICS):
    pval_data[key] = final[key].loc[:, 2000]
    temp = final[key].loc[:, 2000].mean(1)
    temp.name = key[-1]
    quantiles_d[key[:-1]].append(temp)
quantiles = {}
for key in quantiles_d:
    quantiles[key] = pd.concat(quantiles_d[key], axis=1)


plt.rc("figure", figsize=(16, 8))
sns.set_style("darkgrid")
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
for key in quantiles:
    temp = quantiles[key]
    y = temp.index.to_numpy()[:, None]
    x = temp.to_numpy()
    stat = key[0]
    if stat in ("z_t", "z_a"):
        x = -1 * x
    if stat in ("p_u", "p_z"):
        y = 1 - y
    fig, ax = plt.subplots(1, 1)
    plt.plot(x, y)
    plt.title(key)
    pdf.savefig(fig)
    if stat in ("p_u", "p_z"):
        fig, ax = plt.subplots(1, 1)
        plt.plot(np.log(x), y)
        plt.title(f"Log {key[0]}, {key[1]}")
        pdf.savefig(fig)
pdf.close()

pval_results = {}
pval_large_p = {}
pval_small_p = {}
pval_tau_star = {}
pval_tau_min = {}
pval_tau_max = {}
for key in pval_data:
    temp = pval_data[key].copy()
    if key[0] in ("p_z", "p_u"):
        temp.index = 1 - temp.index
        temp = -1 * temp
    temp = temp.sort_index()
    res = fit_pval_model(temp)
    out_key = (stat_names[key[0]],) + key[1:]
    pval_results[out_key] = res
    pval_large_p[out_key] = res.large_p
    pval_small_p[out_key] = res.small_p
    pval_tau_min[out_key] = res.tau_min
    pval_tau_max[out_key] = res.tau_max
    pval_tau_star[out_key] = res.tau_star


header = f'''\
"""
Critical values produced by phillips-ouliaris-simulation.py

{counts}
"""

from math import inf

'''

formatted_code = header + "CV_PARAMETERS = " + format_dict(cv_params)
formatted_code += "\n\nCV_TAU_MIN = " + format_dict(cv_tau_min)
formatted_code += "\n\nPVAL_LARGE_P = " + format_dict(pval_large_p)
formatted_code += "\n\nPVAL_SMALL_P = " + format_dict(pval_small_p)
formatted_code += "\n\nPVAL_TAU_MAX = " + format_dict(pval_tau_max)
formatted_code += "\n\nPVAL_TAU_STAR = " + format_dict(pval_tau_star)
formatted_code += "\n\nPVAL_TAU_MIN = " + format_dict(pval_tau_min)

targets = {TargetVersion.PY36, TargetVersion.PY37, TargetVersion.PY38}
fm = FileMode(target_versions=targets)
formatted_code = format_file_contents(formatted_code, fast=False, mode=fm)

with open("../phillips_ouliaris.py", "w") as po:
    po.write(formatted_code)
