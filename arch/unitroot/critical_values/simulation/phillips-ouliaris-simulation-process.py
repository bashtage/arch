import argparse
from collections import defaultdict
import glob
import os
from typing import cast

from black import FileMode, TargetVersion, format_file_contents
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phillips_ouliaris import FILE_TYPES, ROOT, TRENDS
import seaborn as sns
from shared import fit_smooth_pval_model, format_dict
from statsmodels.regression.linear_model import OLS

META = {"z_a": "negative", "z_t": "negative", "p_u": "positive", "p_z": "positive"}
CRITICAL_VALUES = (1, 5, 10)
WINS: dict[int, int] = defaultdict(int)
# The p-value may be set to 0 or 1 beyond a turning point of a polynomial, which
# is beyond the simulated quantiles, if p (or 1 - p) is no larger than the step
# there. A larger step is only used if no junction gives a monotone p-value.
MAX_TAIL_STEPS = (1e-12, 1e-9, 1e-6)
# The coefficients are ill-conditioned when the statistic is large, as Pz is, and
# rounding them to fewer digits moves the slopes at the junction apart
SIGNIFICANT_DIGITS = 15
# The small p polynomial of each statistic: the number of coefficients and
# whether it is in log|stat|. phillips_ouliaris_pval reads the second from
# PVAL_SMALL_P_LOG.
SMALL_P_MODEL = {
    "z_a": (4, True),
    "z_t": (4, True),
    "p_u": (4, True),
    "p_z": (4, True),
}
# 1. Load data
# 2. Compute critical values


def xval(lhs: np.ndarray, rhs: np.ndarray, log: bool = True, folds: int = 5) -> None:
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
            sigma2 = (sm_log.resid**2).mean()
            predictions[pred_loc, 2] = sgn * np.exp(pred_rhs[:, :3] @ sm_log.params)
            predictions[pred_loc, 3] = sgn * np.exp(
                pred_rhs[:, :3] @ sm_log.params + sigma2 / 2
            )

            lg_log = OLS(log_lhs[locs], rhs[locs]).fit()
            sigma2 = (lg_log.resid**2).mean()
            predictions[pred_loc, 4] = sgn * np.exp(pred_rhs @ lg_log.params)
            predictions[pred_loc, 5] = sgn * np.exp(
                pred_rhs @ lg_log.params + sigma2 / 2
            )
    errors = lhs[:, None] - predictions
    best = int(np.argmin(errors.var(0)))
    WINS[best] += 1


def estimate_cv_regression(
    results: pd.DataFrame, statistic: str
) -> tuple[dict[int, list[float]], float]:
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
        out[cv] = [float(round(val, 5)) for val in params]
    return out, float(tau.min())


parser = argparse.ArgumentParser(
    description="Process the Phillips-Ouliaris simulations"
)
parser.add_argument("--root", default=ROOT, help="Directory of the simulation results")
parser.add_argument(
    "--output",
    default=os.path.join("..", "phillips_ouliaris.py"),
    help="File to write",
)
# The files hold quantiles only, so the number of simulations per file, the
# --ex-size of phillips-ouliaris-simulation.py, is only known from the run
parser.add_argument(
    "--ex-size",
    type=int,
    default=250_000,
    help="Simulations per file, for the docstring of the output",
)
parser.add_argument(
    "--plot",
    action="store_true",
    help="Plot the quantiles at the largest sample size to output.pdf",
)
args = parser.parse_args()

results = defaultdict(list)
num_files = {}
for file_type in FILE_TYPES:
    for trend in TRENDS:
        pattern = f"*-statistic-{file_type}-trend-{trend}-*.hdf"
        # Sorted so that the averages do not depend on the order of the directory
        result_files = sorted(glob.glob(os.path.join(args.root, pattern)))
        num_files[(file_type, trend)] = len(result_files)
        for rf in result_files:
            temp = pd.DataFrame(pd.read_hdf(rf, "results"))
            mi_cols = cast("pd.MultiIndex", temp.columns)
            statistics = mi_cols.levels[2]
            for stat in statistics:
                # TODO: Bug in pandas-stubs prevents valid index types
                index_slice = pd.IndexSlice[:, :, stat]
                single = temp.loc[:, index_slice]
                single.columns = single.columns.droplevel(level=2)
                results[(stat, trend)].append(single)

missing = [key for key, count in num_files.items() if count == 0]
if missing:
    raise FileNotFoundError(f"No result files in {args.root} for {missing}")
nsimulation = {k: args.ex_size * v for k, v in num_files.items()}

joined = defaultdict(list)
for key in results:
    result_dfs = results[key]
    mi_columns = cast("pd.MultiIndex", result_dfs[0].columns)
    stoch_trends = mi_columns.levels[1]
    for st in stoch_trends:
        for df in result_dfs:
            # TODO: Bug in pandas-stubs prevents valid index types
            single = df.loc[:, pd.IndexSlice[:, st]]
            single.columns = single.columns.droplevel(level=1)
            single = single.dropna(axis=1, how="all")
            joined[key + (st,)].append(single)

final = {key: pd.concat(joined[key], axis=1) for key in joined}
stat_names = {"p_z": "Pz", "p_u": "Pu", "z_t": "Zt", "z_a": "Za"}
cv_params = {}
cv_tau_min = {}
for _final_key, final_value in final.items():
    # The results are keyed by the statistic's name in the simulation, z_a, and
    # the constants by its name in arch, Za
    final_key = (stat_names[_final_key[0]],) + _final_key[1:]
    cv_params[final_key], cv_tau_min[final_key] = estimate_cv_regression(
        final_value, _final_key[0]
    )

print("Best methods")
for wins_key in sorted(WINS):
    print(f"{wins_key}: {WINS[wins_key]}")

report = []
for key, n_sim_value in nsimulation.items():
    s = key[0].upper()
    t = key[1]
    report.append(
        f"{s}-type statistics with trend {t} based on {n_sim_value:,} simulations"
    )

counts = "\n".join(report)

quantiles_d = defaultdict(list)
pval_data = {}
# In the order of the results, as the critical values, so that the order of the
# keys in the output does not depend on the hash of the strings
for multi_key in final:
    # TODO: Bug in pandas-stubs prevents valid index types
    # A list so that there is a column for each file, even if there is one
    pval_data[multi_key] = final[multi_key].loc[:, [2000]]
    # TODO: Bug in pandas-stubs prevents valid index types
    temp_series = final[multi_key].loc[:, [2000]].mean(axis=1)
    # This is a series since there are many columns with 2000
    temp_series.name = multi_key[-1]
    quantiles_d[multi_key[:-1]].append(temp_series)
quantiles = {}
for key in quantiles_d:
    selected = cast("np.ndarray", quantiles_d[key])
    quantiles[key] = pd.concat(selected, axis=1)


if args.plot:
    plt.rc("figure", figsize=(16, 8))
    sns.set_style("darkgrid")
    pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
    for quantile_key, quantile_value in quantiles.items():
        temp = quantile_value
        y = temp.index.to_numpy()[:, None]
        x = temp.to_numpy()
        stat = quantile_key[0]
        if stat in ("z_t", "z_a"):
            x = -1 * x
        if stat in ("p_u", "p_z"):
            y = 1 - y
        fig, ax = plt.subplots(1, 1)
        plt.plot(x, y)
        plt.title(", ".join(quantile_key))
        pdf.savefig(fig)
        if stat in ("p_u", "p_z"):
            fig, ax = plt.subplots(1, 1)
            plt.plot(np.log(x), y)
            plt.title(f"Log {quantile_key[0]}, {quantile_key[1]}")
            pdf.savefig(fig)
    pdf.close()

pval_results = {}
pval_large_p = {}
pval_small_p = {}
pval_tau_star = {}
pval_tau_min = {}
pval_tau_max = {}
failures = []
for pval_key, pval_value in pval_data.items():
    pval_series = pval_value.copy()
    if pval_key[0] in ("p_z", "p_u"):
        pval_series.index = 1 - pval_series.index
        pval_series = -1 * pval_series
    pval_series = pval_series.sort_index()
    out_key = (stat_names[pval_key[0]],) + pval_key[1:]
    # The two polynomials are fit jointly so that the p-value is continuous, has a
    # continuous first derivative and is monotone in the statistic
    small_order, use_log = SMALL_P_MODEL[pval_key[0]]
    error: RuntimeError | None = None
    for max_tail_step in MAX_TAIL_STEPS:
        try:
            res = fit_smooth_pval_model(
                pval_series,
                small_order=small_order,
                use_log=use_log,
                max_tail_step=max_tail_step,
                significant_digits=SIGNIFICANT_DIGITS,
            )
        except RuntimeError as exc:
            error = exc
            continue
        if max_tail_step > MAX_TAIL_STEPS[0]:
            print(f"{out_key}: the p-value steps by up to {max_tail_step} in a tail")
        break
    else:
        failures.append(f"{out_key}: {error}")
        continue
    pval_results[out_key] = res
    pval_large_p[out_key] = res.large_p
    pval_small_p[out_key] = res.small_p
    pval_tau_min[out_key] = res.tau_min
    pval_tau_max[out_key] = res.tau_max
    pval_tau_star[out_key] = res.tau_star
if failures:
    raise RuntimeError("No p-value model for\n" + "\n".join(failures))


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
pval_small_p_log = {
    stat_names[stat]: use_log for stat, (_, use_log) in SMALL_P_MODEL.items()
}
formatted_code += "\n\n# Whether the small p polynomials are in log|stat|, or in stat\n"
formatted_code += "PVAL_SMALL_P_LOG = " + format_dict(pval_small_p_log)

targets = {TargetVersion.PY38, TargetVersion.PY39, TargetVersion.PY310}
fm = FileMode(target_versions=targets)
formatted_code = format_file_contents(formatted_code, fast=False, mode=fm)

# CRLF line endings on every platform, as in the repository
with open(args.output, "w", newline="\r\n") as po:
    po.write(formatted_code)
