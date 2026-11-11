"""
Process the ADF z-test simulations into the adf_z values of dickey_fuller.py.

adf_z_critical_values_simulation_joblib.py writes the quantiles of the z-test
statistic to adf-z/adf_z_<trend>-<i>.npz. Run from this directory:

    python adf_z_simlation_process.py

The statistic of a series of length T is nobs * (gamma - 1), computed from the
nobs = T - 1 observations of the regression of ADF, which is the nobs that ADF and
PhillipsPerron give mackinnoncrit. Each simulated quantile is regressed on a
constant and powers of 1/nobs. The regressions of the 1%, 5% and 10% quantiles
are the critical values. The constants of the regressions are the asymptotic
quantiles, and the p-value is fit to them, so that the p-value at an asymptotic
critical value is its size, up to the error of the fit. The values are written to
the last section of dickey_fuller.py, which is replaced, and the sections before it
are kept.
"""

import argparse
import os
import pathlib

from adf_simulation import OUTPUT_PATH, PERCENTILES, TIME_SERIES_LENGTHS, TRENDS
from black import FileMode, TargetVersion, format_file_contents
import numpy as np
import pandas as pd
from scipy import stats
from shared import (
    PvalueResult,
    estimate_response_surface,
    fit_smooth_pval_model,
    format_dict,
)

Z_MARKER = "# Z values from"

parser = argparse.ArgumentParser(description="Process the ADF z-test simulations")
parser.add_argument(
    "--data-path",
    default=str(OUTPUT_PATH),
    help="Directory containing the adf_z_<trend>-<i>.npz files",
)
parser.add_argument(
    "--output",
    default=os.path.join("..", "dickey_fuller.py"),
    help="The dickey_fuller.py to update",
)
# The files hold quantiles only, so the number of simulations per file, EX_SIZE in
# adf_z_critical_values_simulation_joblib.py, is only known from the run
parser.add_argument(
    "--ex-size",
    type=int,
    default=250_000,
    help="Simulations per file, for the comment in the output",
)
args = parser.parse_args()
output = pathlib.Path(args.output)


def pvalue(stat: np.ndarray, pvals: PvalueResult) -> np.ndarray:
    # The small p polynomial is in log|stat|, and tau_star is negative
    small, large = np.array(pvals.small_p), np.array(pvals.large_p)
    below = stat <= pvals.tau_star
    with np.errstate(divide="ignore", over="ignore"):
        return stats.norm.cdf(
            np.where(
                below,
                np.polyval(small[::-1], np.log(np.abs(np.where(below, stat, -1.0)))),
                np.polyval(large[::-1], stat),
            )
        )


data_path = pathlib.Path(args.data_path)
critical_values = (1.0, 5.0, 10.0)
cv_rows = [int(np.argmin(np.abs(PERCENTILES - cv))) for cv in critical_values]
nobs = TIME_SERIES_LENGTHS - 1
rhs = (1.0 / nobs[:, None]) ** np.arange(4)
probs = PERCENTILES / 100.0
counts = {}
adf_z_cv_approx = {}
adf_z_max = {}
adf_z_min = {}
adf_z_star = {}
adf_z_small_p = {}
adf_z_large_p = {}

for trend in TRENDS:
    # Sorted so that the averages do not depend on the order of the directory
    data_files = sorted(data_path.glob(f"adf_z_{trend}-*.npz"))
    if not data_files:
        raise FileNotFoundError(f"No adf_z_{trend}-*.npz files in {data_path}")
    all_results = []
    for df in data_files:
        with np.load(df) as data:
            all_results.append(data["results"])
    results = np.hstack(all_results)
    cols = nobs.tolist() * len(data_files)
    results_df = pd.DataFrame(results, index=pd.Index(probs), columns=cols)
    counts[trend] = len(data_files)

    surface = estimate_response_surface(results_df)
    adf_z_cv_approx[trend] = [
        [float(val) for val in surface.iloc[row]] for row in cv_rows
    ]
    asymptotic = surface.iloc[:, :1]

    # The two p-value polynomials are fit jointly so that the p-value is continuous,
    # has a continuous first derivative and is monotone in the statistic
    pvals = fit_smooth_pval_model(asymptotic)
    adf_z_max[trend] = pvals.tau_max
    adf_z_min[trend] = pvals.tau_min
    adf_z_star[trend] = float(pvals.tau_star)
    adf_z_small_p[trend] = [float(val) for val in pvals.small_p]
    adf_z_large_p[trend] = [float(val) for val in pvals.large_p]

    # The fit at the asymptotic quantiles that it is estimated from
    err = np.abs(pvalue(asymptotic.iloc[:, 0].to_numpy(), pvals) - probs)
    tail = probs <= 0.15
    print(
        f"{trend}: {len(data_files)} files, tau_star {pvals.tau_star}, largest "
        f"error in p {err[tail].max():.1e} if p <= 0.15 and {err[~tail].max():.4f} "
        f"otherwise, relative error at p={probs[0]:g} {err[0] / probs[0]:.2%}"
    )
    asymptotic_cv = np.array([cv[0] for cv in adf_z_cv_approx[trend]])
    print(
        "  p-value at the asymptotic critical values: "
        + ", ".join(f"{val:.6f}" for val in pvalue(asymptotic_cv, pvals))
    )
    # The critical values compared with the average simulated quantiles
    for j, cv in enumerate(critical_values):
        grouped = results_df.iloc[cv_rows[j]].groupby(level=0)
        mean = grouped.mean().to_numpy()
        se = (grouped.std() / np.sqrt(grouped.count())).to_numpy()
        fit_err = rhs @ np.array(adf_z_cv_approx[trend][j]) - mean
        worst_fit = int(np.argmax(np.abs(fit_err / se)))
        print(
            f"  {cv:g}%: {adf_z_cv_approx[trend][j]}, largest error "
            f"{fit_err[worst_fit]:+.4f} at nobs={nobs[worst_fit]} "
            f"({fit_err[worst_fit] / se[worst_fit]:+.1f} s.e.)"
        )

z_section = f"{Z_MARKER} new simulations, {args.ex_size:,} per ex., "
if len(set(counts.values())) == 1:
    z_section += f"{counts[TRENDS[0]]:,} exercises.\n"
else:
    z_section += "exercises\n# " + ", ".join(f"{t} {c:,}" for t, c in counts.items())
    z_section += ".\n"
z_section += "# The critical values and the p-values are fit to the same simulations,\n"
z_section += "# and the p-value at an asymptotic critical value is its size.\n"
z_section += "adf_z_min = " + format_dict(adf_z_min)
z_section += "\n\nadf_z_star = " + format_dict(adf_z_star)
z_section += "\n\nadf_z_max = " + format_dict(adf_z_max)
z_section += "\n\n# The small p parameters are for np.log(np.abs(stat))\n"
z_section += "adf_z_small_p = " + format_dict(adf_z_small_p)
z_section += "\n\nadf_z_large_p = " + format_dict(adf_z_large_p)
z_section += "\n\nadf_z_cv_approx = " + format_dict(adf_z_cv_approx)

lines = output.read_text().splitlines(keepends=True)
z_start = next(i for i, line in enumerate(lines) if line.startswith(Z_MARKER))
formatted_code = "".join(lines[:z_start]) + "\n\n" + z_section

targets = {
    TargetVersion.PY310,
    TargetVersion.PY311,
    TargetVersion.PY312,
    TargetVersion.PY313,
    TargetVersion.PY314,
}
fm = FileMode(target_versions=targets)
formatted_code = format_file_contents(formatted_code, fast=False, mode=fm)

# LF line endings on every platform, as in the repository
with open(output, "w", newline="\n") as cvs:
    cvs.write(formatted_code)
