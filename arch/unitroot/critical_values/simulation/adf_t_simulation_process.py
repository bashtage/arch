"""
Process the ADF t-test simulations into the adf_t values of dickey_fuller.py.

adf_z_critical_values_simulation_joblib.py writes the quantiles of the t-test
statistic to adf-t/adf_t_<trend>-<i>.npz. Run from this directory:

    python adf_t_simulation_process.py

The statistic of a series of length T is computed from the nobs = T - 1
observations of the regression of ADF, which is the nobs that ADF and
PhillipsPerron give mackinnoncrit. Each simulated quantile is regressed on a
constant and powers of 1/nobs. The regressions of the 1%, 5% and 10% quantiles
are the critical values, as for the ADF z-test. The constants of the regressions
are the asymptotic quantiles, and the p-value is fit to them, so that the p-value
at an asymptotic critical value is its size, up to the error of the fit. The
values are for one unit root. They are written to a section of dickey_fuller.py
that is before the z-test values, so that adf_z_simlation_process.py keeps it,
and which is replaced if it exists.
"""

import argparse
import os
import pathlib
import runpy

from adf_simulation import PERCENTILES, T_OUTPUT_PATH, TIME_SERIES_LENGTHS, TRENDS
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

# The small p polynomial is a cubic in the statistic, and not a quadratic as in
# MacKinnon (1994)
SMALL_ORDER = 4
# The p-value may be set to 0 below (or 1 above) the turning point of a
# polynomial, which is far in the tail, if p (or 1 - p) is no larger than this
# there. Larger steps are only used if no junction gives a monotone p-value.
MAX_TAIL_STEPS = (1e-12, 1e-9, 1e-6)
T_MARKER = "# t values from"
Z_MARKER = "# Z values from"

parser = argparse.ArgumentParser(description="Process the ADF t-test simulations")
parser.add_argument(
    "--data-path",
    default=str(T_OUTPUT_PATH),
    help="Directory containing the adf_t_<trend>-<i>.npz files",
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
# MacKinnon (2010), for comparison, from the file that is updated. Importing it from
# arch.unitroot would fail if the file does not have the values that this writes.
tau_2010 = runpy.run_path(str(output))["tau_2010"]


def pvalue(stat: np.ndarray, pvals: PvalueResult) -> np.ndarray:
    small, large = np.array(pvals.small_p), np.array(pvals.large_p)
    with np.errstate(over="ignore"):
        return stats.norm.cdf(
            np.where(
                stat <= pvals.tau_star,
                np.polyval(small[::-1], stat),
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
adf_t_cv_approx = {}
adf_t_max = {}
adf_t_min = {}
adf_t_star = {}
adf_t_small_p = {}
adf_t_large_p = {}

for trend in TRENDS:
    # Sorted so that the averages do not depend on the order of the directory
    data_files = sorted(data_path.glob(f"adf_t_{trend}-*.npz"))
    if not data_files:
        raise FileNotFoundError(f"No adf_t_{trend}-*.npz files in {data_path}")
    all_results = []
    for df in data_files:
        with np.load(df) as data:
            all_results.append(data["results"])
    results = np.hstack(all_results)
    cols = nobs.tolist() * len(data_files)
    results_df = pd.DataFrame(results, index=pd.Index(probs), columns=cols)
    counts[trend] = len(data_files)

    surface = estimate_response_surface(results_df)
    adf_t_cv_approx[trend] = [
        [float(val) for val in surface.iloc[row]] for row in cv_rows
    ]
    asymptotic = surface.iloc[:, :1]

    # The two p-value polynomials are fit jointly so that the p-value is continuous,
    # has a continuous first derivative and is monotone in the statistic
    error: RuntimeError | None = None
    for max_tail_step in MAX_TAIL_STEPS:
        try:
            pvals = fit_smooth_pval_model(
                asymptotic,
                small_order=SMALL_ORDER,
                use_log=False,
                max_tail_step=max_tail_step,
            )
        except RuntimeError as exc:
            error = exc
            continue
        if max_tail_step > MAX_TAIL_STEPS[0]:
            print(f"{trend}: the p-value steps by up to {max_tail_step} in a tail")
        break
    else:
        raise RuntimeError(f"{trend}: {error}")
    adf_t_max[trend] = pvals.tau_max
    adf_t_min[trend] = pvals.tau_min
    adf_t_star[trend] = float(pvals.tau_star)
    adf_t_small_p[trend] = [float(val) for val in pvals.small_p]
    adf_t_large_p[trend] = [float(val) for val in pvals.large_p]

    # The fit at the asymptotic quantiles that it is estimated from
    err = np.abs(pvalue(asymptotic.iloc[:, 0].to_numpy(), pvals) - probs)
    tail = probs <= 0.15
    print(
        f"{trend}: {len(data_files)} files, tau_star {pvals.tau_star}, largest "
        f"error in p {err[tail].max():.1e} if p <= 0.15 and {err[~tail].max():.4f} "
        f"otherwise, relative error at p={probs[0]:g} {err[0] / probs[0]:.2%}"
    )
    asymptotic_cv = np.array([cv[0] for cv in adf_t_cv_approx[trend]])
    print(
        "  p-value at the asymptotic critical values: "
        + ", ".join(f"{val:.6f}" for val in pvalue(asymptotic_cv, pvals))
    )
    # The critical values compared with the average simulated quantiles and with
    # the critical values of MacKinnon (2010)
    for j, cv in enumerate(critical_values):
        grouped = results_df.iloc[cv_rows[j]].groupby(level=0)
        mean = grouped.mean().to_numpy()
        se = (grouped.std() / np.sqrt(grouped.count())).to_numpy()
        fitted = rhs @ np.array(adf_t_cv_approx[trend][j])
        mackinnon = rhs @ tau_2010[trend][0, j]
        fit_err = fitted - mean
        diff = fitted - mackinnon
        worst_fit = int(np.argmax(np.abs(fit_err / se)))
        worst = int(np.argmax(np.abs(diff)))
        print(
            f"  {cv:g}%: {adf_t_cv_approx[trend][j]}, largest error "
            f"{fit_err[worst_fit]:+.4f} at nobs={nobs[worst_fit]} "
            f"({fit_err[worst_fit] / se[worst_fit]:+.1f} s.e.), largest difference "
            f"from MacKinnon (2010) {diff[worst]:+.4f} at nobs={nobs[worst]}"
        )

t_section = f"{T_MARKER} new simulations, {args.ex_size:,} per ex., "
if len(set(counts.values())) == 1:
    t_section += f"{counts[TRENDS[0]]:,} exercises.\n"
else:
    t_section += "exercises\n# " + ", ".join(f"{t} {c:,}" for t, c in counts.items())
    t_section += ".\n"
t_section += (
    "# One unit root. The critical values and the p-values are fit to the same\n"
)
t_section += (
    "# simulations, and the p-value at an asymptotic critical value is its size.\n"
)
t_section += "adf_t_min = " + format_dict(adf_t_min)
t_section += "\n\nadf_t_star = " + format_dict(adf_t_star)
t_section += "\n\nadf_t_max = " + format_dict(adf_t_max)
t_section += "\n\nadf_t_small_p = " + format_dict(adf_t_small_p)
t_section += "\n\nadf_t_large_p = " + format_dict(adf_t_large_p)
t_section += "\n\nadf_t_cv_approx = " + format_dict(adf_t_cv_approx)

lines = output.read_text().splitlines(keepends=True)
z_start = next(i for i, line in enumerate(lines) if line.startswith(Z_MARKER))
t_start = next(
    (i for i, line in enumerate(lines[:z_start]) if line.startswith(T_MARKER)),
    z_start,
)
formatted_code = "".join(lines[:t_start]) + "\n\n" + t_section + "\n\n\n"
formatted_code += "".join(lines[z_start:])

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
