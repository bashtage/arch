"""
Process the output of dfgls_critical_values_simulation.py into dfgls.py.

Run from this directory, after the simulation has written dfgls_c_<i>.npy and
dfgls_ct_<i>.npy, one file per experiment:

    python dfgls_simulation_process.py --data-path <dir> --output ../dfgls.py
"""

import argparse
import os
import pathlib

from black import FileMode, TargetVersion, format_file_contents
from dfgls_critical_values_simulation import (
    EX_SIZE,
    OUTPUT_PATH,
    PERCENTILES,
    TRENDS,
    T,
)
import numpy as np
import pandas as pd
from shared import fit_smooth_pval_model, format_dict
from statsmodels.regression.linear_model import OLS

CRITICAL_VALUES = (1.0, 5.0, 10.0)
# The p-value may be set to 0 below the turning point of the small p polynomial,
# which is far in the tail, if p is no larger than this there
MAX_TAIL_STEP = 1e-12

HEADER = '''\
"""
Contains values used to approximate the critical value and p-value from DFGLS
statistics

These have been computed using the methodology of MacKinnon (1994) and (2010)
simulation. See dfgls_critival_values_simulation for implementation.

{counts}
"""

'''

parser = argparse.ArgumentParser(description="Process the DFGLS simulations")
parser.add_argument(
    "--data-path",
    default=str(OUTPUT_PATH),
    help="Directory containing the dfgls_<trend>_<i>.npy files",
)
parser.add_argument(
    "--output", default=os.path.join("..", "dfgls.py"), help="File to write"
)
args = parser.parse_args()

percentiles = np.array(PERCENTILES)
dfgls_cv_approx = {}
dfgls_large_p = {}
dfgls_small_p = {}
dfgls_tau_star = {}
dfgls_tau_max = {}
dfgls_tau_min = {}
counts = []
for t in TRENDS:
    # Sorted so that the averages do not depend on the order of the directory
    files = sorted(pathlib.Path(args.data_path).glob(f"dfgls_{t}_*.npy"))
    if not files:
        raise FileNotFoundError(f"No dfgls_{t}_*.npy files in {args.data_path}")
    print(f"{t}: {len(files)} files")
    counts.append(
        f"Trend {t} is based on {len(files)} experiments of {EX_SIZE:,} simulations."
    )
    # Each file has the percentiles (columns) for each sample size in T (rows), and
    # results has the percentiles in rows, sample sizes in columns and the
    # experiments in the last dimension
    results = np.stack([np.load(f) for f in files], axis=-1).transpose(1, 0, 2)
    if results.shape[:2] != (percentiles.shape[0], T.shape[0]):
        raise ValueError("The files do not match T and PERCENTILES")

    # For percentiles 1, 5 and 10, regress on a constant, and powers of 1/T
    out = []
    for cv in CRITICAL_VALUES:
        num_ex = results.shape[2]
        loc = np.where(percentiles == cv)[0][0]
        lhs = np.squeeze(results[loc, :, :])
        # Adjust for effective sample size, this is what lookup the code uses
        tau = np.ones((num_ex, 1)).dot(T[None, :]) - 1.0
        tau = tau.T
        lhs = lhs.ravel()
        tau = tau.ravel()
        tau = tau[:, None]
        n = lhs.shape[0]
        rhs = np.ones((n, 1))
        rhs = np.hstack((rhs, 1.0 / tau))
        rhs = np.hstack((rhs, (1.0 / tau) ** 2.0))
        rhs = np.hstack((rhs, (1.0 / tau) ** 3.0))
        res = OLS(lhs, rhs).fit()
        out.append(res.params)

    dfgls_cv_approx[t] = np.array(out)

    # The p-value models are fit to the quantiles of the largest sample size. The
    # two polynomials, a quadratic in the statistic for small p and a cubic for
    # large p, are fit jointly so that the p-value is continuous, has a continuous
    # first derivative and is monotone in the statistic. The 0th percentile, the
    # smallest simulated value, has no normal quantile.
    interior = (percentiles > 0) & (percentiles < 100)
    quantiles = pd.DataFrame(
        results[interior, np.argmax(T), :],
        index=pd.Index(percentiles[interior] / 100.0),
    )
    pvals = fit_smooth_pval_model(
        quantiles, small_order=3, use_log=False, max_tail_step=MAX_TAIL_STEP
    )
    dfgls_tau_max[t] = pvals.tau_max
    dfgls_tau_min[t] = pvals.tau_min
    dfgls_tau_star[t] = pvals.tau_star
    dfgls_large_p[t] = pvals.large_p
    dfgls_small_p[t] = pvals.small_p

bounds = list(dfgls_tau_max.values()) + list(dfgls_tau_min.values())
imports = "array, inf" if not np.all(np.isfinite(bounds)) else "array"
# The arrays are written as numpy prints them
with np.printoptions(precision=8):
    cv_approx = ", ".join(f'"{t}": {dfgls_cv_approx[t]!r}' for t in dfgls_cv_approx)

formatted_code = HEADER.format(counts="\n".join(counts))
formatted_code += f"from numpy import {imports}\n\n"
formatted_code += "dfgls_cv_approx = {" + cv_approx + "}"
formatted_code += "\n\ndfgls_tau_max = " + format_dict(dfgls_tau_max)
formatted_code += "\n\ndfgls_tau_min = " + format_dict(dfgls_tau_min)
formatted_code += "\n\ndfgls_tau_star = " + format_dict(dfgls_tau_star)
formatted_code += "\n\ndfgls_large_p = " + format_dict(dfgls_large_p)
formatted_code += "\n\ndfgls_small_p = " + format_dict(dfgls_small_p)
formatted_code += "\n"

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
with open(args.output, "w", newline="\n") as cvs:
    cvs.write(formatted_code)
