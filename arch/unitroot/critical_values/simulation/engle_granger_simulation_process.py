"""
Process the output of engle_granger_simulation.py into engle_granger.py.

Run from this directory, after the simulation has written
engle-granger-results-trend-<trend>-<i>.npz, one file per experiment:

    python engle_granger_simulation_process.py --data-path <dir>
"""

import argparse
from collections import defaultdict
from collections.abc import MutableMapping
import os
import pathlib
from typing import cast

from black import FileMode, TargetVersion, format_file_contents
import numpy as np
import pandas as pd
from shared import fit_smooth_pval_model, format_dict
from statsmodels.regression.linear_model import OLS

PERCENTILES = list(np.arange(0.1, 1.0, 0.1)) + list(np.arange(1.0, 100.0, 0.5))
PERCENTILES = PERCENTILES[::-1]
# The p-value may be set to 0 below the turning point of the small p polynomial,
# which is far in the tail, if p is no larger than this there
MAX_TAIL_STEP = 1e-12
# The small p polynomial is a cubic in the statistic. It fits the tail better than a
# quadratic: the largest relative error in p at 0.1% is 0.2%, and 0.6% with a
# quadratic.
SMALL_ORDER = 4

parser = argparse.ArgumentParser(description="Process the Engle-Granger simulations")
parser.add_argument(
    "--data-path",
    default="engle-granger",
    help="Directory containing the engle-granger-results-trend-<trend>-<i>.npz files",
)
parser.add_argument(
    "--output", default=os.path.join("..", "engle_granger.py"), help="File to write"
)
# The files hold quantiles only, so the number of simulations per file, EX_SIZE in
# engle_granger_simulation.py, is only known from the run
parser.add_argument(
    "--ex-size",
    type=int,
    default=250_000,
    help="Simulations per file, for the docstring of the output",
)
args = parser.parse_args()

# Sorted so that the averages do not depend on the order of the directory
files = sorted(pathlib.Path(args.data_path).glob("*.npz"))
if not files:
    raise FileNotFoundError(f"No .npz files in {args.data_path}")
input_data = defaultdict(list)
for f in files:
    contents = np.load(f)
    temp = contents["quantiles"]
    temp = np.reshape(temp, temp.shape + (1,), copy=False)
    input_data[contents["trend"][0]].append(temp)
    sample_sizes = contents["sample_sizes"]

final: dict[str, np.ndarray] = {}
for key in input_data:
    final[key] = np.concatenate(input_data[key], -1)

trends = ("nc", "c", "ct", "ctt")
critical_values = (1, 5, 10)
final_cv: dict[str, dict[int, list[list[float]]]] = {}
for trend in trends:
    print(trend)
    results = final[trend]

    # For percentiles 1, 5 and 10, regress on a constant, and powers of 1/T
    out: MutableMapping[int, list[list[float]]] = defaultdict(list)
    for cv in critical_values:
        num_ex = results.shape[-1]
        loc = np.argmin(np.abs(np.array(PERCENTILES) - cv))
        all_lhs = np.squeeze(results[loc])
        tau = np.ones((num_ex, 1)).dot(sample_sizes[None, :])
        tau = tau.T
        tau = tau.ravel()
        tau = tau[:, None]
        n = all_lhs.shape[0]
        rhs = (1.0 / tau) ** np.arange(4)

        for i in range(all_lhs.shape[1]):
            lhs = all_lhs[:, i, :].ravel()
            res = OLS(lhs, rhs).fit()
            params = res.params.copy()
            if res.pvalues[-1] > 0.05:
                params[-1] = 0.00
            out[cv].append(params)
        values = np.array(out[cv]).tolist()
        out[cv] = [[round(cast("float", val), 5) for val in row] for row in values]

    final_cv[trend] = dict(out)

final_cv["n"] = final_cv["nc"]
del final_cv["nc"]
formatted_str = format_dict(final_cv)

# %%
# The p-value models are fit to the quantiles of the largest sample size. The two
# polynomials, both cubics in the statistic, are fit jointly so that the p-value is
# continuous, has a continuous first derivative and is monotone in the statistic.
percentiles = np.array(PERCENTILES) / 100.0
large_p = {}
small_p = {}
tau_max = {}
tau_star = {}
tau_min = {}
for trend in trends:
    out_trend = "n" if trend == "nc" else trend
    # The first statistic is the Dickey-Fuller test of the dependent variable,
    # which has no regressors
    for i in range(1, final[trend].shape[2]):
        quantiles = pd.DataFrame(
            final[trend][:, np.argmax(sample_sizes), i], index=pd.Index(percentiles)
        )
        pvals = fit_smooth_pval_model(
            quantiles,
            small_order=SMALL_ORDER,
            use_log=False,
            max_tail_step=MAX_TAIL_STEP,
        )
        large_p[(out_trend, i)] = pvals.large_p
        small_p[(out_trend, i)] = pvals.small_p
        tau_max[(out_trend, i)] = pvals.tau_max
        tau_star[(out_trend, i)] = pvals.tau_star
        tau_min[(out_trend, i)] = pvals.tau_min

counts = "\n".join(
    f"Trend {'n' if trend == 'nc' else trend} is based on {final[trend].shape[-1]} "
    f"experiments of {args.ex_size:,} simulations."
    for trend in trends
)
bounds = list(tau_max.values()) + list(tau_min.values())
inf_import = "" if np.all(np.isfinite(bounds)) else "from math import inf\n\n"
header = f'''\
"""
Critical values and p-values produced by engle_granger_simulation_process.py

{counts}
"""

{inf_import}import numpy as np

eg_num_variables = np.arange(1, 12)
'''

formatted_code = header + "CV_PARAMETERS = " + formatted_str

pvals_code = (
    "LARGE_PARAMETERS=" + format_dict(large_p),
    "SMALL_PARAMETERS=" + format_dict(small_p),
    "TAU_MAX=" + format_dict(tau_max),
    "TAU_STAR=" + format_dict(tau_star),
    "TAU_MIN=" + format_dict(tau_min),
)

formatted_code += "\n\n" + "\n\n".join(pvals_code) + "\n"

targets = {
    TargetVersion.PY310,
    TargetVersion.PY311,
    TargetVersion.PY312,
    TargetVersion.PY313,
    TargetVersion.PY314,
}
fm = FileMode(target_versions=targets)
blackened = format_file_contents(formatted_code, fast=False, mode=fm)
# LF line endings on every platform, as in the repository
with open(args.output, "w", newline="\n") as eg:
    eg.write(blackened)
