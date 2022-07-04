import glob
import os

from adf_simulation import OUTPUT_PATH, PERCENTILES, TIME_SERIES_LENGTHS, TRENDS
from black import FileMode, TargetVersion, format_file_contents
import numpy as np
import pandas as pd
from shared import estimate_cv_regression, fit_pval_model, format_dict

critical_values = (1.0, 5.0, 10.0)
adf_z_cv_approx = {}
adf_z_max = {}
adf_z_min = {}
adf_z_star = {}
adf_z_small_p = {}
adf_z_large_p = {}

for trend in TRENDS:
    data_files = glob.glob(os.path.join(OUTPUT_PATH, f"adf_z_{trend}-*.npz"))
    percentiles = PERCENTILES
    all_results = []
    for df in data_files:
        with np.load(df) as data:
            all_results.append(data["results"])
    results = np.hstack(all_results)
    cols = TIME_SERIES_LENGTHS.tolist() * len(data_files)
    results_df = pd.DataFrame(
        results, index=pd.Index(PERCENTILES / 100.0), columns=cols
    )

    cv_approx = estimate_cv_regression(results_df, critical_values)
    adf_z_cv_approx[trend] = [cv_approx[cv] for cv in critical_values]

    pvals = fit_pval_model(results_df[2000], small_order=4, use_log=True)
    adf_z_max[trend] = pvals.tau_max
    adf_z_min[trend] = pvals.tau_min
    adf_z_star[trend] = pvals.tau_star
    adf_z_small_p[trend] = pvals.small_p
    adf_z_large_p[trend] = pvals.large_p

formatted_code = "adf_z_min = " + format_dict(adf_z_min)
formatted_code += "\n\nadf_z_star = " + format_dict(adf_z_star)
formatted_code += "\n\nadf_z_max = " + format_dict(adf_z_max)
formatted_code += "\n\n# The small p parameters are for np.log(np.abs(stat))\n"
formatted_code += "adf_z_small_p = " + format_dict(adf_z_small_p)
formatted_code += "\n\nadf_z_large_p = " + format_dict(adf_z_large_p)
formatted_code += "\n\nadf_z_cv_approx = " + format_dict(adf_z_cv_approx)

with open("../dickey_fuller.py") as cvs:
    lines = cvs.readlines()

retain = []
for line in lines:
    if "# Z values from" in line:
        break
    retain.append(line)
retain.append("\n\n# Z values from new simulations, 500 exercises, 250,000 per ex.\n")

formatted_code = "".join(retain) + formatted_code

targets = {TargetVersion.PY36, TargetVersion.PY37, TargetVersion.PY38}
fm = FileMode(target_versions=targets)
formatted_code = format_file_contents(formatted_code, fast=False, mode=fm)

with open("../dickey_fuller.py", "w") as cvs:
    cvs.write(formatted_code)
