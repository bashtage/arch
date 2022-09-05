from io import StringIO

import numpy as np
import pandas as pd

sio = StringIO()
sio.write("from numpy import asarray\n\n")
sio.write("kpss_critical_values = {}\n")

c = pd.read_hdf("kpss_critical_values.h5", "c")
ct = pd.read_hdf("kpss_critical_values.h5", "ct")

data = {"c": c, "ct": ct}
for k in ("c", "ct"):
    v = data[k]
    n = v.shape[0]
    selected = np.zeros((n, 1), dtype=bool)
    selected[0] = True
    selected[-1] = True
    selected[v.index == 10.0] = True
    selected[v.index == 5.0] = True
    selected[v.index == 2.5] = True
    selected[v.index == 1.0] = True
    max_diff = 1.0
    while max_diff > 0.05:
        xp = np.squeeze(np.asarray(v[selected].values))
        yp = np.asarray(v[selected].index, dtype=float)
        x = np.squeeze(np.asarray(v.values))
        y = np.asarray(v.index, dtype=float)
        yi = np.interp(x, xp, yp)
        abs_diff = np.abs(y - yi)
        max_diff = np.max(abs_diff)
        if max_diff > 0.05:
            selected[np.where(abs_diff == max_diff)] = True
    selected[np.asarray(v.index, dtype=float) <= 10.0] = True

    quantiles = list(np.squeeze(v[selected].index.values))
    critical_values = list(np.squeeze(np.asarray(v[selected].values)))
    # Fix for first CV
    critical_values[0] = 0.0
    sio.write(k + " = (")
    count = 0
    for c, q in zip(critical_values, quantiles):
        sio.write("(" + f"{q:0.3f}" + ", " + f"{c:0.4f}" + ")")
        count += 1
        if count % 4 == 0:
            sio.write(",\n    " + " " * len(k))
        else:
            sio.write(", ")
    sio.write(")\n")
    sio.write("kpss_critical_values['" + k + "'] = ")
    sio.write("asarray(" + k + ")")
    sio.write("\n")

sio.seek(0)
print(sio.read())
