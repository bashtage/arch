from arch.compat.pandas import MONTH_END

from typing import cast

import numpy as np
import pandas as pd
import pytest

from arch._typing import ArrayLike2D, Float64Array, Float64Array2D


@pytest.fixture(scope="module", params=[True, False])
def data(request) -> tuple[Float64Array, Float64Array]:
    g = np.random.RandomState([12839028, 3092183, 902813])
    e = g.standard_normal((2000, 2))
    phi = g.random_sample((3, 2, 2))
    phi[:, 0, 0] *= 0.8 / phi[:, 0, 0].sum()
    phi[:, 1, 1] *= 0.8 / phi[:, 1, 1].sum()
    phi[:, 0, 1] *= 0.2 / phi[:, 0, 1].sum()
    phi[:, 1, 0] *= 0.2 / phi[:, 1, 0].sum()
    y = e.copy()
    for i in range(3, y.shape[0]):
        y[i] = e[i]
        for j in range(3):
            y[i] += (phi[j] @ y[i - j - 1].T).T
    y = y[-1000:]
    if request.param:
        df = pd.DataFrame(y, columns=["y", "x"])
        return np.asarray(df.iloc[:, :1], dtype=float), np.asarray(
            df.iloc[:, 1:], dtype=float
        )
    return y[:, :1], y[:, 1:]


@pytest.fixture(scope="module", params=[True, False], ids=["pandas", "numpy"])
def trivariate_data(request) -> tuple[ArrayLike2D, ArrayLike2D]:
    rs = np.random.RandomState([922019, 12882912, 192010, 10189, 109981])
    nobs = 1000
    burn = 100
    e = rs.standard_normal((nobs + burn, 3))
    y = e.copy()
    for i in range(1, 3):
        roots = np.ones(3)
        roots[1:] = rs.random_sample(2)
        ar = -np.poly(roots)[1:]
        lags = np.arange(1, 4)
        for j in range(3, nobs + burn):
            y[j, i] = y[j - lags, i] @ ar + e[j, i]
    y[:, 0] = 10 + 0.75 * y[:, 1] + 0.25 * y[:, 2] + e[:, 0]
    y = y[burn:]
    theta = np.pi * (2 * rs.random_sample(3) - 1)
    rot = np.eye(3)
    idx = 0
    for i in range(3):
        for j in range(i + 1, 3):
            th = theta[idx]
            c = np.cos(th)
            s = np.sin(th)
            r = np.eye(3)
            r[j, j] = r[i, i] = c
            r[i, j] = -s
            r[j, i] = s
            rot = rot @ r
            idx += 1
    y = y @ rot
    if request.param:
        dt_index = pd.date_range("1-1-2000", periods=nobs, freq=MONTH_END)
        cols = [f"y{i}" for i in range(1, 4)]
        data = pd.DataFrame(y, columns=cols, index=dt_index)
        return cast("pd.DataFrame", data.iloc[:, :1]), cast(
            "pd.DataFrame", data.iloc[:, 1:]
        )

    return cast("Float64Array2D", y[:, :1]), cast("Float64Array2D", y[:, 1:])
