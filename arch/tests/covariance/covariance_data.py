from itertools import product

import numpy as np
import pandas as pd
import pytest

DATA_PARAMS = list(product([1, 3], [True, False], [0, 1, 3]))
DATA_IDS = [f"dim: {d}, pandas: {p}, order: {o}" for d, p, o in DATA_PARAMS]


@pytest.fixture(scope="module", params=DATA_PARAMS, ids=DATA_IDS)
def covariance_data(request):
    dim, pandas, order = request.param
    rs = np.random.RandomState([839084, 3823810, 982103, 829108])
    burn = 100
    shape = (burn + 500,)
    if dim > 1:
        shape += (3,)
    rvs = rs.standard_normal(shape)
    phi = np.zeros((order, dim, dim))
    if order > 0:
        phi[0] = np.eye(dim) * 0.4 + 0.1
        for i in range(1, order):
            phi[i] = 0.3 / (i + 1) * np.eye(dim)
        for i in range(order, burn + 500):
            for j in range(order):
                if dim == 1:
                    rvs[i] += np.squeeze(phi[j] * rvs[i - j - 1])
                else:
                    rvs[i] += phi[j] @ rvs[i - j - 1]
    if order > 1:
        p = np.eye(dim * order, dim * order, -dim)
        for j in range(order):
            p[:dim, j * dim : (j + 1) * dim] = phi[j]
        v, _ = np.linalg.eig(p)
        assert np.max(np.abs(v)) < 1
    rvs = rvs[burn:]
    if pandas and dim == 1:
        return pd.Series(rvs, name="x")
    elif pandas:
        df = pd.DataFrame(rvs, columns=[f"x{i}" for i in range(dim)])
        df.to_csv(f"cov-data-order-{order}.csv")
        return df

    return rvs
