import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from arch.bootstrap.base import optimal_block_length


def test_block_length():
    rs = np.random.RandomState(0)
    e = rs.standard_normal(10000 + 100)
    y = e
    for i in range(1, len(e)):
        y[i] = 0.3 * y[i - 1] + e[i]
    s = pd.Series(y[100:], name="x")
    bl = optimal_block_length(s)
    sb, cb = bl.loc["x"]
    assert_allclose(sb, 13.635665, rtol=1e-4)
    assert_allclose(cb, 15.60894, rtol=1e-4)

    df = pd.DataFrame([s, s]).T
    df.columns = ["x", "y"]
    bl = optimal_block_length(df)
    for idx in ("x", "y"):
        sb, cb = bl.loc[idx]
        assert_allclose(sb, 13.635665, rtol=1e-4)
        assert_allclose(cb, 15.60894, rtol=1e-4)

    assert tuple(bl.columns) == ("stationary", "circular")
    assert tuple(bl.index) == ("x", "y")

    bl = optimal_block_length(np.asarray(df))
    assert tuple(bl.index) == (0, 1)
