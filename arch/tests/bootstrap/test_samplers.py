from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import pytest

from arch.bootstrap._samplers_python import (
    stationary_bootstrap_sample,
    stationary_bootstrap_sample_python,
)

try:
    from arch.bootstrap._samplers import (
        stationary_bootstrap_sample as stationary_bootstrap_sample_cython,
    )

    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


class BSData(NamedTuple):
    rng: np.random.RandomState
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    x_df: pd.DataFrame
    y_series: pd.Series
    z_df: pd.DataFrame

    func: Callable[[np.ndarray, int], float | np.ndarray]


@pytest.fixture
def bs_setup():
    rng = np.random.RandomState(1234)
    y = rng.standard_normal(1000)
    x = rng.standard_normal((1000, 2))
    z = rng.standard_normal((1000, 1))

    y_series = pd.Series(y)
    x_df = pd.DataFrame(x)
    z_df = pd.DataFrame(z)

    def func(y, axis=0):
        return y.mean(axis=axis)

    return BSData(rng, x, y, z, x_df, y_series, z_df, func)


@pytest.mark.skipif(not HAS_EXTENSION, reason="Extension not built.")
@pytest.mark.filterwarnings("ignore::arch.compat.numba.PerformanceWarning")
def test_samplers(bs_setup):
    """
    Test all three implementations are identical
    """
    indices = np.array(bs_setup.rng.randint(0, 1000, 1000), dtype=np.int64)
    u = bs_setup.rng.random_sample(1000)
    p = 0.1
    indices_orig = indices.copy()

    numba = stationary_bootstrap_sample(indices, u, p)
    indices = indices_orig.copy()
    python = stationary_bootstrap_sample_python(indices, u, p)
    indices = indices_orig.copy()
    cython = stationary_bootstrap_sample_cython(indices, u, p)
    np.testing.assert_equal(numba, cython)
    np.testing.assert_equal(numba, python)
