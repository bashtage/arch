import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.bootstrap.multiple_comparison import MCS, SPA, StepM


@pytest.mark.parametrize(
    ("method", "canonical"),
    [("r", "r"), ("R", "r"), ("max", "max"), ("MAX", "max"), ("Max", "max")],
)
def test_mcs_method_case_insensitive(method, canonical):
    losses = np.random.default_rng(0).standard_normal((50, 3))
    mcs = MCS(losses, 0.1, reps=10, method=method, seed=1)
    mcs.compute()
    reference = MCS(losses, 0.1, reps=10, method=canonical, seed=1)
    reference.compute()
    assert_allclose(mcs.pvalues["Pvalue"], reference.pvalues["Pvalue"])
    assert list(mcs.pvalues.index) == list(reference.pvalues.index)


@pytest.mark.parametrize(
    ("method", "error"),
    [("foo", ValueError), ("", ValueError), ("maximum", ValueError)]
    + [(None, TypeError), (1, TypeError)],
)
def test_mcs_method_invalid(method, error):
    losses = np.random.default_rng(0).standard_normal((50, 3))
    with pytest.raises(error, match="method must be one of 'r' or 'max'"):
        MCS(losses, 0.1, reps=10, method=method)


def test_mcs_method_changed_after_construction():
    losses = np.random.default_rng(0).standard_normal((50, 3))
    mcs = MCS(losses, 0.1, reps=10, seed=1)
    mcs.method = "other"
    with pytest.raises(ValueError, match="method must be one of 'r' or 'max'"):
        mcs.compute()


@pytest.mark.parametrize(
    ("bootstrap", "block_size", "error"),
    [
        ("sb", 0, ValueError),
        ("cbb", -1, ValueError),
        ("mbb", 0, ValueError),
        ("cbb", 2.5, TypeError),
        ("mbb", 2.5, TypeError),
    ],
)
@pytest.mark.parametrize("procedure", [MCS, SPA, StepM])
def test_multiple_comparison_block_size(procedure, bootstrap, block_size, error):
    losses = np.random.default_rng(0).standard_normal((50, 3))
    if procedure is MCS:
        args = (losses, 0.1)
    else:
        args = (losses[:, 0], losses[:, 1:])
    with pytest.raises(error, match="block_size must be"):
        procedure(*args, block_size=block_size, bootstrap=bootstrap)
