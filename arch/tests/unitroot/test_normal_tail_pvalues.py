from math import erfc, sqrt

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.unitroot import VarianceRatio
from arch.unitroot.cointegration import DynamicOLS


@pytest.fixture
def cointegration_result():
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(100).cumsum()
    y = 2 * x + rng.standard_normal(100)
    result = DynamicOLS(y, x, trend="n", leads=0, lags=0).fit(bandwidth=1)
    result._cov.iloc[:, :] = np.eye(result.full_params.size)
    return result


@pytest.mark.parametrize("stat", [0.0, -2.0, 2.0, -8.0, 8.0, -10.0, 10.0, -12.0, 12.0])
def test_cointegration_pvalues_normal_tail(cointegration_result, stat):
    cointegration_result._params.iloc[:] = stat
    expected = erfc(abs(stat) / sqrt(2))
    assert_allclose(cointegration_result.pvalues, [expected], rtol=5e-14, atol=0)


@pytest.mark.parametrize("stat", [0.0, -2.0, 2.0, -8.0, 8.0, -10.0, 10.0, -12.0, 12.0])
def test_dynamic_ols_full_summary_normal_tail(cointegration_result, stat, monkeypatch):
    cointegration_result._params.iloc[:] = stat
    assert cointegration_result.full_params.size > cointegration_result.params.size
    # The usual four-decimal display would hide the small tail probabilities.
    monkeypatch.setattr("arch.unitroot.cointegration.pval_format", str)
    table = cointegration_result.summary(full=True).tables[1]
    actual = [float(row[4]) for row in table.data[1:]]
    expected = erfc(abs(stat) / sqrt(2))
    assert_allclose(actual, [expected, expected], rtol=5e-14, atol=0)


@pytest.mark.parametrize(
    ("y", "tail"),
    [
        pytest.param(np.arange(101.0), 1, id="positive-tail"),
        pytest.param(np.arange(101.0) % 2, -1, id="negative-tail"),
        pytest.param(np.arange(3.0), 0, id="zero-statistic"),
        pytest.param(
            np.random.default_rng(1234).standard_normal(101).cumsum(),
            None,
            id="ordinary-statistic",
        ),
    ],
)
def test_variance_ratio_normal_tail(y, tail):
    result = VarianceRatio(y, trend="n", robust=False, debiased=False)
    if tail in (-1, 1):
        assert tail * result.stat > 8
    elif tail == 0:
        assert result.stat == 0
    expected = erfc(abs(result.stat) / sqrt(2))
    assert_allclose(result.pvalue, expected, rtol=5e-14, atol=0)
