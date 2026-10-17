"""
Input validation of the cointegration tests and estimators.

The table-driven rows are built on arch.tests.validation_table.
"""

import numpy as np
import pytest

from arch.unitroot.cointegration import phillips_ouliaris

NOBS = 120
RNG = np.random.default_rng(20261017)
X = np.cumsum(RNG.standard_normal((NOBS, 2)), 0)
Y = X @ np.array([1.0, 0.5]) + RNG.standard_normal(NOBS)

TREND_TEXT = {
    "n": "No Trend",
    "c": "Constant",
    "ct": "Constant and Linear Time Trend",
    "ctt": "Constant, Linear and Quadratic Time Trends",
}


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("test_type", ["Za", "Zt", "Pu", "Pz"])
def test_phillips_ouliaris_reports_trend(trend, test_type):
    # The results always reported trend "c" (and the summary "Constant")
    res = phillips_ouliaris(Y, X, trend, test_type=test_type)
    assert res.trend == trend
    lines = [line.strip() for line in res.summary().as_text().splitlines()]
    assert f"Trend: {TREND_TEXT[trend]}" in lines
    assert f"Trend: {trend}" in str(res)
