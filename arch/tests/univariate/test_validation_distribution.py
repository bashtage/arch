"""Input validation of the distributions in arch.univariate.distribution."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats

from arch.univariate.distribution import SkewStudent


def test_skew_student_ppf_zero_dimensional_array():
    # SkewStudent.ppf failed an assert for a 0-d array, which is not a scalar
    # for numpy.isscalar. With no skew the distribution is a standardized t.
    nu = 8.0
    expected = stats.t.ppf(0.4, nu) * np.sqrt((nu - 2) / nu)
    dist = SkewStudent()
    result = dist.ppf(np.array(0.4), [nu, 0.0])
    assert isinstance(result, float)
    assert_allclose(result, expected, rtol=1e-12)
    assert_allclose(dist.ppf(0.4, [nu, 0.0]), expected, rtol=1e-12)
    assert_allclose(dist.ppf(np.array([0.4]), [nu, 0.0]), [expected], rtol=1e-12)
