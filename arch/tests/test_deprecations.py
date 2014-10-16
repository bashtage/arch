from __future__ import absolute_import, division

from unittest import TestCase
import warnings

from numpy.testing import assert_equal


from arch.mean import LS, HARX, ARX, ConstantMean, ZeroMean
from arch.volatility import ARCH, HARCH, EWMAVariance, ConstantVariance, \
    RiskMetrics2006, GARCH, EGARCH
from arch.distribution import Normal, StudentsT

class TestDeprecations(TestCase):
    def test_mean(self):
        models = [LS, HARX, ARX, ConstantMean, ZeroMean]
        for model in models:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mean_model = model()
                assert_equal(len(w), 1)

    def test_volatility(self):
        processes = [ARCH, HARCH, EWMAVariance, ConstantVariance,
                     RiskMetrics2006, GARCH, EGARCH]

        for proc in processes:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                vol_proc = proc()
                assert_equal(len(w), 1)

    def test_distribution(self):
        distributions = [Normal, StudentsT]
        for dist in distributions :
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                error_distribution = dist()
                assert_equal(len(w), 1)
