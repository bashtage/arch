from unittest import TestCase
import scipy.stats as stats
from arch.bootstrap.multiple_comparrison import SPA, StepM
import pandas as pd


class TestSPA(TestCase):
    def test_smoke(self):
        fixed_rng = stats.chi2(1)
        benchmark = fixed_rng.rvs((500, 1))
        models = fixed_rng.rvs((500, 500))
        spa = SPA(benchmark, models, block_size=10)
        spa.compute()

    def test_smoke_pandas(self):
        fixed_rng = stats.chi2(1)
        benchmark = pd.Series(fixed_rng.rvs((500)))
        models = pd.DataFrame(fixed_rng.rvs((500, 500)),
                              columns=['col_' + str(i) for i in range(500)])
        spa = SPA(benchmark, models, block_size=10)
        spa.compute()


class TestStepM(TestCase):
    def test_smoke(self):
        fixed_rng = stats.chi2(1)
        benchmark = fixed_rng.rvs((500, 1))
        models = fixed_rng.rvs((500, 500))
        stepm = StepM(benchmark, models, size=0.66)
        stepm.compute()

    def test_smoke_pandas(self):
        fixed_rng = stats.chi2(1)
        benchmark = pd.Series(fixed_rng.rvs((500)))
        models = pd.DataFrame(fixed_rng.rvs((500, 500)),
                              columns=['col_' + str(i) for i in range(500)])
        stepm = StepM(benchmark, models, size=0.66)
        stepm.compute()