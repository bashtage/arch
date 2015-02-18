from unittest import TestCase

import pandas as pd
from numpy import random, linspace
from numpy.testing import assert_equal, assert_raises, assert_allclose
import scipy.stats as stats

from arch.bootstrap.multiple_comparrison import SPA, StepM


class TestSPA(TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(23456)
        fixed_rng = stats.chi2(10)
        cls.t = t = 1000
        cls.k = k = 500
        cls.benchmark = fixed_rng.rvs(t)
        cls.models = fixed_rng.rvs((t, k))
        index = pd.date_range('2000-01-01', periods=t)
        cls.benchmark_series = pd.Series(fixed_rng.rvs(t), index=index)
        cls.benchmark_df = pd.DataFrame(fixed_rng.rvs((t, 1)), index=index)
        cls.models_df = pd.DataFrame(fixed_rng.rvs((t, k)), index=index)

    def test_smoke(self):
        fixed_rng = stats.chi2(1)
        benchmark = fixed_rng.rvs((500, 1))
        models = fixed_rng.rvs((500, 500))
        spa = SPA(benchmark, models, block_size=10, reps=200)
        spa.compute()

    def test_smoke_pandas(self):
        fixed_rng = stats.chi2(1)
        benchmark = pd.Series(fixed_rng.rvs((500)))
        models = pd.DataFrame(fixed_rng.rvs((500, 500)),
                              columns=['col_' + str(i) for i in range(500)])
        spa = SPA(benchmark, models, block_size=10, reps=200)
        spa.compute()

    def test_spa_nested(self):
        spa = SPA(self.benchmark, self.models, nested=True, reps=100)
        spa.compute()

    def test_spa_errors(self):
        spa = SPA(self.benchmark, self.models)

        with assert_raises(RuntimeError):
            spa.pvalues

        assert_raises(RuntimeError, spa.critical_values)
        assert_raises(RuntimeError, spa.better_models)

        assert_raises(ValueError, SPA, self.benchmark, self.models, bootstrap='unknown')
        spa.compute()
        assert_raises(ValueError, spa.better_models, pvalue_type='unknown')
        assert_raises(ValueError, spa.critical_values, pvalue=1.0)

    def test_str_repr(self):
        spa = SPA(self.benchmark, self.models)
        print(spa)
        print(spa.__repr__())
        print(spa._repr_html_())
        spa = SPA(self.benchmark, self.models, studentize=False, bootstrap='cbb')
        print(spa)
        spa = SPA(self.benchmark, self.models, nested=True, bootstrap='moving_block')
        print(spa)

    def test_seed_reset(self):
        spa = SPA(self.benchmark, self.models, reps=10)
        spa.seed(23456)
        initial_state = spa.bootstrap.random_state
        spa.compute()
        spa.reset()
        assert_equal(spa._pvalues, None)
        assert_equal(spa.bootstrap.random_state, initial_state)

class TestStepM(TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(23456)
        fixed_rng = stats.chi2(10)
        cls.t = t = 1000
        cls.k = k = 500
        cls.benchmark = fixed_rng.rvs(t)
        cls.models = fixed_rng.rvs((t, k))
        index = pd.date_range('2000-01-01', periods=t)
        cls.benchmark_series = pd.Series(fixed_rng.rvs(t), index=index)
        cls.benchmark_df = pd.DataFrame(fixed_rng.rvs((t, 1)), index=index)
        cls.models_df = pd.DataFrame(fixed_rng.rvs((t, k)), index=index)

    def test_smoke(self):
        stepm = StepM(self.benchmark, self.models, size=0.66, reps=200)
        stepm.compute()

    def test_smoke_pandas(self):
        stepm = StepM(self.benchmark_series, self.models, size=0.66, reps=200)
        stepm.compute()
        stepm = StepM(self.benchmark_df, self.models_df, size=0.66, reps=150)
        stepm.compute()
        superior_models = stepm.superior_models


    def test_superior_models(self):
        adj_models = self.models - linspace(-0.4, 0.4, self.k)
        stepm = StepM(self.benchmark, adj_models, reps=200)
        stepm.compute()
        superior_models = stepm.superior_models
        print(superior_models)
        spa = SPA(self.benchmark, adj_models, reps=200)
        spa.compute()
        print(spa.pvalues)
        print(spa.critical_values(0.05))
        spa.better_models(0.05)
        adj_models = self.models_df - linspace(-3.0, 3.0, self.k)
        stepm = StepM(self.benchmark_series, adj_models, reps=200)
        stepm.compute()
        superior_models = stepm.superior_models

    def test_str_repr(self):
        stepm = StepM(self.benchmark_series, self.models, size=0.66)
        print(stepm)
        print(stepm.__repr__())
        print(stepm._repr_html_())
        stepm = StepM(self.benchmark_series, self.models, size=0.66, studentize=False)
        print(stepm)
