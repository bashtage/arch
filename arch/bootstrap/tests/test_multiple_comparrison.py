from __future__ import division
from unittest import TestCase
from nose.tools import assert_true

import pandas as pd
from pandas.util.testing import assert_series_equal
import numpy as np
from numpy import random, linspace
from numpy.testing import assert_equal, assert_raises, assert_allclose
import scipy.stats as stats

from arch.bootstrap import (StationaryBootstrap, CircularBlockBootstrap,
                            MovingBlockBootstrap)
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
        cls.benchmark_series = pd.Series(cls.benchmark, index=index)
        cls.benchmark_df = pd.DataFrame(cls.benchmark, index=index)
        cls.models_df = pd.DataFrame(cls.models, index=index)

    def test_equivalence(self):
        spa = SPA(self.benchmark, self.models, block_size=10, reps=100)
        spa.seed(23456)
        spa.compute()
        numpy_pvalues = spa.pvalues
        spa = SPA(self.benchmark_df, self.models_df, block_size=10, reps=100)
        spa.seed(23456)
        spa.compute()
        pandas_pvalues = spa.pvalues
        assert_series_equal(numpy_pvalues, pandas_pvalues)

    def test_variances_and_selection(self):
        adj_models = self.models + linspace(-2, 0.5, self.k)
        spa = SPA(self.benchmark, adj_models, block_size=10, reps=10)
        spa.seed(23456)
        spa.compute()
        variances = spa._loss_diff_var
        loss_diffs = spa._loss_diff
        demeaned = spa._loss_diff - loss_diffs.mean(0)
        t = loss_diffs.shape[0]
        kernel_weights = np.zeros(t)
        p = 1 / 10.0
        for i in range(1, t):
            kernel_weights[i] = ((1.0 - (i / t)) * ((1 - p) ** i)) + ((i / t) * ((1 - p) ** (t - i)))
        direct_vars = (demeaned ** 2).sum(0) / t
        for i in range(1, t):
            direct_vars += 2 * kernel_weights[i] * (demeaned[:t - i, :] * demeaned[i:, :]).sum(0) / t
        assert_allclose(direct_vars, variances)

        selection_criteria = -1.0 * np.sqrt((direct_vars / t) * 2 * np.log(np.log(t)))
        valid = loss_diffs.mean(0) >= selection_criteria
        assert_equal(valid, spa._valid_columns)

        # Bootstrap variances
        spa = SPA(self.benchmark, self.models, block_size=10, reps=100, nested=True)
        spa.seed(23456)
        spa.compute()
        spa.reset()
        bs = spa.bootstrap.clone(demeaned)
        variances = spa._loss_diff_var
        boostrap_variances = t * bs.var(lambda x: x.mean(0), reps=100, recenter=True)
        assert_allclose(boostrap_variances, variances)

    def test_pvalues_and_critvals(self):
        spa = SPA(self.benchmark, self.models, reps=100)
        spa.compute()
        spa.seed(23456)
        simulated_vals = spa._simulated_vals
        max_stats = np.max(simulated_vals, 0)
        max_loss_diff = np.max(spa._loss_diff.mean(0), 0)
        pvalues = np.mean(max_loss_diff <= max_stats, 0)
        pvalues = pd.Series(pvalues, index=['lower', 'consistent', 'upper'])
        assert_series_equal(pvalues, spa.pvalues)

        crit_vals = np.percentile(max_stats, 90.0, axis=0)
        crit_vals = pd.Series(crit_vals, index=['lower', 'consistent', 'upper'])
        assert_series_equal(spa.critical_values(0.10), crit_vals)

    def test_errors(self):
        spa = SPA(self.benchmark, self.models, reps=100)

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
        expected = 'SPA(studentization: asymptotic, bootstrap: ' + str(spa.bootstrap) + ')'
        assert_equal(str(spa), expected)
        expected = expected[:-1] + ', ID: ' + hex(id(spa)) + ')'
        assert_equal(spa.__repr__(), expected)

        expected = ('<strong>SPA</strong>(' +
                    '<strong>studentization</strong>: asymptotic, ' +
                    '<strong>bootstrap</strong>: ' + str(spa.bootstrap) + ')')

        assert_equal(spa._repr_html_(), expected)
        spa = SPA(self.benchmark, self.models, studentize=False, bootstrap='cbb')
        expected = 'SPA(studentization: none, bootstrap: ' + str(spa.bootstrap) + ')'
        assert_equal(str(spa), expected)

        spa = SPA(self.benchmark, self.models, nested=True, bootstrap='moving_block')
        expected = 'SPA(studentization: bootstrap, bootstrap: ' + str(spa.bootstrap) + ')'
        assert_equal(str(spa), expected)

    def test_seed_reset(self):
        spa = SPA(self.benchmark, self.models, reps=10)
        spa.seed(23456)
        initial_state = spa.bootstrap.random_state
        assert_equal(spa.bootstrap._seed, 23456)
        spa.compute()
        spa.reset()
        assert_equal(spa._pvalues, None)
        assert_equal(spa.bootstrap.random_state, initial_state)

    def test_spa_nested(self):
        spa = SPA(self.benchmark, self.models, nested=True, reps=100)
        spa.compute()

    def test_bootstrap_selection(self):
        spa = SPA(self.benchmark, self.models, bootstrap='sb')
        assert_true(isinstance(spa.bootstrap, StationaryBootstrap))
        spa = SPA(self.benchmark, self.models, bootstrap='cbb')
        assert_true(isinstance(spa.bootstrap, CircularBlockBootstrap))
        spa = SPA(self.benchmark, self.models, bootstrap='circular')
        assert_true(isinstance(spa.bootstrap, CircularBlockBootstrap))
        spa = SPA(self.benchmark, self.models, bootstrap='mbb')
        assert_true(isinstance(spa.bootstrap, MovingBlockBootstrap))
        spa = SPA(self.benchmark, self.models, bootstrap='moving block')
        assert_true(isinstance(spa.bootstrap, MovingBlockBootstrap))


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
        cls.benchmark_series = pd.Series(cls.benchmark, index=index)
        cls.benchmark_df = pd.DataFrame(cls.benchmark, index=index)
        cls.models_df = pd.DataFrame(cls.models,
                                     index=index,
                                     columns=['col_' + str(i) for i in range(cls.k)])

    def test_equivalence(self):
        adj_models = self.models - linspace(-2.0,2.0,self.k)
        stepm = StepM(self.benchmark, adj_models, size=0.20, reps=200)
        stepm.seed(23456)
        stepm.compute()


        adj_models = self.models_df - linspace(-2.0,2.0,self.k)
        stepm_pandas = StepM(self.benchmark_series, adj_models , size=0.20, reps=200)
        stepm_pandas.seed(23456)
        stepm_pandas.compute()
        stepm_pandas.superior_models
        numeric_locs = np.argwhere(adj_models.columns.isin(stepm_pandas.superior_models)).squeeze()
        numeric_locs.sort()
        assert_equal(np.array(stepm.superior_models), numeric_locs)

    def test_superior_models(self):
        adj_models = self.models - linspace(-0.4, 0.4, self.k)
        stepm = StepM(self.benchmark, adj_models, reps=120)
        stepm.compute()
        superior_models = stepm.superior_models
        print(superior_models)
        spa = SPA(self.benchmark, adj_models, reps=120)
        spa.compute()
        print(spa.pvalues)
        print(spa.critical_values(0.05))
        spa.better_models(0.05)
        adj_models = self.models_df - linspace(-3.0, 3.0, self.k)
        stepm = StepM(self.benchmark_series, adj_models, reps=120)
        stepm.compute()
        superior_models = stepm.superior_models

    def test_str_repr(self):
        stepm = StepM(self.benchmark_series, self.models, size=0.10)
        expected = 'StepM(FWER (size): 0.10, studentization: asymptotic, bootstrap: ' + str(stepm.spa.bootstrap) +')'
        assert_equal(str(stepm), expected)
        expected = expected[:-1] + ', ID: ' + hex(id(stepm)) + ')'
        assert_equal(stepm.__repr__(), expected)

        expected = ('<strong>StepM</strong>('
                    '<strong>FWER (size)</strong>: 0.10, '
                    '<strong>studentization</strong>: asymptotic, '
                    '<strong>bootstrap</strong>: ' + str(stepm.spa.bootstrap) +')')
        assert_equal(stepm._repr_html_(), expected)

        stepm = StepM(self.benchmark_series, self.models, size=0.05, studentize=False)
        expected = 'StepM(FWER (size): 0.05, studentization: none, bootstrap: ' + str(stepm.spa.bootstrap) +')'
        assert_equal(expected, str(stepm))