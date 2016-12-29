from __future__ import division

from unittest import TestCase

import numpy as np
from numpy import random, linspace
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
from pandas.util.testing import assert_series_equal, assert_frame_equal
import pytest
import scipy.stats as stats

from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap, \
    MovingBlockBootstrap
from arch.bootstrap.multiple_comparrison import SPA, StepM, MCS


class TestSPA(TestCase):
    @classmethod
    def setup_class(cls):
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
            kernel_weights[i] = ((1.0 - (i / t)) * ((1 - p) ** i)) + \
                                ((i / t) * ((1 - p) ** (t - i)))
        direct_vars = (demeaned ** 2).sum(0) / t
        for i in range(1, t):
            direct_vars += 2 * kernel_weights[i] * \
                           (demeaned[:t - i, :] * demeaned[i:, :]).sum(0) / t
        assert_allclose(direct_vars, variances)

        selection_criteria = -1.0 * np.sqrt((direct_vars / t) * 2 *
                                            np.log(np.log(t)))
        valid = loss_diffs.mean(0) >= selection_criteria
        assert_equal(valid, spa._valid_columns)

        # Bootstrap variances
        spa = SPA(self.benchmark, self.models, block_size=10, reps=100,
                  nested=True)
        spa.seed(23456)
        spa.compute()
        spa.reset()
        bs = spa.bootstrap.clone(demeaned)
        variances = spa._loss_diff_var
        bootstrap_variances = t * bs.var(lambda x: x.mean(0), reps=100,
                                         recenter=True)
        assert_allclose(bootstrap_variances, variances)

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
        crit_vals = pd.Series(crit_vals,
                              index=['lower', 'consistent', 'upper'])
        assert_series_equal(spa.critical_values(0.10), crit_vals)

    def test_errors(self):
        spa = SPA(self.benchmark, self.models, reps=100)

        with pytest.raises(RuntimeError):
            spa.pvalues

        with pytest.raises(RuntimeError):
            spa.critical_values()
        with pytest.raises(RuntimeError):
            spa.better_models()

        with pytest.raises(ValueError):
            SPA(self.benchmark, self.models, bootstrap='unknown')
        spa.compute()
        with pytest.raises(ValueError):
            spa.better_models(pvalue_type='unknown')
        with pytest.raises(ValueError):
            spa.critical_values(pvalue=1.0)

    def test_str_repr(self):
        spa = SPA(self.benchmark, self.models)
        expected = 'SPA(studentization: asymptotic, bootstrap: ' + \
                   str(spa.bootstrap) + ')'
        assert_equal(str(spa), expected)
        expected = expected[:-1] + ', ID: ' + hex(id(spa)) + ')'
        assert_equal(spa.__repr__(), expected)

        expected = ('<strong>SPA</strong>(' +
                    '<strong>studentization</strong>: asymptotic, ' +
                    '<strong>bootstrap</strong>: ' + str(spa.bootstrap) +
                    ', <strong>ID</strong>: ' + hex(id(spa)) + ')')

        assert_equal(spa._repr_html_(), expected)
        spa = SPA(self.benchmark, self.models, studentize=False,
                  bootstrap='cbb')
        expected = 'SPA(studentization: none, bootstrap: ' + \
                   str(spa.bootstrap) + ')'
        assert_equal(str(spa), expected)

        spa = SPA(self.benchmark, self.models, nested=True,
                  bootstrap='moving_block')
        expected = 'SPA(studentization: bootstrap, bootstrap: ' + \
                   str(spa.bootstrap) + ')'
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
        assert isinstance(spa.bootstrap, StationaryBootstrap)
        spa = SPA(self.benchmark, self.models, bootstrap='cbb')
        assert isinstance(spa.bootstrap, CircularBlockBootstrap)
        spa = SPA(self.benchmark, self.models, bootstrap='circular')
        assert isinstance(spa.bootstrap, CircularBlockBootstrap)
        spa = SPA(self.benchmark, self.models, bootstrap='mbb')
        assert isinstance(spa.bootstrap, MovingBlockBootstrap)
        spa = SPA(self.benchmark, self.models, bootstrap='moving block')
        assert isinstance(spa.bootstrap, MovingBlockBootstrap)

    def test_single_model(self):
        spa = SPA(self.benchmark, self.models[:, 0])
        spa.compute()

        spa = SPA(self.benchmark_series, self.models_df.iloc[:, 0])
        spa.compute()


class TestStepM(TestCase):
    @classmethod
    def setup_class(cls):
        random.seed(23456)
        fixed_rng = stats.chi2(10)
        cls.t = t = 1000
        cls.k = k = 500
        cls.benchmark = fixed_rng.rvs(t)
        cls.models = fixed_rng.rvs((t, k))
        index = pd.date_range('2000-01-01', periods=t)
        cls.benchmark_series = pd.Series(cls.benchmark, index=index)
        cls.benchmark_df = pd.DataFrame(cls.benchmark, index=index)
        cols = ['col_' + str(i) for i in range(cls.k)]
        cls.models_df = pd.DataFrame(cls.models,
                                     index=index,
                                     columns=cols)

    def test_equivalence(self):
        adj_models = self.models - linspace(-2.0, 2.0, self.k)
        stepm = StepM(self.benchmark, adj_models, size=0.20, reps=200)
        stepm.seed(23456)
        stepm.compute()

        adj_models = self.models_df - linspace(-2.0, 2.0, self.k)
        stepm_pandas = StepM(self.benchmark_series, adj_models, size=0.20,
                             reps=200)
        stepm_pandas.seed(23456)
        stepm_pandas.compute()
        stepm_pandas.superior_models
        members = adj_models.columns.isin(stepm_pandas.superior_models)
        numeric_locs = np.argwhere(members).squeeze()
        numeric_locs.sort()
        assert_equal(np.array(stepm.superior_models), numeric_locs)

    def test_superior_models(self):
        adj_models = self.models - linspace(-0.4, 0.4, self.k)
        stepm = StepM(self.benchmark, adj_models, reps=120)
        stepm.compute()
        superior_models = stepm.superior_models
        assert len(superior_models) > 0
        spa = SPA(self.benchmark, adj_models, reps=120)
        spa.compute()
        spa.pvalues
        spa.critical_values(0.05)
        spa.better_models(0.05)
        adj_models = self.models_df - linspace(-3.0, 3.0, self.k)
        stepm = StepM(self.benchmark_series, adj_models, reps=120)
        stepm.compute()
        superior_models = stepm.superior_models
        assert len(superior_models) > 0

    def test_str_repr(self):
        stepm = StepM(self.benchmark_series, self.models, size=0.10)
        expected = 'StepM(FWER (size): 0.10, studentization: ' \
                   'asymptotic, bootstrap: ' + str(stepm.spa.bootstrap) + ')'
        assert_equal(str(stepm), expected)
        expected = expected[:-1] + ', ID: ' + hex(id(stepm)) + ')'
        assert_equal(stepm.__repr__(), expected)

        expected = ('<strong>StepM</strong>('
                    '<strong>FWER (size)</strong>: 0.10, '
                    '<strong>studentization</strong>: asymptotic, '
                    '<strong>bootstrap</strong>: ' + str(stepm.spa.bootstrap) +
                    ', ' + '<strong>ID</strong>: ' + hex(id(stepm)) + ')')

        assert_equal(stepm._repr_html_(), expected)

        stepm = StepM(self.benchmark_series, self.models, size=0.05,
                      studentize=False)
        expected = 'StepM(FWER (size): 0.05, studentization: none, ' \
                   'bootstrap: ' + str(stepm.spa.bootstrap) + ')'
        assert_equal(expected, str(stepm))

    def test_single_model(self):
        stepm = StepM(self.benchmark, self.models[:, 0], size=0.10)
        stepm.compute()

        stepm = StepM(self.benchmark_series, self.models_df.iloc[:, 0])
        stepm.compute()

    def test_all_superior(self):
        adj_models = self.models - 100.0
        stepm = StepM(self.benchmark, adj_models, size=0.10)
        stepm.compute()
        assert_equal(len(stepm.superior_models), self.models.shape[1])

    def test_errors(self):
        stepm = StepM(self.benchmark, self.models, size=0.10)
        with pytest.raises(RuntimeError):
            stepm.superior_models

    def test_exact_ties(self):
        adj_models = self.models_df - 100.0
        adj_models.iloc[:, :2] -= adj_models.iloc[:, :2].mean()
        adj_models.iloc[:, :2] += self.benchmark_df.mean().iloc[0]
        stepm = StepM(self.benchmark_df, adj_models, size=0.10)
        stepm.compute()
        assert_equal(len(stepm.superior_models), self.models.shape[1] - 2)


class TestMCS(TestCase):
    @classmethod
    def setup_class(cls):
        random.seed(23456)
        fixed_rng = stats.chi2(10)
        cls.t = t = 1000
        cls.k = k = 50
        cls.losses = fixed_rng.rvs((t, k))
        index = pd.date_range('2000-01-01', periods=t)
        cls.losses_df = pd.DataFrame(cls.losses, index=index)

    def test_r_method(self):
        def r_step(losses, indices):
            # A basic but direct implementation of the r method
            t, k = losses.shape
            b = len(indices)
            mean_diffs = losses.mean(0)
            loss_diffs = np.zeros((k, k))
            variances = np.zeros((k, k))
            bs_diffs = np.zeros(b)
            stat_candidates = []
            for i in range(k):
                for j in range(i, k):
                    if i == j:
                        variances[i, i] = 1.0
                        loss_diffs[i, j] = 0.0
                        continue
                    loss_diffs_vec = losses[:, i] - losses[:, j]
                    loss_diffs_vec = loss_diffs_vec - loss_diffs_vec.mean()
                    loss_diffs[i, j] = mean_diffs[i] - mean_diffs[j]
                    loss_diffs[j, i] = mean_diffs[j] - mean_diffs[i]
                    for n in range(b):
                        # Compute bootstraped versions
                        bs_diffs[n] = loss_diffs_vec[indices[n]].mean()
                    variances[j, i] = variances[i, j] = (bs_diffs ** 2).mean()
                    std_diffs = np.abs(bs_diffs) / np.sqrt(variances[i, j])
                    stat_candidates.append(std_diffs)
            stat_candidates = np.array(stat_candidates).T
            stat_distn = np.max(stat_candidates, 1)
            std_loss_diffs = loss_diffs / np.sqrt(variances)
            stat = np.max(std_loss_diffs)
            pval = np.mean(stat <= stat_distn)
            loc = np.argwhere(std_loss_diffs == stat)
            drop_index = loc.flat[0]
            return pval, drop_index

        losses = self.losses[:, :10]  # Limit size
        mcs = MCS(losses, 0.05, reps=200)
        mcs.seed(23456)
        mcs.compute()
        m = 5  # Number of direct
        pvals = np.zeros(m) * np.nan
        indices = np.zeros(m) * np.nan
        for i in range(m):
            removed = list(indices[np.isfinite(indices)])
            include = list(set(list(range(10))).difference(removed))
            include.sort()
            pval, drop_index = r_step(losses[:, np.array(include)],
                                      mcs._bootsrap_indices)
            pvals[i] = pval if i == 0 else np.max([pvals[i - 1], pval])
            indices[i] = include[drop_index]
        direct = pd.DataFrame(pvals,
                              index=np.array(indices, dtype=np.int64),
                              columns=['Pvalue'])
        direct.index.name = 'Model index'
        assert_frame_equal(mcs.pvalues.iloc[:m], direct)

    def test_max_method(self):
        def max_step(losses, indices):
            # A basic but direct implementation of the max method
            t, k = losses.shape
            b = len(indices)
            loss_errors = losses - losses.mean(0)
            stats = np.zeros((b, k))
            for n in range(b):
                # Compute bootstraped versions
                bs_loss_errors = loss_errors[indices[n]]
                stats[n] = bs_loss_errors.mean(0) - bs_loss_errors.mean()
            variances = (stats ** 2).mean(0)
            std_devs = np.sqrt(variances)
            stat_dist = np.max(stats / std_devs, 1)

            test_stat = (losses.mean(0) - losses.mean())
            std_test_stat = test_stat / std_devs
            test_stat = np.max(std_test_stat)
            pval = (test_stat < stat_dist).mean()
            drop_index = np.argwhere(std_test_stat == test_stat).squeeze()
            return pval, drop_index, std_devs

        losses = self.losses[:, :10]  # Limit size
        mcs = MCS(losses, 0.05, reps=200, method='max')
        mcs.seed(23456)
        mcs.compute()
        m = 8  # Number of direct
        pvals = np.zeros(m) * np.nan
        indices = np.zeros(m) * np.nan
        for i in range(m):
            removed = list(indices[np.isfinite(indices)])
            include = list(set(list(range(10))).difference(removed))
            include.sort()
            pval, drop_index, std_devs = max_step(losses[:, np.array(include)],
                                                  mcs._bootsrap_indices)
            pvals[i] = pval if i == 0 else np.max([pvals[i - 1], pval])
            indices[i] = include[drop_index]
        direct = pd.DataFrame(pvals,
                              index=np.array(indices, dtype=np.int64),
                              columns=['Pvalue'])
        direct.index.name = 'Model index'
        assert_frame_equal(mcs.pvalues.iloc[:m], direct)

    def test_output_types(self):
        mcs = MCS(self.losses_df, 0.05, reps=100, block_size=10, method='r')
        mcs.compute()
        assert_equal(type(mcs.included), list)
        assert_equal(type(mcs.excluded), list)
        assert isinstance(mcs.pvalues, pd.DataFrame)

    def test_errors(self):
        with pytest.raises(ValueError):
            MCS(self.losses[:, 1], 0.05)
        mcs = MCS(self.losses, 0.05, reps=100, block_size=10, method='max',
                  bootstrap='circular')
        mcs.compute()
        mcs = MCS(self.losses, 0.05, reps=100, block_size=10, method='max',
                  bootstrap='moving block')
        mcs.compute()
        with pytest.raises(ValueError):
            MCS(self.losses, 0.05, bootstrap='unknown')

    def test_str_repr(self):
        mcs = MCS(self.losses, 0.05)
        expected = 'MCS(size: 0.05, bootstrap: ' + str(mcs.bootstrap) + ')'
        assert_equal(str(mcs), expected)
        expected = expected[:-1] + ', ID: ' + hex(id(mcs)) + ')'
        assert_equal(mcs.__repr__(), expected)
        expected = ('<strong>MCS</strong>(' +
                    '<strong>size</strong>: 0.05, ' +
                    '<strong>bootstrap</strong>: ' + str(mcs.bootstrap) +
                    ', ' + '<strong>ID</strong>: ' + hex(id(mcs)) + ')')
        assert_equal(mcs._repr_html_(), expected)

    def test_all_models_have_pval(self):
        losses = self.losses_df.iloc[:, :20]
        mcs = MCS(losses, 0.05, reps=200)
        mcs.seed(23456)
        mcs.compute()
        nan_locs = np.isnan(mcs.pvalues.iloc[:, 0])
        assert not nan_locs.any()

    def test_exact_ties(self):
        losses = self.losses_df.iloc[:, :20].copy()
        tied_mean = losses.mean().median()
        losses.iloc[:, 10:] -= losses.iloc[:, 10:].mean()
        losses.iloc[:, 10:] += tied_mean
        mcs = MCS(losses, 0.05, reps=200)
        mcs.seed(23456)
        mcs.compute()
