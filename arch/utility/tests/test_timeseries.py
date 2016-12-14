from __future__ import division

from unittest import TestCase
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.random import randn
from numpy.testing import (assert_equal, assert_array_almost_equal,
                           assert_array_equal)
from pandas.util.testing import assert_frame_equal, assert_produces_warning

from arch.utility.timeseries import add_trend, ColumnNameConflict


class TestAddTrend(TestCase):
    def test_add_trend_prepend(self):
        n = 10
        x = randn(n, 1)
        trend_1 = add_trend(x, trend='ct', prepend=True)
        trend_2 = add_trend(x, trend='ct', prepend=False)
        assert_equal(trend_1[:, :2], trend_2[:, 1:])

    def test_add_time_trend_dataframe(self):
        n = 10
        x = randn(n, 1)
        x = pd.DataFrame(x, columns=['col1'])
        trend_1 = add_trend(x, trend='t')
        assert_array_almost_equal(np.asarray(trend_1['trend']),
                                  np.arange(1.0, n + 1))

    def test_add_trend_prepend_dataframe(self):
        n = 10
        x = randn(n, 1)
        x = pd.DataFrame(x, columns=['col1'])
        trend_1 = add_trend(x, trend='ct', prepend=True)
        trend_2 = add_trend(x, trend='ct', prepend=False)
        assert_frame_equal(trend_1.iloc[:, :2], trend_2.iloc[:, 1:])

    def test_add_trend_duplicate_name(self):
        x = pd.DataFrame(np.zeros((10, 1)), columns=['trend'])
        with warnings.catch_warnings(record=True) as w:
            assert_produces_warning(add_trend(x, trend='ct'),
                                    ColumnNameConflict)
            y = add_trend(x, trend='ct')
            # should produce a single warning

        assert len(w) > 0
        assert 'const' in y.columns
        assert 'trend_0' in y.columns

    def test_add_trend_c(self):
        x = np.zeros((10, 1))
        y = add_trend(x, trend='c')
        assert np.all(y[:, 1] == 1.0)

    def test_add_trend_ct(self):
        n = 20
        x = np.zeros((20, 1))
        y = add_trend(x, trend='ct')
        assert np.all(y[:, 1] == 1.0)
        assert_equal(y[0, 2], 1.0)
        assert_array_almost_equal(np.diff(y[:, 2]), np.ones((n - 1)))

    def test_add_trend_ctt(self):
        n = 10
        x = np.zeros((n, 1))
        y = add_trend(x, trend='ctt')
        assert np.all(y[:, 1] == 1.0)
        assert y[0, 2] == 1.0
        assert_array_almost_equal(np.diff(y[:, 2]), np.ones((n - 1)))
        assert y[0, 3] == 1.0
        assert_array_almost_equal(np.diff(y[:, 3]),
                                  np.arange(3.0, 2.0 * n, 2.0))

    def test_add_trend_t(self):
        n = 20
        x = np.zeros((20, 1))
        y = add_trend(x, trend='t')
        assert y[0, 1] == 1.0
        assert_array_almost_equal(np.diff(y[:, 1]), np.ones((n - 1)))

    def test_add_trend_no_input(self):
        n = 100
        y = add_trend(x=None, trend='ct', nobs=n)
        assert np.all(y[:, 0] == 1.0)
        assert y[0, 1] == 1.0
        assert_array_almost_equal(np.diff(y[:, 1]), np.ones((n - 1)))

    def test_skip_constant(self):
        x = np.ones((100, 1))
        appended = add_trend(x, trend='c', has_constant='add')
        assert_array_equal(np.ones((100, 2)), appended)
        appended = add_trend(x, trend='c', has_constant='skip')
        assert_array_equal(np.ones((100, 1)), appended)

    def test_errors(self):
        n = 100
        with pytest.raises(ValueError):
            add_trend(x=None, trend='unknown', nobs=n)
        with pytest.raises(ValueError):
            add_trend(x=None, trend='ct')
        x = np.ones((100, 1))
        with pytest.raises(ValueError):
            add_trend(x, trend='ct', has_constant='raise')
