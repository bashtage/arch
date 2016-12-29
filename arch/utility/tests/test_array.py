import datetime as dt
import os
from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
from pandas import Series, DataFrame, date_range

from pandas import Timedelta

import pytest
from arch import doc
from arch.utility.array import ensure1d, parse_dataframe, DocStringInheritor, \
    date_to_index, find_index, cutoff_to_index
from arch.univariate.base import implicit_constant
from arch.compat.python import add_metaclass, long


class TestUtils(TestCase):
    def test_ensure1d(self):
        out = ensure1d(1.0, 'y')
        assert_equal(out, np.array([1.0]))
        out = ensure1d(np.arange(5.0), 'y')
        assert_equal(out, np.arange(5.0))
        out = ensure1d(np.arange(5.0)[:, None], 'y')
        assert_equal(out, np.arange(5.0))
        in_array = np.reshape(np.arange(16.0), (4, 4))
        with pytest.raises(ValueError):
            ensure1d(in_array, 'y')

        y = Series(np.arange(5.0))
        ys = ensure1d(y, 'y')
        assert isinstance(ys, np.ndarray)
        ys = ensure1d(y, 'y', True)
        assert isinstance(ys, Series)
        y = DataFrame(y)
        ys = ensure1d(y, 'y')
        assert isinstance(ys, np.ndarray)
        ys = ensure1d(y, 'y', True)
        assert isinstance(ys, Series)
        y = Series(np.arange(5.0), name='series')
        ys = ensure1d(y, 'y')
        assert isinstance(ys, np.ndarray)
        ys = ensure1d(y, 'y', True)
        assert isinstance(ys, Series)
        y = DataFrame(y)
        ys = ensure1d(y, 'y')
        assert isinstance(ys, np.ndarray)
        ys = ensure1d(y, 'y', True)
        assert isinstance(ys, Series)
        y = DataFrame(np.reshape(np.arange(10), (5, 2)))
        with pytest.raises(ValueError):
            ensure1d(y, 'y')

    def test_parse_dataframe(self):
        s = Series(np.arange(10.0), name='variable')
        out = parse_dataframe(s, 'y')
        assert_equal(out[1], np.arange(10.0))
        assert_equal(out[0], ['variable'])
        df = DataFrame(s)
        out = parse_dataframe(df, 'y')
        assert_equal(out[1], np.arange(10.0))
        assert_equal(out[0], ['variable'])
        out = parse_dataframe(np.arange(10.0), 'y')
        assert_equal(out[1], np.arange(10.0))
        assert_equal(out[0], ['y'])

    def test_implicit_constant(self):
        x = np.random.standard_normal((1000, 2))
        assert not implicit_constant(x)
        x[:, 0] = 1.0
        assert implicit_constant(x)
        x = np.random.standard_normal((1000, 3))
        x[:, 0] = x[:, 0] > 0
        x[:, 1] = 1 - x[:, 0]
        assert implicit_constant(x)

    def test_docstring_inheritor(self):
        @add_metaclass(DocStringInheritor)
        class A(object):
            """
            Docstring
            """
            pass

        class B(A):
            pass

        ds = """
            Docstring
            """
        assert_equal(B.__doc__, ds)

    def test_date_to_index(self):
        dr = date_range('20000101', periods=3000, freq='W')
        y = Series(np.arange(3000.0), index=dr)
        date_index = y.index

        index = date_to_index(date_index[0], date_index)
        assert_equal(index, 0)
        index = date_to_index(date_index[-1], date_index)
        assert_equal(index, date_index.shape[0] - 1)

        index = date_to_index('2009-08-02', date_index)
        assert_equal(index, 500)
        index = date_to_index('2009-08-04', date_index)
        assert_equal(index, 501)
        index = date_to_index('2009-08-01', date_index)
        assert_equal(index, 500)
        index = date_to_index(dt.datetime(2009, 8, 1), date_index)
        assert_equal(index, 500)
        with pytest.raises(ValueError):
            date_to_index(dt.date(2009, 8, 1), date_index)
        z = y + 0.0
        z.index = np.arange(3000)
        num_index = z.index
        with pytest.raises(ValueError):
            date_to_index(dt.datetime(2009, 8, 1), num_index)

    def test_date_to_index_timestamp(self):
        dr = date_range('20000101', periods=3000, freq='W')
        y = Series(np.arange(3000.0), index=dr)
        date_index = y.index
        date = y.index[1000]

        try:
            date_pydt = date.to_pydatetime()
        except AttributeError:
            # Old pandas
            date_pydt = date.to_datetime()

        date_npdt = date.to_datetime64()
        date_str = date_pydt.strftime('%Y-%m-%d')
        index = date_to_index(date, date_index)
        index_pydt = date_to_index(date_pydt, date_index)
        index_npdt = date_to_index(date_npdt, date_index)
        index_str = date_to_index(date_str, date_index)
        assert_equal(index, 1000)
        assert_equal(index, index_npdt)
        assert_equal(index, index_pydt)
        assert_equal(index, index_str)

    def test_(self):
        dr = date_range('20000101', periods=3000, freq='W')
        y = Series(np.arange(3000.0), index=dr)
        date_index = y.index

        date = date_index[1000] + Timedelta(1, 'D')
        date_pydt = date.to_pydatetime()
        date_npdt = date.to_datetime64()
        date_str = date_pydt.strftime('%Y-%m-%d')
        index = date_to_index(date, date_index)
        index_pydt = date_to_index(date_pydt, date_index)
        index_npdt = date_to_index(date_npdt, date_index)
        index_str = date_to_index(date_str, date_index)
        assert_equal(index, 1001)
        assert_equal(index, index_npdt)
        assert_equal(index, index_pydt)
        assert_equal(index, index_str)

        date = date_index[0] - Timedelta(1, 'D')
        index = date_to_index(date, date_index)
        assert_equal(index, 0)

        date_pydt = date.to_pydatetime()
        date_npdt = date.to_datetime64()
        date_str = date_pydt.strftime('%Y-%m-%d')
        index_pydt = date_to_index(date_pydt, date_index)
        index_npdt = date_to_index(date_npdt, date_index)
        index_str = date_to_index(date_str, date_index)
        assert_equal(index, index_npdt)
        assert_equal(index, index_pydt)
        assert_equal(index, index_str)

    def test_cutoff_to_index(self):
        dr = date_range('20000101', periods=3000, freq='W')
        y = Series(np.arange(3000.0), index=dr)
        date_index = y.index
        assert cutoff_to_index(1000, date_index, 0) == 1000
        assert cutoff_to_index(long(1000), date_index, 0) == 1000
        assert cutoff_to_index(np.int16(1000), date_index, 0) == 1000
        assert cutoff_to_index(np.int64(1000), date_index, 0) == 1000
        assert cutoff_to_index(date_index[1000], date_index, 0) == 1000
        assert cutoff_to_index(None, date_index, 1000) == 1000

    def test_find_index(self):
        index = date_range('2000-01-01', periods=5000)
        series = Series(np.arange(len(index)), index=index, name='test')
        df = DataFrame(series)
        assert_equal(find_index(series, '2000-01-01'), 0)
        assert_equal(find_index(series, series.index[0]), 0)
        assert_equal(find_index(series, series.index[3000]), 3000)
        assert_equal(find_index(series, series.index[3000].to_pydatetime()),
                     3000)
        npy_date = np.datetime64(series.index[3000].to_pydatetime())
        found_loc = find_index(series, npy_date)
        assert_equal(found_loc, 3000)
        with pytest.raises(ValueError):
            find_index(series, 'bad-date')
        with pytest.raises(ValueError):
            find_index(series, '1900-01-01')

        assert_equal(find_index(df, '2000-01-01'), 0)
        assert_equal(find_index(df, df.index[0]), 0)
        assert_equal(find_index(df, df.index[3000]), 3000)
        assert_equal(find_index(df, df.index[3000].to_pydatetime()), 3000)
        found_loc = find_index(df,
                               np.datetime64(df.index[3000].to_pydatetime()))
        assert_equal(found_loc, 3000)
        with pytest.raises(ValueError):
            find_index(df, 'bad-date')
        with pytest.raises(ValueError):
            find_index(df, '1900-01-01')


@pytest.mark.skipif(os.name != 'nt', reason='XVFB is broken on travis')
class TestDoc(TestCase):
    def test_doc(self):
        doc()
