from unittest import TestCase

from nose.tools import assert_true
from numpy.testing import assert_equal, assert_raises
import numpy as np
from pandas import Series, DataFrame, date_range

from arch.utils import ensure1d, parse_dataframe, DocStringInheritor, \
    date_to_index
from arch.univariate.base import implicit_constant
from arch.compat.python import add_metaclass


class TestUtils(TestCase):
    def test_ensure1d(self):
        out = ensure1d(1.0, 'y')
        assert_equal(out, np.array([1.0]))
        out = ensure1d(np.arange(5.0), 'y')
        assert_equal(out, np.arange(5.0))
        out = ensure1d(np.arange(5.0)[:, None], 'y')
        assert_equal(out, np.arange(5.0))
        in_array = np.reshape(np.arange(16.0), (4, 4))
        assert_raises(ValueError, ensure1d, in_array, 'y')

        y = Series(np.arange(5.0))
        ys = ensure1d(y, 'y')
        assert_true(isinstance(ys, np.ndarray))
        ys = ensure1d(y, 'y', True)
        assert_true(isinstance(ys, Series))
        y = DataFrame(y)
        ys = ensure1d(y, 'y')
        assert_true(isinstance(ys, np.ndarray))
        ys = ensure1d(y, 'y', True)
        assert_true(isinstance(ys, Series))
        y = Series(np.arange(5.0), name='series')
        ys = ensure1d(y, 'y')
        assert_true(isinstance(ys, np.ndarray))
        ys = ensure1d(y, 'y', True)
        assert_true(isinstance(ys, Series))
        y = DataFrame(y)
        ys = ensure1d(y, 'y')
        assert_true(isinstance(ys, np.ndarray))
        ys = ensure1d(y, 'y', True)
        assert_true(isinstance(ys, Series))
        y = DataFrame(np.reshape(np.arange(10), (5, 2)))
        assert_raises(ValueError, ensure1d, y, 'y')

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
        assert_true(not implicit_constant(x))
        x[:, 0] = 1.0
        assert_true(implicit_constant(x))
        x = np.random.standard_normal((1000, 3))
        x[:, 0] = x[:, 0] > 0
        x[:, 1] = 1 - x[:, 0]
        assert_true(implicit_constant(x))

    def test_docstring_inheritor(self):
        @add_metaclass(DocStringInheritor)
        class A:
            """
            Docstring
            """

            def __init__(self):
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
        import datetime as dt
        index = date_to_index(date_index[0], date_index)
        assert_equal(index, 0)
        index = date_to_index(date_index[-1], date_index)
        assert_equal(index, date_index.shape[0] - 1)

        index = date_to_index('2009-08-02', date_index)
        assert_equal(index, 500)
        index = date_to_index('2009-08-04', date_index)
        assert_equal(index, 500)
        index = date_to_index('2009-08-01', date_index)
        assert_equal(index, 499)
        index = date_to_index(dt.datetime(2009, 8, 1), date_index)
        assert_equal(index, 499)
        assert_raises(ValueError, date_to_index, dt.date(2009, 8, 1),
                      date_index)
        z = y + 0.0
        z.index = np.arange(3000)
        num_index = z.index
        assert_raises(ValueError, date_to_index,
                      dt.datetime(2009, 8, 1), num_index)
