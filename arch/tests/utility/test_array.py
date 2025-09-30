from abc import abstractmethod
import datetime as dt

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_equal
from pandas import DataFrame, Series, Timedelta, date_range
import pytest

from arch import doc
from arch.univariate.base import implicit_constant
from arch.utility.array import (
    ConcreteClassMeta,
    DocStringInheritor,
    cutoff_to_index,
    date_to_index,
    ensure1d,
    ensure2d,
    find_index,
    parse_dataframe,
    to_array_1d,
)


@pytest.fixture
def rng():
    return RandomState(12345)


def test_ensure1d():
    out = ensure1d(1.0, "y")
    assert_equal(out, np.array([1.0]))
    out = ensure1d(np.arange(5.0), "y")
    assert_equal(out, np.arange(5.0))
    out = ensure1d(np.arange(5.0)[:, None], "y")
    assert_equal(out, np.arange(5.0))
    in_array = np.reshape(np.arange(16.0), (4, 4))
    with pytest.raises(ValueError, match=r"y must be squeezable"):
        ensure1d(in_array, "y")

    y = Series(np.arange(5.0))
    ys = ensure1d(y, "y")
    assert isinstance(ys, np.ndarray)
    ys = ensure1d(y, "y", True)
    assert isinstance(ys, Series)
    y = DataFrame(y)
    ys = ensure1d(y, "y")
    assert isinstance(ys, np.ndarray)
    ys = ensure1d(y, "y", True)
    assert isinstance(ys, Series)
    y.columns = [1]
    ys = ensure1d(y, "y", True)
    assert isinstance(ys, Series)
    assert ys.name == "1"
    y = Series(np.arange(5.0), name="series")
    ys = ensure1d(y, "y")
    assert isinstance(ys, np.ndarray)
    ys = ensure1d(y, "y", True)
    assert isinstance(ys, Series)
    y = DataFrame(y)
    ys = ensure1d(y, "y")
    assert isinstance(ys, np.ndarray)
    ys = ensure1d(y, "y", True)
    assert isinstance(ys, Series)
    ys.name = 1
    ys = ensure1d(ys, None, True)
    assert isinstance(ys, Series)
    assert ys.name == "1"
    y = DataFrame(np.reshape(np.arange(10), (5, 2)))
    with pytest.raises(ValueError, match=r"y must be squeezable to 1 dimension"):
        ensure1d(y, "y")


def test_ensure2d():
    s = Series([1, 2, 3], name="x")
    df = ensure2d(s, "x")
    assert isinstance(df, DataFrame)

    df2 = ensure2d(df, "x")
    assert df is df2

    npa = ensure2d(s.values, "x")
    assert isinstance(npa, np.ndarray)
    assert npa.ndim == 2

    npa = ensure2d(np.array(1.0), "x")
    assert isinstance(npa, np.ndarray)
    assert npa.ndim == 2

    with pytest.raises(ValueError, match=r"Variable x must be 2d"):
        ensure2d(np.array([[[1]]]), "x")
    with pytest.raises(
        TypeError, match=r"Variable x must be a Series, DataFrame or ndarray"
    ):
        ensure2d([1], "x")


def test_parse_dataframe():
    s = Series(np.arange(10.0), name="variable")
    out = parse_dataframe(s, "y")
    assert_equal(out[1], np.arange(10.0))
    assert_equal(out[0], ["variable"])
    df = DataFrame(s)
    out = parse_dataframe(df, "y")
    assert_equal(out[1], np.arange(10.0))
    assert_equal(out[0], ["variable"])
    out = parse_dataframe(np.arange(10.0), "y")
    assert_equal(out[1], np.arange(10.0))
    assert_equal(out[0], ["y"])

    out = parse_dataframe(None, "name")
    assert out[0] == ["name"]
    assert isinstance(out[1], np.ndarray)
    assert out[1].shape == (0,)


def test_implicit_constant(rng):
    x = rng.standard_normal((1000, 2))
    assert not implicit_constant(x)
    x[:, 0] = 1.0
    assert implicit_constant(x)
    x = rng.standard_normal((1000, 3))
    x[:, 0] = x[:, 0] > 0
    x[:, 1] = 1 - x[:, 0]
    assert implicit_constant(x)


def test_docstring_inheritor():
    class A(metaclass=DocStringInheritor):
        """
        Docstring
        """

    class B(A):
        pass

    assert_equal(B.__doc__, A.__doc__)


def test_date_to_index():
    dr = date_range("20000101", periods=3000, freq="W")
    y = Series(np.arange(3000.0), index=dr)
    date_index = y.index

    index = date_to_index(date_index[0], date_index)
    assert_equal(index, 0)
    index = date_to_index(date_index[-1], date_index)
    assert_equal(index, date_index.shape[0] - 1)

    index = date_to_index("2009-08-02", date_index)
    assert_equal(index, 500)
    index = date_to_index("2009-08-04", date_index)
    assert_equal(index, 501)
    index = date_to_index("2009-08-01", date_index)
    assert_equal(index, 500)
    index = date_to_index(dt.datetime(2009, 8, 1), date_index)
    assert_equal(index, 500)
    with pytest.raises(ValueError, match=r"date must be a datetime"):
        date_to_index(dt.date(2009, 8, 1), date_index)
    z = y + 0.0
    z.index = np.arange(3000)
    num_index = z.index
    with pytest.raises(ValueError, match=r"date_index must be a datetime64 array"):
        date_to_index(dt.datetime(2009, 8, 1), num_index)
    idx = date_range("1999-12-31", periods=3)

    df = DataFrame([1, 2, 3], index=idx[::-1])
    with pytest.raises(ValueError, match=r"date_index is not monotonic and unique"):
        date_to_index(idx[0], df.index)

    df = DataFrame([1, 2, 3], index=[idx[0]] * 3)
    with pytest.raises(ValueError, match=r"date_index is not monotonic and unique"):
        date_to_index(idx[0], df.index)

    with pytest.raises(ValueError, match=r"date: NaT cannot be parsed to a date"):
        date_to_index("NaT", idx)

    #  check whether this also works for a localized datetimeindex
    date_index = date_range("20000101", periods=3000, freq="W", tz="Europe/Berlin")
    index = date_to_index(date_index[0], date_index)
    assert_equal(index, 0)


def test_date_to_index_timestamp():
    dr = date_range("20000101", periods=3000, freq="W")
    y = Series(np.arange(3000.0), index=dr)
    date_index = y.index
    date = y.index[1000]
    date_pydt = date.to_pydatetime()
    date_npdt = date.to_datetime64()
    date_str = date_pydt.strftime("%Y-%m-%d")

    index = date_to_index(date, date_index)
    index_pydt = date_to_index(date_pydt, date_index)
    index_npdt = date_to_index(date_npdt, date_index)
    index_str = date_to_index(date_str, date_index)
    assert_equal(index, 1000)
    assert_equal(index, index_npdt)
    assert_equal(index, index_pydt)
    assert_equal(index, index_str)


def test_():
    dr = date_range("20000101", periods=3000, freq="W")
    y = Series(np.arange(3000.0), index=dr)
    date_index = y.index

    date = date_index[1000] + Timedelta(1, "D")
    date_pydt = date.to_pydatetime()
    date_npdt = date.to_datetime64()
    date_str = date_pydt.strftime("%Y-%m-%d")
    index = date_to_index(date, date_index)
    index_pydt = date_to_index(date_pydt, date_index)
    index_npdt = date_to_index(date_npdt, date_index)
    index_str = date_to_index(date_str, date_index)
    assert_equal(index, 1001)
    assert_equal(index, index_npdt)
    assert_equal(index, index_pydt)
    assert_equal(index, index_str)

    date = date_index[0] - Timedelta(1, "D")
    index = date_to_index(date, date_index)
    assert_equal(index, 0)

    date_pydt = date.to_pydatetime()
    date_npdt = date.to_datetime64()
    date_str = date_pydt.strftime("%Y-%m-%d")
    index_pydt = date_to_index(date_pydt, date_index)
    index_npdt = date_to_index(date_npdt, date_index)
    index_str = date_to_index(date_str, date_index)
    assert_equal(index, index_npdt)
    assert_equal(index, index_pydt)
    assert_equal(index, index_str)


def test_cutoff_to_index():
    dr = date_range("20000101", periods=3000, freq="W")
    y = Series(np.arange(3000.0), index=dr)
    date_index = y.index
    assert cutoff_to_index(1000, date_index, 0) == 1000
    assert cutoff_to_index((1000), date_index, 0) == 1000
    assert cutoff_to_index(np.int16(1000), date_index, 0) == 1000
    assert cutoff_to_index(np.int64(1000), date_index, 0) == 1000
    assert cutoff_to_index(date_index[1000], date_index, 0) == 1000
    assert cutoff_to_index(None, date_index, 1000) == 1000


def test_find_index():
    index = date_range("2000-01-01", periods=5000)
    series = Series(np.arange(len(index)), index=index, name="test")
    df = DataFrame(series)
    assert_equal(find_index(series, "2000-01-01"), 0)
    assert_equal(find_index(series, series.index[0]), 0)
    assert_equal(find_index(series, series.index[3000]), 3000)
    assert_equal(find_index(series, series.index[3000].to_pydatetime()), 3000)
    npy_date = np.datetime64(series.index[3000].to_pydatetime())
    found_loc = find_index(series, npy_date)
    assert_equal(found_loc, 3000)
    with pytest.raises(ValueError, match=r"bad-date cannot"):
        find_index(series, "bad-date")
    with pytest.raises(ValueError, match=r"index not found"):
        find_index(series, "1900-01-01")

    assert_equal(find_index(df, "2000-01-01"), 0)
    assert_equal(find_index(df, df.index[0]), 0)
    assert_equal(find_index(df, df.index[3000]), 3000)
    assert_equal(find_index(df, df.index[3000].to_pydatetime()), 3000)
    found_loc = find_index(df, np.datetime64(df.index[3000].to_pydatetime()))
    assert_equal(found_loc, 3000)
    with pytest.raises(ValueError, match=r"bad-date cannot be converted to datetime"):
        find_index(df, "bad-date")
    with pytest.raises(ValueError, match=r"index not found"):
        find_index(df, "1900-01-01")

    idx = find_index(df, 1)
    assert idx == 1


def test_date_to_index_ndarray():
    dr = date_range("20000101", periods=3000, freq="W")
    y = Series(np.arange(3000.0), index=dr)
    date_index = np.asarray(y.index)

    index = date_to_index(date_index[0], date_index)
    assert_equal(index, 0)
    index = date_to_index(date_index[-1], date_index)
    assert_equal(index, date_index.shape[0] - 1)


def test_doc():
    doc()


def test_concrete_class_meta():
    with pytest.raises(TypeError, match=r"Dummy has not implemented abstrac"):

        class Dummy(metaclass=ConcreteClassMeta):
            @abstractmethod
            def func(self):
                pass


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1.0], dtype=float),
        np.array(0),
        np.array([0, 1]),
        np.array([[2, 3, 4]]),
        Series([1, 2, 3]),
    ],
)
def test_to_array_1d(arr):
    converted = to_array_1d(arr)
    assert isinstance(converted, np.ndarray)
    assert converted.ndim == 1
    assert converted.dtype == np.float64


def test_to_array_1d_err():
    with pytest.raises(ValueError, match=r"x must be 1D"):
        to_array_1d(np.array([[1, 2], [3, 4]]))
    with pytest.raises(TypeError, match=r"x must be a Series or ndarray"):
        to_array_1d(0)
    with pytest.raises(TypeError, match=r"x must be a Series or ndarray"):
        to_array_1d(DataFrame([[0, 1]]))
