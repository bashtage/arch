from abc import abstractmethod
import datetime as dt

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_equal
from pandas import DataFrame, DatetimeIndex, Series, Timedelta, Timestamp, date_range
from pandas.testing import assert_frame_equal, assert_index_equal
import pytest

from arch import arch_model, doc
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

# Pandas deprecations (the Pandas4Warning class only exists in recent pandas) must not
# be triggered by date_to_index
NO_DEPRECATIONS = ["error::DeprecationWarning", "error::FutureWarning"]
try:
    from pandas.errors import Pandas4Warning  # noqa: F401

    NO_DEPRECATIONS.append("error::pandas.errors.Pandas4Warning")
except ImportError:  # pragma: no cover
    pass

TIME_ZONES = [
    "UTC",
    "Europe/Berlin",
    "America/New_York",
    "Asia/Kolkata",
    "Pacific/Auckland",
    "Etc/GMT+5",
]


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


@pytest.mark.parametrize("series", [True, False], ids=["series", "ndarray"])
@pytest.mark.parametrize("name", [None, 1, ("a", "b"), "name"], ids=str)
def test_ensure1d_does_not_modify_series(series, name):
    y = Series(np.arange(5.0), name=name)
    ys = ensure1d(y, "y", series)
    assert y.name is name
    assert_equal(np.asarray(ys), np.arange(5.0))
    if series:
        assert isinstance(ys.name, str)


@pytest.mark.parametrize("series", [True, False], ids=["series", "ndarray"])
def test_ensure1d_does_not_modify_dataframe(series):
    df = DataFrame({0: np.arange(5.0)})
    original = df.copy()
    ensure1d(df, "y", series)
    assert df.columns.tolist() == [0]
    assert_frame_equal(df, original)


def test_ensure2d_does_not_modify_inputs():
    s = Series(np.arange(5.0))
    df = DataFrame({0: np.arange(5.0)})
    ensure2d(s, "x")
    ensure2d(df, "x")
    assert s.name is None
    assert df.columns.tolist() == [0]


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
    with pytest.raises(TypeError, match=r"x must be array_like with integer"):
        ensure2d(["a", "b"], "x")


INT_DTYPES = [np.int8, np.int32, np.int64, np.uint8, np.uint64, np.bool_, np.float32]


@pytest.mark.parametrize("dtype", INT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_ensure1d_float64_ndarray(dtype):
    x = np.array([1, 0, 1, 1, 0]).astype(dtype)
    out = ensure1d(x, "y")
    assert out.dtype == np.float64
    assert_equal(out, [1.0, 0.0, 1.0, 1.0, 0.0])
    assert x.dtype == dtype
    out = ensure1d(x[:, None], "y", series=True)
    assert isinstance(out, Series)
    assert out.dtype == np.float64


@pytest.mark.parametrize("data", [[1, 2, 3], (1, 2, 3), [1.0, 2.0, 3.0], 5, [[1], [2]]])
def test_ensure1d_array_like(data):
    out = ensure1d(data, "y")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.ndim == 1


def test_ensure1d_float64_is_not_copied():
    x = np.arange(5.0)
    assert np.shares_memory(ensure1d(x, "y"), x)
    s = Series(x, name="y")
    assert ensure1d(s, "y", series=True) is s
    df = DataFrame({"y": x})
    assert np.shares_memory(ensure1d(df, "y"), df.to_numpy())


def test_ensure1d_pandas_keeps_labels():
    index = date_range("2000-1-1", periods=4, freq="D", name="date")
    s = Series([1, 2, 3, 4], index=index, name="count")
    out = ensure1d(s, "y", series=True)
    assert out.dtype == np.float64
    assert out.name == "count"
    assert_index_equal(out.index, index)
    assert s.dtype == np.int64
    df = DataFrame({"count": [1, 2, 3, 4]}, index=index)
    out = ensure1d(df, "y", series=True)
    assert out.dtype == np.float64
    assert out.name == "count"
    assert_index_equal(out.index, index)
    assert df["count"].dtype == np.int64


def test_ensure1d_pandas_extension_dtypes():
    s = Series([1, 2, 3], dtype="Int64")
    out = ensure1d(s, "y")
    assert out.dtype == np.float64
    assert_equal(out, [1.0, 2.0, 3.0])
    b = Series([True, False, True], dtype="boolean")
    assert_equal(ensure1d(b, "y"), [1.0, 0.0, 1.0])


def test_ensure1d_object():
    out = ensure1d(np.array([1, 2.5, 3], dtype=object), "y")
    assert out.dtype == np.float64
    assert_equal(out, [1.0, 2.5, 3.0])


@pytest.mark.parametrize(
    "data",
    [
        ["a", "b"],
        np.array([1 + 1j, 2]),
        np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"),
        np.array([1, "a"], dtype=object),
        Series(["a", "b"]),
        DataFrame({"a": ["a", "b"]}),
        Series([1, 2], dtype="category"),
    ],
    ids=str,
)
def test_ensure1d_not_numeric(data):
    with pytest.raises(TypeError, match=r"^my_data must .*(float64|real numbers)"):
        ensure1d(data, "my_data")


def test_ensure1d_error_messages():
    with pytest.raises(TypeError, match=r"data type <U1"):
        ensure1d(["a", "b"], "y")
    with pytest.raises(ValueError, match=r"y must be squeezable.*shape \(2, 2\)"):
        ensure1d(np.ones((2, 2)), "y")
    with pytest.raises(ValueError, match=r"y must be squeezable.*shape \(3, 2\)"):
        ensure1d(DataFrame(np.ones((3, 2))), "y")


@pytest.mark.parametrize("dtype", INT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_ensure2d_float64(dtype):
    x = np.array([[1, 0], [0, 1], [1, 1]]).astype(dtype)
    out = ensure2d(x, "x")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert_equal(out, x.astype(float))
    out = ensure2d(DataFrame(x, columns=["a", "b"], index=[3, 4, 5]), "x")
    assert isinstance(out, DataFrame)
    assert (out.dtypes == np.float64).all()
    assert out.columns.tolist() == ["a", "b"]
    assert out.index.tolist() == [3, 4, 5]
    out = ensure2d(Series(x[:, 0], name="a"), "x")
    assert out.dtypes.tolist() == [np.float64]
    assert out.columns.tolist() == ["a"]


@pytest.mark.parametrize(
    ("data", "shape"),
    [([1, 2, 3], (3, 1)), ([[1, 2], [3, 4]], (2, 2)), (4, (1, 1)), ((1, 2), (2, 1))],
)
def test_ensure2d_array_like(data, shape):
    out = ensure2d(data, "x")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == shape


def test_ensure2d_float64_is_not_copied():
    x = np.arange(6.0).reshape(3, 2)
    assert ensure2d(x, "x") is x
    df = DataFrame(x)
    assert ensure2d(df, "x") is df
    assert np.shares_memory(ensure2d(x[:, 0], "x"), x)


def test_ensure2d_errors():
    with pytest.raises(TypeError, match=r"^my_x must be array_like.*data type <U1"):
        ensure2d([["a", "b"]], "my_x")
    with pytest.raises(TypeError, match=r"^my_x must contain.*DataFrame"):
        ensure2d(DataFrame({"a": ["a", "b"]}), "my_x")
    with pytest.raises(ValueError, match=r"my_x must be 2d.*3 dimensions.*\(1, 1, 1\)"):
        ensure2d(np.ones((1, 1, 1)), "my_x")


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


@pytest.mark.filterwarnings(*NO_DEPRECATIONS)
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

    # A naive Timestamp gives the same result as the equivalent string
    utc_index = date_range("20000101", periods=3000, freq="W", tz="UTC")
    index = date_to_index(Timestamp("2009-08-02"), utc_index)
    assert_equal(index, date_to_index("2009-08-02", utc_index))
    assert_equal(index, 500)
    # A Series of dates is accepted for Timestamps as it is for strings
    index = date_to_index(Timestamp("2009-08-02"), Series(dr))
    assert_equal(index, 500)


@pytest.mark.filterwarnings(*NO_DEPRECATIONS)
@pytest.mark.parametrize("tz", TIME_ZONES)
def test_date_to_index_tz_aware(tz):
    # The instants are fixed in UTC, and so are the dates, which are naive or
    # carry a time zone. The positions are found by hand: the position of the date
    # if it is in the index, otherwise the position where it would be inserted.
    utc = [
        "2021-03-13 12:00",
        "2021-03-14 12:00",
        "2021-03-15 12:00",
        "2021-03-16 12:00",
        "2021-03-17 12:00",
    ]
    date_index = DatetimeIndex(utc).tz_localize("UTC").tz_convert(tz)
    assert date_index.tz is not None
    expected = {
        "2021-03-12": 0,
        "2021-03-13 11:59": 0,
        "2021-03-13 12:00": 0,
        "2021-03-13 12:01": 1,
        "2021-03-14": 1,
        "2021-03-14 12:00": 1,
        "2021-03-14 12:01": 2,
        "2021-03-16 12:00": 3,
        "2021-03-17 11:59": 4,
        "2021-03-17 12:00": 4,
        "2021-03-18": 5,
    }
    for date, position in expected.items():
        assert date_to_index(date, date_index) == position
        assert date_to_index(Timestamp(date), date_index) == position
        assert date_to_index(np.datetime64(date), date_index) == position
        assert date_to_index(Timestamp(date).to_pydatetime(), date_index) == position
        # A Timestamp with a time zone is converted to UTC
        local = Timestamp(date).tz_localize("UTC").tz_convert(tz)
        assert date_to_index(local, date_index) == position


@pytest.mark.filterwarnings(*NO_DEPRECATIONS)
def test_date_to_index_tz_aware_timestamp_other_zone():
    utc = ["2021-03-13 12:00", "2021-03-14 12:00", "2021-03-15 12:00"]
    date_index = DatetimeIndex(utc).tz_localize("UTC").tz_convert("Europe/Berlin")
    # 08:00 in New York is 12:00 UTC on 2021-03-14, since the DST change
    # happened at 02:00 that day
    assert (
        date_to_index(Timestamp("2021-03-14 08:00", tz="America/New_York"), date_index)
        == 1
    )
    # 13:00 in Kolkata (UTC+5:30) is 07:30 UTC on 2021-03-14
    assert (
        date_to_index(Timestamp("2021-03-14 13:00", tz="Asia/Kolkata"), date_index) == 1
    )
    # 18:00 in Kolkata is 12:30 UTC on 2021-03-14
    assert (
        date_to_index(Timestamp("2021-03-14 18:00", tz="Asia/Kolkata"), date_index) == 2
    )


@pytest.mark.filterwarnings(*NO_DEPRECATIONS)
def test_date_to_index_tz_aware_not_monotonic():
    utc = ["2021-03-13 12:00", "2021-03-14 12:00", "2021-03-14 12:00"]
    date_index = DatetimeIndex(utc).tz_localize("UTC").tz_convert("Asia/Kolkata")
    with pytest.raises(ValueError, match=r"date_index is not monotonic and unique"):
        date_to_index("2021-03-14", date_index)
    date_index = DatetimeIndex(utc[::-1]).tz_localize("UTC").tz_convert("Asia/Kolkata")
    with pytest.raises(ValueError, match=r"date_index is not monotonic and unique"):
        date_to_index("2021-03-14", date_index)


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


def test_arch_model_does_not_rename_series():
    y = Series(np.random.default_rng(0).standard_normal(100))
    arch_model(y)
    assert y.name is None
