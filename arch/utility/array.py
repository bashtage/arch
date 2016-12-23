"""
Utility functions that do not explicitly relate to Volatility modeling
"""
from __future__ import print_function, division, absolute_import
from ..compat.pandas import is_datetime64_dtype
from ..compat.python import long

import datetime as dt

import numpy as np
from pandas import DataFrame, Series, to_datetime, NaT, Timestamp

__all__ = ['ensure1d', 'parse_dataframe', 'DocStringInheritor',
           'date_to_index']

deprecation_doc = """
{func} has been moved.  Please use {new_location}.{func}.
"""


def ensure1d(x, name, series=False):
    if isinstance(x, Series):
        if not isinstance(x.name, str):
            x.name = str(x.name)
        if series:
            return x
        else:
            return np.asarray(x)

    if isinstance(x, DataFrame):
        if x.shape[1] != 1:
            raise ValueError(name + ' must be squeezable to 1 dimension')
        else:
            x = Series(x[x.columns[0]], x.index)
            if not isinstance(x.name, str):
                x.name = str(x.name)
        if series:
            return x
        else:
            return np.asarray(x)

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if x.ndim == 0:
        x = x[None]
    elif x.ndim != 1:
        x = np.squeeze(x)
        if x.ndim != 1:
            raise ValueError(name + ' must be squeezable to 1 dimension')

    if series:
        return Series(x, name=name)
    else:
        return np.asarray(x)


def ensure2d(x, name):
    if isinstance(x, Series):
        return DataFrame(x)
    elif isinstance(x, DataFrame):
        return x
    elif isinstance(x, np.ndarray):
        if x.ndim == 0:
            return np.asarray([[x]])
        elif x.ndim == 1:
            return x[:, None]
        elif x.ndim == 2:
            return x
        else:
            raise ValueError(
                'Variable ' + name + 'must be 2d or reshapeable to 2d')
    else:
        raise ValueError('Variable ' + name + 'must be 2d or ' +
                         'reshapeable to 2d')


def parse_dataframe(x, name):
    if x is None:
        return [name], np.empty(0)
    if isinstance(x, DataFrame):
        return x.columns, x.index
    elif isinstance(x, Series):
        return [x.name], x.index
    else:
        if not isinstance(name, list):
            name = [name]
        return name, np.arange(np.squeeze(x).shape[0])


class DocStringInheritor(type):
    """
    A variation on
    http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
    by Paul McGuire
    """

    def __new__(mcs, name, bases, clsdict):
        if not ('__doc__' in clsdict and clsdict['__doc__']):
            for mro_cls in (mro_cls for base in bases for mro_cls in
                            base.mro()):
                doc = mro_cls.__doc__
                if doc:
                    clsdict['__doc__'] = doc
                    break
        for attr, attribute in clsdict.items():
            if not attribute.__doc__:
                for mro_cls in (mro_cls for base in bases for mro_cls in
                                base.mro()
                                if hasattr(mro_cls, attr)):
                    doc = getattr(getattr(mro_cls, attr), '__doc__')
                    if doc:
                        attribute.__doc__ = doc
                        break
        return type.__new__(mcs, name, bases, clsdict)


def date_to_index(date, date_index):
    """
    Looks up a date in an array of dates

    Parameters
    ----------
    date : string, datetime or datetime64
        Date to use when returning the index
    date_index : array
        Index data containing datetime64 values

    Returns
    -------
    index : int
        Index location

    Notes
    -----
    Assumes dates are increasing and unique.
    """
    if not is_datetime64_dtype(date_index):
        raise ValueError('date_index must be a datetime64 array')

    if not np.all((np.diff(date_index.values).astype(dtype=np.int64)) > 0):
        raise ValueError('date_index is not monotonic and unique')
    if not isinstance(date, (dt.datetime, np.datetime64, str)):
        raise ValueError("date must be a datetime, datetime64 or string")
    elif isinstance(date, dt.datetime):
        date = np.datetime64(date)
    elif isinstance(date, str):
        orig_date = date
        try:
            date = np.datetime64(to_datetime(date, errors='coerce'))
        except:
            date = np.datetime64(to_datetime(date, coerce=True))
        if date == NaT:
            raise ValueError('date:' + orig_date +
                             ' cannot be parsed to a date.')

    date_index = np.asarray(date_index)

    locs = np.nonzero(date_index <= date)[0]
    if locs.shape[0] == 0:
        return 0

    loc = locs.max()
    in_array = np.any(date_index == date)
    if not in_array:
        loc += 1

    return loc


def cutoff_to_index(cutoff, index, default):
    """
    Converts a cutoff to a numerical index

    Parameters
    ----------
    cutoff : {None, str, datetime, datetime64, Timestamp)
        The cutoff point to use
    index : Pandas index
        Pandas index
    default : int
        The value to return if cutoff is None

    Returns
    -------
    val : int
        Integer value of
    """
    int_index = default
    if isinstance(cutoff, (str, dt.datetime, np.datetime64, Timestamp)):
        int_index = date_to_index(cutoff, index)
    elif isinstance(cutoff, (int, long)) or issubclass(cutoff.__class__,
                                                       np.integer):
        int_index = cutoff

    return int_index


def find_index(s, index):
    """
    Returns the numeric index for a string or datetime

    Parameters
    ----------
    s : Series or DataFrame
        Series or DataFrame to use in lookup
    index : datetime-like, str
        Index value, either a string convertible to a datetime or a datetime

    Returns
    -------
    loc : int
        Integer location of index value
    """
    if isinstance(index, (int, long, np.int, np.int64)):
        return index
    try:
        date_index = to_datetime(index, errors='coerce')
    except:
        date_index = to_datetime(index, coerce=True)

    if date_index is NaT:
        raise ValueError(index + ' cannot be converted to datetime')
    loc = np.argwhere(s.index == date_index).squeeze()
    if loc.size == 0:
        raise ValueError('index not found')
    return loc
