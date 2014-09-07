"""
Utility functions that do not explicitly relate to Volatility modeling
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from pandas import DataFrame, Series
import pandas as pd

__all__ = ['ensure1d', 'parse_dataframe', 'DocStringInheritor', 'date_to_index']

deprecation_doc = """
{func} has been moved.  Please use {new_location}.{func}.
"""


def ensure1d(x, name, series=False):
    if isinstance(x, pd.Series):
        if not isinstance(x.name, str):
            x.name = str(x.name)
        if series:
            return x
        else:
            return np.asarray(x)

    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(name + ' must be squeezable to 1 dimension')
        else:
            x = pd.Series(x[x.columns[0]], x.index)
            if not isinstance(x.name, str):
                x.name = str(x.name)
        if series:
            return x
        else:
            return np.asarray(x)

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if x.ndim == 0:
        x=x[None]
    elif x.ndim != 1:
        x = np.squeeze(x)
        if x.ndim != 1:
            raise ValueError(name + ' must be squeezable to 1 dimension')

    if series:
        return pd.Series(x,name=name)
    else:
        return np.asarray(x)



def parse_dataframe(x, name):
    if x is None:
        return [name], np.empty(0)
    if isinstance(x, DataFrame):
        return x.columns, x.index
    elif isinstance(x, Series):
        return [x.name], x.index
    else:
        return [name], np.arange(np.squeeze(x).shape[0])


class DocStringInheritor(type):
    '''A variation on
    http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
    by Paul McGuire
    '''
    def __new__(meta, name, bases, clsdict):
        if not('__doc__' in clsdict and clsdict['__doc__']):
            for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
                doc=mro_cls.__doc__
                if doc:
                    clsdict['__doc__']=doc
                    break
        for attr, attribute in clsdict.items():
            if not attribute.__doc__:
                for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()
                                if hasattr(mro_cls, attr)):
                    doc=getattr(getattr(mro_cls,attr),'__doc__')
                    if doc:
                        attribute.__doc__=doc
                        break
        return type.__new__(meta, name, bases, clsdict)

def date_to_index(date, date_index):
    """
    Looks up a

    Parameters
    ----------
    date : string, datetime or datetime64
        Date to use when returning the index
    date_index : 1-d array of datetime64
        Index data containing datetime64 values

    Returns
    -------
    index : int
        Index location

    Notes
    -----
    Assumes dates are increasing and unique.

    Uses last value interpolation if a date is not in the series so that the
    value returned satisfies date_index[index] is the largest date less than or
    equal to date.
    """
    import datetime as dt
    import pandas as pd
    from pandas.core.common import is_datetime64_dtype

    if not is_datetime64_dtype(date_index):
        raise ValueError('date_index must be a datetime64 array')

    if not np.all((np.diff(date_index.values).astype(dtype=np.int64))>0):
        raise ValueError('date_index is not monotonic and unique')
    if not isinstance(date, (dt.datetime, np.datetime64, str)):
        raise ValueError("date must be a datetime, datetime64 or string")
    elif isinstance(date, dt.datetime):
        date = np.datetime64(date)
    elif isinstance(date, str):
        orig_date = date
        date = np.datetime64(pd.to_datetime(date, coerce=True))
        if date == pd.NaT:
            raise ValueError('date:' + orig_date +
                             ' cannot be parsed to a date.')

    date_index = np.asarray(date_index)

    locs = np.nonzero(date_index <= date)[0]
    if locs.shape[0] == 0:
        raise ValueError('All dates in date_index occur after date')
    else:
        return locs.max()
