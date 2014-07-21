"""
Utility functions that do not explicitly relate to Volatility modeling
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from pandas import DataFrame, Series
import pandas as pd

__all__ = ['ensure1d', 'parse_dataframe', 'DocStringInheritor']

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
